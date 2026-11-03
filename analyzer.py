from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks, resample_poly, welch

import config
import ia_manager
import ollama_ai
from common import (InstanceLock, JobCancelledError, ProgressWriter, atomic_write_json, cache_key,
                    check_ffmpeg, ensure_directories, ensure_free_space, priority_kwargs,
                    probe_media, read_json, run_ffmpeg_progress, run_low_priority,
                    terminate_process_tree,
                    setup_logging)

LOG = setup_logging()


def _ideal_clip_count(duration_seconds: float) -> int:
    """Define volume de saída automaticamente: 5 para curtas, até 10 para uma hora+."""
    minimum, maximum = 5, 10
    proportion = min(1.0, max(0.0, duration_seconds / 3600.0))
    return int(round(minimum + (maximum - minimum) * proportion))


def _extract_audio(source: Path, output: Path, duration: float | None = None,
                   progress: ProgressWriter | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-vn", "-ac", "1", "-ar", str(config.ANALYSIS["sample_rate"]),
               "-af", f"highpass=f={config.ANALYSIS['highpass_hz']}", "-c:a", "pcm_s16le", str(output)]
    if progress is not None and duration:
        run_ffmpeg_progress(command, duration, progress, stage="áudio", start_percent=6, end_percent=22)
    else:
        result = run_low_priority(command, timeout=None)
        if result.returncode:
            raise RuntimeError("Falha ao extrair áudio: " + (result.stderr or "erro FFmpeg")[-1800:])


def _audio_features(wav_path: Path, progress: ProgressWriter | None = None) -> tuple[list[dict[str, float]], float]:
    """Gera features por segundo lendo o WAV em blocos; não lê a trilha completa para RAM."""
    rows: list[dict[str, float]] = []
    rate = config.ANALYSIS["sample_rate"]
    frames = rate * int(config.ANALYSIS["audio_block_seconds"])
    with sf.SoundFile(str(wav_path)) as audio:
        seconds = 0
        while True:
            if progress is not None:
                progress.check_cancelled()
            block = audio.read(frames, dtype="float32", always_2d=True)
            if not len(block):
                break
            x = np.mean(block, axis=1, dtype=np.float32)
            rms = float(np.sqrt(np.mean(x * x, dtype=np.float64) + 1e-12))
            peak = float(np.max(np.abs(x))) if x.size else 0.0
            crest = peak / max(rms, 1e-8)
            if x.size >= 128 and rms > 1e-8:
                freqs, psd = welch(x, fs=rate, nperseg=min(2048, x.size), noverlap=0)
                total = float(np.sum(psd)) + 1e-12
                low = float(np.sum(psd[freqs < 150]) / total)
                high = float(np.sum(psd[freqs >= 3000]) / total)
                mid = float(np.sum(psd[(freqs >= 150) & (freqs < 3000)]) / total)
                centroid = float(np.sum(freqs * psd) / total)
            else:
                low, mid, high, centroid = 0.0, 0.0, 0.0, 0.0
            rows.append({"second": float(seconds), "rms": rms, "peak": peak, "crest": crest,
                         "low_ratio": low, "mid_ratio": mid, "high_ratio": high,
                         "centroid": centroid})
            seconds += 1
    if not rows:
        raise RuntimeError("O áudio não contém amostras para análise.")
    return rows, float(len(rows))


def _detect_speech_seconds(wav_path: Path, video_duration: float,
                           progress: ProgressWriter | None = None) -> np.ndarray | None:
    """Roda Silero VAD local por blocos curtos; retorna cobertura de fala em cada segundo."""
    if not config.SCENES.get("speech_vad_enabled", True):
        return None
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        rate = int(config.ANALYSIS["sample_rate"])
        coverage = np.zeros(max(1, int(math.ceil(video_duration))), dtype=np.float32)
        options = VadOptions(
            threshold=float(config.SCENES["vad_threshold"]),
            min_speech_duration_ms=int(config.SCENES["vad_min_speech_duration_ms"]),
            min_silence_duration_ms=int(config.SCENES["vad_min_silence_duration_ms"]),
            speech_pad_ms=int(config.SCENES["vad_speech_pad_ms"]),
        )
        chunk_frames = rate * int(config.SCENES["vad_chunk_seconds"])
        with sf.SoundFile(str(wav_path)) as audio:
            total_frames = max(1, int(audio.frames))
            offset_seconds = 0.0
            while True:
                if progress is not None:
                    progress.check_cancelled()
                block = audio.read(chunk_frames, dtype="float32", always_2d=True)
                if not len(block):
                    break
                mono = np.mean(block, axis=1, dtype=np.float32)
                intervals = get_speech_timestamps(mono, vad_options=options, sampling_rate=rate)
                block_duration = len(mono) / rate
                for interval in intervals:
                    start = offset_seconds + float(interval["start"]) / rate
                    end = offset_seconds + float(interval["end"]) / rate
                    for second in range(max(0, int(start)), min(len(coverage), int(math.ceil(end)))):
                        overlap = max(0.0, min(end, second + 1.0) - max(start, float(second)))
                        coverage[second] = max(coverage[second], float(overlap))
                offset_seconds += block_duration
                if progress is not None:
                    ratio = min(1.0, (audio.tell() / total_frames))
                    progress.update("fala", 22 + 4 * ratio,
                                    f"VAD local: {int(audio.tell() / rate)} / {int(total_frames / rate)}s",
                                    process_pid=None)
        return coverage
    except (JobCancelledError, KeyboardInterrupt):
        raise
    except Exception as exc:
        LOG.warning("VAD local indisponível; usando análise espectral conservadora: %s", exc)
        return None


def _speech_probe_candidates(vad_seconds: np.ndarray | None, duration: float) -> list[dict[str, Any]]:
    """Reserva um probe visual na sequência de fala mais longa de cada partição temporal."""
    if vad_seconds is None or not len(vad_seconds):
        return []
    interval = float(config.SCENES["probe_interval_seconds"])
    zones = min(int(config.SCENES["probe_limit"]), max(1, int(math.ceil(duration / max(1.0, interval)))))
    result = []
    min_run = int(config.SCENES["speech_min_seconds"])
    for zone in range(zones):
        lo = int(duration * zone / zones)
        hi = min(len(vad_seconds), int(math.ceil(duration * (zone + 1) / zones)))
        active = np.asarray(vad_seconds[lo:hi] >= 0.20, dtype=np.bool_)
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for offset, voiced in enumerate(active):
            if voiced and run_start is None:
                run_start = offset
            elif not voiced and run_start is not None:
                if offset - run_start >= min_run:
                    runs.append((run_start, offset))
                run_start = None
        if run_start is not None and len(active) - run_start >= min_run:
            runs.append((run_start, len(active)))
        if not runs:
            continue
        local_start, local_end = max(runs, key=lambda span: span[1] - span[0])
        peak = lo + (local_start + local_end) / 2.0
        result.append({"peak": min(max(0.0, peak), max(0.0, duration - 0.01)),
                       "audio_score": 0.0, "event_type": "impacto", "confidence": 0.30,
                       "classification_reason": "probe visual de fala contínua confirmada pelo VAD",
                       "local": [], "raw_score": 0.0, "score": 0.0,
                       "scene_score": 0.0, "movement_score": 0.0,
                       "red_flash": 0.0, "brightness": 0.0, "discard_reason": ""})
    return result


def _mel_filterbank(rate: int, n_fft: int, bands: int = 32) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([rate / 2]))[0], bands + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / rate).astype(int)
    bank = np.zeros((bands, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, bands + 1):
        left, center, right = bins[m - 1:m + 2]
        for k in range(max(0, left), min(center, bank.shape[1])):
            bank[m - 1, k] = (k - left) / max(1, center - left)
        for k in range(max(0, center), min(right, bank.shape[1])):
            bank[m - 1, k] = (right - k) / max(1, right - center)
    return bank


def _logmel_vector(samples: np.ndarray, rate: int) -> np.ndarray:
    from scipy.signal import stft
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if rate != config.ANALYSIS["sample_rate"]:
        from math import gcd
        divisor = gcd(rate, config.ANALYSIS["sample_rate"])
        x = resample_poly(x, config.ANALYSIS["sample_rate"] // divisor, rate // divisor).astype(np.float32)
        rate = config.ANALYSIS["sample_rate"]
    if x.size < 512:
        x = np.pad(x, (0, 512 - x.size))
    _, _, spec = stft(x, fs=rate, nperseg=512, noverlap=384, boundary=None)
    power = np.abs(spec) ** 2
    mel = _mel_filterbank(rate, 512, 32) @ power
    vector = np.log(np.mean(mel, axis=1) + 1e-10).astype(np.float32)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


def _load_signatures(directory: Path) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    if not directory.exists():
        return signatures
    for path in sorted(directory.glob("*.wav")):
        try:
            info = sf.info(str(path))
            frames = min(info.frames, int(info.samplerate * 3.0))
            samples, rate = sf.read(str(path), start=0, frames=frames, dtype="float32", always_2d=True)
            mono = np.mean(samples, axis=1, dtype=np.float32)
            category = path.stem.lower().split("_")[0]
            aliases = {"parry": "parry_faca", "faca": "parry_faca", "boss": "boss_combate",
                       "grito": "susto_grito", "tiro": "pistola_rifle", "serra": "motosserra"}
            category = aliases.get(category, category)
            signatures.append({"name": path.name, "category": category,
                               "vector": _logmel_vector(mono, rate)})
        except Exception as exc:
            LOG.warning("Assinatura ignorada (%s): %s", path.name, exc)
    return signatures


def _read_audio_window(wav_path: Path, center_sec: float, duration: float = 3.0) -> np.ndarray:
    rate = config.ANALYSIS["sample_rate"]
    start = max(0, int((center_sec - duration / 2) * rate))
    frames = int(duration * rate)
    data, _ = sf.read(str(wav_path), start=start, frames=frames, dtype="float32", always_2d=True)
    return np.mean(data, axis=1, dtype=np.float32) if data.size else np.zeros(512, dtype=np.float32)


def _classify_event(wav_path: Path | None, peak_sec: float, local: list[dict[str, float]],
                    signatures: list[dict[str, Any]], visual: dict[str, float] | None = None) -> tuple[str, float, str]:
    if local:
        rms = max(row["rms"] for row in local)
        crest = max(row["crest"] for row in local)
        low = float(np.mean([row["low_ratio"] for row in local]))
        high = float(np.mean([row["high_ratio"] for row in local]))
        centroid = float(np.mean([row["centroid"] for row in local]))
        sustained = sum(row["rms"] > 0.03 for row in local)
    else:
        rms, crest, low, high, centroid, sustained = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    if wav_path and signatures:
        vector = _logmel_vector(_read_audio_window(wav_path, peak_sec), config.ANALYSIS["sample_rate"])
        matches = []
        for signature in signatures:
            similarity = float(np.dot(vector, signature["vector"]))
            matches.append((similarity, signature["category"], signature["name"]))
        if matches:
            similarity, category, name = max(matches)
            if similarity >= 0.66 and category in config.EVENT_WEIGHTS:
                return category, float(np.clip(similarity, 0.0, 1.0)), f"assinatura log-mel: {name} ({similarity:.2f})"
    # Áudio grave sustentado é ambíguo (motor, música, vento, explosão); não afirma objeto.
    if sustained >= 2 and low >= 0.35 and centroid < 1300:
        return "impacto", 0.30, "grave sustentado; evento/objeto específico não confirmado"
    if crest >= 7.0 and low >= 0.18 and high >= 0.12:
        return "escopeta", 0.52, "transiente forte e banda larga (heurística)"
    if crest >= 5.5 and high >= 0.35:
        return "pistola_rifle", 0.48, "transiente curto agudo (heurística)"
    if low >= 0.48 and rms > 0.05:
        return "explosao", 0.45, "energia grave intensa (heurística)"
    if visual and visual.get("red_flash", 0.0) >= 0.10 and visual.get("scene", 0.0) >= 0.18:
        return "impacto", 0.42, "flash visual + mudança de cena; tipo inconclusivo"
    if high >= 0.30 and rms > 0.05:
        return "susto_grito", 0.32, "energia aguda, possível voz/alarme (heurística fraca)"
    return ("impacto", 0.28, "evento de áudio sem assinatura conclusiva") if rms > 0.025 else ("morno", 0.15, "energia baixa")


def _hud_score(luma: np.ndarray, previous: np.ndarray | None) -> float:
    """Estimativa leve: HUD tende a ter bordas persistentes nos cantos enquanto o centro muda."""
    if previous is None:
        return 0.0
    height, width = luma.shape
    corner_h, corner_w = max(8, int(height * 0.22)), max(8, int(width * 0.22))
    regions = [
        (slice(0, corner_h), slice(0, corner_w)),
        (slice(0, corner_h), slice(width - corner_w, width)),
        (slice(height - corner_h, height), slice(0, corner_w)),
        (slice(height - corner_h, height), slice(width - corner_w, width)),
    ]
    edge_densities, corner_motion = [], []
    delta = np.abs(luma - previous)
    for ys, xs in regions:
        region = luma[ys, xs]
        gx = np.abs(np.diff(region, axis=1))
        gy = np.abs(np.diff(region, axis=0))
        edge_densities.append(float((np.count_nonzero(gx > 18) + np.count_nonzero(gy > 18)) /
                                    max(1, gx.size + gy.size)))
        corner_motion.append(float(np.mean(delta[ys, xs]) / 255.0))
    cy0, cy1 = int(height * 0.2), int(height * 0.8)
    cx0, cx1 = int(width * 0.2), int(width * 0.8)
    center_motion = float(np.mean(delta[cy0:cy1, cx0:cx1]) / 255.0)
    corner_mean = float(np.mean(corner_motion))
    persistent_edges = float(np.mean(sorted(edge_densities)[-2:]))
    activity = float(np.clip(center_motion / 0.08, 0.0, 1.0))
    stability = float(np.clip(1.0 - corner_mean / (center_motion + 0.02), 0.0, 1.0))
    return float(np.clip((persistent_edges / 0.08) * activity * stability, 0.0, 1.0))


def _scan_visual(source: Path, start: float, duration: float,
                 progress: ProgressWriter | None = None,
                 progress_start: float = 28, progress_end: float = 75,
                 stage: str = "visual") -> dict[int, dict[str, float]]:
    """Decodifica RGB 320x180 a 5fps e agrega por segundo, sem acumular quadros."""
    width = int(config.ANALYSIS["visual_width"])
    height = int(round(width * 9 / 16))
    fps = int(config.ANALYSIS["visual_fps"])
    frame_bytes = width * height * 3
    vf = f"fps={fps},scale={width}:{height}:flags=fast_bilinear,format=rgb24"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}",
               "-i", str(source), "-t", f"{max(0.2, duration):.3f}", "-an", "-vf", vf,
               "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                **priority_kwargs())
    if progress is not None:
        progress.update(stage, progress_start, f"Decodificando quadros de {start:.1f}s a {start+duration:.1f}s.",
                        process_pid=process.pid)
    # [soma_luma, max_red_flash, soma_motion, contagem, soma_hud, cortes];
    # uso de memória constante por segundo, mesmo em gravações de várias horas.
    acc: dict[int, list[float]] = {}
    previous = None
    index = 0
    try:
        while True:
            if progress is not None:
                progress.check_cancelled()
            raw = process.stdout.read(frame_bytes) if process.stdout else b""
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            luma = frame[:, :, 0].astype(np.float32) * 0.2126 + frame[:, :, 1].astype(np.float32) * 0.7152 + frame[:, :, 2].astype(np.float32) * 0.0722
            brightness = float(luma.mean() / 255.0)
            red = float(np.mean((frame[:, :, 0] > 125) & (frame[:, :, 0] > frame[:, :, 1] * 1.35) & (frame[:, :, 0] > frame[:, :, 2] * 1.2)))
            motion = float(np.mean(np.abs(luma - previous)) / 255.0) if previous is not None else 0.0
            hud = _hud_score(luma, previous)
            cut = float(previous is not None and motion >= float(config.SCENES["camera_cut_diff_threshold"]))
            second = int(index / fps)
            bucket = acc.setdefault(second, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            bucket[0] += brightness
            bucket[1] = max(bucket[1], red)
            bucket[2] += motion
            bucket[3] += 1
            bucket[4] += hud
            bucket[5] += cut
            previous = luma
            index += 1
            if progress is not None and index % max(1, fps * 30) == 0:
                ratio = min(1.0, (index / fps) / max(1.0, duration))
                progress.update(stage, progress_start + ratio * (progress_end - progress_start),
                                f"Quadros analisados: {int(index / fps)} / {int(duration)}s",
                                process_pid=process.pid)
    except (JobCancelledError, KeyboardInterrupt):
        terminate_process_tree(process.pid)
        if progress is not None:
            progress.update("cancelado", progress.last_percent, "Varredura visual encerrada com segurança.",
                            done=True, cancelled=True, process_pid=None)
        raise
    finally:
        if process.poll() is None:
            terminate_process_tree(process.pid)
            process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
    code = process.returncode
    if code:
        raise RuntimeError("Análise visual FFmpeg incompleta (código %s): %s" % (code, stderr[-1200:]))
    result: dict[int, dict[str, float]] = {}
    prior_luma: float | None = None
    for second in sorted(acc):
        sums = acc[second]
        brightness = float(sums[0] / max(1.0, sums[3]))
        motion = float(sums[2] / max(1.0, sums[3]))
        scene = min(1.0, motion * 2.5 + (abs(brightness - prior_luma) if prior_luma is not None else 0.0))
        result[int(start) + second] = {"brightness": brightness, "red_flash": float(sums[1]),
                                       "motion": motion, "scene": scene,
                                       "hud_score": float(sums[4] / max(1.0, sums[3])),
                                       "camera_cuts": float(sums[5])}
        prior_luma = brightness
    return result


def _visual_at(timeline: dict[int, dict[str, float]], center: float, radius: float = 3.0) -> dict[str, float]:
    values = [timeline[t] for t in range(max(0, int(center - radius)), int(center + radius) + 1) if t in timeline]
    if not values:
        return {"brightness": 0.0, "red_flash": 0.0, "motion": 0.0, "scene": 0.0,
                "hud_score": 0.0, "camera_cuts": 0.0}
    return {
        "brightness": float(np.mean([v["brightness"] for v in values])),
        "red_flash": float(max(v["red_flash"] for v in values)),
        "motion": float(np.mean([v["motion"] for v in values])),
        "scene": float(max(v["scene"] for v in values)),
        "hud_score": float(np.mean([v.get("hud_score", 0.0) for v in values])),
        "camera_cuts": float(sum(v.get("camera_cuts", 0.0) for v in values)),
    }


def _align_audio_rows(rows: list[dict[str, float]], video_duration: float) -> list[dict[str, float]]:
    """Pad/trunca features para que o eixo de áudio sempre cubra o eixo do vídeo."""
    target = max(1, int(math.ceil(video_duration)))
    aligned = list(rows[:target])
    silent = {"rms": 0.0, "peak": 0.0, "crest": 0.0, "low_ratio": 0.0,
              "mid_ratio": 0.0, "high_ratio": 0.0, "centroid": 0.0}
    while len(aligned) < target:
        aligned.append({"second": float(len(aligned)), **silent})
    return aligned


def _make_peak_candidates(audio_rows: list[dict[str, float]], wav_path: Path | None,
                          signatures: list[dict[str, Any]],
                          video_duration: float | None = None) -> list[dict[str, Any]]:
    rms = np.asarray([row["rms"] for row in audio_rows], dtype=np.float32)
    if not rms.size:
        return []
    kernel_size = min(3, len(rms))
    smoothed = np.convolve(rms, np.ones(kernel_size, dtype=np.float32) / kernel_size, mode="same")[:len(rms)]
    duration = float(video_duration or len(rms))
    median = float(np.median(smoothed))
    mad = float(np.median(np.abs(smoothed - median)))
    robust = median + config.ANALYSIS["adaptive_threshold_mad"] * max(mad, 1e-5)
    threshold = max(robust, float(np.percentile(smoothed, 90)) * 0.55, 0.005)
    distance = int(config.ANALYSIS["peak_distance_seconds"])
    peaks, _ = find_peaks(smoothed, height=threshold, distance=max(1, distance), prominence=max(0.003, mad))
    if len(peaks) < max(5, config.ANALYSIS["clip_target_count"]):
        fallback, _ = find_peaks(smoothed, distance=max(1, distance), prominence=max(0.001, mad * 0.5))
        peaks = np.unique(np.r_[peaks, fallback])

    # Além dos transientes, acrescenta o máximo local de cada zona. Isso impede
    # que centenas de tiros no começo eliminem todo candidato do último ato.
    bins = min(120, max(10, int(math.ceil(duration / 60.0))))
    zone_peaks = []
    for zone in range(bins):
        lo = min(len(smoothed) - 1, int(zone * len(smoothed) / bins))
        hi = max(lo + 1, min(len(smoothed), int(math.ceil((zone + 1) * len(smoothed) / bins))))
        local_peak = lo + int(np.argmax(smoothed[lo:hi]))
        if smoothed[local_peak] >= max(0.002, threshold * 0.35):
            zone_peaks.append(local_peak)
    if len(peaks) == 0 and not zone_peaks:
        return []
    peaks = np.unique(np.r_[peaks, zone_peaks])
    global_p95 = max(float(np.percentile(smoothed, 95)), 1e-6)
    raw_candidates = []
    for peak_idx in peaks:
        peak_sec = min(duration, float(peak_idx) + 0.5)
        if peak_sec >= duration:
            peak_sec = max(0.0, duration - 0.01)
        local = audio_rows[max(0, peak_idx - 2):min(len(audio_rows), peak_idx + 3)]
        raw_audio = float(np.clip((float(smoothed[peak_idx]) / global_p95) / 1.4, 0, 1))
        raw_candidates.append({"peak": peak_sec, "audio_score": raw_audio, "local": local})

    # Deduplica e aplica o limite após reservar o melhor candidato de cada zona.
    raw_candidates.sort(key=lambda row: row["audio_score"], reverse=True)
    min_distance = max(2.0, float(config.ANALYSIS["peak_distance_seconds"]))
    unique: list[dict[str, Any]] = []
    zone_best: dict[int, dict[str, Any]] = {}
    for row in raw_candidates:
        zone = min(bins - 1, int(row["peak"] / max(duration, 1e-6) * bins))
        zone_best.setdefault(zone, row)
    ordered = list(zone_best.values()) + raw_candidates
    candidate_limit = max(bins, int(config.ANALYSIS["candidate_limit"]))
    for row in ordered:
        if any(abs(row["peak"] - old["peak"]) < min_distance for old in unique):
            continue
        unique.append(row)
        if len(unique) >= candidate_limit:
            break

    # Só classifica a lista já limitada; assinaturas não multiplicam o custo em
    # milhares de picos redundantes.
    for row in unique:
        event, confidence, reason = _classify_event(wav_path, row["peak"], row["local"], signatures)
        row.update({"event_type": event, "confidence": confidence, "classification_reason": reason,
                    "raw_score": 0.0, "score": 0.0, "scene_score": 0.0, "movement_score": 0.0,
                    "red_flash": 0.0, "brightness": 0.0, "discard_reason": ""})

    unique.sort(key=lambda row: row["peak"])
    grouped: list[list[dict[str, Any]]] = []
    for row in unique:
        if grouped and row["peak"] - grouped[-1][-1]["peak"] <= config.ANALYSIS["combo_window_seconds"]:
            grouped[-1].append(row)
        else:
            grouped.append([row])
    combined = []
    for group in grouped:
        best = max(group, key=lambda row: row["audio_score"])
        if len(group) >= 2:
            best = dict(best)
            best["event_type"] = "combo"
            best["confidence"] = min(1.0, 0.45 + len(group) * 0.1)
            best["classification_reason"] = f"combo: {len(group)} eventos em {group[-1]['peak'] - group[0]['peak']:.1f}s"
            best["peak"] = float(np.average([g["peak"] for g in group], weights=[max(g["audio_score"], 0.01) for g in group]))
            best["group_start"] = group[0]["peak"]
            best["group_end"] = group[-1]["peak"]
        else:
            best["group_start"] = best["peak"]
            best["group_end"] = best["peak"]
        combined.append(best)
    return combined


def _audio_context_scores(audio_rows: list[dict[str, float]], center: float,
                          reference_rms: float,
                          vad_seconds: np.ndarray | None = None) -> dict[str, Any]:
    radius = float(config.SCENES["speech_window_seconds"]) / 2.0
    lo, hi = max(0, int(center - radius)), min(len(audio_rows), int(math.ceil(center + radius)))
    rows = audio_rows[lo:hi]
    if not rows:
        return {"speech_score": 0.0, "speech_seconds": 0, "speech_run_seconds": 0,
                "speech_ratio": 0.0, "speech_continuous": False, "music_score": 0.0,
                "tension_score": 0.0,
                "speech_detector": "silero_vad" if vad_seconds is not None else "espectral"}

    speech_values = []
    for row in rows:
        rms = float(row.get("rms", 0.0))
        second = int(row.get("second", 0))
        vad_coverage = (float(vad_seconds[second]) if vad_seconds is not None and
                        0 <= second < len(vad_seconds) else 0.0)
        if rms < float(config.SCENES["speech_min_rms"]):
            speech_values.append(0.75 + 0.25 * vad_coverage if vad_coverage >= 0.20 else 0.0)
            continue
        mid = float(np.clip((row.get("mid_ratio", 0.0) - 0.25) / 0.55, 0, 1))
        high = float(np.clip(row.get("high_ratio", 0.0) / 0.18, 0, 1))
        centroid = float(row.get("centroid", 0.0))
        centroid_fit = float(np.clip((centroid - 180.0) / 500.0, 0, 1) *
                              np.clip((4200.0 - centroid) / 1500.0, 0, 1))
        level = float(np.clip(rms / max(0.015, reference_rms), 0, 1))
        spectral_score = 0.34 * mid + 0.24 * high + 0.25 * centroid_fit + 0.17 * level
        vad_score = 0.75 + 0.25 * vad_coverage if vad_coverage >= 0.20 else 0.0
        speech_values.append(max(spectral_score, vad_score))

    voice_threshold = float(config.SCENES["speech_voiced_threshold"])
    voiced = [value >= voice_threshold for value in speech_values]
    voiced_seconds = sum(voiced)
    longest_run = run = 0
    for is_voiced in voiced:
        run = run + 1 if is_voiced else 0
        longest_run = max(longest_run, run)
    ratio = voiced_seconds / max(1, len(voiced))
    average_voice = float(np.mean([value for value in speech_values if value >= voice_threshold])) if voiced_seconds else 0.0
    speech_score = float(np.clip(0.35 * min(1.0, voiced_seconds / 6.0) +
                                 0.40 * min(1.0, longest_run / 6.0) + 0.25 * average_voice, 0, 1))
    speech_continuous = (
        voiced_seconds >= int(config.SCENES["speech_min_seconds"])
        and longest_run >= int(config.SCENES["speech_min_seconds"])
        and ratio >= float(config.SCENES["speech_min_ratio"])
    )

    energy = np.asarray([row.get("rms", 0.0) for row in rows], dtype=np.float32)
    spectrum = np.asarray([row.get("mid_ratio", 0.0) + row.get("high_ratio", 0.0) for row in rows], dtype=np.float32)
    energy_level = float(np.clip(float(np.mean(energy)) / max(reference_rms, 1e-5), 0, 1))
    spectral_presence = float(np.clip((float(np.mean(spectrum)) - 0.18) / 0.70, 0, 1))
    music_score = float(np.clip(0.65 * spectral_presence + 0.35 * energy_level, 0, 1))
    energy_variation = float(np.clip(float(np.std(energy)) / max(reference_rms, 1e-5), 0, 1))
    tension_score = float(np.clip(0.60 * energy_level + 0.25 * energy_variation +
                                  0.15 * float(np.max(energy) / max(reference_rms, 1e-5)), 0, 1))
    return {"speech_score": speech_score, "speech_seconds": int(voiced_seconds),
            "speech_run_seconds": int(longest_run), "speech_ratio": float(ratio),
            "speech_continuous": bool(speech_continuous), "music_score": music_score,
            "tension_score": tension_score,
            "speech_detector": "silero_vad" if vad_seconds is not None else "espectral"}


def _classify_cutscene(candidate: dict[str, Any], visual: dict[str, float],
                       audio_context: dict[str, Any]) -> bool:
    hud_score = float(visual.get("hud_score", 0.0))
    camera_cuts = float(visual.get("camera_cuts", 0.0))
    candidate.update({"hud_score": hud_score, "camera_cuts": camera_cuts,
                      "speech_score": float(audio_context["speech_score"]),
                      "speech_seconds": int(audio_context["speech_seconds"]),
                      "speech_run_seconds": int(audio_context["speech_run_seconds"]),
                      "speech_ratio": float(audio_context["speech_ratio"]),
                      "speech_detector": str(audio_context.get("speech_detector", "espectral")),
                      "music_score": float(audio_context["music_score"]),
                      "tension_score": float(audio_context["tension_score"])})
    is_scene = (
        bool(audio_context["speech_continuous"])
        and hud_score < float(config.SCENES["hud_threshold"])
        and camera_cuts >= int(config.SCENES["camera_cut_min_count"])
    )
    if is_scene:
        candidate["event_type"] = "cena"
        candidate["confidence"] = round(float(np.clip(
            0.45 + 0.25 * audio_context["speech_score"] +
            0.15 * min(1.0, camera_cuts / 4.0) +
            0.15 * (1.0 - hud_score / max(0.01, float(config.SCENES["hud_threshold"]))), 0.45, 0.98)), 2)
        candidate["classification_reason"] = (
            f"cena: fala contínua ({audio_context['speech_run_seconds']}s), "
            f"sem HUD ({hud_score:.2f}), {camera_cuts:.0f} cortes de câmera"
        )
    return bool(is_scene)


def _duration_for(candidate: dict[str, Any], all_candidates: list[dict[str, Any]]) -> int:
    center = candidate["peak"]
    neighborhood = [row for row in all_candidates if abs(row["peak"] - center) <= 45]
    density = min(1.0, max(0.0, (len(neighborhood) - 1) / 6.0))
    audio = float(candidate["audio_score"])
    fraction = min(1.0, max(0.0, 0.55 * density + 0.45 * audio))
    return int(round(config.ANALYSIS["clip_min_seconds"] + fraction *
                     (config.ANALYSIS["clip_max_seconds"] - config.ANALYSIS["clip_min_seconds"])))


def _cut_window(candidate: dict[str, Any], all_candidates: list[dict[str, Any]], duration: float,
                fixed_duration: int | None = None) -> tuple[float, float]:
    target = fixed_duration or _duration_for(candidate, all_candidates)
    if candidate.get("ai_boundary_advisory_valid"):
        event_span = max(0.0, float(candidate.get("group_end", candidate["peak"])) -
                         float(candidate.get("group_start", candidate["peak"])))
        target = max(target, min(config.ANALYSIS["clip_max_seconds"],
                                 int(math.ceil(event_span + config.ANALYSIS["end_tail_seconds"] + 4))))
    target = int(np.clip(target, config.ANALYSIS["clip_min_seconds"], config.ANALYSIS["clip_max_seconds"]))
    group_start = float(candidate.get("group_start", candidate["peak"]))
    group_end = float(candidate.get("group_end", candidate["peak"]))
    # Fim no pós-clímax; quando a duração ideal excede a ação, acrescenta contexto
    # antes dela em vez de deixar o vídeo rolando em silêncio depois do momento forte.
    end = min(duration, group_end + config.ANALYSIS["end_tail_seconds"])
    start = max(0.0, end - target)
    if end - start < config.ANALYSIS["clip_min_seconds"]:
        start = max(0.0, end - config.ANALYSIS["clip_min_seconds"])
        if end - start < config.ANALYSIS["clip_min_seconds"]:
            end = min(duration, start + config.ANALYSIS["clip_min_seconds"])
    return float(start), float(end)


def _rank_and_select(candidates: list[dict[str, Any]], duration: float, top_n: int,
                     fixed_duration: int | None = None, scene_mode: str | None = None,
                     max_scene_share: float | None = None) -> list[dict[str, Any]]:
    scene_mode = scene_mode or config.SCENE_MODE
    max_scene_share = float(config.SCENES["max_share"] if max_scene_share is None else max_scene_share)
    if scene_mode not in {"manter", "priorizar-gameplay", "ignorar"}:
        raise ValueError("Modo de cenas inválido. Escolha manter, priorizar-gameplay ou ignorar.")
    if not 0.0 <= max_scene_share <= 1.0:
        raise ValueError("O limite de cenas deve estar entre 0 e 1.")
    scene_limit = 0 if scene_mode == "ignorar" else int(math.floor(top_n * max_scene_share + 1e-9))
    for candidate in candidates:
        start, end = _cut_window(candidate, candidates, duration, fixed_duration)
        candidate["start"], candidate["end"] = start, end
        if candidate["event_type"] == "cena":
            if scene_mode == "ignorar":
                candidate["discard_reason"] = "cena ignorada conforme o modo selecionado"
            weights = config.SCENES["score_weights"]
            weight_sum = max(1e-6, sum(float(value) for value in weights.values()))
            raw = 100.0 * sum(float(weights[key]) * float(candidate.get(f"{key}_score", 0.0))
                              for key in ("speech", "music", "tension")) / weight_sum
            candidate["score_basis"] = "fala+música+tensão"
            if candidate.get("ai_assessed"):
                ai_score = float(np.clip(float(candidate.get("ai_score", 0.0)), 0, 100))
                if candidate.get("ai_engine") == "ollama_vision":
                    raw = 0.65 * ai_score + 0.25 * raw + 0.10 * raw
                    candidate["score_basis"] += " + Ollama visão (peso principal)"
                else:
                    raw = 0.45 * ai_score + 0.55 * raw
                    candidate["score_basis"] += " + Ollama textual/áudio (sem visão)"
            candidate["raw_score"] = raw
            continue
        event_weight = config.EVENT_WEIGHTS.get(candidate["event_type"], 1.0)
        specific_events = {"escopeta", "headshot", "parry_faca", "motosserra", "explosao",
                           "boss_combate", "susto_grito", "pistola_rifle"}
        if candidate["event_type"] in specific_events:
            reason = str(candidate.get("classification_reason", "")).lower()
            trusted = reason.startswith("assinatura log-mel:") or "confirmação visual" in reason
            confident = float(candidate.get("confidence", 0.0) or 0.0) >= float(
                config.ANALYSIS.get("specific_event_min_confidence", 0.82))
            if not (trusted and confident):
                event_weight = 1.0
                candidate["event_score_note"] = "bônus específico removido: evento não confirmado"
        weighted_event = min(1.0, max(0.0, (event_weight - 0.5) / 0.9))
        features = config.SCORE_WEIGHTS
        hud_present = float(candidate.get("hud_score", 0.0)) >= float(config.SCENES["hud_threshold"])
        scene_component = candidate["scene_score"] if hud_present else 0.0
        movement_component = candidate["movement_score"] if hud_present else 0.0
        raw = (features["audio"] * candidate["audio_score"] +
               features["scene"] * scene_component +
               features["movement"] * movement_component +
               features["event_type"] * weighted_event) * 100
        candidate["score_basis"] = "áudio+ação com HUD" if hud_present else "áudio; sem bônus visual (HUD não detectado)"
        if candidate.get("ai_assessed"):
            ai_score = float(np.clip(float(candidate.get("ai_score", 0.0)), 0, 100))
            if candidate.get("ai_engine") == "ollama_vision":
                if candidate.get("event_type") == "cena":
                    local_signal = 100.0 * float(np.mean([
                        candidate.get("speech_score", 0.0), candidate.get("music_score", 0.0),
                        candidate.get("tension_score", 0.0)]))
                else:
                    local_signal = 100.0 * (0.55 * scene_component + 0.45 * movement_component)
                raw = 0.65 * ai_score + 0.25 * raw + 0.10 * local_signal
                candidate["score_basis"] += " + Ollama visão (peso principal)"
            else:
                raw = 0.45 * ai_score + 0.55 * raw
                candidate["score_basis"] += " + Ollama textual/áudio (sem visão)"
            if candidate.get("ai_dead") and candidate.get("event_type") != "cena":
                candidate["discard_reason"] = str(candidate.get("ai_discard_reason") or
                                                   "IA classificou como menu/caminhada sem interesse")
        if scene_mode == "priorizar-gameplay":
            raw = min(100.0, raw * (1.0 + float(config.SCENES["gameplay_priority_bonus"])))
        if movement_component < 0.025 and candidate["audio_score"] < 0.18:
            raw *= 0.52
            candidate["discard_reason"] = "trecho parado e com áudio fraco (possível menu/inventário)"
        candidate["raw_score"] = raw
    candidates.sort(key=lambda row: row["raw_score"], reverse=True)
    # Um candidato preferencial por fatia temporal, então completa por score global.
    selected: list[dict[str, Any]] = []
    max_per_type = config.ANALYSIS["max_same_event_type"]
    min_gap = config.ANALYSIS["min_clip_gap_seconds"]
    buckets = max(1, top_n)
    for bucket in range(buckets):
        lo, hi = duration * bucket / buckets, duration * (bucket + 1) / buckets
        pool = [row for row in candidates if lo <= row["peak"] < hi and not row.get("discard_reason")]
        for row in pool:
            if _admissible(row, selected, max_per_type, min_gap, scene_limit):
                selected.append(row)
                break
    for row in candidates:
        if len(selected) >= top_n:
            break
        if row.get("discard_reason"):
            continue
        if _admissible(row, selected, max_per_type, min_gap, scene_limit):
            selected.append(row)
        else:
            row["selection_note"] = "aguardando segunda passagem: variedade/timestamp"
    # A variedade é uma preferência, não um bloqueio que reduza uma rodada rica
    # em um ou dois rótulos heurísticos genéricos (ex.: todos 'impacto').
    for bucket in range(buckets):
        if len(selected) >= top_n:
            break
        lo, hi = duration * bucket / buckets, duration * (bucket + 1) / buckets
        if any(lo <= row["peak"] < hi for row in selected):
            continue
        pool = [row for row in candidates if lo <= row["peak"] < hi and not row.get("discard_reason")]
        for row in pool:
            if _admissible(row, selected, top_n, min_gap, scene_limit):
                selected.append(row)
                row.pop("selection_note", None)
                break
    for row in candidates:
        if len(selected) >= top_n:
            break
        if not row.get("discard_reason") and _admissible(row, selected, top_n, min_gap, scene_limit):
            selected.append(row)
            row.pop("selection_note", None)
    selected.sort(key=lambda row: row["raw_score"], reverse=True)
    maximum = max([row["raw_score"] for row in selected] or [1.0])
    for row in candidates:
        row["score"] = round(float(np.clip(row["raw_score"] / max(1.0, maximum) * 100, 0, 100)), 1)
    for index, row in enumerate(selected, start=1):
        row["rank"] = index
        row["duration_seconds"] = round(row["end"] - row["start"], 1)
    return selected


def _admissible(candidate: dict[str, Any], selected: list[dict[str, Any]], max_per_type: int,
                min_gap: float, max_scenes: int | None = None) -> bool:
    event_type = candidate["event_type"]
    if event_type == "cena" and max_scenes is not None:
        if sum(item["event_type"] == "cena" for item in selected) >= max_scenes:
            return False
    if sum(item["event_type"] == event_type for item in selected) >= max_per_type:
        return False
    for item in selected:
        if abs(candidate["peak"] - item["peak"]) < min_gap or (candidate["start"] < item["end"] and candidate["end"] > item["start"]):
            return False
    return True


def _write_debug_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = ["rank", "start", "end", "peak", "duration_seconds", "event_type", "confidence",
              "classification_reason", "score", "raw_score", "audio_score", "scene_score", "movement_score",
              "hud_score", "hud_detected", "camera_cuts", "speech_score", "speech_seconds",
              "speech_run_seconds", "speech_detector", "music_score", "tension_score", "score_basis", "red_flash",
              "brightness", "ai_assessed", "ai_engine", "ai_score", "ai_visual_score", "ai_visual_class", "ai_relevant",
              "ai_dead", "ai_confidence", "ai_evidence", "ai_moment_start_seconds", "ai_moment_end_seconds",
              "ai_boundary_advisory_valid", "discard_reason", "selection_note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(candidates, key=lambda x: (x.get("rank", 9999), -x.get("raw_score", 0))):
            writer.writerow(row)


def _serializable(row: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in row.items() if key != "local"}
    return result


def analyze_video(video_path: str | Path, mode: str = "quick", top_n: int | None = None,
                  debug: bool = False, fixed_duration: int | None = None,
                  output_root: str | Path | None = None, use_cache: bool = True,
                  scene_mode: str | None = None, max_scene_share: float | None = None,
                  vision_enabled: bool = True, ollama_base_url: str = "http://localhost:11434",
                  vision_model: str = "llava:7b", text_model: str = "llama3.2:3b",
                  vision_timeout: float = 90.0, max_vision_candidates: int = 18) -> dict[str, Any]:
    start_time = time.time()
    ensure_directories()
    scene_mode = scene_mode or config.SCENE_MODE
    max_scene_share = float(config.SCENES["max_share"] if max_scene_share is None else max_scene_share)
    if scene_mode not in {"manter", "priorizar-gameplay", "ignorar"}:
        raise ValueError("Modo de cenas inválido. Use manter, priorizar-gameplay ou ignorar.")
    if not 0.0 <= max_scene_share <= 1.0:
        raise ValueError("O limite de cenas deve estar entre 0 e 1.")
    ffmpeg_info = check_ffmpeg()
    source = Path(video_path).expanduser().resolve()
    info = probe_media(source)
    top_n = max(1, min(int(top_n if top_n is not None else _ideal_clip_count(info["duration"])), 30))
    output_root = Path(output_root or config.DIRS["sessions"])
    session_id = f"{source.stem[:32]}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
    session_dir = output_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressWriter(session_dir)
    ensure_free_space(config.DIRS["temp"], min(max(info["size_bytes"] // 2, 300_000_000), 5_000_000_000))
    signature_stamp = "|".join(
        f"{item.name}:{item.stat().st_size}:{item.stat().st_mtime_ns}"
        for item in sorted(config.DIRS["signatures"].glob("*.wav"))
    )
    ai_status = (ollama_ai.check_status(base_url=ollama_base_url, model=vision_model, timeout=2.0)
                 if vision_enabled else {"ok": False, "models": [], "model_installed": False})
    ai_settings = {"enabled": bool(vision_enabled), "base_url": ollama_base_url,
                   "vision_model": vision_model, "text_model": text_model,
                   "max_candidates": int(max_vision_candidates), "installed_models": sorted(ai_status.get("models", [])),
                   "ollama_ok": bool(ai_status.get("ok")), "ai_config": config.AI}
    config_stamp = json.dumps({"analysis": config.ANALYSIS, "scenes": config.SCENES,
                               "scene_mode": scene_mode, "max_scene_share": max_scene_share,
                               "weights": config.SCORE_WEIGHTS, "events": config.EVENT_WEIGHTS,
                               "ai": ai_settings},
                              sort_keys=True, default=str)
    cache_mode = f"{mode}:top={top_n}:fixed={fixed_duration}:signatures={signature_stamp}:" \
                 f"scene_mode={scene_mode}:scene_share={max_scene_share}:" \
                 f"ai={json.dumps(ai_settings, sort_keys=True)}:config={config_stamp}:ffmpeg={ffmpeg_info['version']}"
    key = cache_key(info, cache_mode)
    cache_path = config.DIRS["cache"] / f"{key}.json"
    if use_cache and cache_path.exists():
        cached = read_json(cache_path)
        if cached and cached.get("schema") == config.ANALYSIS["cache_schema_version"]:
            LOG.info("Cache válido reutilizado: %s", cache_path.name)
            progress.update("cache", 100, "Análise carregada do cache.", done=True)
            cached["session_id"] = session_id
            cached["session_dir"] = str(session_dir)
            cached["cache_hit"] = True
            atomic_write_json(session_dir / "analysis.json", cached)
            if debug:
                _write_debug_csv(session_dir / config.ANALYSIS["debug_csv_name"], cached.get("candidates", []))
            return cached

    lock = InstanceLock(on_busy=lambda message: progress.update("bloqueado", 100, message, done=True))
    with lock:
        progress.update("preparação", 2, "Validando mídia, filtros e espaço em disco.")
        LOG.info("Análise iniciada: %s | modo=%s | modo de cenas=%s | limite=%.0f%%",
                 source, mode, scene_mode, max_scene_share * 100)
        wav_path = None
        try:
            audio_rows: list[dict[str, float]] = []
            audio_duration = 0.0
            vad_seconds: np.ndarray | None = None
            if info["audio"]:
                wav_path = config.DIRS["temp"] / f"audio_{uuid.uuid4().hex}.wav"
                progress.update("áudio", 6, "Extraindo áudio mono 16 kHz e medindo RMS em blocos.")
                _extract_audio(source, wav_path, info["duration"], progress)
                audio_rows, audio_duration = _audio_features(wav_path, progress)
                audio_rows = _align_audio_rows(audio_rows, info["duration"])
                LOG.info("Áudio analisado em blocos: %.1fs / %d amostras por segundo", audio_duration, len(audio_rows))
                progress.update("fala", 22, "Detectando fala localmente para classificar cutscenes.")
                vad_seconds = _detect_speech_seconds(wav_path, info["duration"], progress)
                LOG.info("Detecção de fala: %s", "Silero VAD local" if vad_seconds is not None else "fallback espectral")
            else:
                LOG.warning("Vídeo sem trilha de áudio; usando detecção visual em toda a duração.")

            signatures = _load_signatures(config.DIRS["signatures"])
            candidates = _make_peak_candidates(audio_rows, wav_path, signatures, info["duration"]) if audio_rows else []
            candidates = [row for row in candidates if 0 <= row["peak"] < info["duration"]]
            if info["audio"]:
                # Em toda partição começo/meio/fim, reserva uma janela visual se a
                # região não tem pico ou se o áudio terminou/ficou silencioso nela.
                probe_zones = max(5, min(10, top_n))
                for zone in range(probe_zones):
                    lo, hi = info["duration"] * zone / probe_zones, info["duration"] * (zone + 1) / probe_zones
                    zone_candidates = [row for row in candidates if lo <= row["peak"] < hi]
                    row_start, row_end = max(0, int(lo)), min(len(audio_rows), int(math.ceil(hi)))
                    zone_rms = max((row["rms"] for row in audio_rows[row_start:row_end]), default=0.0)
                    audio_uncovered = audio_duration < hi - 1.0
                    if zone_candidates and not audio_uncovered and zone_rms >= 0.005:
                        continue
                    probe_start = max(lo, min(hi - 0.01, audio_duration)) if audio_uncovered else lo
                    peak = min(hi - 0.01, probe_start + max(0.1, (hi - probe_start) / 2))
                    candidates.append({"peak": peak, "audio_score": 0.0, "event_type": "impacto",
                                       "confidence": 0.15, "classification_reason": "probe visual em região silenciosa/sem evento de áudio",
                                       "local": [], "raw_score": 0.0, "score": 0.0,
                                       "scene_score": 0.0, "movement_score": 0.0,
                                       "red_flash": 0.0, "brightness": 0.0, "discard_reason": ""})

                # Cutscenes podem ter diálogo/música sem qualquer pico abrupto;
                # sondas regulares dão ao classificador visual a chance de achá-las.
                interval = float(config.SCENES["probe_interval_seconds"])
                probe_count = min(int(config.SCENES["probe_limit"]),
                                  max(1, int(math.ceil(info["duration"] / max(1.0, interval)))))
                for zone in range(probe_count):
                    lo, hi = info["duration"] * zone / probe_count, info["duration"] * (zone + 1) / probe_count
                    peak = (lo + hi) / 2.0
                    candidates.append({"peak": peak, "audio_score": 0.0, "event_type": "impacto",
                                       "confidence": 0.10, "classification_reason": "probe visual periódico para cena falada",
                                       "local": [], "raw_score": 0.0, "score": 0.0,
                                       "scene_score": 0.0, "movement_score": 0.0,
                                       "red_flash": 0.0, "brightness": 0.0, "discard_reason": ""})
                candidates.extend(_speech_probe_candidates(vad_seconds, info["duration"]))
            visual_timeline: dict[int, dict[str, float]] = {}
            if mode == "detailed" or not info["audio"]:
                progress.update("visual", 28, "Analisando vídeo inteiro a 320px/5fps em streaming (modo detalhado).")
                visual_timeline = _scan_visual(source, 0.0, info["duration"], progress, 28, 75)
            elif candidates:
                ranked_audio = sorted(candidates, key=lambda row: row["audio_score"], reverse=True)
                # Reserva ao menos um candidato por zona para evitar que clipes de começo
                # monopolizem a análise visual quando há muitos picos no primeiro ato.
                coverage = []
                zones = max(1, min(top_n, 12))
                for zone in range(zones):
                    lo, hi = info["duration"] * zone / zones, info["duration"] * (zone + 1) / zones
                    zone_rows = [row for row in candidates if lo <= row["peak"] < hi]
                    if zone_rows:
                        coverage.append(max(zone_rows, key=lambda row: row["audio_score"]))
                probes = [row for row in candidates if row.get("classification_reason", "").startswith("probe visual")]
                sample, sample_seen = [], set()
                for row in coverage + probes + ranked_audio:
                    marker = round(float(row["peak"]), 1)
                    if marker not in sample_seen:
                        sample_seen.add(marker)
                        sample.append(row)
                sample = sample[:int(config.ANALYSIS["quick_visual_candidates"])]
                progress.update("visual", 28, f"Medindo cena/movimento em {len(sample)} janelas de pico distribuídas.")
                done = set()
                for idx, candidate in enumerate(sample, start=1):
                    progress.check_cancelled()
                    center = candidate["peak"]
                    begin = max(0.0, center - 3.0)
                    length = min(9.0, info["duration"] - begin)
                    bucket = int(begin // 4)
                    if bucket in done:
                        continue
                    done.add(bucket)
                    window = _scan_visual(source, begin, length, progress, 28, 50)
                    visual_timeline.update(window)
                    if idx % 10 == 0:
                        progress.update("visual", 28 + 22 * idx / max(1, len(sample)),
                                        f"Janelas visuais analisadas: {idx}/{len(sample)}")
            else:
                progress.update("visual", 28, "Sem picos de áudio marcados; amostragem visual integral.")
                visual_timeline = _scan_visual(source, 0.0, info["duration"], progress, 28, 75)

            if not candidates:
                visual_candidates = []
                for sec, metrics in visual_timeline.items():
                    if metrics["motion"] > 0.015 or metrics["scene"] > config.ANALYSIS["scene_threshold"]:
                        row = {"peak": sec + 0.5, "audio_score": 0.0, "event_type": "impacto",
                               "confidence": 0.25, "classification_reason": "movimento/mudança de cena visual",
                               "local": [], "raw_score": 0.0, "score": 0.0,
                               "scene_score": 0.0, "movement_score": 0.0,
                               "red_flash": metrics["red_flash"], "brightness": metrics["brightness"],
                               "discard_reason": ""}
                        row["visual_rank"] = metrics["motion"] * 0.55 + metrics["scene"] * 0.35 + metrics["red_flash"] * 0.10
                        visual_candidates.append(row)
                zones = min(120, max(10, int(math.ceil(info["duration"] / 60.0))))
                zone_best = {}
                for row in visual_candidates:
                    zone = min(zones - 1, int(row["peak"] / max(info["duration"], 1e-6) * zones))
                    if zone not in zone_best or row["visual_rank"] > zone_best[zone]["visual_rank"]:
                        zone_best[zone] = row
                ordered = list(zone_best.values()) + sorted(visual_candidates, key=lambda row: row["visual_rank"], reverse=True)
                seen = set()
                candidates = []
                for row in ordered:
                    marker = round(row["peak"], 1)
                    if marker in seen:
                        continue
                    seen.add(marker)
                    candidates.append(row)
                    if len(candidates) >= config.ANALYSIS["candidate_limit"]:
                        break
                candidates.sort(key=lambda r: r["peak"])

            audio_reference_rms = (max(float(np.percentile([row["rms"] for row in audio_rows], 95)), 0.01)
                                   if audio_rows else 0.01)
            for candidate in candidates:
                metrics = _visual_at(visual_timeline, candidate["peak"],
                                     radius=float(config.SCENES["speech_window_seconds"]) / 2.0)
                candidate["visual_sampled"] = any(
                    second in visual_timeline for second in range(
                        max(0, int(candidate["peak"] - config.SCENES["speech_window_seconds"] / 2)),
                        int(candidate["peak"] + config.SCENES["speech_window_seconds"] / 2) + 1)
                )
                hud_detected = metrics["hud_score"] >= float(config.SCENES["hud_threshold"])
                candidate["hud_score"] = metrics["hud_score"]
                candidate["hud_detected"] = bool(hud_detected)
                candidate["camera_cuts"] = metrics["camera_cuts"]
                # Mudanças de câmera e movimento só são sinais de gameplay se
                # houver evidência de HUD persistente nos cantos.
                candidate["scene_score"] = (float(np.clip(metrics["scene"] / max(0.01, config.ANALYSIS["scene_threshold"]), 0, 1))
                                             if hud_detected else 0.0)
                candidate["movement_score"] = (float(np.clip(metrics["motion"] / 0.12, 0, 1))
                                                if hud_detected else 0.0)
                candidate["red_flash"] = metrics["red_flash"]
                candidate["brightness"] = metrics["brightness"]
                audio_context = _audio_context_scores(audio_rows, candidate["peak"],
                                                     audio_reference_rms, vad_seconds)
                _classify_cutscene(candidate, metrics, audio_context)
                if (candidate["event_type"] != "cena" and hud_detected and
                        candidate["event_type"] in {"impacto", "morno"} and
                        metrics["red_flash"] > 0.12 and metrics["scene"] > 0.16):
                    candidate["event_type"] = "explosao"
                    candidate["classification_reason"] = "flash vermelho/mudança brusca de cena (pista visual)"
                    candidate["confidence"] = max(candidate["confidence"], 0.35)

            progress.update("ia", 76, "Avaliando candidatos com Ollama visual/textual; fallback seguro se indisponível.")
            try:
                ai_info = ia_manager.analyze_candidates(
                    source, candidates, duration=info["duration"], top_n=top_n,
                    base_url=ollama_base_url, vision_model=vision_model, text_model=text_model,
                    timeout=vision_timeout, max_candidates=max_vision_candidates,
                    enabled=vision_enabled, progress=progress, status=ai_status,
                )
            except JobCancelledError:
                raise
            except Exception as ai_error:
                LOG.warning("IA de análise falhou; usando ranking local: %s", ai_error)
                ai_info = {"engine": "heuristic_local", "status": "fallback", "model": "",
                           "candidates_assessed": 0, "candidate_count": len(candidates),
                           "fallback_reason": f"Falha segura da IA: {type(ai_error).__name__}", "decisions": {}}
            for decision in (ai_info.get("decisions") or {}).values():
                if not isinstance(decision, dict):
                    continue
                try:
                    index = int(decision.get("candidate_index", -1))
                except (TypeError, ValueError):
                    continue
                if index < 0 or index >= len(candidates):
                    continue
                candidate = candidates[index]
                engine = str(ai_info.get("engine", ""))
                candidate.update({"ai_assessed": True, "ai_engine": engine,
                                  "ai_candidate_id": str(decision.get("candidate_id", "")),
                                  "ai_confidence": float(decision.get("confidence", 0.0) or 0.0)})
                if engine == "ollama_vision":
                    visual_class = str(decision.get("visual_class", "unknown"))
                    raw_visual_score = float(np.clip(float(decision.get("visual_score", 0.0)), 0, 100))
                    visual_relevant = bool(decision.get("relevant", False))
                    visual_score = raw_visual_score if visual_relevant or visual_class == "cutscene" else min(raw_visual_score, 25.0)
                    candidate.update({"ai_score": visual_score, "ai_visual_class": visual_class,
                                      "ai_visual_score": raw_visual_score, "ai_relevant": visual_relevant,
                                      "ai_evidence": str(decision.get("evidence", ""))[:240],
                                      "ai_peak_seconds": float(decision.get("peak_seconds", candidate["peak"])),
                                      "ai_moment_start_seconds": float(decision.get("moment_start_seconds", candidate["peak"])),
                                      "ai_moment_end_seconds": float(decision.get("moment_end_seconds", candidate["peak"])),
                                      "ai_boundary_advisory_valid": bool(decision.get("boundary_advisory_valid", False))})
                    candidate["ai_dead"] = bool(
                        visual_class in {"menu_loading", "traversal"} and
                        (not decision.get("relevant", False) or visual_score < 25))
                    if candidate["ai_dead"]:
                        candidate["ai_discard_reason"] = "Ollama visão identificou menu/carregamento/caminhada sem ação"
                    if visual_class == "cutscene" and candidate.get("event_type") != "cena":
                        candidate["event_type"] = "cena"
                        candidate["confidence"] = max(float(candidate.get("confidence", 0.0)),
                                                       min(0.95, float(decision.get("confidence", 0.0))))
                        candidate["classification_reason"] = (
                            "cena: Ollama visão identificou cutscene; " + str(decision.get("evidence", ""))[:180])
                    if decision.get("boundary_advisory_valid"):
                        candidate["group_start"] = float(decision["moment_start_seconds"])
                        candidate["group_end"] = float(decision["moment_end_seconds"])
                elif engine == "ollama_text_audio_fallback":
                    priority_score = float(np.clip(float(decision.get("priority_score", 0.0)), 0, 100))
                    if not decision.get("keep", False):
                        priority_score = min(priority_score, 25.0)
                    candidate.update({"ai_score": priority_score, "ai_visual_class": "not_seen",
                                      "ai_relevant": bool(decision.get("keep", False)),
                                      "ai_dead": False, "ai_evidence": str(decision.get("reason", ""))[:220]})
            selected = _rank_and_select(candidates, info["duration"], top_n, fixed_duration,
                                        scene_mode=scene_mode, max_scene_share=max_scene_share)
            duration_stats = time.time() - start_time
            # O renderizador usa os eventos e as leituras de brilho para sincronizar
            # punch-in, reações e correção seletiva sem fazer outra varredura completa.
            for row in selected:
                row["events"] = [{"timestamp": float(event["peak"]), "type": event["event_type"]}
                                 for event in candidates if row["start"] <= event["peak"] <= row["end"]]
                row["brightness_windows"] = [{"start": max(row["start"], float(event["peak"]) - 0.5),
                                              "end": min(row["end"], float(event["peak"]) + 1.5),
                                              "yavg": float(event.get("brightness", 1.0)) * 255.0}
                                             for event in candidates if row["start"] <= event["peak"] <= row["end"] and event.get("visual_sampled")]
            final = {
                "schema": config.ANALYSIS["cache_schema_version"], "source": info,
                "mode": mode, "scene_mode": scene_mode, "max_scene_share": max_scene_share,
                "session_id": session_id, "session_dir": str(session_dir),
                "cache_hit": False, "ffmpeg": ffmpeg_info["version"],
                "created_at": time.time(), "analysis_seconds": round(duration_stats, 2),
                "summary": {"candidate_count": len(candidates), "selected_count": len(selected),
                            "target_count": top_n,
                            "scene_count": sum(row["event_type"] == "cena" for row in selected),
                            "scene_mode": scene_mode, "max_scenes": (0 if scene_mode == "ignorar" else
                                int(math.floor(top_n * max_scene_share + 1e-9))),
                            "audio_present": bool(info["audio"]),
                            "speech_detector": "silero_vad" if vad_seconds is not None else "espectral",
                            "ai_engine": ai_info.get("engine", "heuristic_local"),
                            "ai_status": ai_info.get("status", "fallback"),
                            "ai_model": ai_info.get("model", ""),
                            "ai_candidates_assessed": int(ai_info.get("candidates_assessed", 0) or 0),
                            "ai_fallback_reason": str(ai_info.get("fallback_reason", ""))[:300],
                            "signatures_loaded": len(signatures)},
                "ai_analysis": {key: value for key, value in ai_info.items() if key != "decisions"},
                "clips": [_serializable(row) for row in selected],
                "candidates": [_serializable(row) for row in candidates],
            }
            if debug:
                _write_debug_csv(session_dir / config.ANALYSIS["debug_csv_name"], candidates)
            atomic_write_json(session_dir / "analysis.json", final)
            progress.update("concluído", 100, f"Análise pronta em {duration_stats:.1f}s.", done=True)
            try:
                atomic_write_json(cache_path, final)
            except OSError as cache_error:
                LOG.warning("Não foi possível gravar cache; os resultados da sessão foram preservados: %s", cache_error)
            LOG.info("Análise finalizada em %.1fs; %d/%d candidatos selecionados.", duration_stats, len(selected), len(candidates))
            return final
        except (JobCancelledError, KeyboardInterrupt):
            progress.update("cancelado", progress.last_percent, "Análise cancelada; temporários serão removidos.",
                            done=True, cancelled=True, process_pid=None)
            LOG.info("Análise cancelada pelo usuário.")
            raise
        except Exception as exc:
            LOG.exception("Falha de análise: %s", exc)
            progress.update("erro", progress.last_percent, str(exc), done=True, process_pid=None)
            raise
        finally:
            if wav_path:
                wav_path.unlink(missing_ok=True)


def _format_time(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ClipRush — análise local de gameplay RE4 Remake.")
    parser.add_argument("video", help="Caminho do vídeo de gameplay (MP4/MKV).")
    parser.add_argument("--mode", choices=["quick", "detailed"], default="quick")
    parser.add_argument("--top", type=int, default=None, help="Avançado; sem isto a quantidade é calculada (5–10).")
    parser.add_argument("--duration", type=int, default=None, help="Duração fixa (s); padrão dinâmico 20–90s.")
    parser.add_argument("--scene-mode", choices=["manter", "priorizar-gameplay", "ignorar"],
                        default=config.SCENE_MODE,
                        help="Manter cutscenes, dar bônus leve à gameplay ou ignorar cenas (padrão: manter).")
    parser.add_argument("--max-scene-share", type=float, default=float(config.SCENES["max_share"]),
                        help="Fração máxima do top N que pode ser cena, de 0 a 1 (padrão: 0.30).")
    parser.add_argument("--debug", action="store_true", help="Grava CSV com score, sinais e motivo de descarte.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", default=None, help="Diretório da sessão (padrão output/sessoes).")
    parser.add_argument("--no-vision-ai", action="store_true", help="Desativa Ollama no ranking e usa o motor local.")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="URL local do Ollama (somente loopback).")
    parser.add_argument("--vision-model", default=config.AI["vision_model"], help="Modelo VLM local, padrão llava:7b.")
    parser.add_argument("--text-model", default=config.AI["text_model"], help="Fallback textual local, padrão llama3.2:3b.")
    parser.add_argument("--vision-timeout", type=int, default=config.AI["vision_timeout_seconds"], help="Tempo limite por lote de visão.")
    parser.add_argument("--max-vision-candidates", type=int, default=config.AI["vision_max_candidates"], help="Máximo de janelas amostradas, 6–24.")
    args = parser.parse_args(argv)
    try:
        ff = check_ffmpeg()
        info = probe_media(args.video)
        print(f"Vídeo: {info['duration']/60:.1f} min · {info['width']}x{info['height']} · {info['fps']:.2f} fps")
        if not info["audio"]:
            print("Aviso: sem áudio; será usada análise visual.")
        print(f"FFmpeg: {ff['version']}")
        result = analyze_video(args.video, args.mode, args.top, args.debug, args.duration,
                               args.output, use_cache=not args.no_cache,
                               scene_mode=args.scene_mode, max_scene_share=args.max_scene_share,
                               vision_enabled=not args.no_vision_ai, ollama_base_url=args.ollama_url,
                               vision_model=args.vision_model, text_model=args.text_model,
                               vision_timeout=args.vision_timeout,
                               max_vision_candidates=args.max_vision_candidates)
        print("\nMelhores momentos:")
        for row in result["clips"]:
            print(f"  #{row['rank']:02d} score={row['score']:5.1f} tipo={row['event_type']:<14} "
                  f"conf={row['confidence']:.2f} { _format_time(row['start']) }–{ _format_time(row['end']) } "
                  f"({row['duration_seconds']:.0f}s) — {row['classification_reason']}")
        print(f"Modo de cenas: {result['scene_mode']} · cenas selecionadas: "
              f"{result['summary']['scene_count']}/{result['summary']['max_scenes']} máximo")
        print(f"IA de análise: {result['summary'].get('ai_engine', 'heurística local')} · "
              f"modelo={result['summary'].get('ai_model', '—')} · "
              f"janelas={result['summary'].get('ai_candidates_assessed', 0)}")
        print(f"\nSessão: {result['session_dir']}")
        print(f"JSON: {Path(result['session_dir']) / 'analysis.json'}")
        print(f"Cache reutilizável: {config.DIRS['cache']}")
        if args.debug:
            print(f"CSV: {Path(result['session_dir']) / config.ANALYSIS['debug_csv_name']}")
        return 0
    except Exception as exc:
        LOG.exception("CLI analyzer falhou")
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
