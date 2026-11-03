from __future__ import annotations

import argparse
from functools import lru_cache
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

import config
from common import (InstanceLock, JobCancelledError, ProgressWriter, check_ffmpeg, ensure_directories,
                    ensure_free_space, priority_kwargs, probe_media, run_ffmpeg_progress,
                    run_low_priority,
                    setup_logging)

LOG = setup_logging()
REACTIONS = {
    "escopeta": "ESCOPETA NA CARA!", "headshot": "HEADSHOT!", "parry_faca": "PARRY PERFEITO!",
    "motosserra": "SOBREVIVEU À MOTOSSERRA!", "explosao": "EXPLOSÃO!", "boss_combate": "BATALHA DE CHEFE!",
    "susto_grito": "QUE SUSTO!", "combo": "COMBO INSANO!", "pistola_rifle": "TIRO PRECISO!",
    "impacto": "OLHA ISSO!", "morno": "MOMENTO DA RUN", "cena": "VOCÊ VIU ISSO?",
}


def _escape_ass_text(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ").replace("\r", " ")


def _ass_time(seconds: float) -> str:
    total_centiseconds = int(round(max(0.0, seconds) * 100))
    total_seconds, centiseconds = divmod(total_centiseconds, 100)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def _filter_path(path: Path) -> str:
    # libavfilter escaping, sem shell intermediário: preserva espaços/acentos no Windows.
    value = path.resolve().as_posix()
    value = value.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    value = value.replace(",", "\\,").replace("[", "\\[").replace("]", "\\]")
    return value


def _filter_text(value: str) -> str:
    return _escape_ass_text(value)


def _subtitle_header(width: int, height: int) -> str:
    factor = width / float(config.RENDER["width"])
    title_size = int(round(config.RENDER["title_font_size"] * factor))
    caption_size = int(round(config.RENDER["caption_font_size"] * factor))
    outline = int(config.RENDER["outline_px"])
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,DejaVu Sans,{title_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,{outline},3,8,120,120,0,1
Style: Caption,DejaVu Sans,{caption_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H99000000,-1,0,0,0,100,100,0,0,1,{outline},3,2,100,100,650,1
Style: Reaction,DejaVu Sans,{int(round(58*factor))},&H0000FFFF,&H0000FFFF,&H00000000,&HAA000000,-1,0,0,0,100,100,0,0,1,7,4,8,120,120,0,1
Style: Watermark,DejaVu Sans,{int(round(30*factor))},&H99FFFFFF,&H99FFFFFF,&H99000000,&H55000000,-1,0,0,0,100,100,0,0,1,3,2,7,80,80,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _wrap_title(title: str, max_chars: int | None = None, max_total: int = 40) -> str:
    clean = " ".join(str(title).split())[:max_total]
    if max_chars is None:
        font_size = float(config.RENDER["title_font_size"])
        max_chars = max(12, int(config.RENDER["safe_max_width"] / max(1.0, font_size * 0.62)))
    return "\\N".join(textwrap.wrap(clean, width=max_chars, break_long_words=True, break_on_hyphens=False))


def _map_source_time(source_time: float, hook_start: float, hook_end: float,
                     context_start: float, context_end: float, hook_len: float) -> list[float]:
    mapped: list[float] = []
    if hook_start <= source_time <= hook_end:
        mapped.append(max(0.0, source_time - hook_start))
    if context_start <= source_time <= context_end:
        mapped.append(hook_len + source_time - context_start)
    return mapped


def _write_ass(path: Path, width: int, height: int, title: str, words: list[dict[str, Any]],
               hook_start: float, hook_end: float, context_start: float, context_end: float,
               hook_len: float, event_type: str, event_times: list[float], duration: float,
               reactions: bool = True, watermark: str = "") -> None:
    lines = [_subtitle_header(width, height)]
    factor = width / float(config.RENDER["width"])
    end = max(0.2, min(duration, 2.0))
    title_text = "\\N".join(_filter_text(line) for line in _wrap_title(str(title).upper()).split("\\N"))
    title_y = int(((config.RENDER["safe_title_y_min"] + config.RENDER["safe_title_y_max"]) / 2) * factor)
    caption_y = int(((config.RENDER["safe_caption_y_min"] + config.RENDER["safe_caption_y_max"]) / 2) * factor)
    if title_text:
        lines.append(f"Dialogue: 3,{_ass_time(0.12)},{_ass_time(end)},Title,,0,0,0,,{{\\an8\\pos({width//2},{title_y})}}{title_text}")
    if words:
        for word in words:
            token = str(word.get("text", "")).strip()
            if not token:
                continue
            start = float(word.get("start", 0))
            finish = float(word.get("end", start + 0.2))
            for mapped_start in _map_source_time(start, hook_start, hook_end, context_start, context_end, hook_len):
                # A 0.12 s hold avoids one-frame flicker on very short Whisper tokens.
                mapped_end = min(duration, mapped_start + max(0.12, finish - start))
                impact = token.strip(".,!?…:;").upper() in {"NÃO", "MANO", "INSANO", "CARACA", "CLUTCH", "PARRY", "HEADSHOT", "SÉRIO"}
                color = "\\c&H0000FFFF&" if impact else "\\c&H00FFFFFF&"
                safe = _filter_text(token.upper())
                lines.append(f"Dialogue: 2,{_ass_time(mapped_start)},{_ass_time(mapped_end)},Caption,,0,0,0,,{{\\an2\\pos({width//2},{caption_y}){color}\\fad(40,70)}}{safe}{{\\c&H00FFFFFF&}}")
    if reactions and config.RENDER["reactions_enabled"]:
        reaction = REACTIONS.get(event_type, "OLHA ISSO!")
        for event_time in event_times[:config.RENDER["reaction_max_per_clip"]]:
            for mapped_time in _map_source_time(event_time, hook_start, hook_end, context_start, context_end, hook_len):
                begin, finish = max(0.0, mapped_time - 0.15), min(duration, mapped_time + 1.25)
                if finish > begin:
                    lines.append(f"Dialogue: 2,{_ass_time(begin)},{_ass_time(finish)},Reaction,,0,0,0,,{{\\an8\\pos({width//2},{int(950*factor)})\\fad(80,220)}}{_filter_text(reaction)}")
    if watermark:
        safe_watermark = _filter_text(watermark[:40])
        lines.append(f"Dialogue: 1,{_ass_time(0)},{_ass_time(duration)},Watermark,,0,0,0,,{{\\an7\\pos({int(70*factor)},{int(500*factor)})}}{safe_watermark}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _available_platforms(platform: str, duration: float, start: float) -> list[tuple[str, int]]:
    names = list(config.PLATFORMS) if platform == "all" else [platform]
    result = []
    for name in names:
        policy = config.PLATFORMS[name]
        target = int(round(duration))
        target = max(target, int(policy["min_seconds"]))
        target = min(target, int(policy["max_seconds"]))
        result.append((name, target))
    return result


def _ffmpeg_zoom_expression(events: list[float], hook_start: float, hook_end: float,
                            context_start: float, context_end: float, segment: str) -> str:
    pieces: list[str] = []
    used_times: list[float] = []
    segment_duration = (hook_end - hook_start) if segment == "hook" else (context_end - context_start)
    amplitude = float(config.RENDER["zoom_factor"]) - 1.0
    half = float(config.RENDER["zoom_duration"]) / 2.0
    for event in events:
        if segment == "hook" and hook_start <= event <= hook_end:
            times = [event - hook_start]
        elif segment == "context" and context_start <= event <= context_end:
            times = [event - context_start]
        else:
            times = []
        for at in times:
            if any(abs(at - prior) < config.RENDER["zoom_min_interval"] for prior in used_times):
                continue
            begin, mid, finish = max(0.0, at - half), at, min(segment_duration, at + half)
            if finish <= begin or mid <= begin:
                continue
            ramp_up = max(0.01, mid - begin)
            ramp_down = max(0.01, finish - mid)
            up = f"((t-{begin:.3f})/{ramp_up:.3f})"
            down = f"((t-{mid:.3f})/{ramp_down:.3f})"
            # Smoothstep reduz a sensação de zoom mecânico; a gameplay central/HUD
            # permanece intacta e o movimento afeta apenas o fundo desfocado.
            shape = (f"if(between(t,{begin:.3f},{finish:.3f}),"
                     f"if(lt(t,{mid:.3f}),pow({up},2)*(3-2*{up}),"
                     f"1-pow({down},2)*(3-2*{down})),0)")
            pieces.append(f"({amplitude:.4f}*{shape})")
            used_times.append(at)
    return "1" + "".join(f"+{piece}" for piece in pieces)


def _transcribe(video: Path, start: float, duration: float, language: str) -> list[dict[str, Any]]:
    audio_path = config.DIRS["temp"] / f"fala_{uuid.uuid4().hex}.wav"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{start:.3f}",
               "-i", str(video), "-t", f"{duration:.3f}", "-vn", "-ac", "1", "-ar", "16000",
               "-c:a", "pcm_s16le", str(audio_path)]
    try:
        process = run_low_priority(command)
        if process.returncode:
            LOG.warning("Falha ao extrair áudio para legenda: %s", process.stderr[-800:])
            return []
        samples, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = np.mean(samples, axis=1, dtype=np.float32)
        if mono.size < 16000 or float(np.sqrt(np.mean(mono * mono))) < 0.004:
            return []
        # Whisper é executado no processo atual, já abaixo da prioridade normal.
        if os.name == "nt":
            try:
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
            except Exception:
                pass
        model = _load_whisper_model()
        # ndarray float32: evita o caminho PyAV/av.open afetado por metadata_errors.
        selected_language = None if language == "auto" else language
        segments, _ = model.transcribe(mono, language=selected_language,
                                       word_timestamps=True, vad_filter=True)
        output = []
        for segment in segments:
            for word in (getattr(segment, "words", None) or []):
                token = str(word.word).strip()
                if token:
                    output.append({"start": start + float(word.start), "end": start + float(word.end), "text": token})
        return output
    finally:
        audio_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _load_whisper_model():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("faster-whisper não instalado; execute iniciar.bat.") from exc
    return WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type=config.WHISPER_COMPUTE_TYPE,
                        cpu_threads=config.PERFORMANCE["whisper_cpu_threads"],
                        num_workers=config.PERFORMANCE["whisper_num_workers"],
                        download_root=str(config.DIRS["models"]))


def _stream_map(has_audio: bool, extra_audio: list[Path]) -> tuple[list[str], str]:
    command: list[str] = []
    indices: dict[str, int] = {}
    index = 2
    for name, path in zip(("narration", "music", "sfx"), extra_audio):
        if path:
            if name == "music":
                command += ["-stream_loop", "-1"]
            command += ["-i", str(path)]
            indices[name] = index
            index += 1
    return command, json.dumps(indices)


def _build_filter_graph(width: int, height: int, fg_height: int, hook_len: float,
                        context_len: float, total_duration: float, hook_start: float,
                        hook_end: float, context_start: float, peak: float,
                        event_times: list[float], has_audio: bool, audio_enabled: bool,
                        ass_path: Path, music_index: int | None,
                        narration_index: int | None, sfx_index: int | None,
                        brightness_windows: list[dict[str, float]], loop_tail: float = 0.0) -> tuple[str, list[str]]:
    bg_w = max(2, int(config.RENDER["blur_small_width"] * width / 1080))
    bg_h = max(2, int(config.RENDER["blur_small_height"] * height / 1920))
    context_end = context_start + context_len
    hook_zoom = _ffmpeg_zoom_expression(event_times, hook_start, hook_end, context_start, context_end, "hook")
    context_zoom = _ffmpeg_zoom_expression(event_times, hook_start, hook_end, context_start, context_end, "context")
    focus_x, focus_y = config.RENDER["zoom_focus_x"], config.RENDER["zoom_focus_y"]
    chains = [
        f"[0:v:0]fps={config.RENDER['fps']},scale={width}:{fg_height}:force_original_aspect_ratio=decrease,pad={width}:{fg_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,setpts=PTS-STARTPTS[v0]",
        f"[1:v:0]fps={config.RENDER['fps']},scale={width}:{fg_height}:force_original_aspect_ratio=decrease,pad={width}:{fg_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,setpts=PTS-STARTPTS[v1]",
        f"[0:v:0]fps={config.RENDER['fps']},scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,crop={bg_w}:{bg_h},boxblur={config.RENDER['blur_sigma']}:5,scale=w='trunc({width}*({hook_zoom})/2)*2':h='trunc({height}*({hook_zoom})/2)*2':eval=frame,crop={width}:{height}:(iw-ow)*{focus_x:.3f}:(ih-oh)*{focus_y:.3f},eq=brightness={config.RENDER['background_brightness']}:saturation={config.RENDER['background_saturation']},setpts=PTS-STARTPTS[bg0]",
        f"[1:v:0]fps={config.RENDER['fps']},scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,crop={bg_w}:{bg_h},boxblur={config.RENDER['blur_sigma']}:5,scale=w='trunc({width}*({context_zoom})/2)*2':h='trunc({height}*({context_zoom})/2)*2':eval=frame,crop={width}:{height}:(iw-ow)*{focus_x:.3f}:(ih-oh)*{focus_y:.3f},eq=brightness={config.RENDER['background_brightness']}:saturation={config.RENDER['background_saturation']},setpts=PTS-STARTPTS[bg1]",
        "[bg0][bg1]concat=n=2:v=1:a=0[bgbase]",
    ]
    if has_audio:
        chains += [
            f"[0:a:0]atrim=duration={hook_len:.3f},asetpts=PTS-STARTPTS,aresample=48000[a0]",
            f"[1:a:0]atrim=duration={context_len:.3f},asetpts=PTS-STARTPTS,aresample=48000[a1]",
            "[v0][a0][v1][a1]concat=n=2:v=1:a=1[vbase][abase]",
        ]
    else:
        chains.append("[v0][v1]concat=n=2:v=1:a=0[vbase]")
    if loop_tail > 0:
        chains += [
            "[vbase]split=2[vfull][vfirst]",
            "[vfirst]trim=end_frame=1,setpts=PTS-STARTPTS[vloop]",
            "[vfull][vloop]concat=n=2:v=1:a=0[vcat]",
            f"[bgbase]tpad=stop_mode=clone:stop_duration={loop_tail:.5f}[bg]",
        ]
        if has_audio:
            chains += [
                "[abase]asplit=2[afull][afirst]",
                f"[afirst]atrim=duration={loop_tail:.5f},asetpts=PTS-STARTPTS[aloop]",
                "[afull][aloop]concat=n=2:v=0:a=1[acat]",
            ]
    else:
        chains += ["[vbase]null[vcat]", "[bgbase]null[bg]"]
        if has_audio:
            chains.append("[abase]anull[acat]")

    # Correção seletiva, baseada no YAVG medido no analyzer e mapeada aos dois relógios.
    brightness_terms = []
    threshold = float(config.RENDER["dark_brightness_threshold"])
    max_correction = float(config.RENDER["max_brightness_correction"])
    for item in brightness_windows:
        yavg = float(item.get("yavg", item.get("brightness", 1.0)))
        fraction = yavg / 255.0 if yavg > 1.0 else yavg
        if fraction >= threshold:
            continue
        correction = min(max_correction, max(0.0, (threshold - fraction) * 0.8))
        src_a, src_b = float(item.get("start", peak - 0.5)), float(item.get("end", peak + 1.5))
        for segment, seg_start, seg_end, offset in (("hook", hook_start, hook_end, 0.0),
                                                      ("context", context_start, context_end, hook_len)):
            a, b = max(src_a, seg_start), min(src_b, seg_end)
            if b > a:
                out_a, out_b = offset + a - seg_start, offset + b - seg_start
                brightness_terms.append(f"if(between(t,{out_a:.3f},{out_b:.3f}),{correction:.4f},0)")
    if brightness_terms:
        expression = "+".join(f"({term})" for term in brightness_terms)
        chains.append(f"[vcat]eq=brightness='{expression}':eval=frame[vbright]")
        visual_label = "vbright"
    else:
        visual_label = "vcat"
    fade_seconds = max(0.0, float(config.RENDER["intro_fade_seconds"]))
    chains += [
        f"[bg][{visual_label}]overlay=(W-w)/2:(H-h)/2:shortest=1[composed]",
        f"[composed]ass=filename='{_filter_path(ass_path)}':fontsdir='{_filter_path(config.DIRS['fonts'])}',fade=t=in:st=0:d={fade_seconds:.3f}:color=black[outv]",
    ]
    maps = ["[outv]"]
    if audio_enabled:
        if has_audio:
            chains.append(f"[acat]loudnorm=I={config.LOUDNESS['integrated_lufs']}:TP={config.LOUDNESS['true_peak_db']}:LRA={config.LOUDNESS['lra']}:print_format=summary,alimiter=limit=0.95[game_pre]")
        else:
            chains.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={total_duration:.3f}[game_pre]")
        chains.append(f"[game_pre]afade=t=in:st=0:d={config.RENDER['audio_fade_seconds']},afade=t=out:st={max(0,total_duration-config.RENDER['audio_fade_seconds']):.3f}:d={config.RENDER['audio_fade_seconds']}[game]")
        game_label = "game"
        if narration_index is not None:
            chains.append(f"[{narration_index}:a:0]atrim=duration={total_duration:.3f},asetpts=PTS-STARTPTS,volume=1.0[narration]")
            chains.append("[game][narration]amix=inputs=2:duration=first:dropout_transition=1[main]")
            game_label = "main"
        if music_index is not None:
            volume = 10 ** (config.RENDER["music_volume_db"] / 20.0)
            chains.append(f"[{music_index}:a:0]atrim=duration={total_duration:.3f},asetpts=PTS-STARTPTS,volume={volume:.5f},afade=t=out:st={max(0,total_duration-0.5):.3f}:d=0.5[music]")
            chains.append(f"[{game_label}]asplit=2[mixbase][sidechain]")
            chains.append("[music][sidechain]sidechaincompress=threshold=0.03:ratio=6:attack=20:release=300[ducked]")
            chains.append("[mixbase][ducked]amix=inputs=2:duration=first:dropout_transition=1[audio_mix]")
            game_label = "audio_mix"
        if sfx_index is not None:
            chains.append("[" + str(sfx_index) + ":a:0]atrim=duration=2,asetpts=PTS-STARTPTS,volume=0.45,adelay=100|100[sfx]")
            chains.append(f"[{game_label}][sfx]amix=inputs=2:duration=first:dropout_transition=0.2[finalaudio]")
            game_label = "finalaudio"
        maps.append(f"[{game_label}]")
    return ";".join(chains), maps


def _render_cover(rendered_video: Path, destination: Path, timestamp: float = 1.0) -> None:
    """Extrai a capa do MP4 final no timestamp do próprio corte (por padrão +1,0 s)."""
    if not rendered_video.is_file():
        raise FileNotFoundError(f"Vídeo final não encontrado para extrair a capa: {rendered_video}")
    timestamp = max(0.0, float(timestamp))
    # select decodifica desde o início e escolhe o primeiro frame com PTS >= timestamp;
    # evita seeks aproximados a keyframes e garante que a capa corresponde ao render final.
    select = f"select=gte(t\\,{timestamp:.3f})"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(rendered_video),
               "-vf", select, "-frames:v", "1", "-q:v", "2", str(destination)]
    result = run_low_priority(command)
    if result.returncode or not destination.exists() or destination.stat().st_size < 1024:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Não foi possível extrair uma miniatura válida do MP4 em +"
                           f"{timestamp:.1f}s: " + (result.stderr or "FFmpeg não gerou uma imagem útil.")[-900:])


def render_clip(source_path: str | Path, start: float, end: float, peak: float,
                output_dir: str | Path, platform: str = "shorts", title: str = "MOMENTO INSANO",
                event_type: str = "impacto", captions: bool = False, language: str = "pt",
                music: str | Path | None = None, narration: str | Path | None = None,
                sfx: str | Path | None = None, watermark: str = "", reaction: bool = True,
                loop: bool = False, draft: bool = False, economy: bool = False,
                more_events: list[float] | None = None,
                brightness_windows: list[dict[str, float]] | None = None,
                create_cover: bool = True) -> list[dict[str, Any]]:
    ensure_directories()
    check_ffmpeg()
    source = Path(source_path).expanduser().resolve()
    info = probe_media(source)
    if not (0 <= start < end <= info["duration"] + 0.05):
        raise ValueError(f"Corte inválido: início/fim {start:.2f}–{end:.2f}s para vídeo de {info['duration']:.2f}s.")
    if platform not in {*config.PLATFORMS.keys(), "all"}:
        raise ValueError("Plataforma inválida; escolha shorts, reels, tiktok ou all.")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_duration = end - start
    selected_platforms = _available_platforms(platform, requested_duration, start)
    has_audio = bool(info["audio"])
    extra_paths = {"music": Path(music).resolve() if music else None,
                   "narration": Path(narration).resolve() if narration else None,
                   "sfx": Path(sfx).resolve() if sfx else None}
    for name, path in extra_paths.items():
        if path and not path.is_file():
            raise FileNotFoundError(f"Arquivo opcional não encontrado ({name}): {path}")
    if not has_audio and not any(extra_paths.values()):
        LOG.warning("Vídeo sem áudio: renderizando sem normalização/loudnorm, somente trilha visual.")

    event_times = sorted(set([float(peak)] + [float(t) for t in (more_events or [])]))
    action_end = max([float(peak)] + [t for t in event_times if start <= t <= end])
    brightness_windows = brightness_windows or []
    outputs = []
    with InstanceLock():
        for platform_name, target_duration in selected_platforms:
            # Hook/replay de pico no início; o restante continua do início natural do contexto.
            loop_tail = (1.0 / float(config.RENDER["fps"])) if loop else 0.0
            render_duration = target_duration - loop_tail
            hook_len = min(float(config.RENDER["hook_seconds"]), render_duration * 0.10)
            context_len = render_duration - hook_len
            hook_start = max(0.0, min(peak - config.RENDER["hook_before_peak"], info["duration"] - hook_len))
            hook_end = hook_start + hook_len
            desired_context_end = min(info["duration"], action_end + float(config.ANALYSIS["end_tail_seconds"]))
            context_start = max(0.0, desired_context_end - context_len)
            context_start = min(context_start, max(0.0, info["duration"] - context_len))
            context_end = context_start + context_len
            if info["duration"] + 0.05 < context_len:
                message = (f"A origem não tem contexto suficiente para {target_duration}s em {platform_name}; "
                           f"o trecho natural necessário é {context_len:.1f}s.")
                if platform == "all":
                    LOG.warning("Destino %s ignorado: %s", platform_name, message)
                    continue
                raise ValueError(message)
            ensure_free_space(output_dir, max(250_000_000, int(target_duration * info["width"] * info["height"] * 0.12)))
            width = config.RENDER["draft_width"] if draft else config.RENDER["width"]
            height = config.RENDER["draft_height"] if draft else config.RENDER["height"]
            fg_height = int(round(width * 9 / 16))
            threads = config.PERFORMANCE["economy_threads"] if economy or config.PERFORMANCE["economy_mode"] else config.PERFORMANCE["threads"]
            preset = config.PERFORMANCE["draft_preset"] if draft else config.PERFORMANCE["ffmpeg_preset"]
            stub = f"{source.stem[:38]}_{platform_name}_{int(start):06d}_{uuid.uuid4().hex[:5]}"
            final_path = output_dir / f"{stub}.mp4"
            partial_path = output_dir / f"{stub}.partial.mp4"
            ass_path = config.DIRS["temp"] / f"captions_{uuid.uuid4().hex}.ass"
            words: list[dict[str, Any]] = []
            session_dir = output_dir / f"sessao_{time.strftime('%Y%m%d_%H%M%S')}_{platform_name}_{uuid.uuid4().hex[:4]}"
            progress = ProgressWriter(session_dir)
            progress.update("render", 10, f"Preparando {config.PLATFORMS[platform_name]['label']}.")
            try:
                progress.check_cancelled()
                if captions and has_audio:
                    LOG.info("Transcrevendo janela de %.1fs com faster-whisper tiny.", context_len)
                    words = _transcribe(source, context_start, context_len, language)
                    progress.check_cancelled()
                if not words:
                    LOG.info("Nenhuma fala transcrita ou legendas desligadas; arquivo ASS sem palavras.")
                _write_ass(ass_path, width, height, title, words, hook_start, hook_end,
                           context_start, context_end, hook_len, event_type, event_times,
                           float(target_duration), reactions=reaction, watermark=watermark or
                           (config.RENDER["watermark_text"] if config.RENDER["watermark_enabled"] else ""))

                # Inputs opcionais seguem ordem fixa, paths em argv sem shell e com espaços/acentos suportados.
                input_args = []
                if extra_paths["narration"]:
                    input_args += ["-i", str(extra_paths["narration"])]
                if extra_paths["music"]:
                    input_args += ["-stream_loop", "-1", "-i", str(extra_paths["music"])]
                if extra_paths["sfx"]:
                    input_args += ["-i", str(extra_paths["sfx"])]
                # indices 2.. seguem narração, música, SFX nesta ordem, ignorando arquivos ausentes.
                indices = {}
                next_idx = 2
                for name in ("narration", "music", "sfx"):
                    if extra_paths[name]:
                        indices[name] = next_idx
                        next_idx += 1
                graph, maps = _build_filter_graph(
                    width, height, fg_height, hook_len, context_len, float(target_duration),
                    hook_start, hook_end, context_start, peak, event_times, has_audio,
                    bool(has_audio or any(extra_paths.values())), ass_path,
                    indices.get("music"), indices.get("narration"), indices.get("sfx"),
                    brightness_windows, loop_tail=loop_tail,
                )
                command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{hook_start:.3f}",
                           "-t", f"{hook_len:.3f}", "-i", str(source), "-ss", f"{context_start:.3f}",
                           "-t", f"{context_len:.3f}", "-i", str(source)]
                command += input_args
                command += ["-filter_complex", graph, "-map", maps[0]]
                if len(maps) > 1:
                    command += ["-map", maps[1], "-c:a", "aac", "-b:a", config.RENDER["audio_bitrate"]]
                else:
                    command += ["-an"]
                command += ["-t", f"{target_duration:.3f}", "-c:v", "libx264", "-preset", preset,
                            "-crf", str(config.RENDER["crf"]), "-pix_fmt", "yuv420p",
                            "-r", str(config.RENDER["fps"]), "-fps_mode", "cfr",
                            "-threads", str(threads), "-filter_threads", str(threads),
                            "-movflags", "+faststart", "-map_metadata", "-1", str(partial_path)]
                LOG.info("Início do FFmpeg: plataforma=%s clipe=%.1fs threads=%d prévia=%s", platform_name, target_duration, threads, draft)
                run_ffmpeg_progress(command, float(target_duration), progress, start_percent=10, end_percent=95)
                if not partial_path.exists() or partial_path.stat().st_size == 0:
                    raise RuntimeError("FFmpeg não criou um arquivo final válido.")
                os.replace(partial_path, final_path)
                cover_path = output_dir / f"{stub}_capa.jpg"
                if create_cover:
                    _render_cover(final_path, cover_path, timestamp=1.0)
                progress.update("concluído", 100, f"Arquivo pronto: {final_path.name}", done=True)
                outputs.append({"platform": platform_name, "video": str(final_path),
                                "cover": str(cover_path) if cover_path.exists() else None,
                                "duration": target_duration, "captions_words": len(words),
                                "hook_start": hook_start, "hook_end": hook_end,
                                "context_start": context_start, "context_end": context_end,
                                "action_end": action_end})
                LOG.info("Render concluído: %s", final_path)
                if economy or config.PERFORMANCE["economy_mode"]:
                    time.sleep(float(config.PERFORMANCE["economy_pause_seconds"]))
            except Exception as exc:
                partial_path.unlink(missing_ok=True)
                LOG.exception("Render falhou (%s): %s", platform_name, exc)
                if isinstance(exc, JobCancelledError):
                    progress.update("cancelado", progress.last_percent, str(exc), done=True,
                                    cancelled=True, process_pid=None)
                else:
                    progress.update("erro", progress.last_percent, str(exc), done=True, process_pid=None)
                if platform == "all" and isinstance(exc, ValueError) and (
                    "Não há contexto suficiente" in str(exc) or "origem é curta demais" in str(exc)
                ):
                    LOG.warning("Destino %s ignorado: %s", platform_name, exc)
                    continue
                raise
            finally:
                ass_path.unlink(missing_ok=True)
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ClipRush — renderiza clipe 9:16 por CPU, sem webcam.")
    parser.add_argument("video", nargs="?", help="Arquivo fonte de gameplay (ou use --analysis-json).")
    parser.add_argument("--analysis-json", default=None, help="analysis.json gerado pelo analyzer.")
    parser.add_argument("--rank", type=int, default=1, help="Ranking do clipe em analysis.json (começa em 1).")
    parser.add_argument("--start", type=float, default=None, help="Início do contexto, em segundos.")
    parser.add_argument("--end", type=float, default=None, help="Fim/duração candidata, em segundos.")
    parser.add_argument("--peak", type=float, default=None, help="Timestamp de pico de ação/áudio.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--platform", choices=["shorts", "reels", "tiktok", "all"], default="shorts")
    parser.add_argument("--title", default="MOMENTO INSANO")
    parser.add_argument("--event-type", choices=list(config.EVENT_WEIGHTS), default=None)
    parser.add_argument("--events-json", default=None, help="JSON opcional com lista de timestamps de eventos adicionais.")
    parser.add_argument("--brightness-json", default=None, help="Lista opcional de janelas escuras {start,end,yavg}.")
    parser.add_argument("--captions", action="store_true")
    parser.add_argument("--language", default="pt", choices=["pt", "en", "auto"])
    parser.add_argument("--music", default=None)
    parser.add_argument("--narration", default=None)
    parser.add_argument("--sfx", default=None)
    parser.add_argument("--watermark", default="")
    parser.add_argument("--no-reaction", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--draft", action="store_true")
    parser.add_argument("--economy", action="store_true")
    parser.add_argument("--no-cover", action="store_true")
    args = parser.parse_args(argv)
    try:
        extras: list[float] = []
        brightness_windows: list[dict[str, float]] = []
        source, start, end, peak = args.video, args.start, args.end, args.peak
        event_type = args.event_type
        analysis_session_dir = None
        if args.analysis_json:
            analysis = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
            if not isinstance(analysis, dict) or not isinstance(analysis.get("clips"), list):
                raise ValueError("analysis.json inválido: esperava um objeto com a lista `clips`.")
            source_data = analysis.get("source") or {}
            if not isinstance(source_data, dict):
                raise ValueError("analysis.json inválido: `source` deve ser um objeto com o caminho do vídeo.")
            clips = analysis.get("clips", [])
            chosen = next((row for row in clips if int(row.get("rank", 0)) == args.rank), None)
            if chosen is None and 1 <= args.rank <= len(clips):
                chosen = clips[args.rank - 1]
            if chosen is None:
                raise ValueError(f"Ranking {args.rank} não existe no JSON de análise.")
            source = source_data.get("path") or source
            analysis_session_dir = analysis.get("session_dir")
            start, end, peak = chosen["start"], chosen["end"], chosen["peak"]
            if event_type is None:
                from postkit import safe_event_type
                event_type = safe_event_type(chosen)
            extras = [float(item.get("timestamp", item)) if isinstance(item, dict) else float(item)
                      for item in chosen.get("events", [])]
            brightness_windows = chosen.get("brightness_windows", [])
        if args.events_json:
            loaded_events = json.loads(Path(args.events_json).read_text(encoding="utf-8"))
            extras.extend(float(item.get("timestamp", item)) if isinstance(item, dict) else float(item)
                          for item in loaded_events)
        if args.brightness_json:
            brightness_windows.extend(json.loads(Path(args.brightness_json).read_text(encoding="utf-8")))
        if not source:
            parser.error("Informe um vídeo ou --analysis-json.")
        if start is None or end is None or peak is None:
            parser.error("Informe --start, --end e --peak, ou use --analysis-json.")
        event_type = event_type or "impacto"
        output_dir = args.output_dir
        if output_dir is None:
            safe_session = False
            if analysis_session_dir:
                try:
                    Path(analysis_session_dir).resolve().relative_to(config.DIRS["sessions"].resolve())
                    safe_session = True
                except (ValueError, OSError):
                    safe_session = False
            if safe_session:
                output_dir = str(Path(analysis_session_dir) / "clips")
            else:
                output_dir = str(config.DIRS["sessions"] / f"render_{Path(source).stem[:32]}_{time.strftime('%Y%m%d_%H%M%S')}")
        outputs = render_clip(source, start, end, peak, output_dir,
                              platform=args.platform, title=args.title, event_type=event_type,
                              captions=args.captions, language=args.language, music=args.music,
                              narration=args.narration, sfx=args.sfx, watermark=args.watermark,
                              reaction=not args.no_reaction, loop=args.loop, draft=args.draft,
                              economy=args.economy, more_events=extras,
                              brightness_windows=brightness_windows,
                              create_cover=not args.no_cover)
        if args.platform == "all" and not outputs:
            raise RuntimeError("Nenhuma plataforma pôde ser renderizada; verifique duração e contexto disponível.")
        for item in outputs:
            print(f"OK {item['platform']}: {item['video']} ({item['duration']}s)")
            if item["cover"]:
                print(f"CAPA: {item['cover']}")
        return 0
    except Exception as exc:
        LOG.exception("CLI renderer falhou")
        print(f"ERRO: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
