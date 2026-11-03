from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from analyzer import analyze_video
from analyzer import (_audio_context_scores, _classify_cutscene, _classify_event, _detect_speech_seconds, _hud_score,
                      _ideal_clip_count, _make_peak_candidates, _rank_and_select)
from analyzer import _speech_probe_candidates
from common import JobCancelledError, ProgressWriter, probe_media, run_ffmpeg_progress
import renderer


class MotorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="cliprush-test-")
        cls.work = Path(cls.temp.name)
        cls.video = cls.work / "gameplay_sintetica.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=70",
            "-f", "lavfi", "-i", "sine=frequency=420:sample_rate=16000:duration=70",
            "-filter:a", "volume='if(between(t,6,9),5,0.06)+if(between(t,32,35),5,0.06)+if(between(t,62,65),5,0.06)':eval=frame",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", "-shortest", str(cls.video),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("Não foi possível gerar fixture FFmpeg: " + result.stderr[-1500:])
        cls.truncated_video = cls.work / "gameplay_audio_truncado.mp4"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15:duration=70",
            "-f", "lavfi", "-i", "sine=frequency=420:sample_rate=16000:duration=20",
            "-map", "0:v:0", "-map", "1:a:0", "-t", "70",
            "-vf", "drawbox=x=5:y=5:w=40:h=30:color=white:t=fill,drawbox=x=275:y=5:w=40:h=30:color=white:t=fill",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "64k", str(cls.truncated_video),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError("Não foi possível gerar fixture de áudio truncado: " + result.stderr[-1500:])

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_probe_and_analysis_cover_start_middle_end(self):
        original = copy.deepcopy(config.ANALYSIS)
        try:
            config.ANALYSIS.update({
                "clip_min_seconds": 12, "clip_max_seconds": 18,
                "clip_target_count": 3, "start_lead_seconds": 3, "end_tail_seconds": 3,
                "min_clip_gap_seconds": 6, "max_same_event_type": 10,
                "quick_visual_candidates": 12, "candidate_limit": 80,
            })
            info = probe_media(self.video)
            self.assertGreaterEqual(info["duration"], 69)
            self.assertTrue(info["audio"])
            result = analyze_video(self.video, mode="quick", top_n=3, debug=True,
                                   output_root=self.work / "sessions", use_cache=False)
            clips = result["clips"]
            self.assertEqual(len(clips), 3)
            peaks = sorted(row["peak"] for row in clips)
            self.assertLess(peaks[0], 20, f"corte inicial ausente: {peaks}")
            self.assertTrue(20 <= peaks[1] <= 50, f"corte do meio ausente: {peaks}")
            self.assertGreater(peaks[2], 50, f"corte final ausente: {peaks}")
            self.assertEqual([row["rank"] for row in clips], [1, 2, 3])
            self.assertTrue(all(0 <= row["score"] <= 100 for row in clips))
            self.assertTrue(all(row["duration_seconds"] >= 12 for row in clips))
            self.assertTrue(Path(result["session_dir"], "calibracao.csv").exists())
            self.assertTrue(Path(result["session_dir"], "progress.json").exists())
        finally:
            config.ANALYSIS.clear()
            config.ANALYSIS.update(original)

    def test_automatic_count_and_generic_type_fallback(self):
        self.assertEqual(_ideal_clip_count(60), 5)
        self.assertEqual(_ideal_clip_count(3600), 10)
        original = copy.deepcopy(config.ANALYSIS)
        try:
            config.ANALYSIS.update({"clip_min_seconds": 5, "clip_max_seconds": 10,
                                    "min_clip_gap_seconds": 2, "max_same_event_type": 3,
                                    "start_lead_seconds": 1, "end_tail_seconds": 1})
            candidates = [{"peak": 6 + 12 * i, "audio_score": 0.8 - i * 0.01,
                           "scene_score": 0.5, "movement_score": 0.5,
                           "event_type": "impacto", "discard_reason": ""}
                          for i in range(10)]
            selected = _rank_and_select(candidates, duration=120, top_n=10, fixed_duration=6)
            self.assertEqual(len(selected), 10)
            self.assertEqual([row["rank"] for row in selected], list(range(1, 11)))
        finally:
            config.ANALYSIS.clear()
            config.ANALYSIS.update(original)

    def test_sustained_low_frequency_audio_does_not_claim_chainsaw(self):
        rows = [{"rms": 0.08, "crest": 2.0, "low_ratio": 0.62,
                 "high_ratio": 0.05, "centroid": 620.0} for _ in range(4)]
        event, confidence, reason = _classify_event(None, 2.0, rows, [])
        self.assertEqual(event, "impacto")
        self.assertLess(confidence, 0.5)
        self.assertIn("não confirmado", reason)

    def test_unconfirmed_specific_event_does_not_get_ranking_bonus(self):
        base = {"peak": 40.0, "audio_score": 0.7, "scene_score": 0.0,
                "movement_score": 0.0, "hud_score": 0.0, "event_type": "impacto",
                "confidence": 0.3, "classification_reason": "evento inconclusivo", "discard_reason": ""}
        inferred = {**base, "event_type": "motosserra", "confidence": 0.58,
                    "classification_reason": "grave sustentado/áspero (heurística)"}
        score_base = _rank_and_select([copy.deepcopy(base)], 200, 1, fixed_duration=20)[0]["raw_score"]
        score_inferred = _rank_and_select([copy.deepcopy(inferred)], 200, 1, fixed_duration=20)[0]["raw_score"]
        self.assertAlmostEqual(score_base, score_inferred)

    def test_one_hour_audio_candidates_cover_the_full_timeline(self):
        rows = []
        for second in range(3600):
            active = second % 60 == 30
            rows.append({"second": float(second), "rms": 0.30 if active else 0.01,
                         "peak": 0.90 if active else 0.04, "crest": 8.0 if active else 2.0,
                         "low_ratio": 0.20, "mid_ratio": 0.30,
                         "high_ratio": 0.50 if active else 0.10, "centroid": 900.0})
        candidates = _make_peak_candidates(rows, None, [], video_duration=3600.0)
        minute_bins = {int(row["peak"] // 60) for row in candidates}
        self.assertGreaterEqual(len(candidates), 55)
        self.assertLess(min(row["peak"] for row in candidates), 60)
        self.assertGreater(max(row["peak"] for row in candidates), 3540)
        self.assertGreaterEqual(len(minute_bins), 55)

    def test_cutscene_detection_requires_continuous_speech_no_hud_and_cuts(self):
        speech_rows = [{"rms": 0.04, "low_ratio": 0.10, "mid_ratio": 0.70,
                        "high_ratio": 0.20, "centroid": 1400.0} for _ in range(9)]
        audio = _audio_context_scores(speech_rows, center=4.5, reference_rms=0.04)
        self.assertTrue(audio["speech_continuous"])
        candidate = {"event_type": "combo", "confidence": 0.6}
        is_scene = _classify_cutscene(candidate, {"hud_score": 0.05, "camera_cuts": 4}, audio)
        self.assertTrue(is_scene)
        self.assertEqual(candidate["event_type"], "cena")
        self.assertIn("fala contínua", candidate["classification_reason"])

        gameplay = {"event_type": "combo", "confidence": 0.6}
        self.assertFalse(_classify_cutscene(gameplay, {"hud_score": 0.9, "camera_cuts": 4}, audio))
        self.assertEqual(gameplay["event_type"], "combo")
        no_cuts = {"event_type": "combo", "confidence": 0.6}
        self.assertFalse(_classify_cutscene(no_cuts, {"hud_score": 0.05, "camera_cuts": 0}, audio))

        quiet_rows = [{"second": float(i), "rms": 0.0, "mid_ratio": 0.0,
                       "high_ratio": 0.0, "centroid": 0.0} for i in range(9)]
        vad_audio = _audio_context_scores(quiet_rows, 4.5, 0.01, np.ones(9, dtype=np.float32))
        self.assertTrue(vad_audio["speech_continuous"])
        self.assertEqual(vad_audio["speech_detector"], "silero_vad")

    def test_silero_vad_returns_no_speech_for_silent_wav(self):
        silent = self.work / "silent-vad.wav"
        sf.write(silent, np.zeros(2 * 16000, dtype=np.float32), 16000)
        coverage = _detect_speech_seconds(silent, 2.0)
        self.assertIsNotNone(coverage)
        self.assertEqual(float(np.sum(coverage)), 0.0)

    def test_vad_speech_interval_creates_a_visual_probe(self):
        coverage = np.zeros(120, dtype=np.float32)
        coverage[35:45] = 0.8
        probes = _speech_probe_candidates(coverage, 120.0)
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["peak"], 40.0)
        self.assertIn("VAD", probes[0]["classification_reason"])

    def test_hud_heuristic_detects_persistent_corner_overlay(self):
        previous = np.full((180, 320), 50, dtype=np.float32)
        yy, xx = np.indices((36, 70))
        pattern = np.where((xx // 3 + yy // 3) % 2, 230.0, 20.0).astype(np.float32)
        previous[:36, :70] = pattern
        current = previous.copy()
        current[45:135, 80:240] = 180.0
        score_with_hud = _hud_score(current, previous)
        score_without_hud = _hud_score(np.pad(np.full((90, 160), 180.0, dtype=np.float32),
                                               ((45, 45), (80, 80)), constant_values=50),
                                       np.full((180, 320), 50, dtype=np.float32))
        self.assertGreaterEqual(score_with_hud, config.SCENES["hud_threshold"])
        self.assertLess(score_without_hud, config.SCENES["hud_threshold"])

    def test_scene_modes_cap_ignore_and_prioritize_gameplay(self):
        candidates = []
        for index in range(12):
            is_scene = index < 4
            candidates.append({"peak": 10 + 100 * index, "audio_score": 0.8,
                               "scene_score": 0.7, "movement_score": 0.6,
                               "hud_score": 0.0 if is_scene else 0.8,
                               "speech_score": 0.9 if is_scene else 0.0,
                               "music_score": 0.8 if is_scene else 0.0,
                               "tension_score": 0.7 if is_scene else 0.0,
                               "event_type": "cena" if is_scene else "impacto",
                               "confidence": 0.8, "discard_reason": ""})
        limited = _rank_and_select(copy.deepcopy(candidates), 1300, 10, fixed_duration=20,
                                   scene_mode="manter", max_scene_share=0.30)
        self.assertEqual(len(limited), 10)
        self.assertLessEqual(sum(row["event_type"] == "cena" for row in limited), 3)
        ignored_candidates = copy.deepcopy(candidates)
        ignored = _rank_and_select(ignored_candidates, 1300, 10, fixed_duration=20,
                                   scene_mode="ignorar", max_scene_share=1.0)
        self.assertFalse(any(row["event_type"] == "cena" for row in ignored))
        ignored_scenes = [row for row in ignored_candidates if row["event_type"] == "cena"]
        self.assertTrue(all(row["discard_reason"] and row["score"] > 0 for row in ignored_scenes))

        action = {"peak": 20, "audio_score": 0.4, "scene_score": 0.8,
                  "movement_score": 0.8, "hud_score": 0.0, "event_type": "impacto",
                  "confidence": 0.5, "discard_reason": ""}
        baseline = _rank_and_select([copy.deepcopy(action)], 200, 1, fixed_duration=20,
                                   scene_mode="manter", max_scene_share=1.0)[0]["raw_score"]
        prioritized = _rank_and_select([copy.deepcopy(action)], 200, 1, fixed_duration=20,
                                       scene_mode="priorizar-gameplay", max_scene_share=1.0)[0]["raw_score"]
        self.assertGreater(prioritized, baseline)

    def test_scene_score_uses_speech_music_tension_not_camera_cuts(self):
        scene = {"peak": 50, "audio_score": 1.0, "scene_score": 1.0,
                 "movement_score": 1.0, "hud_score": 0.0, "camera_cuts": 30,
                 "speech_score": 0.8, "music_score": 0.5, "tension_score": 0.4,
                 "event_type": "cena", "confidence": 0.9, "discard_reason": ""}
        score_many_cuts = _rank_and_select([copy.deepcopy(scene)], 200, 1, fixed_duration=20,
                                           scene_mode="manter", max_scene_share=1.0)[0]["raw_score"]
        scene["camera_cuts"] = 0
        scene["scene_score"] = scene["movement_score"] = 0.0
        score_no_cuts = _rank_and_select([copy.deepcopy(scene)], 200, 1, fixed_duration=20,
                                         scene_mode="manter", max_scene_share=1.0)[0]["raw_score"]
        self.assertAlmostEqual(score_many_cuts, score_no_cuts)
        self.assertAlmostEqual(score_no_cuts, 62.5)

        action = {"peak": 50, "audio_score": 0.3, "scene_score": 1.0,
                  "movement_score": 1.0, "hud_score": 0.0, "camera_cuts": 10,
                  "event_type": "impacto", "confidence": 0.5, "discard_reason": ""}
        without_hud = _rank_and_select([copy.deepcopy(action)], 200, 1, fixed_duration=20,
                                       scene_mode="manter", max_scene_share=1.0)[0]["raw_score"]
        action["hud_score"] = 0.9
        with_hud = _rank_and_select([copy.deepcopy(action)], 200, 1, fixed_duration=20,
                                    scene_mode="manter", max_scene_share=1.0)[0]["raw_score"]
        self.assertGreater(with_hud, without_hud)

    def test_scene_reaction_is_neutral(self):
        ass = self.work / "scene-reaction.ass"
        renderer._write_ass(ass, 1080, 1920, "CENA", [], 7, 9, 4, 20, 2,
                            "cena", [8.0], 18.0, reactions=True)
        text = ass.read_text(encoding="utf-8-sig")
        self.assertIn("VOCÊ VIU ISSO?", text)
        self.assertNotIn("COMBO INSANO!", text)

    def test_video_tail_is_not_dropped_when_audio_ends_early(self):
        original = copy.deepcopy(config.ANALYSIS)
        try:
            config.ANALYSIS.update({"clip_min_seconds": 12, "clip_max_seconds": 18,
                                    "min_clip_gap_seconds": 6, "quick_visual_candidates": 12,
                                    "candidate_limit": 80})
            info = probe_media(self.truncated_video)
            self.assertGreater(info["duration"], 65)
            result = analyze_video(self.truncated_video, mode="quick", top_n=3,
                                   output_root=self.work / "truncated-sessions", use_cache=False)
            probes = [row for row in result["candidates"] if row["audio_score"] == 0.0]
            self.assertTrue(probes, "não foram criados probes no trecho sem áudio")
            self.assertTrue(any(row["peak"] > 50 for row in probes))
            self.assertTrue(any(row["peak"] > 50 for row in result["clips"]),
                            f"nenhum clipe selecionado no fim: {[row['peak'] for row in result['clips']]}")
        finally:
            config.ANALYSIS.clear()
            config.ANALYSIS.update(original)

    def test_cancel_flag_stops_ffmpeg_and_sets_cancelled_state(self):
        session = self.work / "cancel-test"
        session.mkdir()
        progress = ProgressWriter(session)
        progress.request_cancel()
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-re", "-f", "lavfi",
                   "-i", "testsrc2=size=64x64:rate=10:duration=5", "-t", "5", "-f", "null", "-"]
        with self.assertRaises(JobCancelledError):
            run_ffmpeg_progress(command, 5, progress, start_percent=0, end_percent=90)
        state = json.loads(progress.path.read_text(encoding="utf-8"))
        self.assertTrue(state["cancelled"])
        self.assertTrue(state["done"])
        self.assertIsNone(state["process_pid"])

    def test_whisper_receives_numpy_audio_without_metadata_errors(self):
        calls = {}

        class FakeWord:
            start, end, word = 0.25, 0.75, "clutch"

        class FakeSegment:
            words = [FakeWord()]

        class FakeWhisper:
            def transcribe(self, audio, **kwargs):
                calls["is_numpy"] = isinstance(audio, np.ndarray)
                calls["kwargs"] = kwargs
                return iter([FakeSegment()]), None

        fake_module = types.ModuleType("faster_whisper")
        fake_module.WhisperModel = lambda *args, **kwargs: FakeWhisper()
        old_module = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = fake_module
        renderer._load_whisper_model.cache_clear()
        try:
            words = renderer._transcribe(self.video, 2.0, 12.0, "pt")
            self.assertTrue(calls["is_numpy"])
            self.assertNotIn("metadata_errors", calls["kwargs"])
            self.assertEqual(words[0]["text"], "clutch")
            self.assertAlmostEqual(words[0]["start"], 2.25, places=2)
        finally:
            renderer._load_whisper_model.cache_clear()
            if old_module is None:
                sys.modules.pop("faster_whisper", None)
            else:
                sys.modules["faster_whisper"] = old_module

    def test_safe_zones_in_ass(self):
        ass = self.work / "safe-zone.ass"
        renderer._write_ass(
            ass, 1080, 1920, "PARRY NO LIMITE",
            [{"start": 8.0, "end": 8.4, "text": "perfeito"}],
            hook_start=7.0, hook_end=8.8, context_start=4.0, context_end=27.0,
            hook_len=1.8, event_type="parry_faca", event_times=[8.0],
            duration=25.0, reactions=True,
        )
        text = ass.read_text(encoding="utf-8-sig")
        self.assertIn(r"\pos(540,275)", text)     # Título dentro de y=150–400.
        self.assertIn(r"\pos(540,1250)", text)    # Legenda dentro de y=1150–1350.
        self.assertIn("PARRY NO LIMITE", text)
        self.assertIn("PERFEITO", text)
        # Confirma pixels realmente rasterizados, não apenas os metadados do ASS.
        filter_ass = renderer._filter_path(ass)
        font_dir = renderer._filter_path(config.DIRS["fonts"])
        for frame_index, y0, y1, label in ((15, 150, 400, "título"), (33, 1150, 1350, "legenda")):
            vf = f"ass=filename='{filter_ass}':fontsdir='{font_dir}',select='eq(n,{frame_index})'"
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "color=c=black:s=1080x1920:r=30:d=2.5", "-vf", vf, "-fps_mode", "vfr",
                "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ]
            rendered = subprocess.run(command, capture_output=True, timeout=30)
            self.assertEqual(rendered.returncode, 0, rendered.stderr.decode("utf-8", errors="replace"))
            frame = np.frombuffer(rendered.stdout, dtype=np.uint8).reshape((1920, 1080, 3))
            band = frame[y0:y1]
            self.assertGreater(int(np.count_nonzero(np.max(band, axis=2) > 80)), 10,
                               f"quadros rasterizados não encontraram {label} na zona segura")

    def test_optional_narration_music_and_sfx_mix(self):
        rate = 48000
        t = np.arange(rate * 3, dtype=np.float32) / rate
        narration = self.work / "narracao.wav"
        music = self.work / "musica.wav"
        sfx = self.work / "efeito.wav"
        sf.write(narration, 0.12 * np.sin(2 * np.pi * 220 * t), rate)
        sf.write(music, 0.08 * np.sin(2 * np.pi * 110 * t), rate)
        sf.write(sfx, 0.18 * np.sin(2 * np.pi * 900 * t[:rate]), rate)
        outputs = renderer.render_clip(
            self.video, start=4, end=16, peak=8, output_dir=self.work / "mix-render",
            platform="shorts", title="MIX TESTE", event_type="combo", draft=True,
            narration=narration, music=music, sfx=sfx, create_cover=False,
        )
        self.assertEqual(len(outputs), 1)
        media = probe_media(outputs[0]["video"])
        self.assertTrue(media["audio"])
        self.assertEqual((media["width"], media["height"]), (540, 960))

    def test_analysis_json_cli_renders_ranked_clip(self):
        analysis_path = self.work / "analysis.json"
        payload = {
            "source": {"path": str(self.video)},
            "session_dir": str(self.work / "outside-session"),
            "clips": [{"rank": 1, "start": 4.0, "end": 16.0, "peak": 8.0,
                       "event_type": "combo", "events": [{"timestamp": 8.0}],
                       "brightness_windows": []}],
        }
        analysis_path.write_text(json.dumps(payload), encoding="utf-8")
        output_dir = self.work / "cli-render"
        code = renderer.main(["--analysis-json", str(analysis_path), "--rank", "1",
                              "--platform", "shorts", "--draft", "--no-cover",
                              "--output-dir", str(output_dir)])
        self.assertEqual(code, 0)
        self.assertEqual(len(list(output_dir.glob("*.mp4"))), 1)

    def test_render_vertical_with_blur_hook_audio_cover_and_loop(self):
        output_dir = self.work / "rendered"
        output_dir.mkdir()
        outputs = renderer.render_clip(
            self.video, start=4.0, end=28.0, peak=8.0,
            output_dir=output_dir, platform="shorts", title="OLHA ESSE MOMENTO",
            event_type="combo", captions=False, draft=True, loop=True,
            create_cover=True,
        )
        self.assertEqual(len(outputs), 1)
        video_path = Path(outputs[0]["video"])
        cover_path = Path(outputs[0]["cover"])
        media = probe_media(video_path)
        self.assertEqual((media["width"], media["height"]), (540, 960))
        self.assertTrue(media["audio"])
        self.assertGreaterEqual(media["duration"], 23.8)
        self.assertLessEqual(media["duration"], 24.2)
        self.assertTrue(cover_path.is_file())
        cover_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "json", str(cover_path)],
            capture_output=True, text=True, check=True,
        )
        cover_stream = json.loads(cover_probe.stdout)["streams"][0]
        self.assertEqual((cover_stream["width"], cover_stream["height"]), (540, 960))

        def decode_rgb_frame(path: Path, timestamp: float | None = None) -> np.ndarray:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path)]
            if timestamp is None:
                command += ["-frames:v", "1"]
            else:
                command += ["-vf", f"select=gte(t\\,{timestamp:.3f})", "-frames:v", "1"]
            command += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
            decoded = subprocess.run(command, capture_output=True, check=True, timeout=30)
            return np.frombuffer(decoded.stdout, dtype=np.uint8).reshape((960, 540, 3))

        cover_frame = decode_rgb_frame(cover_path)
        rendered_plus_one = decode_rgb_frame(video_path, 1.0)
        self.assertLess(float(np.mean(np.abs(cover_frame.astype(np.int16) - rendered_plus_one.astype(np.int16)))),
                        12.0, "A capa deve corresponder ao frame final do render em +1,0 s.")
        first_frame = decode_rgb_frame(video_path, 0.0)
        self.assertLess(float(first_frame.mean()), float(rendered_plus_one.mean()) * 0.50,
                        "A abertura deve começar em preto e revelar a imagem suavemente, sem flash branco.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
