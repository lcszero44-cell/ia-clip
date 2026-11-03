from __future__ import annotations

import json
import io
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import ia_manager
import analyzer
import queue_service
from PIL import Image


def _candidates(count: int = 12) -> list[dict]:
    return [
        {"peak": float(index * 10), "audio_score": index / max(1, count),
         "scene_score": 0.2, "movement_score": 0.4, "speech_score": 0.1,
         "speech_seconds": 0, "music_score": 0.2, "tension_score": 0.3,
         "hud_detected": index % 2 == 0, "event_type": "impacto"}
        for index in range(count)
    ]


class IAManagerTests(unittest.TestCase):
    def test_local_endpoint_validation_rejects_remote_hosts(self):
        self.assertEqual(ia_manager._local_url("http://localhost:11434"), "http://localhost:11434")
        self.assertEqual(ia_manager._local_url("http://127.0.0.1:11434/"), "http://127.0.0.1:11434")
        self.assertIsNone(ia_manager._local_url("https://localhost:11434"))
        self.assertIsNone(ia_manager._local_url("http://ollama.example:11434"))

    def test_vision_request_sends_contact_sheets_and_validates_decisions(self):
        captured = []

        def fake_post(base_url, body, timeout):
            captured.append((base_url, body, timeout))
            prompt = json.loads(body["messages"][1]["content"])
            rows = [{
                "candidate_id": item["candidate_id"], "visual_class": "action", "relevant": True,
                "visual_score": 88, "confidence": 0.9, "peak_offset": 1.0,
                "moment_start_offset": -7.0, "moment_end_offset": 8.0,
                "evidence": "Ação observável nos quadros com HUD preservada.",
            } for item in prompt["candidates"]]
            return {"message": {"content": json.dumps({"decisions": rows}, ensure_ascii=False)}}

        status = {"ok": True, "model_installed": True, "models": ["llava:7b", "llama3.2:3b"]}
        with patch.object(ia_manager, "_extract_contact_sheet", return_value=(b"small-jpeg-contact-sheet", [-2.0, 0.0, 2.0])), \
             patch.object(ia_manager, "_post_json", side_effect=fake_post):
            result = ia_manager.analyze_candidates(
                "/local/private/partida.mp4", _candidates(), duration=120, top_n=5,
                max_candidates=6, status=status,
            )

        self.assertEqual(result["engine"], "ollama_vision")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["candidates_assessed"], 6)
        self.assertEqual(len(result["decisions"]), 6)
        self.assertEqual(len(captured), 2)
        for _url, body, timeout in captured:
            self.assertEqual(_url, "http://localhost:11434")
            self.assertEqual(body["model"], "llava:7b")
            self.assertIn("images", body["messages"][1])
            self.assertTrue(all(isinstance(image, str) and image for image in body["messages"][1]["images"]))
            self.assertNotIn("/local/private/partida.mp4", json.dumps(body))
            self.assertGreater(timeout, 0)
        self.assertTrue(all(row["boundary_advisory_valid"] for row in result["decisions"].values()))

    def test_textual_ranking_is_explicitly_not_visual(self):
        def fake_post(_base, body, _timeout):
            payload = json.loads(body["messages"][1]["content"])
            decisions = [{"candidate_id": row["candidate_id"], "priority_score": 76.0,
                          "keep": True, "reason": "Pico de áudio/movimento acima do grupo."}
                         for row in payload["candidates"]]
            return {"message": {"content": json.dumps({"decisions": decisions}, ensure_ascii=False)}}

        status = {"ok": True, "model_installed": False, "models": ["llama3.2:3b"]}
        with patch.object(ia_manager, "_post_json", side_effect=fake_post):
            result = ia_manager.analyze_candidates(
                "gameplay.mp4", _candidates(8), duration=80, top_n=5,
                vision_model="llava:7b", text_model="llama3.2:3b",
                max_candidates=6, status=status,
            )
        self.assertEqual(result["engine"], "ollama_text_audio_fallback")
        self.assertIn("não instalado", result["fallback_reason"])
        self.assertEqual(len(result["decisions"]), 6)
        self.assertTrue(all(row["keep"] for row in result["decisions"].values()))

    def test_uses_installed_llama_vision_when_configured_llava_is_missing(self):
        status = {"ok": True, "model_installed": False, "models": ["llama3.2-vision:11b"]}
        def fake_post(_base, body, _timeout):
            payload = json.loads(body["messages"][1]["content"])
            decisions = [{"candidate_id": row["candidate_id"], "visual_class": "action",
                          "relevant": True, "visual_score": 70, "confidence": 0.8,
                          "peak_offset": 0, "moment_start_offset": -5, "moment_end_offset": 7,
                          "evidence": "Ação visível."} for row in payload["candidates"]]
            return {"message": {"content": json.dumps({"decisions": decisions})}}
        with patch.object(ia_manager, "_extract_contact_sheet", return_value=(b"jpg", [-2, 0, 2])), \
             patch.object(ia_manager, "_post_json", side_effect=fake_post):
            result = ia_manager.analyze_candidates(
                "gameplay.mp4", _candidates(6), duration=60, top_n=5, max_candidates=6,
                vision_model="llava:7b", status=status,
            )
        self.assertEqual(result["engine"], "ollama_vision")
        self.assertEqual(result["model"], "llama3.2-vision:11b")

    def test_offline_fallback_is_clean_and_nonfatal(self):
        result = ia_manager.analyze_candidates(
            "gameplay.mp4", _candidates(6), duration=60, top_n=5,
            status={"ok": False, "models": [], "error": "offline"},
        )
        self.assertEqual(result["engine"], "heuristic_local")
        self.assertEqual(result["status"], "fallback")
        self.assertIn("Ollama local indisponível", result["fallback_reason"])
        self.assertEqual(result["decisions"], {})

    def test_candidate_sampling_covers_the_full_timeline(self):
        candidates = [{"peak": float(i * 60), "audio_score": 0.2, "scene_score": 0.1,
                       "movement_score": 0.1, "speech_score": 0.0, "tension_score": 0.0}
                      for i in range(60)]
        selected = ia_manager._choose_candidates(candidates, duration=3600, max_count=18)
        peaks = [row["peak"] for _, row in selected]
        self.assertGreaterEqual(min(peaks), 0)
        self.assertGreaterEqual(max(peaks), 3000)
        self.assertLessEqual(len(selected), 18)

    def test_ai_boundaries_are_included_in_final_cut_window(self):
        candidate = {"peak": 50.0, "audio_score": 0.1, "group_start": 45.0, "group_end": 65.0,
                     "ai_boundary_advisory_valid": True}
        start, end = analyzer._cut_window(candidate, [candidate], duration=120.0, fixed_duration=20)
        self.assertLessEqual(start, candidate["group_start"])
        self.assertGreaterEqual(end, candidate["group_end"])
        self.assertLessEqual(end - start, 90)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg não está disponível")
    def test_extract_contact_sheet_from_real_synthetic_video(self):
        with tempfile.TemporaryDirectory(prefix="cliprush-keyframes-") as temp:
            source = Path(temp) / "synthetic.mp4"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                "testsrc=size=640x360:rate=12", "-t", "8", "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-threads", "1", str(source),
            ], check=True, capture_output=True)
            image_bytes, offsets = ia_manager._extract_contact_sheet(source, 4.0)
        self.assertTrue(image_bytes)
        self.assertEqual(offsets, [-4.0, -2.0, 0.0])
        image = Image.open(io.BytesIO(image_bytes))
        self.assertEqual(image.size, (960, 180))

    def test_editorial_safety_rejects_unconfirmed_specific_event(self):
        copy = {"titles": ["A motosserra apareceu!", "O que veio depois?", "Isso mudou a run"],
                "hook": "Fique até o final.", "description": "Um momento de gameplay.",
                "cta": "O que você faria?", "hashtags": "#Gameplay", "pinned_comment": "Comente."}
        self.assertFalse(ia_manager.editorial_copy_is_safe(copy, "impacto"))
        self.assertTrue(ia_manager.editorial_copy_is_safe(copy, "motosserra"))

    def test_auto_render_persists_asset_content_and_post(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "gameplay.mp4"
            source.write_bytes(b"local game source")
            work = root / "job"
            work.mkdir()
            result_path = root / "analysis" / "analysis.json"
            result_path.parent.mkdir()
            result_path.write_text("{}", encoding="utf-8")
            video = root / "analysis" / "auto_render" / "clip.mp4"
            cover = root / "analysis" / "auto_render" / "cover.jpg"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fake mp4")
            cover.write_bytes(b"fake jpg")
            clip = {"rank": 1, "start": 1.0, "end": 30.0, "peak": 8.0,
                    "score": 87.0, "duration_seconds": 29.0, "event_type": "impacto",
                    "confidence": 0.3, "events": [], "brightness_windows": []}
            job = {"id": "job123", "source_path": str(source), "work_dir": str(work)}
            result = {"clips": [clip], "summary": {"ai_engine": "ollama_vision"}}
            options = {"platform": "shorts", "auto_render": True, "ollama_enabled": False,
                       "game_name": "Jogo teste", "captions": False}
            with patch("library.load_library", return_value={"items": []}), \
                 patch("renderer.render_clip", return_value=[{
                     "platform": "shorts", "video": str(video), "cover": str(cover), "duration": 29.0,
                 }]) as render, \
                 patch("queue_service.save_rendered_asset") as save_asset, \
                 patch("queue_service.save_clip_content") as save_content, \
                 patch("queue_service.create_post") as create_post:
                summary = queue_service._auto_render_analysis(job, result, result_path, options)

        self.assertEqual(len(summary["rendered"]), 1)
        self.assertEqual(summary["failed"], [])
        render.assert_called_once()
        save_asset.assert_called_once()
        save_content.assert_called_once()
        create_post.assert_called_once()
        self.assertEqual(create_post.call_args.args[0]["source_job_id"], "job123")
        self.assertEqual(create_post.call_args.args[0]["video_path"], str(video.resolve()))


if __name__ == "__main__":
    unittest.main()
