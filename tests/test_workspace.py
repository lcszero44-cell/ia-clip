from __future__ import annotations

import copy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import common
import local_store
import queue_service
from library import build_library_document, validate_library_document
from postkit import generate_post_kit, safe_event_type
from scripts.seed_library import CATALOG


class LocalWorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cliprush-workspace-")
        self.root = Path(self.temp.name)
        self.old_data = config.DIRS["data"]
        self.old_db = local_store.DB_PATH
        self.old_settings = local_store.SETTINGS_PATH
        config.DIRS["data"] = self.root / "data"
        local_store.DB_PATH = config.DIRS["data"] / "cliprush_local.db"
        local_store.SETTINGS_PATH = config.DIRS["data"] / "settings.json"
        local_store._INITIALIZED = False
        local_store.init_storage()

    def tearDown(self):
        local_store._INITIALIZED = False
        config.DIRS["data"] = self.old_data
        local_store.DB_PATH = self.old_db
        local_store.SETTINGS_PATH = self.old_settings
        self.temp.cleanup()

    def test_settings_roundtrip_and_clamping(self):
        saved = local_store.save_settings({"scene_mode": "ignorar", "max_scene_share": 4.0,
                                          "threads": 50, "game_name": "  Meu jogo  ",
                                          "ollama_enabled": True, "ollama_base_url": "http://localhost:11434/",
                                          "ollama_model": "llama3.2:3b", "ollama_timeout": 150,
                                          "vision_ai_enabled": True, "vision_model": "  llama3.2-vision:11b  ",
                                          "vision_timeout": 600, "max_vision_candidates": 99,
                                          "auto_render": True})
        loaded = local_store.load_settings()
        self.assertEqual(loaded["scene_mode"], "ignorar")
        self.assertEqual(loaded["max_scene_share"], 1.0)
        self.assertEqual(loaded["threads"], 8)
        self.assertEqual(loaded["game_name"], "Meu jogo")
        self.assertTrue(loaded["ollama_enabled"])
        self.assertEqual(loaded["ollama_base_url"], "http://localhost:11434")
        self.assertEqual(loaded["ollama_model"], "llama3.2:3b")
        self.assertEqual(loaded["ollama_timeout"], 90)
        self.assertTrue(loaded["vision_ai_enabled"])
        self.assertEqual(loaded["vision_model"], "llama3.2-vision:11b")
        self.assertEqual(loaded["vision_timeout"], 300)
        self.assertEqual(loaded["max_vision_candidates"], 18)
        self.assertTrue(loaded["auto_render"])
        self.assertTrue(local_store.SETTINGS_PATH.is_file())

    def test_atomic_json_retries_transient_windows_permission_error(self):
        target = self.root / "progress.json"
        real_replace = common.os.replace
        attempts = {"count": 0}

        def transient_lock(source, destination):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise PermissionError(5, "acesso temporariamente negado")
            return real_replace(source, destination)

        with patch("common.os.replace", side_effect=transient_lock):
            common.atomic_write_json(target, {"percent": 50})
        self.assertEqual(attempts["count"], 3)
        self.assertEqual(common.read_json(target), {"percent": 50})
        self.assertEqual(list(self.root.glob("progress.json.*.tmp")), [])

    def test_queue_claim_cancel_and_status(self):
        source = self.root / "gameplay.mp4"
        source.write_bytes(b"fake-video")
        job_id = local_store.insert_queue_job(source, self.root / "work", {"analysis_mode": "quick"}, "testjob001")
        self.assertEqual(job_id, "testjob001")
        claimed = queue_service.claim_next_job()
        self.assertEqual(claimed["id"], job_id)
        self.assertEqual(claimed["status"], "processando")
        self.assertEqual(queue_service.cancel_job(job_id), "cancelamento_solicitado")
        self.assertTrue(queue_service.requested_cancel(job_id))
        queue_service.update_job(job_id, status="cancelado", error="teste", finished=True)
        row = local_store.get_queue_job(job_id)
        self.assertEqual(row["status"], "cancelado")
        self.assertEqual(row["error"], "teste")

    def test_nested_render_progress_is_discovered_and_cancelled(self):
        work = self.root / "work"
        nested = work / "session" / "auto_render" / "clip_session"
        nested.mkdir(parents=True)
        progress_path = nested / "progress.json"
        common.atomic_write_json(progress_path, {"stage": "render", "percent": 42, "message": "FFmpeg"})
        progress = queue_service._progress_for(work)
        self.assertEqual(progress["stage"], "render")
        self.assertEqual(Path(progress["progress_path"]), progress_path)
        self.assertTrue(queue_service._write_cancel_flag(work))
        self.assertTrue((nested / "cancel.flag").is_file())

    def test_worker_persists_completed_analysis_path(self):
        source = self.root / "source.mp4"
        source.write_bytes(b"fake-video")
        work_dir = self.root / "worker-output"
        job_id = local_store.insert_queue_job(source, work_dir, {"analysis_mode": "quick"})

        def fake_analyze(_source, **kwargs):
            session_dir = Path(kwargs["output_root"]) / "session-test"
            session_dir.mkdir(parents=True)
            (session_dir / "analysis.json").write_text('{"clips": []}', encoding="utf-8")
            return {"session_id": "session-test", "session_dir": str(session_dir),
                    "summary": {"candidate_count": 3}, "mode": "quick", "scene_mode": "manter"}

        claimed = queue_service.claim_next_job()
        fake_module = types.SimpleNamespace(analyze_video=fake_analyze)
        with patch.dict(sys.modules, {"analyzer": fake_module}), patch("queue_service._set_worker_priority"):
            queue_service._run_job(claimed)
        completed = local_store.get_queue_job(job_id)
        self.assertEqual(completed["status"], "concluido")
        self.assertTrue(Path(completed["result_path"]).is_file())
        self.assertEqual(completed["result"]["summary"]["candidate_count"], 3)

    def test_calendar_metrics_and_checklist_persist(self):
        post_id = local_store.create_post({"title": "Teste editorial", "platform": "shorts",
                                           "status": "rascunho", "game_name": "Jogo"})
        local_store.update_post(post_id, {"status": "agendado", "scheduled_at": "2026-10-19T18:30"})
        self.assertEqual(local_store.get_post(post_id)["scheduled_at"], "2026-10-19T18:30")
        metrics_id = local_store.add_metrics(post_id, {"measured_at": "2026-10-20", "views": 1000,
                                                       "likes": 120, "comments": 8, "shares": 10,
                                                       "saves": 12, "retention_pct": 68})
        rows = local_store.list_metrics(post_id)
        self.assertEqual(rows[0]["id"], metrics_id)
        self.assertEqual(rows[0]["views"], 1000)
        items = [("hook", "Gancho"), ("hud", "HUD")]
        checklist = local_store.ensure_checklist(post_id, items)
        self.assertEqual(len(checklist), 2)
        local_store.save_checklist(post_id, {"hook": {"checked": True, "note": "ok"},
                                             "hud": {"checked": False, "note": "rever"}})
        checklist = local_store.ensure_checklist(post_id, items)
        self.assertTrue(checklist[0]["checked"])
        self.assertEqual(checklist[0]["note"], "ok")
        local_store.delete_post(post_id)
        self.assertIsNone(local_store.get_post(post_id))
        self.assertEqual(local_store.list_metrics(post_id), [])

    def test_rendered_asset_lookup_is_durable(self):
        analysis_path = self.root / "analysis.json"
        analysis_path.write_text("{}", encoding="utf-8")
        video = self.root / "clip.mp4"
        video.write_bytes(b"video")
        cover = self.root / "cover.jpg"
        cover.write_bytes(b"cover")
        local_store.save_rendered_asset(analysis_path, 2, "shorts", video, cover, "TÍTULO")
        rows = local_store.list_rendered_assets(analysis_path, 2, "shorts")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["title"], "TÍTULO")
        self.assertEqual(rows[0]["video_path"], str(video.resolve()))

    def test_clip_content_editor_roundtrip(self):
        analysis_path = self.root / "analysis.json"
        analysis_path.write_text("{}", encoding="utf-8")
        local_store.save_clip_content(analysis_path, 2, "shorts", {
            "game_name": "Meu jogo", "overlay_title": "JOGADA LIMPA",
            "publish_title": "Escape no último segundo", "caption": "Escape no último segundo\nTexto",
            "hashtags": "#gaming", "pinned_comment": "Qual foi sua parte favorita?",
            "confirmed_event": "", "ai_titles": ["Título A", "Título B", "Título C"],
        })
        saved = local_store.get_clip_content(analysis_path, 2, "shorts")
        self.assertEqual(saved["overlay_title"], "JOGADA LIMPA")
        self.assertEqual(saved["publish_title"], "Escape no último segundo")
        self.assertEqual(saved["game_name"], "Meu jogo")
        self.assertEqual(saved["ai_titles"], ["Título A", "Título B", "Título C"])

    def test_legacy_database_migrates_ai_titles_without_data_loss(self):
        old_db_path = local_store.DB_PATH
        legacy_db_path = self.root / "legacy.db"
        try:
            local_store.DB_PATH = legacy_db_path
            local_store._INITIALIZED = False
            with local_store.connect() as db:
                db.execute("""CREATE TABLE clip_content (
                    analysis_path TEXT NOT NULL, clip_rank INTEGER NOT NULL, platform TEXT NOT NULL,
                    game_name TEXT NOT NULL DEFAULT '', overlay_title TEXT NOT NULL DEFAULT '',
                    publish_title TEXT NOT NULL DEFAULT '', caption TEXT NOT NULL DEFAULT '',
                    hashtags TEXT NOT NULL DEFAULT '', pinned_comment TEXT NOT NULL DEFAULT '',
                    confirmed_event TEXT NOT NULL DEFAULT '', updated_at REAL NOT NULL,
                    PRIMARY KEY (analysis_path, clip_rank, platform))""")
                db.execute("INSERT INTO clip_content (analysis_path, clip_rank, platform, publish_title, updated_at) "
                           "VALUES (?, ?, ?, ?, ?)", ("old-analysis", 1, "shorts", "Título antigo", 1.0))
            local_store._INITIALIZED = False
            local_store.init_storage()
            with local_store.connect() as db:
                columns = {row["name"] for row in db.execute("PRAGMA table_info(clip_content)")}
                row = db.execute("SELECT publish_title, ai_titles_json FROM clip_content").fetchone()
            self.assertIn("ai_titles_json", columns)
            self.assertEqual(row["publish_title"], "Título antigo")
            self.assertEqual(row["ai_titles_json"], "[]")
        finally:
            local_store.DB_PATH = old_db_path
            local_store._INITIALIZED = False
            local_store.init_storage()

    def test_unconfirmed_chainsaw_never_enters_generated_title(self):
        clip = {"event_type": "motosserra", "confidence": 0.58,
                "classification_reason": "grave sustentado/áspero (heurística)", "peak": 14.0}
        self.assertEqual(safe_event_type(clip), "impacto")
        kit = generate_post_kit(clip, "Meu jogo", "shorts", build_library_document(CATALOG))
        text = " ".join([*kit["titles"], kit["caption"], kit["reaction_text"]]).lower()
        self.assertNotIn("motosserra", text)
        tags = kit["hashtags"].lower()
        self.assertNotIn("#re4", tags)
        self.assertNotIn("#residentevil", tags)
        confirmed = {**clip, "user_confirmed_event": "motosserra"}
        self.assertEqual(safe_event_type(confirmed), "motosserra")
        confirmed_kit = generate_post_kit(confirmed, "Resident Evil 4", "shorts",
                                          build_library_document(CATALOG))
        self.assertEqual(confirmed_kit["event_type"], "motosserra")

    def test_editorial_library_count_and_cutscene_kit(self):
        document = build_library_document(CATALOG)
        report = validate_library_document(document)
        self.assertEqual(report["item_count"], len(document["items"]))
        self.assertGreaterEqual(report["item_count"], 200)
        self.assertGreaterEqual(report["category_counts"]["titulos_cena"], 15)
        scene = {"event_type": "cena", "peak": 128.0, "rank": 1, "score": 82.0}
        kit = generate_post_kit(scene, "Meu jogo", "shorts", document)
        self.assertEqual(len(kit["titles"]), 3)
        self.assertEqual(kit["event_type"], "cena")
        self.assertNotIn("COMBO INSANO!", kit["reaction_text"])
        self.assertTrue(kit["reaction_text"])
        with self.assertRaises(ValueError):
            validate_library_document({"items": document["items"][:10]})

    def test_library_rejects_duplicate_ids_and_text(self):
        document = build_library_document(CATALOG)
        broken = copy.deepcopy(document)
        broken["items"][1]["id"] = broken["items"][0]["id"]
        with self.assertRaises(ValueError):
            validate_library_document(broken)
        broken = copy.deepcopy(document)
        broken["items"][1]["text"] = broken["items"][0]["text"]
        broken["items"][1]["category"] = broken["items"][0]["category"]
        with self.assertRaises(ValueError):
            validate_library_document(broken)


if __name__ == "__main__":
    unittest.main()
