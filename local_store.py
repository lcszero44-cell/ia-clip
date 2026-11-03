from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import config
from common import atomic_write_json, read_json

DB_PATH = config.DIRS["data"] / "cliprush_local.db"
SETTINGS_PATH = config.DIRS["data"] / "settings.json"
_INITIALIZED = False

DEFAULT_SETTINGS: dict[str, Any] = {
    "analysis_mode": "quick",
    "automatic_count": True,
    "top_n": 10,
    "scene_mode": config.SCENE_MODE,
    "max_scene_share": float(config.SCENES["max_share"]),
    "debug": True,
    "platform": "shorts",
    "captions": False,
    "draft": True,
    "economy_mode": bool(config.PERFORMANCE["economy_mode"]),
    "threads": int(config.PERFORMANCE["threads"]),
    "priority_below_normal": bool(config.PERFORMANCE["priority_below_normal"]),
    "game_name": "",
    "channel_name": "",
    "brand_handle": "",
    "default_hashtags": "",
    "content_warning": "",
    "ollama_enabled": True,
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "llama3.2:3b",
    "ollama_timeout": 25,
    "vision_ai_enabled": True,
    "vision_model": "llava:7b",
    "vision_timeout": 90,
    "max_vision_candidates": 18,
    "auto_render": True,
}

POST_STATUSES = ("rascunho", "roteiro", "agendado", "publicado", "pausado")
QUEUE_STATUSES = ("pendente", "processando", "concluido", "falhou", "cancelado")


class _ClosingConnection(sqlite3.Connection):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def connect() -> sqlite3.Connection:
    config.DIRS["data"].mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=20, isolation_level=None, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 20000")
    return connection


def init_storage() -> None:
    global _INITIALIZED
    if _INITIALIZED:
        return
    with connect() as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA synchronous = NORMAL")
        db.executescript("""
        CREATE TABLE IF NOT EXISTS queue_jobs (
            id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            work_dir TEXT NOT NULL,
            options_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pendente',
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            result_path TEXT,
            result_json TEXT,
            error TEXT,
            created_at REAL NOT NULL,
            started_at REAL,
            finished_at REAL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_queue_status_created ON queue_jobs(status, created_at);
        CREATE TABLE IF NOT EXISTS posts (
            id TEXT PRIMARY KEY,
            source_job_id TEXT,
            clip_rank INTEGER,
            platform TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'rascunho',
            scheduled_at TEXT,
            game_name TEXT NOT NULL DEFAULT '',
            event_type TEXT NOT NULL DEFAULT '',
            score REAL NOT NULL DEFAULT 0,
            title TEXT NOT NULL DEFAULT '',
            caption TEXT NOT NULL DEFAULT '',
            hashtags TEXT NOT NULL DEFAULT '',
            pinned_comment TEXT NOT NULL DEFAULT '',
            video_path TEXT NOT NULL DEFAULT '',
            notes TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_posts_schedule ON posts(scheduled_at, status);
        CREATE TABLE IF NOT EXISTS post_metrics (
            id TEXT PRIMARY KEY,
            post_id TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            measured_at TEXT NOT NULL,
            views INTEGER NOT NULL DEFAULT 0,
            likes INTEGER NOT NULL DEFAULT 0,
            comments INTEGER NOT NULL DEFAULT 0,
            shares INTEGER NOT NULL DEFAULT 0,
            saves INTEGER NOT NULL DEFAULT 0,
            followers_gained INTEGER NOT NULL DEFAULT 0,
            watch_minutes REAL NOT NULL DEFAULT 0,
            retention_pct REAL,
            notes TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_metrics_post_date ON post_metrics(post_id, measured_at);
        CREATE TABLE IF NOT EXISTS post_checklist (
            post_id TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
            item_key TEXT NOT NULL,
            checked INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL,
            PRIMARY KEY(post_id, item_key)
        );
        CREATE TABLE IF NOT EXISTS rendered_assets (
            id TEXT PRIMARY KEY,
            analysis_path TEXT NOT NULL,
            clip_rank INTEGER NOT NULL,
            platform TEXT NOT NULL,
            video_path TEXT NOT NULL,
            cover_path TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_renders_lookup ON rendered_assets(analysis_path, clip_rank, platform, created_at);
        CREATE TABLE IF NOT EXISTS clip_content (
            analysis_path TEXT NOT NULL,
            clip_rank INTEGER NOT NULL,
            platform TEXT NOT NULL,
            game_name TEXT NOT NULL DEFAULT '',
            overlay_title TEXT NOT NULL DEFAULT '',
            publish_title TEXT NOT NULL DEFAULT '',
            caption TEXT NOT NULL DEFAULT '',
            hashtags TEXT NOT NULL DEFAULT '',
            pinned_comment TEXT NOT NULL DEFAULT '',
            confirmed_event TEXT NOT NULL DEFAULT '',
            ai_titles_json TEXT NOT NULL DEFAULT '[]',
            updated_at REAL NOT NULL,
            PRIMARY KEY (analysis_path, clip_rank, platform)
        );
        """)
        clip_content_columns = {str(row["name"]) for row in db.execute("PRAGMA table_info(clip_content)")}
        if "ai_titles_json" not in clip_content_columns:
            db.execute("ALTER TABLE clip_content ADD COLUMN ai_titles_json TEXT NOT NULL DEFAULT '[]'")
    _INITIALIZED = True
    if not SETTINGS_PATH.exists():
        save_settings(DEFAULT_SETTINGS)


def load_settings() -> dict[str, Any]:
    init_storage()
    saved = read_json(SETTINGS_PATH, {})
    values = dict(DEFAULT_SETTINGS)
    if isinstance(saved, dict):
        values.update({key: value for key, value in saved.items() if key in DEFAULT_SETTINGS})
    return _validate_settings(values)


def _validate_settings(values: dict[str, Any]) -> dict[str, Any]:
    result = dict(DEFAULT_SETTINGS)
    result.update({key: value for key, value in values.items() if key in DEFAULT_SETTINGS})
    if result["analysis_mode"] not in {"quick", "detailed"}:
        result["analysis_mode"] = "quick"
    if result["scene_mode"] not in {"manter", "priorizar-gameplay", "ignorar"}:
        result["scene_mode"] = config.SCENE_MODE
    if result["platform"] not in config.PLATFORMS:
        result["platform"] = "shorts"
    try:
        result["top_n"] = max(1, min(30, int(result["top_n"])))
    except (TypeError, ValueError):
        result["top_n"] = 10
    try:
        result["max_scene_share"] = max(0.0, min(1.0, float(result["max_scene_share"])))
    except (TypeError, ValueError):
        result["max_scene_share"] = float(config.SCENES["max_share"])
    try:
        result["threads"] = max(1, min(8, int(result["threads"])))
    except (TypeError, ValueError):
        result["threads"] = int(config.PERFORMANCE["threads"])
    for key in ("automatic_count", "debug", "captions", "draft", "economy_mode", "priority_below_normal",
                "ollama_enabled", "vision_ai_enabled", "auto_render"):
        result[key] = bool(result[key])
    for key in ("game_name", "channel_name", "brand_handle", "default_hashtags", "content_warning",
                "ollama_base_url", "ollama_model", "vision_model"):
        result[key] = str(result[key] or "").strip()
    result["ollama_base_url"] = result["ollama_base_url"].rstrip("/") or "http://localhost:11434"
    result["ollama_model"] = result["ollama_model"] or "llama3.2:3b"
    try:
        result["ollama_timeout"] = max(5, min(90, int(result["ollama_timeout"])))
    except (TypeError, ValueError):
        result["ollama_timeout"] = 25
    result["vision_model"] = result["vision_model"] or "llava:7b"
    try:
        result["vision_timeout"] = max(15, min(300, int(result["vision_timeout"])))
    except (TypeError, ValueError):
        result["vision_timeout"] = 90
    try:
        result["max_vision_candidates"] = max(6, min(18, int(result["max_vision_candidates"])))
    except (TypeError, ValueError):
        result["max_vision_candidates"] = 18
    return result


def save_settings(values: dict[str, Any]) -> dict[str, Any]:
    config.DIRS["data"].mkdir(parents=True, exist_ok=True)
    result = _validate_settings(values)
    atomic_write_json(SETTINGS_PATH, result)
    return result


def insert_queue_job(source_path: str | Path, work_dir: str | Path,
                     options: dict[str, Any], job_id: str | None = None) -> str:
    init_storage()
    job_id = job_id or uuid.uuid4().hex[:12]
    now = time.time()
    with connect() as db:
        db.execute("""INSERT INTO queue_jobs
            (id, source_path, work_dir, options_json, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pendente', ?, ?)""",
            (job_id, str(Path(source_path).resolve()), str(Path(work_dir).resolve()),
             json.dumps(options, ensure_ascii=False), now, now))
    return job_id


def get_queue_job(job_id: str) -> dict[str, Any] | None:
    init_storage()
    with connect() as db:
        row = db.execute("SELECT * FROM queue_jobs WHERE id = ?", (job_id,)).fetchone()
    return _queue_row(row) if row else None


def list_queue_jobs(limit: int = 100) -> list[dict[str, Any]]:
    init_storage()
    with connect() as db:
        rows = db.execute("SELECT * FROM queue_jobs ORDER BY created_at DESC LIMIT ?",
                           (max(1, min(500, int(limit))),)).fetchall()
    return [_queue_row(row) for row in rows]


def _queue_row(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    try:
        value["options"] = json.loads(value.pop("options_json"))
    except (ValueError, TypeError):
        value["options"] = {}
    value["cancel_requested"] = bool(value["cancel_requested"])
    try:
        value["result"] = json.loads(value.pop("result_json")) if value.get("result_json") else None
    except (ValueError, TypeError):
        value["result"] = None
    return value


def create_post(data: dict[str, Any]) -> str:
    init_storage()
    post_id = uuid.uuid4().hex[:12]
    now = time.time()
    values = {
        "id": post_id,
        "source_job_id": data.get("source_job_id"),
        "clip_rank": data.get("clip_rank"),
        "platform": data.get("platform", "shorts"),
        "status": data.get("status", "rascunho"),
        "scheduled_at": data.get("scheduled_at"),
        "game_name": str(data.get("game_name", "")),
        "event_type": str(data.get("event_type", "")),
        "score": float(data.get("score", 0.0) or 0.0),
        "title": str(data.get("title", "")),
        "caption": str(data.get("caption", "")),
        "hashtags": str(data.get("hashtags", "")),
        "pinned_comment": str(data.get("pinned_comment", "")),
        "video_path": str(data.get("video_path", "")),
        "notes": str(data.get("notes", "")),
        "created_at": now,
        "updated_at": now,
    }
    if values["status"] not in POST_STATUSES:
        raise ValueError("Status editorial inválido.")
    if values["platform"] not in config.PLATFORMS:
        raise ValueError("Plataforma inválida.")
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    with connect() as db:
        db.execute(f"INSERT INTO posts ({columns}) VALUES ({placeholders})", tuple(values.values()))
    return post_id


def list_posts(status: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
    init_storage()
    with connect() as db:
        if status:
            rows = db.execute("SELECT * FROM posts WHERE status = ? ORDER BY COALESCE(scheduled_at, '9999') LIMIT ?",
                              (status, max(1, min(5000, int(limit))))).fetchall()
        else:
            rows = db.execute("SELECT * FROM posts ORDER BY COALESCE(scheduled_at, '9999'), created_at DESC LIMIT ?",
                              (max(1, min(5000, int(limit))),)).fetchall()
    return [dict(row) for row in rows]


def get_post(post_id: str) -> dict[str, Any] | None:
    init_storage()
    with connect() as db:
        row = db.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    return dict(row) if row else None


def update_post(post_id: str, changes: dict[str, Any]) -> None:
    allowed = {"platform", "status", "scheduled_at", "game_name", "event_type", "score",
               "title", "caption", "hashtags", "pinned_comment", "video_path", "notes"}
    clean = {key: value for key, value in changes.items() if key in allowed}
    if not clean:
        return
    if "status" in clean and clean["status"] not in POST_STATUSES:
        raise ValueError("Status editorial inválido.")
    if "platform" in clean and clean["platform"] not in config.PLATFORMS:
        raise ValueError("Plataforma inválida.")
    clean["updated_at"] = time.time()
    sql = ", ".join(f"{key} = ?" for key in clean)
    with connect() as db:
        db.execute(f"UPDATE posts SET {sql} WHERE id = ?", (*clean.values(), post_id))


def delete_post(post_id: str) -> None:
    with connect() as db:
        db.execute("DELETE FROM posts WHERE id = ?", (post_id,))


def add_metrics(post_id: str, metrics: dict[str, Any]) -> str:
    if not get_post(post_id):
        raise ValueError("Post não encontrado.")
    metrics_id = uuid.uuid4().hex[:12]
    values = {
        "id": metrics_id, "post_id": post_id,
        "measured_at": str(metrics.get("measured_at") or time.strftime("%Y-%m-%d")),
        "views": max(0, int(metrics.get("views", 0))),
        "likes": max(0, int(metrics.get("likes", 0))),
        "comments": max(0, int(metrics.get("comments", 0))),
        "shares": max(0, int(metrics.get("shares", 0))),
        "saves": max(0, int(metrics.get("saves", 0))),
        "followers_gained": max(0, int(metrics.get("followers_gained", 0))),
        "watch_minutes": max(0.0, float(metrics.get("watch_minutes", 0.0))),
        "retention_pct": metrics.get("retention_pct"),
        "notes": str(metrics.get("notes", "")), "created_at": time.time(),
    }
    if values["retention_pct"] is not None:
        values["retention_pct"] = max(0.0, min(100.0, float(values["retention_pct"])))
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    with connect() as db:
        db.execute(f"INSERT INTO post_metrics ({columns}) VALUES ({placeholders})", tuple(values.values()))
    return metrics_id


def list_metrics(post_id: str | None = None, limit: int = 5000) -> list[dict[str, Any]]:
    limit = max(1, min(50000, int(limit)))
    with connect() as db:
        if post_id:
            rows = db.execute("SELECT * FROM post_metrics WHERE post_id = ? ORDER BY measured_at DESC LIMIT ?",
                              (post_id, limit)).fetchall()
        else:
            rows = db.execute("SELECT * FROM post_metrics ORDER BY measured_at DESC LIMIT ?",
                              (limit,)).fetchall()
    return [dict(row) for row in reversed(rows)]


def ensure_checklist(post_id: str, items: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not get_post(post_id):
        raise ValueError("Post não encontrado.")
    now = time.time()
    with connect() as db:
        db.executemany("INSERT OR IGNORE INTO post_checklist (post_id, item_key, checked, note, updated_at) VALUES (?, ?, 0, '', ?)",
                       [(post_id, key, now) for key, _label in items])
        rows = db.execute("SELECT * FROM post_checklist WHERE post_id = ?", (post_id,)).fetchall()
    by_key = {row["item_key"]: dict(row) for row in rows}
    return [{**by_key[key], "label": label} for key, label in items]


def save_checklist(post_id: str, values: dict[str, dict[str, Any]]) -> None:
    now = time.time()
    with connect() as db:
        db.executemany("""INSERT INTO post_checklist (post_id, item_key, checked, note, updated_at)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(post_id, item_key) DO UPDATE SET
            checked=excluded.checked, note=excluded.note, updated_at=excluded.updated_at""",
            [(post_id, key, int(bool(value.get("checked"))), str(value.get("note", "")), now)
             for key, value in values.items()])


def save_rendered_asset(analysis_path: str | Path, clip_rank: int, platform: str,
                        video_path: str | Path, cover_path: str | Path | None,
                        title: str) -> str:
    asset_id = uuid.uuid4().hex[:12]
    with connect() as db:
        db.execute("""INSERT INTO rendered_assets
            (id, analysis_path, clip_rank, platform, video_path, cover_path, title, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (asset_id, str(Path(analysis_path).resolve()), int(clip_rank), platform,
             str(Path(video_path).resolve()), str(Path(cover_path).resolve()) if cover_path else "",
             title, time.time()))
    return asset_id


def list_rendered_assets(analysis_path: str | Path | None = None,
                         clip_rank: int | None = None, platform: str | None = None,
                         limit: int = 500) -> list[dict[str, Any]]:
    clauses, params = [], []
    if analysis_path:
        clauses.append("analysis_path=?")
        params.append(str(Path(analysis_path).resolve()))
    if clip_rank is not None:
        clauses.append("clip_rank=?")
        params.append(int(clip_rank))
    if platform:
        clauses.append("platform=?")
        params.append(platform)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(max(1, min(5000, int(limit))))
    with connect() as db:
        rows = db.execute(f"SELECT * FROM rendered_assets{where} ORDER BY created_at DESC LIMIT ?", params).fetchall()
    return [dict(row) for row in rows]


def get_clip_content(analysis_path: str | Path, clip_rank: int, platform: str) -> dict[str, Any]:
    init_storage()
    key = (str(Path(analysis_path).resolve()), int(clip_rank), str(platform))
    with connect() as db:
        row = db.execute("SELECT * FROM clip_content WHERE analysis_path=? AND clip_rank=? AND platform=?", key).fetchone()
    if row:
        result = dict(row)
        try:
            titles = json.loads(result.pop("ai_titles_json", "[]"))
            result["ai_titles"] = titles if isinstance(titles, list) else []
        except (TypeError, ValueError, json.JSONDecodeError):
            result["ai_titles"] = []
        return result
    return {"analysis_path": key[0], "clip_rank": key[1], "platform": platform, "game_name": "",
            "overlay_title": "", "publish_title": "", "caption": "", "hashtags": "", "ai_titles": [],
            "pinned_comment": "", "confirmed_event": ""}


def save_clip_content(analysis_path: str | Path, clip_rank: int, platform: str,
                      content: dict[str, Any]) -> None:
    init_storage()
    if platform not in config.PLATFORMS:
        raise ValueError("Plataforma inválida para os metadados do clipe.")
    rank = int(clip_rank)
    if rank < 1:
        raise ValueError("O número do clipe precisa ser positivo.")
    confirmed_event = str(content.get("confirmed_event", "") or "").strip()
    if confirmed_event and confirmed_event not in config.EVENT_WEIGHTS:
        raise ValueError("Evento confirmado inválido.")
    raw_ai_titles = content.get("ai_titles", [])
    ai_titles = [str(title).strip()[:120] for title in raw_ai_titles
                 if isinstance(title, str) and title.strip()][:3] if isinstance(raw_ai_titles, list) else []
    values = {
        "analysis_path": str(Path(analysis_path).resolve()), "clip_rank": rank,
        "platform": platform,
        "game_name": str(content.get("game_name", "") or "").strip(),
        "overlay_title": str(content.get("overlay_title", "") or "").strip(),
        "publish_title": str(content.get("publish_title", "") or "").strip(),
        "caption": str(content.get("caption", "") or "").strip(),
        "hashtags": str(content.get("hashtags", "") or "").strip(),
        "pinned_comment": str(content.get("pinned_comment", "") or "").strip(),
        "confirmed_event": confirmed_event,
        "ai_titles_json": json.dumps(ai_titles, ensure_ascii=False), "updated_at": time.time(),
    }
    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    updates = ", ".join(f"{key}=excluded.{key}" for key in values if key not in {"analysis_path", "clip_rank", "platform"})
    with connect() as db:
        db.execute(f"INSERT INTO clip_content ({columns}) VALUES ({placeholders}) "
                   f"ON CONFLICT(analysis_path,clip_rank,platform) DO UPDATE SET {updates}",
                   tuple(values.values()))
