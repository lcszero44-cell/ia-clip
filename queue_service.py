from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import config
from common import (InstanceLock, JobCancelledError, ProgressWriter, atomic_write_json,
                    priority_kwargs, read_json)
from local_store import (QUEUE_STATUSES, connect, create_post, get_queue_job, init_storage,
                         insert_queue_job, list_queue_jobs, save_clip_content,
                         save_rendered_asset)

QUEUE_PID_PATH = config.DIRS["data"] / "queue_worker.json"
QUEUE_LOG_PATH = config.DIRS["logs"] / "queue_worker.log"
QUEUE_LAUNCH_LOCK_PATH = config.DIRS["data"] / "queue_worker.launch.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
            kernel.OpenProcess.restype = ctypes.c_void_p
            kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
            kernel.GetExitCodeProcess.restype = ctypes.c_bool
            kernel.CloseHandle.argtypes = [ctypes.c_void_p]
            process = kernel.OpenProcess(0x1000, False, pid)
            if not process:
                return False
            exit_code = ctypes.c_ulong()
            ok = kernel.GetExitCodeProcess(process, ctypes.byref(exit_code))
            kernel.CloseHandle(process)
            return bool(ok and exit_code.value == 259)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def worker_pid() -> int | None:
    payload = read_json(QUEUE_PID_PATH, {}) or {}
    try:
        pid = int(payload.get("pid", 0))
    except (TypeError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def _progress_for(work_dir: str | Path) -> dict[str, Any]:
    root = Path(work_dir)
    candidates = list(root.rglob("progress.json"))
    if not candidates:
        return {}
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    payload = read_json(path, {}) or {}
    payload["progress_path"] = str(path)
    return payload


def enqueue_analysis(source_path: str | Path, options: dict[str, Any]) -> str:
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Gameplay não encontrada: {source}")
    job_id = uuid.uuid4().hex[:12]
    work_dir = config.DIRS["sessions"] / "fila" / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    created_id = insert_queue_job(source, work_dir, options, job_id=job_id)
    try:
        ensure_worker()
    except Exception:
        # O item permanece na fila para que o usuário possa iniciar novamente.
        raise
    return created_id


def get_jobs(limit: int = 100) -> list[dict[str, Any]]:
    jobs = list_queue_jobs(limit)
    for job in jobs:
        job["progress"] = _progress_for(job["work_dir"])
        if job["status"] == "processando" and job["progress"]:
            job["percent"] = float(job["progress"].get("percent", 0.0))
            job["stage"] = str(job["progress"].get("stage", "análise"))
            job["message"] = str(job["progress"].get("message", "Processando"))
        else:
            job["percent"] = 100.0 if job["status"] == "concluido" else 0.0
            job["stage"] = job["status"]
            job["message"] = job.get("error") or ""
    return jobs


def _write_cancel_flag(work_dir: str | Path) -> bool:
    progress_files = list(Path(work_dir).rglob("progress.json"))
    if not progress_files:
        return False
    for progress_path in progress_files:
        atomic_write_json(progress_path.with_name("cancel.flag"), {"requested_at": time.time()})
    return True


def cancel_job(job_id: str) -> str:
    init_storage()
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM queue_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            db.rollback()
            raise ValueError("Job não encontrado.")
        status = row["status"]
        if status == "pendente":
            db.execute("UPDATE queue_jobs SET status='cancelado', finished_at=?, updated_at=? WHERE id=?",
                       (time.time(), time.time(), job_id))
            db.commit()
            return "cancelado"
        if status == "processando":
            db.execute("UPDATE queue_jobs SET cancel_requested=1, updated_at=? WHERE id=?",
                       (time.time(), job_id))
            work_dir = row["work_dir"]
            db.commit()
            _write_cancel_flag(work_dir)
            return "cancelamento_solicitado"
        db.commit()
        return status


def retry_job(job_id: str) -> str:
    job = get_queue_job(job_id)
    if not job or job["status"] not in {"falhou", "cancelado"}:
        raise ValueError("Apenas jobs cancelados ou com falha podem ser reenfileirados.")
    source = Path(job["source_path"])
    if not source.is_file():
        raise FileNotFoundError("O arquivo de origem não está mais disponível em uploads/.")
    new_work_dir = config.DIRS["sessions"] / "fila" / f"{job_id}_r_{int(time.time())}"
    new_work_dir.mkdir(parents=True, exist_ok=True)
    new_id = insert_queue_job(source, new_work_dir, job["options"])
    ensure_worker()
    return new_id


def remove_finished_jobs() -> int:
    init_storage()
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        cursor = db.execute("DELETE FROM queue_jobs WHERE status IN ('concluido','falhou','cancelado')")
        deleted = cursor.rowcount
        db.commit()
    return int(deleted)


def recover_orphaned_jobs() -> int:
    """Marca como interrompidos apenas jobs que ficaram ativos sem worker vivo."""
    init_storage()
    with InstanceLock(QUEUE_LAUNCH_LOCK_PATH):
        if worker_pid() is not None:
            return 0
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id FROM queue_jobs WHERE status='processando' AND updated_at < ?",
                              (time.time() - 30,)).fetchall()
            now = time.time()
            for row in rows:
                db.execute("UPDATE queue_jobs SET status='falhou', error=?, finished_at=?, updated_at=? WHERE id=?",
                           ("Worker local foi encerrado inesperadamente; use Reenfileirar.", now, now, row["id"]))
            db.commit()
        QUEUE_PID_PATH.unlink(missing_ok=True)
        return len(rows)


def ensure_worker() -> int | None:
    """Inicia um único worker persistente; o worker drena a fila e aguarda novas tarefas por 10 s."""
    init_storage()
    launch_lock = InstanceLock(QUEUE_LAUNCH_LOCK_PATH)
    with launch_lock:
        pid = worker_pid()
        if pid is not None:
            return pid
        with connect() as db:
            pending = db.execute("SELECT 1 FROM queue_jobs WHERE status='pendente' LIMIT 1").fetchone()
        if not pending:
            return None
        kwargs = priority_kwargs()
        if os.name == "nt":
            kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
        QUEUE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with QUEUE_LOG_PATH.open("a", encoding="utf-8") as logfile:
            process = subprocess.Popen(
                [sys.executable, str(config.ROOT / "queue_worker.py")],
                cwd=str(config.ROOT), stdin=subprocess.DEVNULL,
                stdout=logfile, stderr=subprocess.STDOUT, **kwargs,
            )
        atomic_write_json(QUEUE_PID_PATH, {"pid": process.pid, "started_at": time.time()})
        return process.pid


def claim_next_job() -> dict[str, Any] | None:
    init_storage()
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM queue_jobs WHERE status='pendente' ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            db.commit()
            return None
        now = time.time()
        db.execute("UPDATE queue_jobs SET status='processando', started_at=?, updated_at=? WHERE id=? AND status='pendente'",
                   (now, now, row["id"]))
        claimed = db.execute("SELECT * FROM queue_jobs WHERE id=?", (row["id"],)).fetchone()
        db.commit()
    job = dict(claimed)
    try:
        job["options"] = json.loads(job.pop("options_json"))
    except (ValueError, TypeError):
        job["options"] = {}
    job["cancel_requested"] = bool(job["cancel_requested"])
    return job


def update_job(job_id: str, *, status: str | None = None, result_path: str | None = None,
               result: dict[str, Any] | None = None, error: str | None = None,
               finished: bool = False) -> None:
    allowed_status = {None, *QUEUE_STATUSES}
    if status not in allowed_status:
        raise ValueError("Status de fila inválido.")
    fields: dict[str, Any] = {"updated_at": time.time()}
    if status is not None:
        fields["status"] = status
    if result_path is not None:
        fields["result_path"] = result_path
    if result is not None:
        fields["result_json"] = json.dumps(result, ensure_ascii=False)
    if error is not None:
        fields["error"] = str(error)[:4000]
    if finished:
        fields["finished_at"] = time.time()
    sql = ", ".join(f"{name}=?" for name in fields)
    with connect() as db:
        db.execute(f"UPDATE queue_jobs SET {sql} WHERE id=?", (*fields.values(), job_id))


def requested_cancel(job_id: str) -> bool:
    with connect() as db:
        row = db.execute("SELECT cancel_requested FROM queue_jobs WHERE id=?", (job_id,)).fetchone()
    return bool(row and row["cancel_requested"])


def _set_worker_priority(below_normal: bool) -> None:
    try:
        if os.name == "nt":
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel.SetPriorityClass.restype = ctypes.c_bool
            handle = kernel.GetCurrentProcess()
            target_class = 0x00004000 if below_normal else 0x00000020
            kernel.SetPriorityClass(handle, target_class)
        elif hasattr(os, "getpriority") and hasattr(os, "setpriority"):
            current = os.getpriority(os.PRIO_PROCESS, 0)
            target = max(current, 5) if below_normal else min(current, 0)
            if current != target:
                os.setpriority(os.PRIO_PROCESS, 0, target)
    except (OSError, AttributeError):
        pass


def _auto_editorial_kit(clip: dict[str, Any], options: dict[str, Any], library: dict[str, Any]) -> dict[str, Any]:
    """Cria kit offline e tenta melhorá-lo com o Llama textual, sem inventar eventos."""
    import ia_manager
    import ollama_ai
    from postkit import generate_post_kit, safe_event_type

    game_name = str(options.get("game_name", "") or "")
    platform = str(options.get("platform", "shorts") or "shorts")
    safe_clip = {**clip, "event_type": safe_event_type(clip)}
    fallback = generate_post_kit(
        safe_clip, game_name, platform, library,
        channel_name=str(options.get("channel_name", "")),
        brand_handle=str(options.get("brand_handle", "")),
        default_hashtags=str(options.get("default_hashtags", "")),
        content_warning=str(options.get("content_warning", "")),
    )
    if not options.get("ollama_enabled", True):
        return fallback
    visual_context = str(clip.get("ai_evidence", "") or "")[:240]
    visual_class = str(clip.get("ai_visual_class", "") or "")
    summary = (f"Evidência visual classificada como {visual_class}: {visual_context}" if visual_context
               else "Sem descrição visual validada; usar somente o tipo seguro e os sinais locais.")
    context = {
        "game_name": game_name[:100], "platform": platform,
        "evento_seguro": safe_clip["event_type"], "score": float(clip.get("score", 0) or 0),
        "duration_seconds": float(clip.get("duration_seconds", 0) or 0),
        "scene_speech": (f"cutscene; fala detectada por {float(clip.get('speech_seconds', 0) or 0):.0f} segundos"
                         if safe_clip["event_type"] == "cena" else "gameplay; evento específico não confirmado"),
        "summary": summary,
    }
    try:
        ai_copy = ollama_ai.generate_editorial_copy(
            context, base_url=str(options.get("ollama_base_url", "http://localhost:11434")),
            model=str(options.get("text_model", options.get("ollama_model", "llama3.2:3b"))),
            timeout=float(options.get("ollama_timeout", 25)),
        )
    except Exception:
        ai_copy = None
    if not isinstance(ai_copy, dict) or not ia_manager.editorial_copy_is_safe(ai_copy, safe_clip["event_type"]):
        return fallback
    merged = {**fallback, **ai_copy}
    tags: list[str] = []
    import re
    for value in (str(merged.get("hashtags", "")), str(fallback.get("hashtags", ""))):
        for tag in re.findall(r"#[\w]+", value, flags=re.UNICODE):
            if tag.lower() not in {item.lower() for item in tags}:
                tags.append(tag)
    merged["hashtags"] = " ".join(tags[:12])
    merged["caption"] = "\n\n".join(part for part in (
        str(merged.get("titles", fallback["titles"])[0]), str(merged.get("hook", "")),
        str(merged.get("description", "")), str(merged.get("cta", "")), merged["hashtags"],
        ("Aviso: " + str(options.get("content_warning", "")).strip()) if str(options.get("content_warning", "")).strip() else "",
    ) if part)
    merged["event_type"] = fallback["event_type"]
    merged["reaction_text"] = fallback["reaction_text"]
    return merged


def _auto_render_analysis(job: dict[str, Any], result: dict[str, Any], result_path: Path,
                          options: dict[str, Any]) -> dict[str, Any]:
    """Renderiza em série os clipes selecionados e os persiste no histórico local."""
    import config
    from library import load_library
    from postkit import safe_event_type
    from renderer import render_clip

    source = Path(job["source_path"]).resolve()
    session_dir = result_path.resolve().parent
    output_dir = session_dir / "auto_render"
    output_dir.mkdir(parents=True, exist_ok=True)
    platform = str(options.get("platform", "shorts") or "shorts")
    clips = list(result.get("clips", []))
    library = load_library()
    progress = ProgressWriter(job["work_dir"])
    completed, failed = [], []
    total = max(1, len(clips))
    for index, clip in enumerate(clips, start=1):
        progress.check_cancelled()
        rank = int(clip.get("rank", index))
        progress.update("render_auto", 78 + 18 * (index - 1) / total,
                        f"Gerando MP4/capa e kit do corte #{rank:02d} ({index}/{len(clips)}).",
                        current_clip=index, total_clips=len(clips))
        try:
            kit = _auto_editorial_kit(clip, options, library)
            event_type = safe_event_type(clip)
            overlay_title = str(kit.get("reaction_text") or "MOMENTO INTENSO")[:40]
            outputs = render_clip(
                source_path=source, start=float(clip["start"]), end=float(clip["end"]),
                peak=float(clip["peak"]), output_dir=output_dir, platform=platform,
                title=overlay_title, event_type=event_type,
                captions=bool(options.get("captions", False)),
                language=config.ANALYSIS["speech_language"], reaction=True,
                more_events=[float(item["timestamp"]) for item in clip.get("events", [])],
                brightness_windows=clip.get("brightness_windows", []), draft=False,
                economy=bool(options.get("economy_mode", config.PERFORMANCE["economy_mode"])),
            )
            if not outputs:
                raise RuntimeError("O render não produziu saída para a plataforma escolhida.")
            for output in outputs:
                video = Path(output["video"]).resolve()
                cover = Path(output["cover"]).resolve() if output.get("cover") else None
                title = str((kit.get("titles") or [""])[0]).strip()
                caption = str(kit.get("caption", "")).strip()
                lines = caption.splitlines()
                if lines:
                    lines[0] = title
                    caption = "\n".join(lines)
                else:
                    caption = title
                save_rendered_asset(result_path, rank, output["platform"], video, cover, overlay_title)
                save_clip_content(result_path, rank, output["platform"], {
                    "game_name": str(options.get("game_name", "")),
                    "overlay_title": overlay_title, "publish_title": title,
                    "caption": caption, "hashtags": str(kit.get("hashtags", "")),
                    "pinned_comment": str(kit.get("pinned_comment", "")),
                    "ai_titles": list(kit.get("titles", [])),
                })
                create_post({
                    "source_job_id": job["id"], "clip_rank": rank, "platform": output["platform"],
                    "status": "rascunho", "game_name": str(options.get("game_name", "")),
                    "event_type": event_type, "score": float(clip.get("score", 0) or 0),
                    "title": title, "caption": caption, "hashtags": str(kit.get("hashtags", "")),
                    "pinned_comment": str(kit.get("pinned_comment", "")), "video_path": str(video),
                    "notes": f"analysis_path={result_path}; ai_engine={result.get('summary', {}).get('ai_engine', 'heuristic_local')}",
                })
                completed.append({"rank": rank, "platform": output["platform"], "video": str(video),
                                  "cover": str(cover) if cover else "", "title": title})
        except JobCancelledError:
            raise
        except Exception as exc:
            failed.append({"rank": rank, "error": f"{type(exc).__name__}: {exc}"[:500]})
    progress.update("concluido", 100,
                    f"Análise e render automático concluídos: {len(completed)} MP4(s); {len(failed)} falha(s).",
                    current_clip=len(clips), total_clips=len(clips), done=True)
    return {"rendered": completed, "failed": failed, "auto_rendered": bool(clips)}


# Keep the worker lifecycle in this module so all UI and CLI paths share one implementation.
def run_worker(idle_seconds: int = 10) -> int:
    init_storage()
    pid = os.getpid()
    atomic_write_json(QUEUE_PID_PATH, {"pid": pid, "started_at": time.time()})
    idle_ticks = 0
    completed_count = 0
    try:
        idle_limit = max(1, int(idle_seconds / 2))
        while True:
            job = claim_next_job()
            if job is None:
                idle_ticks += 1
                if idle_ticks >= idle_limit:
                    with InstanceLock(QUEUE_LAUNCH_LOCK_PATH):
                        with connect() as db:
                            pending = db.execute("SELECT 1 FROM queue_jobs WHERE status='pendente' LIMIT 1").fetchone()
                        if pending:
                            idle_ticks = 0
                            continue
                        payload = read_json(QUEUE_PID_PATH, {}) or {}
                        if int(payload.get("pid", 0) or 0) == pid:
                            QUEUE_PID_PATH.unlink(missing_ok=True)
                    break
                time.sleep(2)
                continue
            idle_ticks = 0
            _run_job(job)
            completed_count += 1
            if config.PERFORMANCE["economy_mode"]:
                time.sleep(max(0.0, float(config.PERFORMANCE["economy_pause_seconds"])))
    finally:
        payload = read_json(QUEUE_PID_PATH, {}) or {}
        if int(payload.get("pid", 0) or 0) == pid:
            QUEUE_PID_PATH.unlink(missing_ok=True)
    return completed_count


def _run_job(job: dict[str, Any]) -> None:
    from analyzer import analyze_video

    job_id = job["id"]
    work_dir = Path(job["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    if job["cancel_requested"] or requested_cancel(job_id):
        update_job(job_id, status="cancelado", error="Cancelado antes do início.", finished=True)
        return
    source = Path(job["source_path"])
    if not source.is_file():
        update_job(job_id, status="falhou", error=f"Gameplay não encontrada: {source}", finished=True)
        return

    monitor_stop = threading.Event()

    def cancellation_monitor() -> None:
        while not monitor_stop.wait(0.4):
            if not requested_cancel(job_id):
                continue
            _write_cancel_flag(work_dir)

    monitor = threading.Thread(target=cancellation_monitor, name=f"cancel-watch-{job_id}", daemon=True)
    monitor.start()
    try:
        options = job.get("options", {})
        config.PERFORMANCE["threads"] = max(1, min(8, int(options.get("threads", config.PERFORMANCE["threads"]))))
        config.PERFORMANCE["economy_mode"] = bool(options.get("economy_mode", config.PERFORMANCE["economy_mode"]))
        config.PERFORMANCE["priority_below_normal"] = bool(options.get("priority_below_normal", config.PERFORMANCE["priority_below_normal"]))
        _set_worker_priority(config.PERFORMANCE["priority_below_normal"])
        result = analyze_video(
            source,
            mode=options.get("analysis_mode", "quick"),
            top_n=options.get("top_n"),
            debug=bool(options.get("debug", True)),
            output_root=work_dir,
            use_cache=bool(options.get("use_cache", True)),
            scene_mode=options.get("scene_mode", config.SCENE_MODE),
            max_scene_share=float(options.get("max_scene_share", config.SCENES["max_share"])),
            vision_enabled=bool(options.get("vision_ai_enabled", True)),
            ollama_base_url=str(options.get("ollama_base_url", "http://localhost:11434")),
            vision_model=str(options.get("vision_model", config.AI["vision_model"])),
            text_model=str(options.get("text_model", options.get("ollama_model", config.AI["text_model"]))),
            vision_timeout=float(options.get("vision_timeout", config.AI["vision_timeout_seconds"])),
            max_vision_candidates=int(options.get("max_vision_candidates", config.AI["vision_max_candidates"])),
        )
        result_path = Path(result["session_dir"]) / "analysis.json"
        auto_render_summary = {"rendered": [], "failed": [], "auto_rendered": False}
        if bool(options.get("auto_render", True)):
            try:
                auto_render_summary = _auto_render_analysis(job, result, result_path, options)
            except JobCancelledError:
                raise
            except Exception as render_error:
                LOG.exception("Render automático interrompido após análise concluída")
                auto_render_summary = {"rendered": [], "failed": [{"rank": 0,
                    "error": f"Falha no pipeline de render: {type(render_error).__name__}: {render_error}"[:500]}],
                    "auto_rendered": True}
        compact_result = {"session_id": result.get("session_id"),
                          "summary": result.get("summary", {}),
                          "mode": result.get("mode"), "scene_mode": result.get("scene_mode"),
                          "auto_rendered": auto_render_summary.get("auto_rendered", False),
                          "rendered_count": len(auto_render_summary.get("rendered", [])),
                          "render_failed_count": len(auto_render_summary.get("failed", [])),
                          "rendered": auto_render_summary.get("rendered", []),
                          "render_failures": auto_render_summary.get("failed", [])}
        update_job(job_id, status="concluido", result_path=str(result_path), result=compact_result,
                   finished=True)
    except JobCancelledError as exc:
        update_job(job_id, status="cancelado", error=str(exc), finished=True)
    except Exception as exc:
        update_job(job_id, status="falhou", error=f"{type(exc).__name__}: {exc}", finished=True)
    finally:
        monitor_stop.set()
        monitor.join(timeout=2)
