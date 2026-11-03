from __future__ import annotations

import json
import logging
import logging.handlers
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import config


class JobCancelledError(RuntimeError):
    """O usuário solicitou cancelamento seguro do job atual."""


def ensure_directories() -> None:
    for directory in config.DIRS.values():
        Path(directory).mkdir(parents=True, exist_ok=True)


def setup_logging() -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger("cliprush")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.handlers.RotatingFileHandler(
        config.DIRS["logs"] / "app.log", maxBytes=5_000_000, backupCount=4, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def atomic_write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=target.name + ".", suffix=".tmp", delete=False) as tmp:
            temp_name = tmp.name
            json.dump(data, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
        for attempt in range(8):
            try:
                os.replace(temp_name, target)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                # Defender, indexadores e scanners podem segurar o destino por alguns ms.
                time.sleep(min(0.5, 0.05 * (2 ** attempt)))
    finally:
        if temp_name and os.path.exists(temp_name):
            try:
                os.unlink(temp_name)
            except OSError:
                # Um scanner pode manter o temporário aberto; não sobrepor a exceção original.
                pass


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def priority_kwargs() -> dict[str, Any]:
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
        if config.PERFORMANCE["priority_below_normal"]:
            flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)
        return {"creationflags": flags}
    def lower_priority() -> None:
        try:
            os.nice(5)
        except (AttributeError, OSError):
            pass
    return {"preexec_fn": lower_priority if config.PERFORMANCE["priority_below_normal"] else None,
            "start_new_session": True}


def run_low_priority(args: list[str], *, timeout: float | None = None,
                     capture_output: bool = True, text: bool = True,
                     check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, timeout=timeout, capture_output=capture_output, text=text,
                          check=check, **priority_kwargs())


def run_ffmpeg_progress(args: list[str], duration: float, progress: "ProgressWriter",
                        stage: str = "render", start_percent: float = 10.0,
                        end_percent: float = 95.0) -> subprocess.CompletedProcess:
    """Lê -progress pipe:1 em streaming e grava percentagem/ETA atômicos para a UI."""
    command = list(args) + ["-progress", "pipe:1", "-nostats"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               **priority_kwargs())
    last_out = 0.0
    lines = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            if progress.cancel_requested():
                raise JobCancelledError("Cancelamento solicitado pelo usuário.")
            line = line.strip()
            if line.startswith("out_time_ms=") or line.startswith("out_time_us="):
                try:
                    last_out = int(line.partition("=")[2]) / 1_000_000.0
                    ratio = max(0.0, min(1.0, last_out / max(0.1, duration)))
                    percent = start_percent + ratio * (end_percent - start_percent)
                    progress.update(stage, percent, f"FFmpeg: {last_out:.1f}/{duration:.1f}s", process_pid=process.pid)
                except ValueError:
                    pass
            elif line:
                lines.append(line)
        stderr = process.stderr.read() if process.stderr else ""
        returncode = process.wait()
    except (JobCancelledError, KeyboardInterrupt):
        terminate_process_tree(process.pid)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        progress.update("cancelado", progress.last_percent, "Processo FFmpeg encerrado com segurança.",
                        done=True, cancelled=True, process_pid=None)
        raise
    finally:
        if process.poll() is None:
            terminate_process_tree(process.pid)
            process.wait()
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()
    if returncode:
        raise RuntimeError(f"FFmpeg encerrou com código {returncode}: {stderr[-2400:]}")
    progress.update(stage, end_percent, "FFmpeg concluiu a renderização.", process_pid=None)
    return subprocess.CompletedProcess(command, returncode, "\n".join(lines), stderr)


def terminate_process_tree(pid: int) -> None:
    """Encerra a árvore inteira para suportar cancelamento sem filhos FFmpeg órfãos."""
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True, check=False)
    else:
        try:
            os.killpg(pid, 15)
            time.sleep(0.25)
            try:
                os.killpg(pid, 9)
            except OSError:
                pass
        except OSError:
            try:
                os.kill(pid, 9)
            except OSError:
                pass


def check_ffmpeg() -> dict[str, Any]:
    logger = setup_logging()
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise RuntimeError(f"{binary} não foi encontrado no PATH. Instale o FFmpeg full e reinicie o terminal.")
    result = run_low_priority(["ffmpeg", "-hide_banner", "-filters"], timeout=30)
    if result.returncode != 0:
        raise RuntimeError("Não foi possível consultar os filtros do FFmpeg.")
    available = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0][0] in ".TSCAV|":
            available.add(parts[1])
    missing = sorted(set(config.REQUIRED_FFMPEG_FILTERS) - available)
    if missing:
        raise RuntimeError("O build FFmpeg não inclui os filtros: " + ", ".join(missing) +
                           ". Instale o pacote full (Windows: gyan.dev) e adicione bin ao PATH.")
    version = run_low_priority(["ffmpeg", "-version"], timeout=15).stdout.splitlines()[0]
    logger.info("FFmpeg validado: %s", version)
    return {"version": version, "filters": sorted(available)}


def probe_media(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {source}")
    result = run_low_priority(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)], timeout=60)
    if result.returncode:
        raise RuntimeError(f"ffprobe não conseguiu ler a mídia: {result.stderr[-1200:]}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe retornou JSON inválido.") from exc
    video = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), None)
    audio = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), None)
    if video is None:
        raise RuntimeError("Arquivo recusado: nenhuma trilha de vídeo foi encontrada.")
    fmt = payload.get("format", {})
    duration = float(video.get("duration") or fmt.get("duration") or 0)
    if duration <= 0:
        raise RuntimeError("Não foi possível determinar a duração do vídeo.")
    def fps_value(value: str) -> float:
        try:
            n, d = value.split("/", 1)
            return float(n) / float(d) if float(d) else 0
        except (ValueError, ZeroDivisionError, AttributeError):
            return 0.0
    return {
        "path": str(source), "duration": duration, "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)), "fps": fps_value(video.get("avg_frame_rate", "0/1")),
        "video_codec": video.get("codec_name", "unknown"),
        "audio": audio is not None, "audio_codec": audio.get("codec_name") if audio else None,
        "size_bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns,
    }


def ensure_free_space(path: str | Path, required_bytes: int) -> None:
    usage = shutil.disk_usage(Path(path).resolve())
    if usage.free < required_bytes:
        raise OSError(f"Espaço insuficiente: necessários {required_bytes / 1e9:.2f} GB, "
                       f"disponíveis {usage.free / 1e9:.2f} GB.")


def cache_key(info: dict[str, Any], mode: str) -> str:
    identity = f"{info['path']}|{info['size_bytes']}|{info['mtime_ns']}|{mode}|{config.ANALYSIS['cache_schema_version']}"
    import hashlib
    return hashlib.sha256(identity.encode("utf-8", errors="replace")).hexdigest()[:24]


class InstanceLock:
    """Lock simples com detecção de PID morto; não apaga locks pertencentes a processo ativo."""
    def __init__(self, path: str | Path | None = None, on_busy: Any = None):
        self.path = Path(path or (config.DIRS["temp"] / "cliprush.lock"))
        self.owned = False
        self.on_busy = on_busy

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except PermissionError:
            return True
        except OSError:
            return False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "created": time.time()}
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(payload, file)
                    file.flush()
                    os.fsync(file.fileno())
                self.owned = True
                return
            except FileExistsError:
                current = read_json(self.path, {}) or {}
                try:
                    pid = int(current.get("pid", -1))
                except (TypeError, ValueError):
                    pid = -1
                if self._alive(pid):
                    message = f"Já há um ClipRush em execução (PID {pid}). Aguarde o job atual."
                    if self.on_busy:
                        self.on_busy(message)
                    raise RuntimeError(message)
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if not current and age < 10:
                    message = "Outro processo está inicializando o lock do ClipRush; tente novamente em alguns segundos."
                    if self.on_busy:
                        self.on_busy(message)
                    raise RuntimeError(message)
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
        raise RuntimeError("Não foi possível adquirir o lock do ClipRush; tente novamente.")

    def release(self) -> None:
        if self.owned:
            current = read_json(self.path, {}) or {}
            if int(current.get("pid", -1)) == os.getpid():
                self.path.unlink(missing_ok=True)
            self.owned = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_exc):
        self.release()


class ProgressWriter:
    def __init__(self, session_dir: str | Path):
        self.path = Path(session_dir) / "progress.json"
        self.started = time.time()
        self._steps: dict[str, float] = {}
        self.last_percent = 0.0

    @property
    def cancel_path(self) -> Path:
        return self.path.with_name("cancel.flag")

    def request_cancel(self) -> None:
        atomic_write_json(self.cancel_path, {"requested_at": time.time()})

    def cancel_requested(self) -> bool:
        return self.cancel_path.exists()

    def check_cancelled(self) -> None:
        if self.cancel_requested():
            raise JobCancelledError("Cancelamento solicitado pelo usuário.")

    def update(self, stage: str, percent: float, message: str = "", current_clip: int = 0,
               total_clips: int = 0, done: bool = False, cancelled: bool = False,
               process_pid: int | None = None) -> None:
        now = time.time()
        self._steps[stage] = self._steps.get(stage, 0.0) + 0.0
        percent = max(self.last_percent, min(100.0, max(0.0, float(percent))))
        self.last_percent = percent
        elapsed = now - self.started
        fraction = max(0.001, min(1.0, percent / 100.0))
        remaining = max(0.0, elapsed * (1.0 - fraction) / fraction)
        atomic_write_json(self.path, {
            "stage": stage, "percent": round(float(percent), 1), "message": message,
            "elapsed_seconds": round(elapsed, 1), "remaining_seconds": round(remaining, 1),
            "current_clip": current_clip, "total_clips": total_clips,
            "process_pid": process_pid,
            "done": done, "cancelled": cancelled, "updated_at": now,
        })
