"""IA local para análise multimodal de candidatos do ClipRush.

Envia somente miniaturas JPEG, métricas temporais resumidas e IDs opacos ao
servidor Ollama local. Nunca envia vídeo, áudio bruto, nome de arquivo ou caminho.
Quando o VLM não existe/falha, tenta ranquear as métricas com o modelo textual;
se isso também falhar, o chamador mantém o ranking heurístico local.
"""
from __future__ import annotations

import base64
import ipaddress
import io
import json
import math
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

import ollama_ai

DEFAULT_VISION_MODEL = "llava:7b"
DEFAULT_TEXT_MODEL = "llama3.2:3b"
MAX_VISION_CANDIDATES = 18
BATCH_SIZE = 3
FRAME_WIDTH = 320
FRAME_HEIGHT = 180

_VISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "visual_class": {"type": "string", "enum": ["action", "cutscene", "menu_loading", "traversal", "unknown"]},
                    "relevant": {"type": "boolean"},
                    "visual_score": {"type": "number", "minimum": 0, "maximum": 100},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "peak_offset": {"type": "number", "minimum": -5, "maximum": 5},
                    "moment_start_offset": {"type": "number", "minimum": -18, "maximum": 0},
                    "moment_end_offset": {"type": "number", "minimum": 0, "maximum": 18},
                    "evidence": {"type": "string"},
                },
                "required": ["candidate_id", "visual_class", "relevant", "visual_score", "confidence",
                             "peak_offset", "moment_start_offset", "moment_end_offset", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}

_TEXT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "priority_score": {"type": "number", "minimum": 0, "maximum": 100},
                    "keep": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["candidate_id", "priority_score", "keep", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["decisions"],
    "additionalProperties": False,
}


def _local_url(base_url: str) -> str | None:
    try:
        parsed = urllib.parse.urlsplit(str(base_url or "").strip())
        if parsed.scheme.lower() != "http" or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            return None
        if parsed.path not in ("", "/"):
            return None
        host = parsed.hostname.lower().rstrip(".")
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = host == "localhost"
        return parsed._replace(path="", query="", fragment="").geturl().rstrip("/") if local else None
    except (TypeError, ValueError):
        return None


def _model_matches(requested: str, installed: list[str]) -> str | None:
    wanted = str(requested or "").strip().lower()
    if not wanted:
        return None
    for name in installed:
        lowered = name.lower()
        if lowered == wanted or (":" not in wanted and lowered.startswith(wanted + ":")):
            return name
    return None


def _post_json(base_url: str, body: dict[str, Any], timeout: float) -> dict[str, Any] | None:
    url = _local_url(base_url)
    if not url:
        return None
    try:
        request = urllib.request.Request(
            url + "/api/chat", data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=max(2.0, float(timeout))) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _safe_num(value: Any, low: float, high: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and low <= number <= high else None


def _extract_contact_sheet(source: Path, peak: float) -> tuple[bytes | None, list[float]]:
    """Extrai três frames próximos ao pico em um JPEG pequeno, sem decodificar a run toda."""
    start = max(0.0, float(peak) - 4.0)
    offsets = [round(start + 0.0 - peak, 1), round(start + 2.0 - peak, 1), round(start + 4.0 - peak, 1)]
    vf = (f"fps=0.5,scale={FRAME_WIDTH}:{FRAME_HEIGHT}:force_original_aspect_ratio=decrease:flags=fast_bilinear,"
          f"pad={FRAME_WIDTH}:{FRAME_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black,tile=3x1")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(source),
               "-t", "6.0", "-an", "-vf", vf, "-frames:v", "1", "-q:v", "5",
               "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, check=False)
        if result.returncode or not result.stdout:
            return None, offsets
        image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
        # Contact sheets are internal model inputs, not user-facing artwork. Labels
        # anchor each panel to its relative timestamp and reduce temporal ambiguity.
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        for index, offset in enumerate(offsets):
            x = index * FRAME_WIDTH + 6
            label = f"t{offset:+.1f}s"
            draw.rectangle((x - 3, 3, x + 68, 24), fill=(0, 0, 0))
            draw.text((x, 5), label, fill=(255, 255, 255), font=font)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=75, optimize=True)
        return output.getvalue(), offsets
    except Exception:
        return None, offsets


def _choose_candidates(candidates: list[dict[str, Any]], duration: float, max_count: int) -> list[tuple[int, dict[str, Any]]]:
    if not candidates:
        return []
    limit = max(1, min(MAX_VISION_CANDIDATES, int(max_count)))
    def prior(row: dict[str, Any]) -> float:
        return (0.38 * float(row.get("audio_score", 0.0) or 0.0) +
                0.18 * float(row.get("scene_score", 0.0) or 0.0) +
                0.14 * float(row.get("movement_score", 0.0) or 0.0) +
                0.12 * float(row.get("speech_score", 0.0) or 0.0) +
                0.10 * float(row.get("tension_score", 0.0) or 0.0) +
                0.08 * float(row.get("confidence", 0.0) or 0.0))
    indexed = list(enumerate(candidates))
    # Reserve globally distributed probes first so the opening minutes cannot
    # monopolize visual inference on long recordings.
    zones = min(limit, max(5, min(12, limit // 2 + 1)))
    chosen: list[tuple[int, dict[str, Any]]] = []
    seen: set[int] = set()
    for zone in range(zones):
        lo, hi = duration * zone / zones, duration * (zone + 1) / zones
        pool = [(i, row) for i, row in indexed if lo <= float(row.get("peak", 0.0)) < hi]
        if pool:
            best = max(pool, key=lambda pair: prior(pair[1]))
            chosen.append(best)
            seen.add(best[0])
    for item in sorted(indexed, key=lambda pair: prior(pair[1]), reverse=True):
        if item[0] not in seen:
            chosen.append(item)
            seen.add(item[0])
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def _vision_decisions(base_url: str, model: str, timeout: float,
                      batch: list[tuple[str, dict[str, Any], bytes, list[float]]], final: bool) -> list[dict[str, Any]]:
    image_payloads = [base64.b64encode(item[2]).decode("ascii") for item in batch]
    candidates_context = []
    for candidate_id, candidate, _image, offsets in batch:
        candidates_context.append({
            "candidate_id": candidate_id,
            "peak_seconds": round(float(candidate.get("peak", 0.0)), 1),
            "frame_order": ["left", "center", "right"],
            "relative_frame_seconds": offsets,
            "signals": {
                "audio_peak_score": round(float(candidate.get("audio_score", 0.0) or 0.0), 3),
                "movement_score": round(float(candidate.get("movement_score", 0.0) or 0.0), 3),
                "hud_detected": bool(candidate.get("hud_detected", False)),
                "speech_score": round(float(candidate.get("speech_score", 0.0) or 0.0), 3),
                "speech_seconds": int(candidate.get("speech_seconds", 0) or 0),
                "music_score": round(float(candidate.get("music_score", 0.0) or 0.0), 3),
                "tension_score": round(float(candidate.get("tension_score", 0.0) or 0.0), 3),
            },
        })
    system = (
        "You are a conservative gameplay moment selector. Each image is a three-frame contact sheet for the "
        "candidate with the same position in the candidates array; panels run left-to-right and show relative time. "
        "Use visible evidence only. Distinguish gameplay action, story cutscene, menu/loading, traversal, and unknown. "
        "Do not invent kills, headshots, weapons, combos, dialogue, or outcomes. A high camera-cut count alone is not "
        "virality. Mark long uneventful traversal/menu/loading as not relevant. Return only the JSON schema requested. "
        "Scores measure visible moment strength, not guaranteed virality. Offsets are advisory and relative to peak_seconds."
    )
    user_payload = {"candidates": candidates_context, "decision_schema": _VISION_SCHEMA}
    body = {
        "model": model, "stream": False, "format": _VISION_SCHEMA,
        "keep_alive": "0" if final else "2m",
        "options": {"num_ctx": 4096, "num_gpu": 0, "num_thread": 4, "temperature": 0.0, "num_predict": 500},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False), "images": image_payloads},
        ],
    }
    response = _post_json(base_url, body, timeout)
    message = response.get("message") if isinstance(response, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return []
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return []
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list):
        return []
    known = {item[0] for item in batch}
    valid: list[dict[str, Any]] = []
    allowed_classes = {"action", "cutscene", "menu_loading", "traversal", "unknown"}
    for row in decisions:
        if not isinstance(row, dict) or row.get("candidate_id") not in known:
            continue
        score = _safe_num(row.get("visual_score"), 0, 100)
        confidence = _safe_num(row.get("confidence"), 0, 1)
        peak_offset = _safe_num(row.get("peak_offset"), -5, 5)
        start_offset = _safe_num(row.get("moment_start_offset"), -18, 0)
        end_offset = _safe_num(row.get("moment_end_offset"), 0, 18)
        visual_class = row.get("visual_class")
        relevant = row.get("relevant")
        reason = row.get("evidence")
        if (score is None or confidence is None or peak_offset is None or start_offset is None or
                end_offset is None or visual_class not in allowed_classes or not isinstance(relevant, bool) or
                not isinstance(reason, str)):
            continue
        clean_reason = " ".join(reason.split())[:240]
        valid.append({"candidate_id": row["candidate_id"], "visual_class": visual_class,
                      "relevant": relevant, "visual_score": score, "confidence": confidence,
                      "peak_offset": peak_offset, "moment_start_offset": start_offset,
                      "moment_end_offset": end_offset, "evidence": clean_reason})
    return valid


def _text_rank(base_url: str, model: str, timeout: float,
               batch: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    summary = []
    for candidate_id, row in batch:
        summary.append({
            "candidate_id": candidate_id,
            "audio_peak_score": round(float(row.get("audio_score", 0.0) or 0.0), 3),
            "movement_score": round(float(row.get("movement_score", 0.0) or 0.0), 3),
            "scene_score": round(float(row.get("scene_score", 0.0) or 0.0), 3),
            "hud_detected": bool(row.get("hud_detected", False)),
            "speech_score": round(float(row.get("speech_score", 0.0) or 0.0), 3),
            "speech_seconds": int(row.get("speech_seconds", 0) or 0),
            "music_score": round(float(row.get("music_score", 0.0) or 0.0), 3),
            "tension_score": round(float(row.get("tension_score", 0.0) or 0.0), 3),
            "heuristic_type": str(row.get("event_type", "impacto"))[:24],
        })
    system = (
        "Você auxilia o ranking de momentos de gameplay usando apenas métricas locais de áudio/movimento. "
        "Você não viu os quadros e não deve afirmar que ocorreu uma eliminação, headshot, explosão, combo ou fala. "
        "Dê prioridade a picos/ação/fala/tensão com evidência numérica, não crie timestamps. Responda somente o JSON schema."
    )
    body = {
        "model": model, "stream": False, "format": _TEXT_SCHEMA, "keep_alive": "0",
        "options": {"num_ctx": 2048, "num_gpu": 0, "num_thread": 4, "temperature": 0.0, "num_predict": 350},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"candidates": summary, "schema": _TEXT_SCHEMA}, ensure_ascii=False)},
        ],
    }
    response = _post_json(base_url, body, timeout)
    message = response.get("message") if isinstance(response, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        return []
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return []
    decisions = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decisions, list):
        return []
    known = {item[0] for item in batch}
    valid = []
    for row in decisions:
        if not isinstance(row, dict) or row.get("candidate_id") not in known or not isinstance(row.get("keep"), bool):
            continue
        score = _safe_num(row.get("priority_score"), 0, 100)
        reason = row.get("reason")
        if score is None or not isinstance(reason, str):
            continue
        valid.append({"candidate_id": row["candidate_id"], "priority_score": score,
                      "keep": row["keep"], "reason": " ".join(reason.split())[:220]})
    return valid


def analyze_candidates(
    source_path: str | Path,
    candidates: list[dict[str, Any]],
    *,
    duration: float,
    top_n: int,
    base_url: str = ollama_ai.DEFAULT_BASE_URL,
    vision_model: str = DEFAULT_VISION_MODEL,
    text_model: str = DEFAULT_TEXT_MODEL,
    timeout: float = 90.0,
    max_candidates: int = MAX_VISION_CANDIDATES,
    enabled: bool = True,
    progress: Any | None = None,
    status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Avalia candidatos com visão, texto/áudio ou informa fallback local.

    As decisões do modelo são ligadas apenas aos IDs criados aqui. Timestamps são
    relativos, limitados e sempre reconferidos pelo chamador antes do render.
    """
    fallback = {"engine": "heuristic_local", "status": "fallback", "model": "",
                "candidates_assessed": 0, "fallback_reason": "IA desativada ou sem candidatos", "decisions": {}}
    if not enabled or not candidates:
        return fallback
    status = status or ollama_ai.check_status(base_url=base_url, model=vision_model, timeout=2.0)
    if not status.get("ok"):
        fallback["fallback_reason"] = "Ollama local indisponível; usado motor local de áudio/VAD/visual."
        return fallback
    installed = list(status.get("models", []))
    vision_installed = _model_matches(vision_model, installed)
    if not vision_installed:
        for alternative in ("llava:7b", "llama3.2-vision:11b", "llama3.2-vision", "llava"):
            vision_installed = _model_matches(alternative, installed)
            if vision_installed:
                break
    text_installed = _model_matches(text_model, installed)
    selected = _choose_candidates(candidates, duration, max_candidates)
    if not selected:
        fallback["fallback_reason"] = "Nenhum candidato elegível para análise de IA."
        return fallback

    by_id = {f"c{i:04d}": (index, row) for i, (index, row) in enumerate(selected, start=1)}
    decisions_by_id: dict[str, dict[str, Any]] = {}
    source = Path(source_path).expanduser().resolve()
    if vision_installed:
        eligible: list[tuple[str, dict[str, Any], bytes, list[float]]] = []
        for candidate_id, (_index, candidate) in by_id.items():
            if progress is not None:
                progress.check_cancelled()
            frame, offsets = _extract_contact_sheet(source, float(candidate.get("peak", 0.0)))
            if frame:
                eligible.append((candidate_id, candidate, frame, offsets))
        batches = [eligible[i:i + BATCH_SIZE] for i in range(0, len(eligible), BATCH_SIZE)]
        for batch_index, batch in enumerate(batches):
            if progress is not None:
                progress.check_cancelled()
                progress.update("ia_visual", 78 + 15 * batch_index / max(1, len(batches)),
                                f"Ollama visão: lote {batch_index + 1}/{len(batches)} ({len(batch)} janelas).")
            rows = _vision_decisions(base_url, vision_installed, timeout, batch,
                                     final=(batch_index == len(batches) - 1))
            for row in rows:
                decisions_by_id[row["candidate_id"]] = row
        for candidate_id, decision in decisions_by_id.items():
            index, candidate = by_id[candidate_id]
            peak = float(candidate.get("peak", 0.0))
            peak = max(0.0, min(float(duration), peak + float(decision["peak_offset"])))
            action_start = max(0.0, min(float(duration), float(candidate.get("peak", 0.0)) + decision["moment_start_offset"]))
            action_end = max(0.0, min(float(duration), float(candidate.get("peak", 0.0)) + decision["moment_end_offset"]))
            if action_start >= action_end or not action_start <= float(candidate.get("peak", 0.0)) <= action_end:
                action_start = max(0.0, float(candidate.get("peak", 0.0)) - 3.0)
                action_end = min(float(duration), float(candidate.get("peak", 0.0)) + 3.0)
                decision["boundary_advisory_valid"] = False
            else:
                decision["boundary_advisory_valid"] = True
            decision.update({"candidate_index": index, "peak_seconds": peak,
                             "moment_start_seconds": action_start, "moment_end_seconds": action_end})
        if decisions_by_id:
            return {"engine": "ollama_vision", "status": "success" if len(decisions_by_id) == len(eligible) else "partial",
                    "model": vision_installed, "candidates_assessed": len(decisions_by_id),
                    "candidate_count": len(candidates), "fallback_reason": "" if decisions_by_id else "Resposta visual inválida; usado ranking local.",
                    "decisions": decisions_by_id}
        # If a vision model exists but cannot return validated decisions, continue to the text fallback.
        vision_failure = "O modelo visual retornou resposta ausente/inválida ou os frames não puderam ser extraídos."
    else:
        vision_failure = f"Modelo visual {vision_model!r} não instalado."

    if text_installed:
        text_batch = [(candidate_id, row) for candidate_id, (_index, row) in by_id.items()]
        text_decisions: dict[str, dict[str, Any]] = {}
        for start in range(0, len(text_batch), 12):
            batch = text_batch[start:start + 12]
            if progress is not None:
                progress.check_cancelled()
                progress.update("ia_texto", 85 + 10 * start / max(1, len(text_batch)),
                                "Fallback IA textual: classificando sinais de áudio/VAD e movimento, sem visão.")
            for decision in _text_rank(base_url, text_installed, min(timeout, 45.0), batch):
                text_decisions[decision["candidate_id"]] = decision
        for candidate_id, decision in text_decisions.items():
            index, _candidate = by_id[candidate_id]
            decision["candidate_index"] = index
        if text_decisions:
            return {"engine": "ollama_text_audio_fallback", "status": "fallback_text", "model": text_installed,
                    "candidates_assessed": len(text_decisions), "candidate_count": len(candidates),
                    "fallback_reason": vision_failure, "decisions": text_decisions}
    return {**fallback, "candidate_count": len(candidates),
            "fallback_reason": vision_failure + " O modelo textual também não respondeu; usado motor heurístico local."}


def editorial_copy_is_safe(copy: dict[str, Any], event_type: str) -> bool:
    """Impede títulos que afirmem eventos específicos não confirmados pela análise."""
    required = {"titles", "hook", "description", "cta", "hashtags", "pinned_comment"}
    if set(copy) != required or not isinstance(copy.get("titles"), list) or len(copy["titles"]) != 3:
        return False
    if any(not isinstance(copy.get(key), str) or not copy[key].strip() for key in required - {"titles"}):
        return False
    if any(not isinstance(title, str) or not title.strip() for title in copy["titles"]):
        return False
    specific_terms = {
        "escopeta": ("escopeta", "shotgun"), "headshot": ("headshot", "tiro na cabeça"),
        "parry_faca": ("parry", "aparar com a faca"), "motosserra": ("motosserra", "serra elétrica", "chainsaw"),
        "explosao": ("explosão", "explosao", "granada"), "boss_combate": ("chefe", "boss"),
        "susto_grito": ("susto", "grito", "jumpscare"), "pistola_rifle": ("pistola", "rifle", "tiro certeiro"),
        "combo": ("combo",),
    }
    text = " ".join(str(copy.get(key, "")) for key in ("titles", "hook", "description", "cta", "hashtags", "pinned_comment")).lower()
    return all(event_type == allowed or not any(term in text for term in terms)
               for allowed, terms in specific_terms.items())


__all__ = ["DEFAULT_VISION_MODEL", "DEFAULT_TEXT_MODEL", "MAX_VISION_CANDIDATES",
           "analyze_candidates", "editorial_copy_is_safe"]
