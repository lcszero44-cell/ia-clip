from __future__ import annotations

import calendar
import csv
import hashlib
import io
import json
import re
import shutil
import time
import uuid
import zipfile
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
import local_store
import ollama_ai
import queue_service
from common import atomic_write_json, ensure_directories, read_json
from library import LIBRARY_PATH, load_library, save_library, validate_library_document
from postkit import generate_post_kit, safe_event_type
from queue_service import enqueue_analysis, get_jobs
from renderer import REACTIONS, render_clip

TYPE_LABELS = {
    "cena": "CENA / CUTSCENE", "escopeta": "AÇÃO · ESCOPETA", "headshot": "AÇÃO · HEADSHOT",
    "parry_faca": "AÇÃO · PARRY", "motosserra": "AÇÃO · MOTOSSERRA", "explosao": "AÇÃO · EXPLOSÃO",
    "boss_combate": "AÇÃO · CHEFE", "susto_grito": "AÇÃO · SUSTO/GRITO", "pistola_rifle": "AÇÃO · TIRO",
    "combo": "AÇÃO · COMBO", "impacto": "AÇÃO · IMPACTO", "morno": "MOMENTO · BAIXA INTENSIDADE",
}
SCENE_MODES = {
    "Manter cenas (padrão)": "manter",
    "Priorizar gameplay (+bônus leve)": "priorizar-gameplay",
    "Ignorar cenas": "ignorar",
}
STATUS_LABELS = {
    "pendente": "Na fila", "processando": "Em processamento", "concluido": "Concluído",
    "falhou": "Falhou", "cancelado": "Cancelado", "rascunho": "Rascunho",
    "roteiro": "Em roteiro", "agendado": "Agendado", "publicado": "Publicado", "pausado": "Pausado",
}


@st.cache_data(ttl=15, show_spinner=False)
def _cached_ollama_status(base_url: str, model: str) -> dict[str, Any]:
    return ollama_ai.check_status(base_url=base_url, model=model, timeout=1.0)


def _cached_visual_ollama_status(base_url: str, preferred_model: str) -> dict[str, Any]:
    status = _cached_ollama_status(base_url, preferred_model)
    if not status.get("ok") or status.get("model_installed"):
        return status
    available = [str(name) for name in status.get("models", [])]
    preferences = [preferred_model, "llava:7b", "llama3.2-vision:11b", "llama3.2-vision", "llava"]
    for preference in preferences:
        match = next((name for name in available
                      if name == preference or name.startswith(preference.rstrip(":") + ":")), None)
        if match:
            return {**status, "model_installed": True, "model": match,
                    "requested_model": preferred_model, "auto_detected": match != preferred_model}
    return status


def _copy_button(label: str, value: str, key: str) -> None:
    """Botão de cópia no navegador; o texto é escapado antes de entrar no iframe."""
    safe_value = json.dumps(str(value or ""), ensure_ascii=False)
    safe_value = safe_value.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    safe_label = (label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                  .replace('"', "&quot;"))
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(key))[:120] or "copy"
    component_html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
      html,body{{margin:0;background:transparent;font-family:Inter,'Segoe UI',Arial,sans-serif}}
      button{{width:100%;height:42px;border:1px solid #49366e;border-radius:10px;background:#211a35;
      color:#f4efff;font-weight:700;font-size:13px;cursor:pointer;transition:.15s ease}}
      button:hover{{border-color:#a855f7;background:#30224c}}button:focus{{outline:2px solid #c084fc}}
      span{{display:block;height:14px;padding-top:2px;color:#86efac;font-size:11px;text-align:center}}
      </style></head><body data-copy-id="{safe_id}"><button type="button" onclick="copyText()">{safe_label}</button><span id="done"></span>
      <script>const text={safe_value};async function copyText(){{const status=document.getElementById('done');
      try{{await navigator.clipboard.writeText(text);status.textContent='Copiado para a área de transferência';}}
      catch(e){{const t=document.createElement('textarea');t.value=text;document.body.appendChild(t);t.select();
      const ok=document.execCommand('copy');t.remove();status.textContent=ok?'Copiado':'Copie pelo bloco de texto';}}
      }}</script></body></html>"""
    components.html(component_html, height=60, scrolling=False)


def _ai_copy_is_safe(copy: dict[str, Any], event_type: str) -> bool:
    """Descarta texto que transforme evento incerto em uma ação específica inventada."""
    required = {"titles", "hook", "description", "cta", "hashtags", "pinned_comment"}
    if set(copy) != required or not isinstance(copy.get("titles"), list) or len(copy["titles"]) != 3:
        return False
    if any(not isinstance(copy.get(key), str) or not copy[key].strip()
           for key in required - {"titles"}):
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
    text = " ".join(str(copy.get(key, "")) for key in
                    ("titles", "hook", "description", "cta", "hashtags", "pinned_comment")).lower()
    return all(event_type == allowed or not any(term in text for term in terms)
               for allowed, terms in specific_terms.items())


def _generate_editorial_kit(clip: dict[str, Any], game_name: str, platform: str,
                            library_doc: dict[str, Any], settings: dict[str, Any], *,
                            channel_name: str = "", brand_handle: str = "",
                            default_hashtags: str = "", content_warning: str = "") -> tuple[dict[str, Any], bool]:
    """Gera um kit com Ollama sob demanda; qualquer falha volta ao kit offline."""
    safe_clip = {**clip, "event_type": safe_event_type(clip)}
    fallback = generate_post_kit(safe_clip, game_name, platform, library_doc,
                                 channel_name=channel_name, brand_handle=brand_handle,
                                 default_hashtags=default_hashtags, content_warning=content_warning)
    if not settings.get("ollama_enabled", True):
        return fallback, False
    context = {
        "game_name": str(game_name or "")[:100], "platform": platform,
        "evento_seguro": safe_clip["event_type"], "score": float(clip.get("score", 0) or 0),
        "duration_seconds": float(clip.get("duration_seconds", 0) or 0),
        "scene_speech": (f"cena da história; fala detectada por {float(clip.get('speech_seconds', 0) or 0):.0f} segundos"
                         if safe_clip["event_type"] == "cena" else "gameplay; texto deve permanecer genérico se o evento não for específico"),
        "summary": (f"Evidência visual Ollama: {str(clip.get('ai_evidence', '') or 'não disponível')[:220]}; "
                    f"classe visual {str(clip.get('ai_visual_class', 'não analisada'))}; "
                    f"score {float(clip.get('score', 0) or 0):.0f}/100; não invente jogadas."),
    }
    try:
        ai_copy = ollama_ai.generate_editorial_copy(
            context, base_url=settings.get("ollama_base_url", "http://localhost:11434"),
            model=settings.get("ollama_model", "llama3.2:3b"),
            timeout=float(settings.get("ollama_timeout", 25)),
        )
    except Exception:
        ai_copy = None
    if not isinstance(ai_copy, dict) or not _ai_copy_is_safe(ai_copy, safe_clip["event_type"]):
        return fallback, False
    result = {**fallback, **ai_copy}
    # Preserva as hashtags confiáveis do kit local e dos settings mesmo que o modelo varie.
    tags = []
    for value in (str(result.get("hashtags", "")), str(fallback.get("hashtags", ""))):
        for tag in re.findall(r"#[\w]+", value, flags=re.UNICODE):
            if tag.lower() not in {item.lower() for item in tags}:
                tags.append(tag)
    result["hashtags"] = " ".join(tags[:12])
    result["caption"] = "\n\n".join(part for part in (
        result["titles"][0], result["hook"], result["description"], result["cta"], result["hashtags"],
        f"Aviso: {content_warning.strip()}" if content_warning.strip() else "",
    ) if part)
    result["reaction_text"] = fallback["reaction_text"]
    result["event_type"] = fallback["event_type"]
    return result, True


def _save_kit_content(analysis_path: Path, rank: int, platform: str, content: dict[str, Any],
                      game_name: str, kit: dict[str, Any]) -> None:
    title = str(kit["titles"][0]).strip()
    caption = str(kit.get("caption", "")).strip()
    lines = caption.splitlines()
    if lines:
        lines[0] = title
        caption = "\n".join(lines)
    else:
        caption = title
    local_store.save_clip_content(analysis_path, rank, platform, {
        **content, "game_name": game_name.strip(), "publish_title": title,
        "caption": caption, "hashtags": str(kit.get("hashtags", "")).strip(),
        "pinned_comment": str(kit.get("pinned_comment", "")).strip(),
        "ai_titles": list(kit.get("titles", [])),
    })


def _apply_visual_theme() -> None:
    st.markdown("""<style>
    :root{--cr-violet:#9b5cff;--cr-violet-light:#c084fc;--cr-panel:#121321;--cr-edge:#2a2940}
    .stApp{background:radial-gradient(ellipse at 12% 0%,rgba(111,49,173,.16),transparent 38%),#090a12}
    html,body,[class*="css"]{font-family:Inter,'Segoe UI',sans-serif}
    [data-testid="stHeader"]{background:rgba(9,10,18,.75)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#10101d 0%,#0b0c14 100%);border-right:1px solid #242337}
    [data-testid="stSidebar"] h3{color:#e9ddff}
    h1,h2,h3{letter-spacing:-.025em}
    .stTabs [data-baseweb="tab-list"]{gap:5px;border-bottom:1px solid var(--cr-edge)}
    .stTabs [data-baseweb="tab"]{height:46px;border-radius:10px 10px 0 0;color:#b6b4c8;font-weight:600}
    .stTabs [aria-selected="true"]{color:#f0e6ff!important;border-bottom:2px solid var(--cr-violet)!important}
    [data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--cr-edge)!important;background:rgba(18,19,33,.72)}
    [data-testid="stMetric"]{background:#121321;border:1px solid var(--cr-edge);padding:12px;border-radius:12px}
    div[data-testid="stButton"]>button[kind="primary"]{background:linear-gradient(90deg,#7439ce,#9b5cff);border:0;font-weight:700}
    div[data-testid="stButton"]>button{border-radius:10px;transition:transform .12s ease,border-color .12s ease}
    div[data-testid="stButton"]>button:hover{border-color:#a879ff;transform:translateY(-1px)}
    [data-testid="stExpander"]{border-color:var(--cr-edge);border-radius:12px}
    [data-testid="stCaptionContainer"]{color:#a5a3b7}
    </style>""", unsafe_allow_html=True)
CHECKLIST_ITEMS = [
    ("hook", "Gancho claro nos primeiros segundos"),
    ("contexto", "Contexto suficiente para entender o momento"),
    ("hud", "HUD/vida/mapa continuam visíveis e sem crop"),
    ("safe_zone", "Título e legendas dentro das áreas seguras"),
    ("legendas", "Legendas revisadas, acentos e sincronização corretos"),
    ("audio", "Áudio limpo, sem clipping e volume equilibrado"),
    ("cor", "Cor e brilho conferidos no vídeo exportado"),
    ("capa", "Capa/frame inicial legível no celular"),
    ("titulo", "Título fiel ao conteúdo, sem promessa enganosa"),
    ("descricao", "Descrição e hashtags revisadas para a plataforma"),
    ("direitos", "Música, trilha e materiais de terceiros autorizados"),
    ("spoiler", "Spoilers sinalizados ou removidos quando necessário"),
    ("idade", "Violência/linguagem e classificação etária consideradas"),
    ("export", "Formato, duração e arquivo MP4 conferidos"),
    ("publicacao", "Publicação registrada no calendário e link guardado"),
    ("metricas", "Métricas iniciais anotadas após publicar"),
]


def _format_time(seconds: float) -> str:
    value = max(0, int(float(seconds or 0)))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


def _short_time(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    try:
        value = max(0, int(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    return f"{value // 60:02d}:{value % 60:02d}"


def _safe_stem(value: str) -> str:
    return re.sub(r"[^\w.-]+", "_", Path(value).stem, flags=re.UNICODE).strip("._")[:80] or "gameplay"


def _save_upload(upload: Any) -> Path:
    ensure_directories()
    suffix = Path(upload.name).suffix.lower()
    if suffix not in {".mp4", ".mkv", ".mov", ".m4v", ".avi"}:
        raise ValueError("Formato não aceito. Envie MP4, MKV, MOV, M4V ou AVI.")
    destination = config.DIRS["uploads"] / f"{_safe_stem(upload.name)}_{uuid.uuid4().hex[:8]}{suffix}"
    upload.seek(0)
    with destination.open("wb") as output:
        shutil.copyfileobj(upload, output, length=8 * 1024 * 1024)
    return destination


def _apply_performance(settings: dict[str, Any]) -> None:
    config.PERFORMANCE["threads"] = max(1, min(8, int(settings.get("threads", config.PERFORMANCE["threads"]))))
    config.PERFORMANCE["economy_mode"] = bool(settings.get("economy_mode", config.PERFORMANCE["economy_mode"]))
    config.PERFORMANCE["priority_below_normal"] = bool(settings.get("priority_below_normal", config.PERFORMANCE["priority_below_normal"]))


def _state_settings(prefix: str = "cfg_") -> dict[str, Any]:
    keys = list(local_store.DEFAULT_SETTINGS)
    return {key: st.session_state.get(f"{prefix}{key}", local_store.DEFAULT_SETTINGS[key]) for key in keys}


def _set_state_defaults(settings: dict[str, Any]) -> None:
    if st.session_state.pop("_apply_saved_settings", False):
        for key, value in settings.items():
            st.session_state[f"cfg_{key}"] = value
            st.session_state[f"prefs_{key}"] = value
        st.session_state["cfg_scene_mode_label"] = next(
            (label for label, mode in SCENE_MODES.items() if mode == settings["scene_mode"]),
            list(SCENE_MODES)[0],
        )
        st.session_state["cfg_scene_share_pct"] = int(round(settings["max_scene_share"] * 100))
        st.session_state["cfg_top_n"] = int(settings["top_n"])
        st.session_state["prefs_scene_label"] = next(
            (label for label, mode in SCENE_MODES.items() if mode == settings["scene_mode"]),
            list(SCENE_MODES)[0],
        )
        st.session_state["prefs_max_scene_share"] = int(round(settings["max_scene_share"] * 100))
    for key, value in settings.items():
        st.session_state.setdefault(f"cfg_{key}", value)
        st.session_state.setdefault(f"prefs_{key}", value)
    st.session_state.setdefault(
        "cfg_scene_mode_label",
        next((label for label, mode in SCENE_MODES.items() if mode == settings["scene_mode"]), list(SCENE_MODES)[0]),
    )
    st.session_state.setdefault("cfg_scene_share_pct", int(round(settings["max_scene_share"] * 100)))
    st.session_state.setdefault("cfg_top_n", int(settings["top_n"]))


def _queue_options(settings: dict[str, Any]) -> dict[str, Any]:
    automatic = bool(settings["automatic_count"])
    return {
        "analysis_mode": settings["analysis_mode"],
        "top_n": None if automatic else int(settings["top_n"]),
        "scene_mode": settings["scene_mode"],
        "max_scene_share": float(settings["max_scene_share"]),
        "debug": bool(settings["debug"]),
        "threads": int(settings["threads"]),
        "economy_mode": bool(settings["economy_mode"]),
        "priority_below_normal": bool(settings["priority_below_normal"]),
        "use_cache": True,
        "vision_ai_enabled": bool(settings.get("vision_ai_enabled", True)),
        "vision_model": str(settings.get("vision_model", "llava:7b")),
        "text_model": str(settings.get("ollama_model", "llama3.2:3b")),
        "ollama_base_url": str(settings.get("ollama_base_url", "http://localhost:11434")),
        "ollama_enabled": bool(settings.get("ollama_enabled", True)),
        "ollama_timeout": int(settings.get("ollama_timeout", 25)),
        "vision_timeout": int(settings.get("vision_timeout", 90)),
        "max_vision_candidates": int(settings.get("max_vision_candidates", 18)),
        "auto_render": bool(settings.get("auto_render", True)),
        "platform": str(settings.get("platform", "shorts")),
        "captions": bool(settings.get("captions", False)),
        "game_name": str(settings.get("game_name", "")),
        "channel_name": str(settings.get("channel_name", "")),
        "brand_handle": str(settings.get("brand_handle", "")),
        "default_hashtags": str(settings.get("default_hashtags", "")),
        "content_warning": str(settings.get("content_warning", "")),
    }


def _available_analyses() -> list[dict[str, Any]]:
    session_root = config.DIRS["sessions"]
    if not session_root.exists():
        return []
    paths = sorted(session_root.rglob("analysis.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    entries = []
    seen = set()
    for path in paths[:500]:
        resolved = str(path.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        payload = read_json(path)
        if not isinstance(payload, dict) or not isinstance(payload.get("clips"), list):
            continue
        source = payload.get("source", {})
        source_path = str(source.get("path", "")) if isinstance(source, dict) else ""
        source_name = Path(source_path).name if source_path else "origem desconhecida"
        session_id = str(payload.get("session_id", path.parent.name))
        clips = payload.get("clips", [])
        entries.append({
            "path": path, "payload": payload, "source_path": source_path,
            "label": f"{source_name} · {len(clips)} cortes · {session_id}",
            "mtime": path.stat().st_mtime,
        })
    return entries


def _analysis_selector(label: str, key: str) -> dict[str, Any] | None:
    entries = _available_analyses()
    if not entries:
        st.info("Ainda não há análises concluídas. Adicione uma gameplay à fila primeiro.")
        return None
    labels = [item["label"] for item in entries]
    selected_label = st.selectbox(label, labels, key=key)
    return entries[labels.index(selected_label)]


def _find_existing_render(analysis_path: Path, clip: dict[str, Any], platform: str) -> dict[str, Any] | None:
    assets = local_store.list_rendered_assets(analysis_path, int(clip.get("rank", 0)), platform)
    for asset in assets:
        if Path(asset["video_path"]).is_file():
            return {"video": asset["video_path"], "cover": asset["cover_path"] or None,
                    "title": asset["title"]}
    # Compatibilidade com renders V3 anteriores que ainda não tinham catálogo SQLite.
    session_dir = Path(analysis_path).parent
    start_marker = f"_{platform}_{int(float(clip.get('start', 0))):06d}_"
    for video in sorted(session_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        if start_marker in video.name and ".partial." not in video.name:
            cover = video.with_name(video.stem + "_capa.jpg")
            return {"video": str(video), "cover": str(cover) if cover.is_file() else None,
                    "title": REACTIONS.get(str(clip.get("event_type")), "MOMENTO DA RUN")}
    return None


def _render_clip_for_analysis(entry: dict[str, Any], clip: dict[str, Any], platform: str,
                              captions: bool, draft: bool, economy: bool,
                              title_override: str | None = None,
                              confirmed_event: str = "") -> dict[str, Any]:
    payload = entry["payload"]
    source = Path(entry["source_path"])
    if not source.is_file():
        raise FileNotFoundError("A gameplay original não está disponível. Restaure-a em uploads/ ou use o caminho original.")
    session_dir = entry["path"].resolve().parent
    output_dir = session_dir / "interface_render"
    content = local_store.get_clip_content(entry["path"], int(clip.get("rank", 0)), platform)
    effective_clip = {**clip, "confirmed_event": confirmed_event or content.get("confirmed_event", "")}
    event_type = safe_event_type(effective_clip)
    title = str(title_override or content.get("overlay_title") or
                REACTIONS.get(event_type, "MOMENTO INTENSO")).strip()
    output = render_clip(
        source_path=source, start=float(clip["start"]), end=float(clip["end"]), peak=float(clip["peak"]),
        output_dir=output_dir, platform=platform, title=title,
        event_type=event_type, captions=captions,
        language=config.ANALYSIS["speech_language"], reaction=True,
        more_events=[float(item["timestamp"]) for item in clip.get("events", [])],
        brightness_windows=clip.get("brightness_windows", []), draft=draft, economy=economy,
    )
    if not output:
        raise RuntimeError("O renderizador não produziu um MP4 para esse destino/duração.")
    item = output[0]
    local_store.save_rendered_asset(entry["path"], int(clip.get("rank", 0)), platform,
                                   item["video"], item.get("cover"), title)
    content.update({"overlay_title": title, "confirmed_event": effective_clip["confirmed_event"]})
    local_store.save_clip_content(entry["path"], int(clip.get("rank", 0)), platform, content)
    return item


def _display_clip_results(entry: dict[str, Any], settings: dict[str, Any]) -> None:
    payload = entry["payload"]
    clips = payload.get("clips", [])
    summary = payload.get("summary", {})
    if not clips:
        st.warning("Esta análise não tem clipes selecionados.")
        return
    cols = st.columns(4)
    cols[0].metric("Cortes", f"{len(clips)} / {summary.get('target_count', len(clips))}")
    cols[1].metric("Cenas", f"{summary.get('scene_count', 0)} / {summary.get('max_scenes', 0)}")
    cols[2].metric("Candidatos", summary.get("candidate_count", len(payload.get("candidates", []))))
    cols[3].metric("Modo", payload.get("scene_mode", config.SCENE_MODE))
    ai_engine = summary.get("ai_engine", "heuristic_local")
    if ai_engine == "ollama_vision":
        st.success(f"Ranking multimodal ativo · {summary.get('ai_model') or 'Ollama Vision'} · "
                   f"{summary.get('ai_candidates_assessed', 0)} janelas visuais avaliadas.")
    elif ai_engine == "ollama_text_audio_fallback":
        st.info(f"Fallback IA textual/áudio ativo · {summary.get('ai_model') or 'Ollama'}; o modelo não viu os frames. "
                "A seleção visual continua apoiada no motor local.")
    else:
        st.warning("Esta análise foi classificada pelo motor local, sem resposta válida do Ollama visual/textual. "
                   f"{summary.get('ai_fallback_reason', '')}")
    platform = st.selectbox("Destino de renderização", list(config.PLATFORMS),
                            index=list(config.PLATFORMS).index(settings.get("platform", "shorts")),
                            format_func=lambda value: config.PLATFORMS[value]["label"],
                            key=f"result_platform_{hashlib.md5(str(entry['path']).encode()).hexdigest()[:8]}")
    digest = hashlib.sha256(str(entry["path"].resolve()).encode("utf-8")).hexdigest()[:10]
    preview_key = f"results-preview-{digest}-{platform}"
    ranks = [int(clip.get("rank", index + 1)) for index, clip in enumerate(clips)]
    if st.session_state.get(preview_key) not in ranks:
        st.session_state[preview_key] = ranks[0]
    preview_rank = int(st.session_state[preview_key])
    contents = {rank: local_store.get_clip_content(entry["path"], rank, platform) for rank in ranks}
    def _label_for(item: dict[str, Any]) -> str:
        rank = int(item.get("rank", 0))
        content = contents.get(rank, {})
        effective = safe_event_type({**item, "confirmed_event": content.get("confirmed_event", "")})
        raw = str(item.get("event_type", "impacto"))
        label = TYPE_LABELS.get(effective, effective.replace("_", " ").upper())
        if raw != effective and not content.get("confirmed_event"):
            label += " · TIPO NÃO CONFIRMADO"
        return f"#{rank:02d} · {label} · {_format_time(item.get('peak', 0))}"
    nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
    current_index = ranks.index(preview_rank)
    if nav_prev.button("← Clipe anterior", disabled=current_index == 0,
                        key=f"results-prev-{digest}-{platform}"):
        st.session_state[preview_key] = ranks[max(0, current_index - 1)]
        st.rerun()
    nav_info.caption(f"Visualizando/baixando: {_label_for(next(item for item in clips if int(item.get('rank', 0)) == preview_rank))}")
    if nav_next.button("Próximo clipe →", disabled=current_index >= len(ranks) - 1,
                       key=f"results-next-{digest}-{platform}"):
        st.session_state[preview_key] = ranks[min(len(ranks) - 1, current_index + 1)]
        st.rerun()
    library_doc = load_library()
    for clip in clips:
        rank = int(clip.get("rank", 0))
        content = contents.get(rank) or local_store.get_clip_content(entry["path"], rank, platform)
        event_type = safe_event_type({**clip, "confirmed_event": content.get("confirmed_event", "")})
        raw_event_type = str(clip.get("event_type", "impacto"))
        is_scene = event_type == "cena"
        badge = "#9b59d0" if is_scene else "#218c60"
        event_label = TYPE_LABELS.get(event_type, event_type.replace("_", " ").upper())
        if raw_event_type != event_type and not content.get("confirmed_event"):
            event_label += " · TIPO NÃO CONFIRMADO"
        kit_clip = {**clip, "event_type": event_type,
                    "confirmed_event": content.get("confirmed_event", "")}
        game_name_default = content.get("game_name") or settings.get("game_name", "")
        kit = generate_post_kit(kit_clip, game_name_default, platform, library_doc,
                                channel_name=settings.get("channel_name", ""),
                                brand_handle=settings.get("brand_handle", ""),
                                default_hashtags=settings.get("default_hashtags", ""),
                                content_warning=settings.get("content_warning", ""))
        existing = _find_existing_render(entry["path"], clip, platform)
        auto_overlay = (existing.get("title") if existing and raw_event_type == event_type
                        else REACTIONS.get(event_type, "MOMENTO INTENSO"))
        overlay_default = content.get("overlay_title") or auto_overlay or "OLHA ISSO!"
        publish_title_default = content.get("publish_title") or kit["titles"][0]
        caption_default = content.get("caption") or kit["caption"]
        hashtags_default = content.get("hashtags") or kit["hashtags"]
        pinned_default = content.get("pinned_comment") or kit["pinned_comment"]
        editor_key = f"clip-editor-{digest}-{rank}-{platform}"
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.markdown(f"<span style='background:{badge};color:white;padding:5px 10px;border-radius:999px;font-weight:800'>{event_label}</span> &nbsp; **#{rank:02d}**", unsafe_allow_html=True)
            left.markdown(f"**{_format_time(clip.get('start', 0))} – {_format_time(clip.get('end', 0))}** · "
                          f"pico {_format_time(clip.get('peak', 0))} · {clip.get('duration_seconds', 0):.0f}s")
            left.caption(clip.get("classification_reason", "Classificação heurística"))
            right.metric("Score relativo", f"{float(clip.get('score', 0)):.1f}/100")
            if clip.get("ai_engine") == "ollama_vision":
                st.caption(f"IA visual · nota {float(clip.get('ai_score', 0) or 0):.0f}/100 · {clip.get('ai_visual_class', 'unknown')} · {clip.get('ai_evidence', '')}")
            elif clip.get("ai_engine") == "ollama_text_audio_fallback":
                st.caption(f"Fallback IA textual/áudio (sem visão) · nota {float(clip.get('ai_score', 0) or 0):.0f}/100 · {clip.get('ai_evidence', '')}")
            if is_scene:
                st.caption(f"Cena · fala {clip.get('speech_seconds', 0)}s ({clip.get('speech_detector', 'espectral')}) · "
                           f"HUD {'detectado' if clip.get('hud_detected') else 'não detectado'} · "
                           f"cortes {clip.get('camera_cuts', 0)} · score fala/música/tensão "
                           f"{clip.get('speech_score', 0):.2f}/{clip.get('music_score', 0):.2f}/{clip.get('tension_score', 0):.2f}")
            if raw_event_type != event_type and not content.get("confirmed_event"):
                st.caption(f"O detector sugeriu {TYPE_LABELS.get(raw_event_type, raw_event_type)}, mas a evidência não foi suficiente; títulos automáticos usarão texto neutro. Você pode confirmar o evento manualmente.")
            if rank != preview_rank and st.button(f"Selecionar para prévia · #{rank:02d}",
                                                   key=f"select-preview-{digest}-{rank}-{platform}"):
                st.session_state[preview_key] = rank
                st.rerun()
            with st.expander("Editar texto do vídeo e dados de postagem", expanded=(rank == preview_rank)):
                event_options = {"Automático seguro (neutro se incerto)": ""}
                for event_key, event_name in TYPE_LABELS.items():
                    event_options[f"Confirmado por mim · {event_name}"] = event_key
                option_labels = list(event_options)
                saved_event = str(content.get("confirmed_event", "") or "")
                selected_event = next((label for label, value in event_options.items() if value == saved_event), option_labels[0])
                result_notice_key = f"ai-notice-results-{digest}-{rank}-{platform}"
                result_notice = st.session_state.pop(result_notice_key, "")
                if result_notice == "ollama":
                    st.success(f"Texto criado por {settings.get('ollama_model', 'llama3.2:3b')} e salvo neste corte.")
                elif result_notice == "offline":
                    st.info("Ollama indisponível ou resposta não validada; sugestões heurísticas offline salvas sem interromper o app.")
                if st.button("✨ Gerar títulos e descrição com IA local", key=f"ai-results-{digest}-{rank}-{platform}",
                             disabled=not settings.get("ollama_enabled", True)):
                    with st.spinner("Consultando Ollama local; se estiver indisponível, aplico as sugestões offline…"):
                        ai_clip = {**clip, "confirmed_event": content.get("confirmed_event", "")}
                        ai_kit, used_ollama = _generate_editorial_kit(
                            ai_clip, game_name_default, platform, library_doc, settings,
                            channel_name=settings.get("channel_name", ""),
                            brand_handle=settings.get("brand_handle", ""),
                            default_hashtags=settings.get("default_hashtags", ""),
                            content_warning=settings.get("content_warning", ""))
                        _save_kit_content(entry["path"], rank, platform, content, game_name_default, ai_kit)
                    st.session_state[result_notice_key] = "ollama" if used_ollama else "offline"
                    st.rerun()
                with st.form(editor_key):
                    confirmed_label = st.selectbox("Evento confirmado manualmente (opcional)", option_labels,
                                                   index=option_labels.index(selected_event), key=f"event-{editor_key}")
                    game_name = st.text_input("Jogo/campanha (opcional; revise se o padrão não corresponder)",
                                              value=game_name_default, key=f"game-{editor_key}")
                    overlay_title = st.text_input("Texto sobreposto no vídeo e na capa (até 40 caracteres)",
                                                  value=overlay_default, max_chars=40,
                                                  key=f"overlay-{editor_key}")
                    publish_title = st.text_input("Título para YouTube/TikTok", value=publish_title_default,
                                                  max_chars=int(config.PLATFORMS[platform]["title_max_chars"]),
                                                  key=f"publish-title-{editor_key}")
                    caption = st.text_area("Descrição / legenda", value=caption_default, height=150,
                                           key=f"caption-{editor_key}")
                    hashtags = st.text_input("Hashtags", value=hashtags_default, key=f"hashtags-{editor_key}")
                    pinned = st.text_input("Comentário fixado sugerido", value=pinned_default,
                                           key=f"pinned-{editor_key}")
                    save_edits = st.form_submit_button("Salvar edição")
                    render_edited = st.form_submit_button("Salvar e renderizar com este texto")
                if save_edits or render_edited:
                    confirmed_event = event_options[confirmed_label]
                    caption_text = caption.strip()
                    caption_lines = caption_text.splitlines()
                    if caption_lines:
                        caption_lines[0] = publish_title.strip()
                        caption_text = "\n".join(caption_lines)
                    else:
                        caption_text = publish_title.strip()
                    content.update({"game_name": game_name.strip(), "overlay_title": overlay_title.strip(),
                                    "publish_title": publish_title.strip(), "caption": caption_text,
                                    "hashtags": hashtags.strip(), "pinned_comment": pinned.strip(),
                                    "confirmed_event": confirmed_event})
                    local_store.save_clip_content(entry["path"], rank, platform, content)
                    if render_edited:
                        try:
                            with st.status(f"Renderizando o corte #{rank:02d} com o texto escolhido…", expanded=True):
                                _apply_performance(settings)
                            output = _render_clip_for_analysis(entry, clip, platform, settings["captions"],
                                                               settings["draft"], settings["economy_mode"],
                                                               title_override=overlay_title,
                                                               confirmed_event=confirmed_event)
                            st.session_state[preview_key] = rank
                            st.success(f"Render atualizado: {Path(output['video']).name}")
                        except Exception as exc:
                            st.error(f"Falha ao renderizar com o texto escolhido: {exc}")
                            st.stop()
                    else:
                        st.success("Edição salva localmente. O texto sobreposto muda no próximo render; o MP4 existente não é alterado.")
                    st.rerun()
                copy_caption = caption_default.strip()
                copy_lines = copy_caption.splitlines()
                if copy_lines and copy_lines[0].strip() == publish_title_default.strip():
                    copy_caption = "\n".join(copy_lines[1:]).strip()
                st.markdown("**Cópia rápida**")
                copy_cols = st.columns(3)
                with copy_cols[0]:
                    _copy_button("Copiar título", publish_title_default, f"copy-title-{editor_key}")
                with copy_cols[1]:
                    _copy_button("Copiar descrição", copy_caption, f"copy-description-{editor_key}")
                with copy_cols[2]:
                    _copy_button("Copiar hashtags", hashtags_default, f"copy-hashtags-{editor_key}")
            if existing:
                if rank == preview_rank:
                    video = Path(existing["video"])
                    st.video(str(video))
                    st.download_button("Baixar MP4", video.read_bytes(), file_name=video.name,
                                       mime="video/mp4", key=f"clip-download-{entry['path'].stem}-{rank}-{platform}")
                    if existing.get("cover") and Path(existing["cover"]).is_file():
                        st.image(existing["cover"], caption="Capa vertical", width=180)
                else:
                    st.caption("Render pronto · use Selecionar para prévia neste card para assistir/baixar.")
            else:
                if st.button(f"Renderizar e pré-visualizar · #{rank:02d}",
                             key=f"clip-render-{entry['path'].stem}-{rank}-{platform}"):
                    try:
                        with st.status(f"Renderizando corte #{rank:02d}…", expanded=True):
                            _apply_performance(settings)
                            output = _render_clip_for_analysis(entry, clip, platform,
                                                               settings["captions"], settings["draft"],
                                                               settings["economy_mode"])
                        st.session_state[preview_key] = rank
                        st.success(f"Render pronto: {Path(output['video']).name}")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Falha no render: {exc}")
            with st.expander("Publicação rápida · copiar e abrir upload", expanded=False):
                links = st.columns(2)
                links[0].link_button("Abrir YouTube Studio (Upload)",
                                     "https://studio.youtube.com/channel/UC/videos/upload?d=ud",
                                     help="Abre a página de upload; escolha o MP4 e confirme a publicação no YouTube.",
                                     use_container_width=True)
                links[1].link_button("Abrir TikTok Upload",
                                     "https://www.tiktok.com/creator-center/upload",
                                     help="Abre o envio do TikTok; escolha o MP4 e confirme a publicação.",
                                     use_container_width=True)
                fast_description = str(caption_default).strip()
                fast_lines = fast_description.splitlines()
                if fast_lines and fast_lines[0].strip() == str(publish_title_default).strip():
                    fast_description = "\n".join(fast_lines[1:]).strip()
                fast_cols = st.columns(3)
                with fast_cols[0]:
                    _copy_button("Copiar título", publish_title_default, f"fast-title-{digest}-{rank}-{platform}")
                with fast_cols[1]:
                    _copy_button("Copiar descrição", fast_description, f"fast-description-{digest}-{rank}-{platform}")
                with fast_cols[2]:
                    _copy_button("Copiar hashtags", hashtags_default, f"fast-hashtags-{digest}-{rank}-{platform}")
                st.caption("O navegador não permite que uma URL pré-selecione um arquivo local; escolha o MP4 na página de upload.")
    with st.expander("Auditoria da análise e candidatos", expanded=False):
        candidates = payload.get("candidates", [])
        if candidates:
            table = [{"Tipo exibido": TYPE_LABELS.get(safe_event_type(item), item.get("event_type", "")),
                      "Detecção bruta": TYPE_LABELS.get(item.get("event_type", ""), item.get("event_type", "")),
                      "Rank": item.get("rank", "—"), "Score": item.get("score", 0),
                      "Score bruto": item.get("raw_score", 0), "Pico": _short_time(item.get("peak")),
                      "Confiança": item.get("confidence", 0),
                      "Fala": item.get("speech_detector", "—"), "HUD": bool(item.get("hud_detected", False)),
                      "Motivo": item.get("classification_reason", ""), "Descarte": item.get("discard_reason", "")}
                     for item in candidates]
            st.dataframe(table, use_container_width=True, hide_index=True)
        else:
            st.caption("O arquivo de análise não contém uma lista de candidatos.")


def _create_tab(settings: dict[str, Any]) -> None:
    st.header("Criar cortes")
    st.caption("Três passos: escolha a gameplay, confira o perfil e coloque a análise na fila local.")
    st.subheader("1 · Selecione o arquivo")
    with st.container(border=True):
        upload = st.file_uploader("Gameplay (MP4, MKV, MOV, M4V ou AVI)",
                                  type=["mp4", "mkv", "mov", "m4v", "avi"], key="create_upload")
        local_path = st.text_input("Ou use um caminho local — recomendado para arquivos grandes",
                                   placeholder=r"C:\Gameplays\minha_partida.mp4", key="create_local_path")
    st.subheader("2 · Configuração rápida")
    with st.container(border=True):
        col1, col2, col3 = st.columns([1.15, 1.2, 1])
        with col1:
            st.radio("Profundidade da análise", ["quick", "detailed"], key="cfg_analysis_mode",
                     horizontal=True, format_func=lambda value: "Rápida" if value == "quick" else "Detalhada")
            st.checkbox("Escolher automaticamente 5–10 cortes", key="cfg_automatic_count")
        with col2:
            st.selectbox("Cutscenes", list(SCENE_MODES), key="cfg_scene_mode_label",
                         index=list(SCENE_MODES.values()).index(settings["scene_mode"]),
                         format_func=lambda label: label)
        with col3:
            st.selectbox("Plataforma principal", list(config.PLATFORMS), key="cfg_platform",
                         index=list(config.PLATFORMS).index(settings["platform"]),
                         format_func=lambda value: config.PLATFORMS[value]["label"])
        auto_cols = st.columns([1, 1.5])
        with auto_cols[0]:
            st.checkbox("IA local na análise", key="cfg_vision_ai_enabled",
                        help="Visão Ollama quando llava:7b ou llama3.2-vision estiver instalado; senão, ranking textual de áudio e fallback local.")
        with auto_cols[1]:
            st.checkbox("Renderizar MP4 final + capa automaticamente", key="cfg_auto_render",
                        help="Após analisar, gera em série o MP4 1080×1920 e a capa, e salva kit/histórico no SQLite.")
        vision_status = _cached_visual_ollama_status(settings.get("ollama_base_url", "http://localhost:11434"),
                                                      settings.get("vision_model", "llava:7b"))
        if vision_status.get("ok") and vision_status.get("model_installed"):
            st.success(f"Visão Ollama pronta · {vision_status.get('model', settings.get('vision_model', 'llava:7b'))} · análise serial CPU-first")
        elif vision_status.get("ok"):
            st.info(f"Modelo visual {settings.get('vision_model', 'llava:7b')} não instalado. A IA textual/áudio e o motor local continuam funcionando. Para habilitar visão: `ollama pull {settings.get('vision_model', 'llava:7b')}`.")
        else:
            st.info("Ollama offline: fallback automático para áudio/VAD e ranking local; nada é enviado para serviços externos.")
        with st.expander("Mais opções (auditoria e desempenho)", expanded=False):
            if not st.session_state["cfg_automatic_count"]:
                st.slider("Quantidade de cortes", 1, 30, key="cfg_top_n")
            a, b, c = st.columns(3)
            a.slider("Limite de cutscenes no top N (%)", 0, 100, step=5, key="cfg_scene_share_pct")
            b.checkbox("CSV de auditoria", key="cfg_debug")
            c.checkbox("Prévia econômica (540×960)", key="cfg_draft")
            d, e, f = st.columns(3)
            d.checkbox("Legendas locais Whisper tiny", key="cfg_captions")
            e.checkbox("Modo econômico de CPU", key="cfg_economy_mode")
            f.checkbox("Prioridade reduzida", key="cfg_priority_below_normal")
            st.slider("Threads FFmpeg", 1, 8, key="cfg_threads")
    st.subheader("3 · Enfileirar")
    if st.button("🎮 Analisar e gerar cortes", type="primary",
                 disabled=upload is None and not local_path.strip(), key="create_enqueue"):
        try:
            if upload is not None:
                source = _save_upload(upload)
            else:
                source = Path(local_path.strip().strip('"')).expanduser().resolve()
                if source.suffix.lower() not in {".mp4", ".mkv", ".mov", ".m4v", ".avi"}:
                    raise ValueError("O caminho local deve apontar para MP4, MKV, MOV, M4V ou AVI.")
                if not source.is_file():
                    raise FileNotFoundError(f"Arquivo local não encontrado: {source}")
            current = _state_settings()
            current["scene_mode"] = SCENE_MODES[st.session_state["cfg_scene_mode_label"]]
            current["max_scene_share"] = st.session_state["cfg_scene_share_pct"] / 100.0
            current["platform"] = st.session_state["cfg_platform"]
            current["top_n"] = st.session_state.get("cfg_top_n", current["top_n"])
            current["automatic_count"] = st.session_state["cfg_automatic_count"]
            current["vision_ai_enabled"] = st.session_state["cfg_vision_ai_enabled"]
            current["auto_render"] = st.session_state["cfg_auto_render"]
            local_store.save_settings(current)
            options = _queue_options(current)
            job_id = enqueue_analysis(source, options)
            st.success(f"Job {job_id} adicionado à fila: {source.name}")
            st.info("Acompanhe a análise, o render serial e o kit automático na aba **Fila**. O histórico de MP4s fica na aba **Histórico**.")
        except Exception as exc:
            st.error(f"Não foi possível enfileirar: {exc}")
    st.divider()
    st.markdown("#### Análises recentes")
    entries = _available_analyses()[:6]
    if entries:
        for item in entries:
            st.caption(item["label"])
    else:
        st.caption("Nenhuma análise concluída ainda.")


@st.fragment(run_every="2s")
def _queue_live_panel() -> None:
    try:
        queue_service.recover_orphaned_jobs()
        queue_service.ensure_worker()
    except Exception as exc:
        st.warning(f"Worker: {exc}")
    jobs = get_jobs(100)
    pid = queue_service.worker_pid()
    st.caption(f"Worker local: {'ativo · PID ' + str(pid) if pid else 'ocioso'} · atualização automática a cada 2 s")
    if not jobs:
        st.info("A fila está vazia. Adicione uma ou mais gameplays para começar.")
        return
    for job in jobs:
        source_name = Path(job["source_path"]).name
        label = STATUS_LABELS.get(job["status"], job["status"])
        with st.container(border=True):
            a, b = st.columns([4, 1])
            a.markdown(f"**{source_name}** · `{job['id']}` · {label}")
            b.caption(datetime.fromtimestamp(job["created_at"]).strftime("%d/%m %H:%M"))
            if job["status"] in {"processando", "concluido"}:
                st.progress(max(0, min(100, int(job.get("percent", 100 if job["status"] == "concluido" else 0)))),
                            text=f"{job.get('stage', '')} · {job.get('message', '')}"[:180])
            if job["status"] == "concluido" and isinstance(job.get("result"), dict):
                result = job["result"]
                summary = result.get("summary", {}) if isinstance(result.get("summary"), dict) else {}
                engine = summary.get("ai_engine", "heuristic_local")
                engine_label = {"ollama_vision": "Ollama visão", "ollama_text_audio_fallback": "Llama textual/áudio (sem visão)",
                                "heuristic_local": "fallback local"}.get(engine, str(engine))
                st.caption(f"Análise: {engine_label} · {int(summary.get('ai_candidates_assessed', 0) or 0)} janelas IA · "
                           f"{int(summary.get('selected_count', 0) or 0)} cortes selecionados.")
                if result.get("auto_rendered"):
                    ready, failed = int(result.get("rendered_count", 0) or 0), int(result.get("render_failed_count", 0) or 0)
                    st.success(f"Render automático: {ready} MP4(s) prontos; {failed} falha(s). Consulte Resultados ou Histórico.")
                if summary.get("ai_fallback_reason"):
                    st.caption(f"Nota de fallback: {summary['ai_fallback_reason']}")
            elif job.get("error"):
                st.error(job["error"])
            controls = st.columns([1, 1, 4])
            if job["status"] in {"pendente", "processando"}:
                if controls[0].button("Cancelar", key=f"cancel-{job['id']}"):
                    try:
                        queue_service.cancel_job(job["id"])
                        st.toast("Cancelamento solicitado.")
                    except Exception as exc:
                        st.error(f"Não foi possível cancelar: {exc}")
            elif job["status"] in {"falhou", "cancelado"}:
                if controls[0].button("Reenfileirar", key=f"retry-{job['id']}"):
                    try:
                        new_id = queue_service.retry_job(job["id"])
                        st.toast(f"Novo job {new_id} iniciado.")
                    except Exception as exc:
                        st.error(f"Não foi possível reenfileirar: {exc}")
            if job["status"] == "concluido" and job.get("result_path"):
                result_path = Path(job["result_path"])
                if result_path.is_file():
                    controls[1].download_button("JSON", result_path.read_bytes(),
                                                file_name=f"analysis_{job['id']}.json",
                                                mime="application/json", key=f"jobjson-{job['id']}")
                    controls[2].caption(f"Saída: {result_path.parent}")


def _queue_panel() -> None:
    _queue_live_panel()


def _queue_tab(settings: dict[str, Any]) -> None:
    st.header("Fila de processamento")
    st.caption("Um worker local serializa jobs, mantém progresso entre reruns do Streamlit e respeita cancelamento cooperativo.")
    uploads = st.file_uploader("Adicionar vários arquivos por upload", type=["mp4", "mkv", "mov", "m4v", "avi"],
                               accept_multiple_files=True, key="queue_uploads")
    local_paths = st.text_area("Ou caminhos locais (um caminho por linha; ideal para arquivos longos)",
                               placeholder="C:\\Gameplays\\sessao1.mp4\nC:\\Gameplays\\sessao2.mkv",
                               key="queue_local_paths")
    parsed_paths = [line.strip().strip('"') for line in local_paths.splitlines() if line.strip()]
    if st.button("Enfileirar arquivos selecionados", type="primary", disabled=not uploads and not parsed_paths,
                 key="queue_multi_add"):
        added = []
        try:
            current = _state_settings()
            current["scene_mode"] = SCENE_MODES.get(st.session_state.get("cfg_scene_mode_label", "Manter cenas (padrão)"), current["scene_mode"])
            current["max_scene_share"] = float(st.session_state.get("cfg_scene_share_pct", int(current["max_scene_share"] * 100))) / 100.0
            current["platform"] = st.session_state.get("cfg_platform", current["platform"])
            local_store.save_settings(current)
            options = _queue_options(current)
            sources = [_save_upload(upload) for upload in (uploads or [])]
            for raw_path in parsed_paths:
                path = Path(raw_path).expanduser().resolve()
                if path.suffix.lower() not in {".mp4", ".mkv", ".mov", ".m4v", ".avi"}:
                    raise ValueError(f"Formato inválido no caminho: {path.name}")
                if not path.is_file():
                    raise FileNotFoundError(f"Arquivo não encontrado: {path}")
                sources.append(path)
            for path in sources:
                added.append((path.name, enqueue_analysis(path, options)))
            st.success("Arquivos adicionados: " + ", ".join(f"{name} ({job})" for name, job in added))
        except Exception as exc:
            st.error(f"Falha ao adicionar arquivo(s): {exc}")
    _queue_panel()
    jobs = get_jobs(500)
    clear_col, log_col = st.columns(2)
    if clear_col.button("Limpar histórico concluído/falhou/cancelado", key="clear_queue_history"):
        count = queue_service.remove_finished_jobs()
        st.success(f"Removidos {count} registros; os arquivos de saída não foram apagados.")
        st.rerun()
    with log_col.expander("Log do worker"):
        log_path = config.DIRS["logs"] / "queue_worker.log"
        if log_path.is_file():
            st.code("\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]))
        else:
            st.caption("Ainda não há log do worker.")
    st.caption(f"Jobs persistentes registrados: {len(jobs)} · banco local `data/cliprush_local.db`.")


def _results_tab(settings: dict[str, Any]) -> None:
    st.header("Resultados e renders")
    st.caption("Análises atuais e sessões antigas em output/sessoes são descobertas automaticamente.")
    entry = _analysis_selector("Análise", "results_analysis")
    if entry:
        _display_clip_results(entry, settings)
        raw = entry["path"].read_bytes()
        st.download_button("Baixar JSON completo da análise", raw,
                           file_name=f"cliprush_{entry['path'].parent.name}.json",
                           mime="application/json", key="download-analysis-json")
        csv_path = entry["path"].parent / config.ANALYSIS["debug_csv_name"]
        if csv_path.is_file():
            st.download_button("Baixar CSV de auditoria", csv_path.read_bytes(),
                               file_name=f"cliprush_{entry['path'].parent.name}.csv",
                               mime="text/csv", key="download-analysis-csv")


def _kit_tab(settings: dict[str, Any]) -> None:
    st.header("Kit editorial de postagem")
    st.caption("Gere textos com Ollama local quando quiser; se ele não responder, o kit offline continua disponível. Revise os detalhes antes de postar.")
    entry = _analysis_selector("Escolha a análise", "kit_analysis")
    if not entry:
        return
    clips = entry["payload"].get("clips", [])
    if not clips:
        st.warning("A análise selecionada não possui cortes.")
        return
    clip_labels = [f"#{item.get('rank', i + 1):02d} · {TYPE_LABELS.get(safe_event_type(item), item.get('event_type', ''))} · "
                   f"{_format_time(item.get('peak', 0))} · score {item.get('score', 0):.1f}"
                   for i, item in enumerate(clips)]
    selected_label = st.selectbox("Corte base", clip_labels, key="kit_clip")
    clip = clips[clip_labels.index(selected_label)]
    col1, col2 = st.columns(2)
    platform = col1.selectbox("Plataforma", list(config.PLATFORMS),
                              index=list(config.PLATFORMS).index(settings["platform"]),
                              format_func=lambda value: config.PLATFORMS[value]["label"], key="kit_platform")
    content = local_store.get_clip_content(entry["path"], int(clip.get("rank", 0)), platform)
    unique = f"{hashlib.sha256(str(entry['path'].resolve()).encode('utf-8')).hexdigest()[:10]}_{clip.get('rank', 0)}_{platform}"
    game_name = col2.text_input("Jogo / campanha", value=content.get("game_name") or settings["game_name"], key=f"kit-game-{unique}")
    library_doc = load_library()
    kit_clip = {**clip, "confirmed_event": content.get("confirmed_event", "")}
    kit_clip["event_type"] = safe_event_type(kit_clip)
    kit = generate_post_kit(kit_clip, game_name, platform, library_doc,
                            channel_name=settings["channel_name"], brand_handle=settings["brand_handle"],
                            default_hashtags=settings["default_hashtags"],
                            content_warning=settings["content_warning"])
    kit_notice_key = f"ai-notice-kit-{unique}"
    kit_notice = st.session_state.pop(kit_notice_key, "")
    if kit_notice == "ollama":
        st.success(f"Texto atualizado com {settings.get('ollama_model', 'llama3.2:3b')}.")
    elif kit_notice == "offline":
        st.info("Ollama indisponível ou resposta não validada; as sugestões offline foram aplicadas.")
    if st.button("✨ Gerar / atualizar textos com IA local", key=f"ai-kit-{unique}",
                 disabled=not settings.get("ollama_enabled", True), type="primary"):
        with st.spinner("Gerando sugestões em português com o modelo local…"):
            ai_kit, used_ollama = _generate_editorial_kit(
                kit_clip, game_name, platform, library_doc, settings,
                channel_name=settings["channel_name"], brand_handle=settings["brand_handle"],
                default_hashtags=settings["default_hashtags"], content_warning=settings["content_warning"])
            _save_kit_content(entry["path"], int(clip.get("rank", 0)), platform, content, game_name, ai_kit)
        st.session_state[kit_notice_key] = "ollama" if used_ollama else "offline"
        st.rerun()
    st.markdown("#### 3 opções de título")
    title_cols = st.columns(3)
    display_titles = content.get("ai_titles") or kit["titles"]
    for index, title in enumerate(display_titles[:3]):
        with title_cols[index]:
            st.code(title, language=None)
    st.caption(f"Gancho: {kit['hook']}")
    st.caption(f"Texto de reação para a tela: **{kit['reaction_text']}** · neutro para cutscenes")
    asset = None
    assets = local_store.list_rendered_assets(entry["path"], int(clip.get("rank", 0)), platform)
    for candidate in assets:
        if Path(candidate["video_path"]).is_file():
            asset = candidate
            break
    if not asset:
        fallback = _find_existing_render(entry["path"], clip, platform)
        if fallback:
            asset = {"video_path": fallback["video"], "cover_path": fallback.get("cover", "")}
    if asset:
        video = Path(asset["video_path"])
        st.video(str(video))
        st.caption(f"Render selecionado: {video.name}")
        st.download_button("Baixar MP4 para upload", video.read_bytes(), file_name=video.name,
                           mime="video/mp4", key=f"kit-download-{unique}")
        link_cols = st.columns(2)
        link_cols[0].link_button("Abrir YouTube Studio (Upload)",
                                 "https://studio.youtube.com/channel/UC/videos/upload?d=ud",
                                 use_container_width=True)
        link_cols[1].link_button("Abrir TikTok Upload", "https://www.tiktok.com/creator-center/upload",
                                 use_container_width=True)
        st.caption("Para proteger seus arquivos, a página de upload não aceita que o ClipRush preencha o arquivo via link. Escolha o MP4 e confirme manualmente.")
    else:
        st.info("Ainda não há MP4 renderizado para este destino. Você pode gerar o clipe na aba Resultados e voltar aqui.")
    with st.form(f"postkit-save-{unique}"):
        title = st.text_input("Título para publicação", value=content.get("publish_title") or kit["titles"][0],
                              max_chars=int(config.PLATFORMS[platform]["title_max_chars"]), key=f"kit-title-{unique}")
        caption = st.text_area("Legenda / descrição pronta para colar", value=content.get("caption") or kit["caption"], height=210,
                               key=f"kit-caption-{unique}")
        hashtags = st.text_input("Hashtags", value=content.get("hashtags") or kit["hashtags"], key=f"kit-tags-{unique}")
        pinned = st.text_area("Comentário fixado sugerido", value=content.get("pinned_comment") or kit["pinned_comment"], height=70,
                              key=f"kit-pinned-{unique}")
        schedule_now = st.checkbox("Agendar este post agora", key=f"kit-schedule-{unique}")
        sched_date = st.date_input("Data", value=date.today() + timedelta(days=1), key=f"kit-date-{unique}")
        sched_time = st.time_input("Horário", value=dt_time(18, 0), key=f"kit-time-{unique}")
        submitted = st.form_submit_button("Salvar como rascunho / adicionar ao calendário")
    if submitted:
        caption_text = caption.strip()
        lines = caption_text.splitlines()
        if lines:
            lines[0] = title.strip()
            caption_text = "\n".join(lines)
        else:
            caption_text = title.strip()
        local_store.save_clip_content(entry["path"], int(clip.get("rank", 0)), platform,
                                      {**content, "game_name": game_name.strip(),
                                       "publish_title": title.strip(), "caption": caption_text,
                                       "hashtags": hashtags.strip(), "pinned_comment": pinned.strip()})
        scheduled_at = datetime.combine(sched_date, sched_time).isoformat(timespec="minutes") if schedule_now else None
        status = "agendado" if schedule_now else "rascunho"
        post_id = local_store.create_post({
            "source_job_id": entry["payload"].get("session_id"), "clip_rank": clip.get("rank"),
            "platform": platform, "status": status, "scheduled_at": scheduled_at,
            "game_name": game_name, "event_type": kit["event_type"],
            "score": clip.get("score", 0), "title": title.strip(), "caption": caption_text,
            "hashtags": hashtags, "pinned_comment": pinned,
            "video_path": asset["video_path"] if asset else "",
        })
        st.success(f"Post {post_id} salvo como {STATUS_LABELS[status].lower()}.")
    effective_caption = caption.strip()
    effective_lines = effective_caption.splitlines()
    if effective_lines:
        effective_lines[0] = title.strip()
        effective_caption = "\n".join(effective_lines)
    else:
        effective_caption = title.strip()
    description_to_copy = effective_caption
    description_lines = description_to_copy.splitlines()
    if description_lines and description_lines[0].strip() == title.strip():
        description_to_copy = "\n".join(description_lines[1:]).strip()
    st.markdown("**Copiar rapidamente**")
    copy_cols = st.columns(3)
    with copy_cols[0]:
        _copy_button("Copiar título", title, f"kit-copy-title-{unique}")
    with copy_cols[1]:
        _copy_button("Copiar descrição", description_to_copy, f"kit-copy-description-{unique}")
    with copy_cols[2]:
        _copy_button("Copiar hashtags", hashtags, f"kit-copy-hashtags-{unique}")
    with st.expander("Ver texto completo e comentário fixado", expanded=False):
        st.code(effective_caption, language=None)
        st.markdown("**Comentário fixado**")
        st.code(pinned, language=None)


def _history_tab() -> None:
    st.header("Histórico de vídeos e kits")
    st.caption("Cada MP4, capa, título, descrição, hashtags e comentário fica salvo localmente no SQLite e na pasta output/.")
    posts = local_store.list_posts(limit=500)
    if not posts:
        st.info("Ainda não há vídeos no histórico. Enfileire uma gameplay com render automático ativado.")
        return
    assets = local_store.list_rendered_assets(limit=5000)
    assets_by_path = {str(Path(row.get("video_path", "")).resolve()): row for row in assets
                      if row.get("video_path")}
    col1, col2, col3 = st.columns(3)
    col1.metric("Itens no histórico", len(posts))
    col2.metric("Vídeos existentes", sum(Path(post.get("video_path", "")).is_file() for post in posts))
    col3.metric("Rascunhos", sum(post.get("status") == "rascunho" for post in posts))
    for post in posts:
        video = Path(post.get("video_path", "")) if post.get("video_path") else None
        label = (f"{post.get('title') or 'Sem título'} · {config.PLATFORMS.get(post.get('platform'), {}).get('label', post.get('platform'))} "
                 f"· corte #{int(post.get('clip_rank') or 0):02d} · {post.get('status', 'rascunho')}")
        with st.expander(label, expanded=False):
            if video and video.is_file():
                asset = assets_by_path.get(str(video.resolve()), {})
                st.video(str(video))
                if asset.get("cover_path") and Path(asset["cover_path"]).is_file():
                    st.image(asset["cover_path"], caption="Capa gerada", width=180)
                st.caption(f"Arquivo: `{video}` · {video.stat().st_size / (1024 * 1024):.1f} MB")
                if video.stat().st_size <= 250 * 1024 * 1024:
                    st.download_button("Baixar MP4", video.read_bytes(), file_name=video.name,
                                       mime="video/mp4", key=f"history-download-{post['id']}")
            else:
                st.warning("O caminho do MP4 não está disponível neste dispositivo.")
            st.caption(f"Score {float(post.get('score', 0) or 0):.1f}/100 · evento seguro: {post.get('event_type', 'impacto')} · jogo: {post.get('game_name') or 'não informado'}")
            copy_caption = str(post.get("caption", ""))
            title = str(post.get("title", ""))
            description = copy_caption
            lines = description.splitlines()
            if lines and lines[0].strip() == title.strip():
                description = "\n".join(lines[1:]).strip()
            st.markdown("**Cópia rápida**")
            cols = st.columns(3)
            with cols[0]:
                _copy_button("Copiar título", title, f"history-title-{post['id']}")
            with cols[1]:
                _copy_button("Copiar descrição", description, f"history-description-{post['id']}")
            with cols[2]:
                _copy_button("Copiar hashtags", str(post.get("hashtags", "")), f"history-hashtags-{post['id']}")
            links = st.columns(2)
            links[0].link_button("Abrir YouTube Studio (Upload)",
                                 "https://studio.youtube.com/channel/UC/videos/upload?d=ud",
                                 use_container_width=True)
            links[1].link_button("Abrir TikTok Upload", "https://www.tiktok.com/creator-center/upload",
                                 use_container_width=True)
            st.caption("Por segurança do navegador, o site de upload não recebe automaticamente o arquivo local; selecione o MP4 acima ou na página de envio.")


def _calendar_tab() -> None:
    st.header("Calendário editorial")
    st.caption("Planeje rascunhos, roteiros, posts agendados e publicados; os dados ficam no banco local.")
    selected_month = st.date_input("Mês de referência", value=date.today().replace(day=1), key="calendar_month")
    posts = local_store.list_posts()
    days: dict[date, list[dict[str, Any]]] = {}
    for post in posts:
        scheduled = post.get("scheduled_at")
        if not scheduled:
            continue
        try:
            day = datetime.fromisoformat(scheduled).date()
        except ValueError:
            continue
        if day.year == selected_month.year and day.month == selected_month.month:
            days.setdefault(day, []).append(post)
    headings = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
    cols = st.columns(7)
    for col, heading in zip(cols, headings):
        col.markdown(f"**{heading}**")
    for week in calendar.monthcalendar(selected_month.year, selected_month.month):
        cols = st.columns(7)
        for col, day_num in zip(cols, week):
            if not day_num:
                col.caption(" ")
                continue
            current_day = date(selected_month.year, selected_month.month, day_num)
            is_today = current_day == date.today()
            col.markdown(f"**{day_num}{' · hoje' if is_today else ''}**")
            day_posts = days.get(current_day, [])
            for post in day_posts[:3]:
                time_part = post["scheduled_at"][11:16] if len(post["scheduled_at"]) > 15 else ""
                col.caption(f"{time_part} {post['platform'].upper()} · {STATUS_LABELS.get(post['status'], post['status'])}")
                col.caption(str(post["title"])[:44])
            if len(day_posts) > 3:
                col.caption(f"+{len(day_posts) - 3} posts")
    st.divider()
    st.subheader("Editar / agendar um post")
    if not posts:
        st.info("Crie um kit na aba correspondente; ele será salvo como rascunho e aparecerá aqui.")
        return
    post_labels = [f"{post['title'] or '(sem título)'} · {post['platform']} · {STATUS_LABELS.get(post['status'], post['status'])} · {post['id']}"
                   for post in posts]
    selected_label = st.selectbox("Post", post_labels, key="calendar_post_selector")
    post = posts[post_labels.index(selected_label)]
    schedule = None
    try:
        schedule = datetime.fromisoformat(post["scheduled_at"]) if post.get("scheduled_at") else None
    except ValueError:
        schedule = None
    with st.form(f"calendar-edit-{post['id']}"):
        title = st.text_input("Título", value=post["title"])
        status = st.selectbox("Status", local_store.POST_STATUSES,
                              index=local_store.POST_STATUSES.index(post["status"]) if post["status"] in local_store.POST_STATUSES else 0,
                              format_func=lambda value: STATUS_LABELS.get(value, value))
        caption = st.text_area("Legenda", value=post["caption"], height=150)
        hashtags = st.text_input("Hashtags", value=post["hashtags"])
        pinned = st.text_input("Comentário fixado", value=post["pinned_comment"])
        scheduled_date = st.date_input("Data agendada", value=schedule.date() if schedule else date.today())
        scheduled_time = st.time_input("Horário", value=schedule.time() if schedule else dt_time(18, 0))
        notes = st.text_area("Notas editoriais", value=post["notes"], height=70)
        update = st.form_submit_button("Salvar alterações")
    if update:
        scheduled_at = datetime.combine(scheduled_date, scheduled_time).isoformat(timespec="minutes") if status in {"agendado", "publicado"} else None
        local_store.update_post(post["id"], {"title": title, "status": status, "scheduled_at": scheduled_at,
                                              "caption": caption, "hashtags": hashtags,
                                              "pinned_comment": pinned, "notes": notes})
        st.success("Post atualizado no calendário.")
        st.rerun()
    if st.button("Excluir post e seus registros de métricas/checklist", key=f"delete-post-{post['id']}"):
        local_store.delete_post(post["id"])
        st.success("Post excluído.")
        st.rerun()


def _metrics_tab() -> None:
    st.header("Métricas de publicação")
    st.caption("Registro manual local. Nenhuma API de TikTok/YouTube é acessada nem publicada automaticamente.")
    posts = local_store.list_posts()
    if posts:
        labels = [f"{post['title'] or '(sem título)'} · {post['platform']} · {post['id']}" for post in posts]
        with st.form("metrics_entry"):
            selected = st.selectbox("Post", labels)
            post = posts[labels.index(selected)]
            measured = st.date_input("Data da coleta", value=date.today())
            a, b, c = st.columns(3)
            views = a.number_input("Visualizações", min_value=0, step=100, value=0)
            likes = b.number_input("Curtidas", min_value=0, step=10, value=0)
            comments = c.number_input("Comentários", min_value=0, step=1, value=0)
            d, e, f = st.columns(3)
            shares = d.number_input("Compartilhamentos", min_value=0, step=1, value=0)
            saves = e.number_input("Salvamentos", min_value=0, step=1, value=0)
            followers = f.number_input("Seguidores ganhos", min_value=0, step=1, value=0)
            g, h, i = st.columns(3)
            watch_minutes = g.number_input("Tempo assistido (min)", min_value=0.0, step=1.0, value=0.0)
            retention = h.number_input("Retenção média (%)", min_value=0.0, max_value=100.0, step=1.0, value=0.0)
            notes = i.text_input("Observações")
            save_metrics = st.form_submit_button("Salvar snapshot de métricas")
        if save_metrics:
            local_store.add_metrics(post["id"], {
                "measured_at": measured.isoformat(), "views": views, "likes": likes,
                "comments": comments, "shares": shares, "saves": saves,
                "followers_gained": followers, "watch_minutes": watch_minutes,
                "retention_pct": retention if retention > 0 else None, "notes": notes,
            })
            st.success("Snapshot salvo localmente.")
            st.rerun()
    else:
        st.info("Salve um kit editorial como rascunho para começar a registrar métricas por post.")
    metrics = local_store.list_metrics()
    latest_by_post: dict[str, dict[str, Any]] = {}
    for item in metrics:
        latest_by_post[item["post_id"]] = item
    total_views = sum(item["views"] for item in latest_by_post.values())
    interactions = sum(item["likes"] + item["comments"] + item["shares"] + item["saves"]
                       for item in latest_by_post.values())
    er = interactions * 100.0 / total_views if total_views else 0.0
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Posts com snapshot", len(latest_by_post))
    col2.metric("Visualizações (último snapshot/post)", f"{total_views:,}".replace(",", "."))
    col3.metric("Interações no último snapshot", f"{interactions:,}".replace(",", "."))
    col4.metric("Engajamento por view", f"{er:.2f}%")
    if metrics:
        post_by_id = {post["id"]: post for post in posts}
        records = []
        for post_id, item in latest_by_post.items():
            post = post_by_id.get(post_id, {})
            inter = item["likes"] + item["comments"] + item["shares"] + item["saves"]
            records.append({"Post": post.get("title", post_id), "Plataforma": post.get("platform", ""),
                            "Data": item["measured_at"], "Views": item["views"], "Likes": item["likes"],
                            "Comentários": item["comments"], "Compartilhamentos": item["shares"],
                            "Salvamentos": item["saves"], "Engajamento (%)": round(inter * 100 / item["views"], 2) if item["views"] else 0,
                            "Retenção (%)": item["retention_pct"]})
        df = pd.DataFrame(records).sort_values("Views", ascending=False)
        st.subheader("Desempenho por post — último snapshot de cada um")
        st.dataframe(df, use_container_width=True, hide_index=True)
        if not df.empty:
            chart_df = df.set_index("Post")[["Views", "Likes", "Comentários", "Compartilhamentos"]]
            st.bar_chart(chart_df)
        all_rows = []
        for item in metrics:
            post = post_by_id.get(item["post_id"], {})
            all_rows.append({**item, "title": post.get("title", ""), "platform": post.get("platform", "")})
        export = pd.DataFrame(all_rows).to_csv(index=False).encode("utf-8-sig")
        st.download_button("Baixar histórico CSV", export, file_name="cliprush_metricas.csv", mime="text/csv")
        st.caption("Snapshots repetidos não são somados: os cartões e o ranking usam o registro mais recente de cada post.")


def _checklist_tab() -> None:
    st.header("Checklist de publicação")
    st.caption("Checklist persistente por post para revisão de edição, áudio, direitos, spoilers e publicação.")
    posts = local_store.list_posts()
    if not posts:
        st.info("Crie um rascunho no Kit editorial para iniciar o checklist.")
        return
    labels = [f"{post['title'] or '(sem título)'} · {post['platform']} · {post['id']}" for post in posts]
    selected = st.selectbox("Post para revisar", labels, key="checklist_post_select")
    post = posts[labels.index(selected)]
    existing = local_store.ensure_checklist(post["id"], CHECKLIST_ITEMS)
    state = {item["item_key"]: item for item in existing}
    checked_count = sum(bool(item["checked"]) for item in existing)
    st.progress(checked_count / len(CHECKLIST_ITEMS), text=f"{checked_count}/{len(CHECKLIST_ITEMS)} itens concluídos")
    with st.form(f"checklist-form-{post['id']}"):
        cols = st.columns(2)
        values = {}
        for index, (key, label) in enumerate(CHECKLIST_ITEMS):
            item = state.get(key, {})
            checked = cols[index % 2].checkbox(label, value=bool(item.get("checked")), key=f"chk-{post['id']}-{key}")
            values[key] = {"checked": checked, "note": str(item.get("note", ""))}
        notes = st.text_area("Notas da revisão", value=post.get("notes", ""), height=100)
        submitted = st.form_submit_button("Salvar checklist")
    if submitted:
        local_store.save_checklist(post["id"], values)
        local_store.update_post(post["id"], {"notes": notes})
        st.success("Checklist salvo.")
        st.rerun()
    missing_video = post.get("video_path") and not Path(post["video_path"]).is_file()
    if missing_video:
        st.warning("O caminho do MP4 associado ao post não existe mais. Reassocie ou renderize novamente antes de publicar.")


def _library_tab() -> None:
    st.header("Biblioteca editorial JSON")
    document = load_library()
    summary = validate_library_document(document)
    col1, col2, col3 = st.columns(3)
    col1.metric("Itens validados", summary["item_count"])
    col2.metric("Categorias", len(summary["category_counts"]))
    col3.metric("Mínimo da biblioteca", "200 itens")
    st.caption(f"Arquivo: `{LIBRARY_PATH}` · contagem calculada pelo validador, não digitada manualmente.")
    st.json(summary["category_counts"])
    st.download_button("Baixar biblioteca JSON", json.dumps(document, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="biblioteca_editorial.json", mime="application/json")
    editor_rows = []
    for item in document["items"]:
        editor_rows.append({
            "id": item["id"], "category": item["category"], "text": item["text"],
            "active": bool(item["active"]), "platforms": ",".join(item.get("platforms", [])),
            "tags": ",".join(item.get("tags", [])),
        })
    revision = str(int(float(document.get("updated_at", 0))))
    edited = st.data_editor(
        pd.DataFrame(editor_rows), num_rows="dynamic", use_container_width=True, height=540,
        key=f"library-editor-{revision}",
        column_config={
            "category": st.column_config.SelectboxColumn("Categoria", options=sorted(set(document["category_counts"]) | {"customizados"}), required=True),
            "text": st.column_config.TextColumn("Texto / template", width="large", required=True),
            "active": st.column_config.CheckboxColumn("Ativo", default=True),
            "platforms": st.column_config.TextColumn("Plataformas (vírgula)", width="medium"),
            "tags": st.column_config.TextColumn("Tags (vírgula)", width="small"),
        },
    )
    if st.button("Validar e salvar alterações da biblioteca", type="primary", key="save-library"):
        records = edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
        items = []
        for row in records:
            def cell_text(name: str, default: str = "") -> str:
                value = row.get(name, default)
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return default
                return str(value).strip()
            if not cell_text("text"):
                continue
            raw_platforms = cell_text("platforms", "shorts,reels,tiktok") or "shorts,reels,tiktok"
            raw_tags = cell_text("tags")
            active_value = row.get("active", True)
            if active_value is None or (isinstance(active_value, float) and pd.isna(active_value)):
                active_value = True
            items.append({
                "id": cell_text("id"),
                "category": cell_text("category", "customizados") or "customizados",
                "text": cell_text("text"),
                "active": bool(active_value),
                "platforms": [value.strip() for value in raw_platforms.split(",") if value.strip()],
                "tags": [value.strip() for value in raw_tags.split(",") if value.strip()],
            })
        try:
            saved = save_library(items)
            st.success(f"Biblioteca salva e validada: {saved['item_count']} itens.")
            st.rerun()
        except Exception as exc:
            st.error(f"Não foi possível salvar. A validação exige pelo menos 200 itens e os mínimos por categoria. Detalhes: {exc}")
    st.markdown("#### Regras do validador")
    st.caption("IDs e textos duplicados são rejeitados; plataformas são verificadas; mínimo global 200 e mínimos individuais por categoria. Rode `python scripts/validate_library.py` para validar fora do painel.")


def _settings_tab(settings: dict[str, Any]) -> None:
    st.header("Configurações persistentes")
    st.caption("Preferências em `data/settings.json`; banco editorial em `data/cliprush_local.db`. Ambos ficam no computador e não são incluídos em ZIPs de distribuição.")
    with st.form("persistent-settings"):
        left, right = st.columns(2)
        with left:
            analysis_mode = st.selectbox("Análise padrão", ["quick", "detailed"],
                                         index=0 if settings["analysis_mode"] == "quick" else 1,
                                         format_func=lambda value: "Rápida" if value == "quick" else "Detalhada",
                                         key="prefs_analysis_mode")
            automatic = st.checkbox("Quantidade automática (5–10)", settings["automatic_count"], key="prefs_automatic_count")
            top_n = st.slider("Quantidade avançada", 1, 30, settings["top_n"], key="prefs_top_n")
            scene_label = next((label for label, value in SCENE_MODES.items() if value == settings["scene_mode"]), list(SCENE_MODES)[0])
            selected_scene_label = st.selectbox("Modo cutscenes", list(SCENE_MODES), index=list(SCENE_MODES).index(scene_label), key="prefs_scene_label")
            share = st.slider("Limite máximo de cenas (%)", 0, 100, int(settings["max_scene_share"] * 100), 5, key="prefs_max_scene_share")
            debug = st.checkbox("Gerar CSV de auditoria", settings["debug"], key="prefs_debug")
        with right:
            platform = st.selectbox("Plataforma padrão", list(config.PLATFORMS),
                                    index=list(config.PLATFORMS).index(settings["platform"]),
                                    format_func=lambda value: config.PLATFORMS[value]["label"], key="prefs_platform")
            captions = st.checkbox("Legendas locais por padrão", settings["captions"], key="prefs_captions")
            draft = st.checkbox("Prévia econômica por padrão", settings["draft"], key="prefs_draft")
            economy = st.checkbox("Perfil econômico CPU", settings["economy_mode"], key="prefs_economy_mode")
            threads = st.slider("Threads FFmpeg", 1, 8, settings["threads"], key="prefs_threads")
            low_priority = st.checkbox("Prioridade reduzida", settings["priority_below_normal"], key="prefs_priority_below_normal")
        game = st.text_input("Jogo/campanha padrão", settings["game_name"], key="prefs_game_name")
        brand, handle = st.columns(2)
        channel = brand.text_input("Nome do canal", settings["channel_name"], key="prefs_channel_name")
        brand_handle = handle.text_input("@handle (opcional)", settings["brand_handle"], key="prefs_brand_handle")
        hashtags = st.text_input("Hashtags extras padrão", settings["default_hashtags"], key="prefs_default_hashtags")
        warning = st.text_input("Aviso padrão de conteúdo (opcional)", settings["content_warning"], key="prefs_content_warning")
        save = st.form_submit_button("Salvar e aplicar agora", type="primary")
    if save:
        new_settings = {
            "analysis_mode": analysis_mode, "automatic_count": automatic, "top_n": top_n,
            "scene_mode": SCENE_MODES[selected_scene_label], "max_scene_share": share / 100.0,
            "debug": debug, "platform": platform, "captions": captions, "draft": draft,
            "economy_mode": economy, "threads": threads, "priority_below_normal": low_priority,
            "game_name": game, "channel_name": channel, "brand_handle": brand_handle,
            "default_hashtags": hashtags, "content_warning": warning,
        }
        local_store.save_settings(new_settings)
        st.session_state["_apply_saved_settings"] = True
        st.success("Preferências salvas localmente.")
        st.rerun()
    st.divider()
    st.subheader("IA local · Ollama")
    ai_status = _cached_ollama_status(settings.get("ollama_base_url", "http://localhost:11434"),
                                      settings.get("ollama_model", "llama3.2:3b"))
    vision_status = _cached_visual_ollama_status(settings.get("ollama_base_url", "http://localhost:11434"),
                                                  settings.get("vision_model", "llava:7b"))
    if ai_status.get("ok") and ai_status.get("model_installed"):
        st.success(f"Ollama conectado · modelo {settings.get('ollama_model', 'llama3.2:3b')} instalado")
    elif ai_status.get("ok"):
        st.info("Ollama está ativo, mas o modelo selecionado não foi encontrado; os textos continuarão no modo offline.")
    else:
        st.info("Ollama indisponível no momento. O kit editorial usa automaticamente o gerador local offline, sem interromper o app.")
    if vision_status.get("ok") and vision_status.get("model_installed"):
        st.success(f"Ollama visão · {vision_status.get('model', settings.get('vision_model', 'llava:7b'))} disponível para keyframes e re-ranking multimodal.")
    elif vision_status.get("ok"):
        st.info(f"O modelo visual não está instalado. Para habilitar a análise por imagem, execute no computador: `ollama pull {settings.get('vision_model', 'llava:7b')}`. O modelo textual atual não interpreta imagens.")
    st.caption("A análise envia apenas miniaturas reduzidas e métricas temporais para o Ollama local (localhost). Os modelos trabalham em série, até 18 janelas por gameplay; sem vision, o app usa ranking Llama textual sobre áudio/VAD/movimento e mantém o fallback heurístico.")
    with st.form("ollama-settings"):
        ai_enabled = st.checkbox("Ativar assistência editorial Ollama", settings.get("ollama_enabled", True),
                                 key="prefs_ollama_enabled")
        vision_enabled = st.checkbox("Usar IA local no ranking (visão quando disponível)",
                                     settings.get("vision_ai_enabled", True), key="prefs_vision_ai_enabled")
        auto_render = st.checkbox("Gerar MP4 final e capa automaticamente após análise",
                                  settings.get("auto_render", True), key="prefs_auto_render")
        ai_left, ai_right = st.columns([1.3, 1])
        with ai_left:
            ollama_url = st.text_input("Endereço local", settings.get("ollama_base_url", "http://localhost:11434"),
                                       key="prefs_ollama_base_url", help="Padrão: http://localhost:11434")
        with ai_right:
            ollama_model = st.text_input("Modelo", settings.get("ollama_model", "llama3.2:3b"),
                                         key="prefs_ollama_model")
        ollama_timeout = st.slider("Tempo máximo para resposta (segundos)", 5, 90,
                                   int(settings.get("ollama_timeout", 25)), 5,
                                   key="prefs_ollama_timeout")
        vision_cols = st.columns([1.3, 1])
        with vision_cols[0]:
            vision_model = st.text_input("Modelo visual Ollama", settings.get("vision_model", "llava:7b"),
                                         key="prefs_vision_model", help="Perfis sugeridos: llava:7b (4,7 GB) ou llama3.2-vision:11b (7,8 GB).")
        with vision_cols[1]:
            max_vision_candidates = st.slider("Janelas visuais por análise", 6, 18,
                                               int(settings.get("max_vision_candidates", 18)), 3,
                                               key="prefs_max_vision_candidates")
        vision_timeout = st.slider("Tempo limite por lote visual (s)", 15, 300,
                                   int(settings.get("vision_timeout", 90)), 15,
                                   key="prefs_vision_timeout")
        save_ollama = st.form_submit_button("Salvar IA local")
    if save_ollama:
        if not ollama_url.strip() or not ollama_model.strip():
            st.warning("Informe o endereço e o modelo, ou desative a assistência editorial.")
        else:
            new_settings = local_store.save_settings({**settings, "ollama_enabled": ai_enabled,
                "ollama_base_url": ollama_url.strip(), "ollama_model": ollama_model.strip(),
                "ollama_timeout": ollama_timeout, "vision_ai_enabled": vision_enabled,
                "auto_render": auto_render, "vision_model": vision_model.strip(),
                "max_vision_candidates": max_vision_candidates, "vision_timeout": vision_timeout})
            for key, value in new_settings.items():
                st.session_state[f"cfg_{key}"] = value
                st.session_state[f"prefs_{key}"] = value
            _cached_ollama_status.clear()
            st.success("Preferências da IA local salvas.")
            st.rerun()
    st.code(f"ollama pull {settings.get('vision_model', 'llava:7b')}", language="powershell")
    st.caption("O download do modelo visual pode ocupar vários GB; ele não é iniciado automaticamente pelo app. O processamento é CPU-first e serial para reduzir pico de RAM.")
    if st.button("Atualizar status dos modelos Ollama", key="refresh-ollama-status"):
        _cached_ollama_status.clear()
        st.rerun()
    st.divider()
    st.subheader("Backup local")
    if st.button("Preparar backup de preferências, banco e biblioteca", key="prepare-backup"):
        backup = _make_backup()
        st.download_button("Baixar backup ZIP", backup, file_name=f"cliprush_backup_{date.today().isoformat()}.zip",
                           mime="application/zip", key="download-backup")
    st.caption("Backups e métricas ficam no dispositivo; para uma cópia de segurança periódica, baixe o ZIP e guarde-o fora da pasta do app.")


def _make_backup() -> bytes:
    local_store.init_storage()
    with local_store.connect() as db:
        db.execute("PRAGMA wal_checkpoint(FULL)")
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (local_store.DB_PATH, local_store.SETTINGS_PATH, LIBRARY_PATH):
            if path.is_file():
                archive.write(path, arcname=f"cliprush_backup/{path.name}")
    memory.seek(0)
    return memory.read()


def main() -> None:
    st.set_page_config(page_title="ClipRush Gamer AI v3", page_icon="🎮", layout="wide")
    ensure_directories()
    local_store.init_storage()
    settings = local_store.load_settings()
    _set_state_defaults(settings)
    settings = _state_settings()
    settings["scene_mode"] = SCENE_MODES.get(st.session_state.get("cfg_scene_mode_label", ""), settings["scene_mode"])
    settings["max_scene_share"] = float(st.session_state.get("cfg_scene_share_pct", int(settings["max_scene_share"] * 100))) / 100.0
    settings["platform"] = st.session_state.get("cfg_platform", settings["platform"])
    _apply_performance(settings)
    _apply_visual_theme()

    st.title("🎮 ClipRush Gamer AI")
    st.caption("Workspace local para gameplay PS4/PC · sem webcam · fila CPU-first e arquivos privados no seu computador")
    with st.sidebar:
        st.markdown("### ClipRush · local")
        st.caption("CPU multi-core · fila serial · preferências e histórico persistentes")
        st.markdown("#### Status dos serviços")
        ffmpeg_ok = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
        st.caption(f"{'🟢' if ffmpeg_ok else '🔴'} FFmpeg / ffprobe {'prontos' if ffmpeg_ok else 'não encontrados'}")
        ollama_status = _cached_ollama_status(settings.get("ollama_base_url", "http://localhost:11434"),
                                              settings.get("ollama_model", "llama3.2:3b"))
        if ollama_status.get("ok") and ollama_status.get("model_installed"):
            st.caption(f"🟢 Ollama · {settings.get('ollama_model', 'llama3.2:3b')}")
        elif ollama_status.get("ok"):
            st.caption("🟠 Ollama ativo · modelo escolhido não encontrado (fallback offline)")
        else:
            st.caption("⚪ Ollama offline · fallback editorial automático")
        vision_status = _cached_visual_ollama_status(settings.get("ollama_base_url", "http://localhost:11434"),
                                                      settings.get("vision_model", "llava:7b"))
        if vision_status.get("ok") and vision_status.get("model_installed"):
            st.caption(f"🟣 IA visual · {vision_status.get('model', settings.get('vision_model', 'llava:7b'))} pronta")
        elif vision_status.get("ok"):
            st.caption(f"🟠 IA visual indisponível · instale {settings.get('vision_model', 'llava:7b')}")
        else:
            st.caption("⚪ IA visual offline · fallback áudio/texto")
        try:
            free_gb = shutil.disk_usage(config.ROOT).free / (1024 ** 3)
            st.caption(f"💾 Espaço livre · {free_gb:.1f} GB")
        except OSError:
            st.caption("💾 Espaço em disco indisponível")
        with st.expander("Ajustes rápidos", expanded=False):
            with st.form("sidebar-quick-settings"):
                quick_platform = st.selectbox("Plataforma", list(config.PLATFORMS),
                    index=list(config.PLATFORMS).index(settings["platform"]),
                    format_func=lambda value: config.PLATFORMS[value]["label"], key="sidebar-platform")
                quick_mode = st.selectbox("Profundidade", ["quick", "detailed"],
                    index=0 if settings["analysis_mode"] == "quick" else 1,
                    format_func=lambda value: "Rápida" if value == "quick" else "Detalhada", key="sidebar-mode")
                quick_scene_label = next((label for label, mode in SCENE_MODES.items()
                                          if mode == settings["scene_mode"]), list(SCENE_MODES)[0])
                quick_scene = st.selectbox("Cutscenes", list(SCENE_MODES),
                    index=list(SCENE_MODES).index(quick_scene_label), key="sidebar-scene")
                apply_quick = st.form_submit_button("Aplicar ajustes")
            if apply_quick:
                saved = local_store.save_settings({**local_store.load_settings(),
                    "platform": quick_platform, "analysis_mode": quick_mode,
                    "scene_mode": SCENE_MODES[quick_scene]})
                for key, value in saved.items():
                    st.session_state[f"cfg_{key}"] = value
                    st.session_state[f"prefs_{key}"] = value
                st.session_state["_apply_saved_settings"] = True
                st.rerun()
        if st.button("Recarregar preferências salvas", key="reload-settings"):
            st.session_state["_apply_saved_settings"] = True
            st.rerun()
        st.caption(f"Biblioteca editorial: {LIBRARY_PATH.name}")
        st.caption("Somente miniaturas reduzidas/métricas são enviadas ao Ollama em localhost; o vídeo e o áudio bruto permanecem locais.")

    tabs = st.tabs(["🎬 Criar", "📥 Fila", "🎞️ Resultados", "🗂️ Histórico", "📝 Kit", "🗓️ Calendário",
                    "📊 Métricas", "☑️ Checklist", "📚 Biblioteca", "⚙️ Configurações"])
    with tabs[0]:
        _create_tab(settings)
    with tabs[1]:
        _queue_tab(settings)
    with tabs[2]:
        _results_tab(settings)
    with tabs[3]:
        _history_tab()
    with tabs[4]:
        _kit_tab(settings)
    with tabs[5]:
        _calendar_tab()
    with tabs[6]:
        _metrics_tab()
    with tabs[7]:
        _checklist_tab()
    with tabs[8]:
        _library_tab()
    with tabs[9]:
        _settings_tab(settings)


if __name__ == "__main__":
    main()
