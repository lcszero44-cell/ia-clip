from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

import config

EVENT_LABELS = {
    "cena": "cena da história", "escopeta": "jogada de escopeta", "headshot": "headshot",
    "parry_faca": "parry", "motosserra": "encontro com a motosserra", "explosao": "explosão",
    "boss_combate": "luta contra o chefe", "susto_grito": "susto", "pistola_rifle": "tiro certeiro",
    "combo": "combo", "impacto": "momento intenso", "morno": "momento da gameplay",
}

SPECIFIC_EVENTS = {
    "escopeta", "headshot", "parry_faca", "motosserra", "explosao",
    "boss_combate", "susto_grito", "pistola_rifle",
}


def safe_event_type(clip: dict[str, Any]) -> str:
    """Retorna evento seguro para texto; inferências fracas viram rótulo neutro."""
    detected = str(clip.get("event_type", "impacto"))
    confirmed = str(clip.get("confirmed_event") or clip.get("user_confirmed_event") or "").strip()
    if confirmed in config.EVENT_WEIGHTS:
        return confirmed
    if detected not in SPECIFIC_EVENTS:
        return detected
    confidence = float(clip.get("confidence", 0.0) or 0.0)
    reason = str(clip.get("classification_reason", "")).strip().lower()
    trusted_reason = reason.startswith("assinatura log-mel:") or "confirmação visual" in reason
    if confidence >= float(config.ANALYSIS.get("specific_event_min_confidence", 0.82)) and trusted_reason:
        return detected
    return "impacto"


def _items(library: dict[str, Any], category: str, platform: str) -> list[dict[str, Any]]:
    return [item for item in library.get("items", [])
            if item.get("active", True) and item.get("category") == category
            and (not item.get("platforms") or platform in item["platforms"])]


def _format(text: str, values: dict[str, str]) -> str:
    return text.format_map(defaultdict(str, values)).strip()


def _pick(items: list[dict[str, Any]], seed: int, count: int = 1) -> list[str]:
    if not items:
        return []
    start = seed % len(items)
    result = []
    for step in range(len(items)):
        value = str(items[(start + step * 7) % len(items)].get("text", "")).strip()
        if value and value not in result:
            result.append(value)
        if len(result) >= count:
            break
    return result


def generate_post_kit(clip: dict[str, Any], game_name: str, platform: str,
                      library: dict[str, Any], channel_name: str = "",
                      brand_handle: str = "", default_hashtags: str = "",
                      content_warning: str = "") -> dict[str, Any]:
    if platform not in config.PLATFORMS:
        raise ValueError("Plataforma de postagem inválida.")
    detected_event_type = str(clip.get("event_type", "impacto"))
    event_type = safe_event_type(clip)
    is_scene = event_type == "cena"
    event = EVENT_LABELS.get(event_type, "momento da gameplay")
    seed_material = f"{clip.get('peak', 0)}|{event_type}|{game_name}|{platform}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:10], 16)
    values = {"game": game_name.strip() or "este jogo", "evento": event,
              "channel": channel_name.strip(), "marca": brand_handle.strip()}

    title_category = "titulos_cena" if is_scene else "titulos_acao"
    titles = [_format(text, values) for text in _pick(_items(library, title_category, platform), seed, 3)]
    if not titles:
        titles = (["A cena que mudou o rumo da história", "Você percebeu esse detalhe?", "Um momento importante da campanha"]
                  if is_scene else ["A jogada que salvou a run", "O combate mais tenso da partida", "Como eu escapei por um fio"])
    hooks = [_format(text, values) for text in _pick(_items(library, "ganchos", platform), seed // 3, 1)]
    descriptions = [_format(text, values) for text in _pick(_items(library, "descricoes", platform), seed // 5, 1)]
    ctas = [_format(text, values) for text in _pick(_items(library, "ctas", platform), seed // 11, 1)]
    comments = [_format(text, values) for text in _pick(_items(library, "comentarios_fixados", platform), seed // 13, 1)]
    reactions = _pick(_items(library, "reacoes_cena" if is_scene else "reacoes_acao", platform), seed // 17, 1)
    hashtag_sets = _pick(_items(library, "conjuntos_hashtags", platform), seed // 19, 1)

    game_lower = game_name.strip().lower()
    is_resident_evil = "resident evil" in game_lower or "residentevil" in game_lower
    is_re4 = is_resident_evil and ("4" in game_lower or "re4" in game_lower)
    if descriptions and "survival horror" in descriptions[0].lower() and not is_resident_evil:
        descriptions = []

    tags = list(config.PLATFORMS[platform]["hashtags"])
    if hashtag_sets:
        library_tags = re.findall(r"#[\w]+", hashtag_sets[0], flags=re.UNICODE)
        for tag in library_tags:
            normalized = tag.lower()
            if normalized in {"#re4", "#re4remake", "#residentevil4"} and not is_re4:
                continue
            if normalized == "#residentevil" and not is_resident_evil:
                continue
            if normalized == "#survivalhorror" and not (is_resident_evil or "survival horror" in game_lower):
                continue
            tags.append(tag)
    tags.extend(re.findall(r"#[\w]+", default_hashtags or "", flags=re.UNICODE))
    unique_tags = []
    for tag in tags:
        normalized = tag.lower()
        if normalized not in {value.lower() for value in unique_tags}:
            unique_tags.append(tag)
    hashtag_text = " ".join(unique_tags[:12])
    reaction = reactions[0] if reactions else ("VOCÊ VIU ISSO?" if is_scene else "QUE MOMENTO!")
    hook = hooks[0] if hooks else ("Presta atenção nesta cena." if is_scene else "Espera até ver esta jogada.")
    description = descriptions[0] if descriptions else f"Um trecho de {values['game']}, sem webcam e com a tela do jogo preservada."
    cta = ctas[0] if ctas else "O que você faria nesta situação? Comenta aqui."
    pinned = comments[0] if comments else "Qual detalhe você notou primeiro?"
    warning = content_warning.strip()
    caption = "\n\n".join(part for part in [titles[0], hook, description, cta, hashtag_text,
                                                 f"Aviso: {warning}" if warning else ""] if part)
    return {
        "titles": titles[:3], "hook": hook, "description": description, "cta": cta,
        "hashtags": hashtag_text, "caption": caption, "pinned_comment": pinned,
        "reaction_text": reaction, "content_warning": warning,
        "platform": platform, "event_type": event_type,
        "detected_event_type": detected_event_type,
        "event_confirmed": bool(clip.get("confirmed_event") or clip.get("user_confirmed_event")),
    }
