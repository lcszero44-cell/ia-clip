from __future__ import annotations

import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import config
from common import atomic_write_json, read_json

LIBRARY_PATH = config.DIRS["library"] / "biblioteca_editorial.json"
MINIMUM_TOTAL = 200
MINIMUM_BY_CATEGORY = {
    "ganchos": 25,
    "titulos_acao": 35,
    "titulos_cena": 15,
    "descricoes": 25,
    "ctas": 20,
    "comentarios_fixados": 15,
    "reacoes_acao": 15,
    "reacoes_cena": 10,
    "conjuntos_hashtags": 15,
    "roteiros_narracao": 15,
}
ALLOWED_FIELDS = {"id", "category", "text", "active", "platforms", "tags"}


def _slug(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_") or "item"


def build_library_document(catalog: dict[str, list[str]]) -> dict[str, Any]:
    items = []
    counts = Counter()
    for category, texts in catalog.items():
        for index, text in enumerate(texts, start=1):
            items.append({
                "id": f"{_slug(category)}_{index:03d}",
                "category": category,
                "text": str(text).strip(),
                "active": True,
                "platforms": ["shorts", "reels", "tiktok"],
                "tags": [],
            })
            counts[category] += 1
    document = {
        "schema_version": 1,
        "language": "pt-BR",
        "description": "Biblioteca editorial local ClipRush Gamer AI; editável e sem serviços externos.",
        "minimum_count": MINIMUM_TOTAL,
        "item_count": len(items),
        "category_counts": dict(sorted(counts.items())),
        "updated_at": time.time(),
        "items": items,
    }
    validate_library_document(document)
    return document


def validate_library_document(document: Any, enforce_minimum: bool = True) -> dict[str, Any]:
    if not isinstance(document, dict) or not isinstance(document.get("items"), list):
        raise ValueError("A biblioteca deve ser um objeto JSON com uma lista `items`.")
    items = document["items"]
    ids: set[str] = set()
    texts: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"O item {index} não é um objeto JSON.")
        missing = {"id", "category", "text", "active"} - set(item)
        if missing:
            raise ValueError(f"O item {index} não contém os campos: {', '.join(sorted(missing))}.")
        item_id = str(item["id"]).strip()
        category = str(item["category"]).strip()
        text = str(item["text"]).strip()
        if not item_id or not category or len(text) < 3:
            raise ValueError(f"O item {index} precisa de ID, categoria e texto com ao menos 3 caracteres.")
        if item_id in ids:
            raise ValueError(f"ID duplicado na biblioteca: {item_id}.")
        ids.add(item_id)
        normalized_text = " ".join(text.casefold().split())
        pair = (category, normalized_text)
        if pair in texts:
            raise ValueError(f"Texto duplicado na categoria {category}: {text[:60]}.")
        texts.add(pair)
        if not isinstance(item.get("active"), bool):
            raise ValueError(f"`active` deve ser booleano no item {item_id}.")
        platforms = item.get("platforms", [])
        if not isinstance(platforms, list) or any(value not in config.PLATFORMS for value in platforms):
            raise ValueError(f"Plataformas inválidas no item {item_id}.")
        tags = item.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(value, str) for value in tags):
            raise ValueError(f"`tags` precisa ser uma lista de textos no item {item_id}.")
        counts[category] += 1
    if enforce_minimum:
        if len(items) < MINIMUM_TOTAL:
            raise ValueError(f"Biblioteca incompleta: {len(items)} itens; mínimo validado {MINIMUM_TOTAL}.")
        deficits = {category: minimum - counts[category]
                    for category, minimum in MINIMUM_BY_CATEGORY.items()
                    if counts[category] < minimum}
        if deficits:
            details = ", ".join(f"{category}: faltam {missing}" for category, missing in deficits.items())
            raise ValueError("Categorias abaixo do mínimo: " + details)
    return {"item_count": len(items), "category_counts": dict(sorted(counts.items()))}


def load_library() -> dict[str, Any]:
    if not LIBRARY_PATH.is_file():
        raise FileNotFoundError(f"Biblioteca não encontrada: {LIBRARY_PATH}. Rode scripts/seed_library.py.")
    document = read_json(LIBRARY_PATH)
    summary = validate_library_document(document)
    document["item_count"] = summary["item_count"]
    document["category_counts"] = summary["category_counts"]
    return document


def save_library(items: list[dict[str, Any]]) -> dict[str, Any]:
    LIBRARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    used_ids: set[str] = set()
    for index, row in enumerate(items, start=1):
        item = {key: row[key] for key in ALLOWED_FIELDS if key in row}
        category = str(item.get("category", "customizados")).strip() or "customizados"
        text = str(item.get("text", "")).strip()
        identifier = str(item.get("id", "")).strip() or f"{_slug(category)}_{int(time.time())}_{index:03d}"
        if identifier in used_ids:
            identifier = f"{identifier}_{index}"
        used_ids.add(identifier)
        normalized.append({
            "id": identifier, "category": category, "text": text,
            "active": bool(item.get("active", True)),
            "platforms": list(item.get("platforms", ["shorts", "reels", "tiktok"])),
            "tags": [str(tag) for tag in item.get("tags", []) if str(tag).strip()],
        })
    summary = validate_library_document({"items": normalized})
    document = {
        "schema_version": 1, "language": "pt-BR",
        "description": "Biblioteca editorial local ClipRush Gamer AI; editável e sem serviços externos.",
        "minimum_count": MINIMUM_TOTAL, "item_count": summary["item_count"],
        "category_counts": summary["category_counts"], "updated_at": time.time(),
        "items": normalized,
    }
    atomic_write_json(LIBRARY_PATH, document)
    return document
