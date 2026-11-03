"""Integração opcional e local com Ollama para texto editorial.

O módulo não conhece arquivos de mídia e nunca envia caminhos locais. A API do
Ollama só é acessada quando o endereço informado é HTTP e aponta para localhost
ou para um endereço de loopback.
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2:3b"

# Os limites mantêm a resposta pequena para modelos locais e para uso em posts.
_MAX_METADATA_TEXT = {
    "jogo": 120,
    "plataforma": 60,
    "evento": 80,
    "cena_fala": 280,
    "resumo": 500,
}
_MAX_OUTPUT_TEXT = {
    "hook": 180,
    "description": 500,
    "cta": 120,
    "hashtags": 240,
    "pinned_comment": 240,
}
_MAX_TITLE = 80

# Somente estes aliases são aceitos; caminhos e classificação bruta não fazem
# parte do contrato de dados enviado ao modelo.
_METADATA_ALIASES = {
    "jogo": ("jogo", "game", "game_name"),
    "plataforma": ("plataforma", "platform"),
    "evento": ("evento_seguro", "evento", "event", "safe_event"),
    "score": ("score", "pontuacao", "pontuação"),
    "duracao": ("duracao", "duração", "duration", "duration_seconds"),
    "cena_fala": (
        "cena_fala",
        "cena/fala",
        "scene_speech",
        "scene",
        "fala",
        "speech",
    ),
    "resumo": ("resumo", "resumo_textual", "summary"),
}

_EDITORIAL_SYSTEM_PROMPT = """
Você é um editor de conteúdo de jogos. Responda exclusivamente em PT-BR e
produza somente metadados editoriais curtos para publicação. Nunca invente
jogadas, eventos, falas, resultados ou fatos que não estejam nos metadados
fornecidos. Não transforme uma inferência em fato.

Crie três títulos claramente diferentes e fortes, com curiosidade, tensão,
consequência ou uma pergunta de loop aberto que faça a pessoa querer assistir.
Use palavras concretas que correspondam à evidência fornecida; evite frases
genéricas como “momento insano”, “olha isso” ou “gameplay incrível” quando não
acrescentarem contexto. Seja chamativo sem prometer um evento que não foi visto.
O hook deve capturar atenção nos primeiros segundos, sem clickbait falso.

Não altere score, duração, classificação, ranking, tipo de evento ou reação de
cena: esses dados pertencem ao analisador e não são uma tarefa sua. Não faça
análise do vídeo, não peça caminhos de arquivo e não escreva explicações fora
do JSON. Gere exatamente as chaves titles, hook, description, cta, hashtags e
pinned_comment; titles deve ter exatamente três itens curtos não vazios.
""".strip()


def _result(ok: bool, model_installed: bool, models: list[str], message: str) -> dict[str, Any]:
    """Monta o formato estável retornado por ``check_status``."""

    return {
        "ok": bool(ok),
        "model_installed": bool(model_installed),
        "models": models,
        "message": message,
    }


def _validated_base_url(base_url: str) -> str | None:
    """Aceita somente HTTP local sem credenciais, consulta ou fragmento."""

    if not isinstance(base_url, str):
        return None
    candidate = base_url.strip()
    try:
        parsed = urllib.parse.urlsplit(candidate)
        if parsed.scheme.lower() != "http" or not parsed.hostname:
            return None
        if parsed.username is not None or parsed.password is not None:
            return None
        if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            return None
        # Acessar .port também valida portas malformadas sem fazer DNS.
        _ = parsed.port
        host = parsed.hostname.lower().rstrip(".")
        is_localhost = host == "localhost"
        try:
            is_loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            is_loopback = False
        if not (is_localhost or is_loopback):
            return None
    except (TypeError, ValueError):
        return None
    return candidate.rstrip("/")


def _valid_model(model: str) -> str | None:
    """Valida o identificador do modelo sem adicionar credenciais ou dados."""

    if not isinstance(model, str):
        return None
    value = model.strip()
    if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
        return None
    return value


def _request_json(request: urllib.request.Request, timeout: float) -> Any | None:
    """Executa uma requisição JSON e converte qualquer falha em ``None``."""

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            return None
        return json.loads(raw)
    except Exception:
        # A integração é opcional: indisponibilidade local não pode derrubar o app.
        return None


def check_status(
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 1.0,
) -> dict:
    """Consulta ``/api/tags`` sem lançar exceções.

    O campo ``ok`` indica que o endpoint respondeu com JSON contendo uma lista
    de modelos; ``model_installed`` é independente e pode ser falso quando o
    Ollama está disponível, mas ainda não possui o modelo solicitado.
    """

    try:
        url = _validated_base_url(base_url)
        valid_model = _valid_model(model)
        if url is None:
            return _result(False, False, [], "URL rejeitada: use um endereço HTTP local.")
        if valid_model is None:
            return _result(False, False, [], "Modelo inválido.")

        request = urllib.request.Request(
            url + "/api/tags",
            headers={"Accept": "application/json"},
            method="GET",
        )
        payload = _request_json(request, timeout)
        if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
            return _result(False, False, [], "Resposta inválida de /api/tags.")

        models: list[str] = []
        for item in payload["models"]:
            if isinstance(item, dict) and isinstance(item.get("name"), str):
                name = item["name"].strip()
                if name and name not in models:
                    models.append(name)
        installed = valid_model in models
        if installed:
            message = "Ollama disponível e modelo instalado."
        else:
            message = f"Ollama disponível, mas o modelo {valid_model!r} não está instalado."
        return _result(True, installed, models, message)
    except Exception:
        # Inclui mocks ou objetos de resposta atípicos; status nunca deve lançar.
        return _result(False, False, [], "Não foi possível consultar o Ollama local.")


def _looks_like_local_path(value: str) -> bool:
    """Identifica caminhos locais para impedir vazamento em texto permitido."""

    if value.startswith(("/", "\\")) or value.lower().startswith("file://"):
        return True
    if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("//"):
        return True
    # Também bloqueia um caminho absoluto embutido em um resumo ou fala.
    return bool(
        re.search(
            r"(?i)(?:^|[\s(])(?:/home/|/tmp/|/var/|/mnt/|/users/|[a-z]:[\\/]|\\\\)",
            value,
        )
    )


def _safe_text(value: Any, limit: int) -> str | None:
    """Normaliza texto curto e rejeita conteúdo que pareça caminho local."""

    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text or len(text) > limit or _looks_like_local_path(text):
        return None
    if any(ord(char) < 32 for char in text):
        return None
    return text


def _safe_number(value: Any, *, minimum: float, maximum: float) -> int | float | None:
    """Aceita apenas números finitos dentro do domínio editorial esperado."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        return None
    if isinstance(value, int) or numeric.is_integer():
        return int(numeric)
    return numeric


def _safe_metadata(context: Any) -> dict[str, Any] | None:
    """Extrai somente os sete campos de metadados permitidos pelo contrato."""

    if not isinstance(context, dict):
        return None
    metadata: dict[str, Any] = {}

    for field in ("jogo", "plataforma", "evento", "cena_fala", "resumo"):
        for alias in _METADATA_ALIASES[field]:
            if alias in context:
                cleaned = _safe_text(context[alias], _MAX_METADATA_TEXT[field])
                if cleaned is not None:
                    metadata[field] = cleaned
                break

    if "score" in context or any(alias in context for alias in _METADATA_ALIASES["score"]):
        value = next(
            (context[alias] for alias in _METADATA_ALIASES["score"] if alias in context),
            None,
        )
        cleaned_score = _safe_number(value, minimum=0, maximum=100)
        if cleaned_score is not None:
            metadata["score"] = cleaned_score

    if "duracao" in context or any(alias in context for alias in _METADATA_ALIASES["duracao"]):
        value = next(
            (context[alias] for alias in _METADATA_ALIASES["duracao"] if alias in context),
            None,
        )
        cleaned_duration = _safe_number(value, minimum=0, maximum=24 * 60 * 60)
        if cleaned_duration is not None:
            metadata["duracao"] = cleaned_duration

    return metadata or None


def _valid_output_text(value: Any, limit: int) -> str | None:
    """Valida uma string editorial não vazia e dentro do limite declarado."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > limit:
        return None
    if any(ord(char) < 32 and char not in "\n\t" for char in text):
        return None
    return text


def _validate_editorial_output(value: Any) -> dict[str, Any] | None:
    """Garante o objeto exato que pode sair da integração para o app."""

    required = {"titles", "hook", "description", "cta", "hashtags", "pinned_comment"}
    if not isinstance(value, dict) or set(value) != required:
        return None
    titles = value.get("titles")
    if not isinstance(titles, list) or len(titles) != 3:
        return None

    clean_titles: list[str] = []
    for title in titles:
        clean_title = _valid_output_text(title, _MAX_TITLE)
        if clean_title is None:
            return None
        clean_titles.append(clean_title)

    clean: dict[str, Any] = {"titles": clean_titles}
    for field, limit in _MAX_OUTPUT_TEXT.items():
        text = _valid_output_text(value.get(field), limit)
        if text is None:
            return None
        clean[field] = text
    return clean


def generate_editorial_copy(
    context: Any,
    base_url: str = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    timeout: float = 25.0,
) -> dict | None:
    """Gera copy curta via ``/api/chat`` ou retorna ``None`` em qualquer falha."""

    try:
        url = _validated_base_url(base_url)
        valid_model = _valid_model(model)
        metadata = _safe_metadata(context)
        if url is None or valid_model is None or metadata is None:
            return None

        metadata_text = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        user_prompt = (
            "Use somente estes metadados permitidos, sem acrescentar fatos: "
            + metadata_text
            + "\nGere apenas os campos editoriais pedidos pelo sistema."
        )
        body = {
            "model": valid_model,
            "stream": False,
            "format": "json",
            "keep_alive": "2m",
            "options": {
                "num_ctx": 4096,
                "num_gpu": 0,
                "num_thread": 4,
                "temperature": 0.35,
                "num_predict": 220,
            },
            "messages": [
                {"role": "system", "content": _EDITORIAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url + "/api/chat",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response = _request_json(request, timeout)
        if not isinstance(response, dict):
            return None
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            return None
        generated = json.loads(message["content"])
        return _validate_editorial_output(generated)
    except Exception:
        # Conexão, timeout, modelo inexistente e JSON malformado são opcionais.
        return None


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "check_status",
    "generate_editorial_copy",
]
