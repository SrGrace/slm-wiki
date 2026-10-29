from __future__ import annotations

import json
import re
import ssl
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self

import httpx
import truststore

from slm_wiki.config import Settings


class LLMError(RuntimeError):
    pass


def _strip_fences(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    return m.group(1).strip() if m else text


def _message_text(msg: dict[str, Any]) -> str:
    text = (msg.get("content") or "").strip()
    if not text:
        text = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
    return text


def short_model_id(mid: str) -> str:
    if "/" in mid or mid.endswith(".gguf"):
        return Path(mid).name
    return mid


_DEPLOY_SUFFIXES = ("-azure", "-bedrock", "-ptu", "-payg")


def _unsuffixed(mid: str) -> str:
    for s in _DEPLOY_SUFFIXES:
        if mid.endswith(s):
            return mid[: -len(s)]
    return mid


def _version(mid: str) -> tuple[int, ...]:
    m = re.search(r"\d+(?:\.\d+)*", _unsuffixed(mid))
    return tuple(int(p) for p in m.group(0).split(".")) if m else (0,)


def _family(mid: str) -> str:
    """Blank the first version so claude-opus-4.5 and claude-opus-5.5 are one family."""
    return re.sub(r"\d+(?:\.\d+)*", "x", _unsuffixed(mid).lower(), count=1)


def _deploy_rank(mid: str) -> int:
    for i, s in enumerate(_DEPLOY_SUFFIXES, start=1):
        if mid.endswith(s):
            return i
    return 0


def latest_chat_models(rows: list[dict]) -> list[dict[str, str]]:
    """Latest available chat model per family.

    Drops deprecated/unavailable, aliases, embeddings, image models, and the Azure/Bedrock/PTU
    copy when the plain id exists. ponytail: 'latest' is max version number in the id, not a release date.
    """
    best: dict[str, tuple[tuple, str, str]] = {}
    for m in rows:
        mid = str(m.get("id") or "")
        if (
            not mid
            or m.get("is_alias")
            or m.get("model_status") != "available"
            or m.get("model_type") != "chat_completion"
            or "image" in mid.lower()
        ):
            continue
        rank = (_version(mid), -_deploy_rank(mid))
        cur = best.get(_family(mid))
        if cur is None or rank > cur[0]:
            best[_family(mid)] = (rank, mid, str(m.get("name") or mid))
    return sorted(({"id": mid, "name": name} for _, mid, name in best.values()), key=lambda x: x["name"].lower())


def fetch_mga_models(settings: Settings) -> list[dict[str, str]]:
    if not settings.mga_api_key:
        return []
    with httpx.Client(
        base_url=settings.mga_base_url.rstrip("/"),
        verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
        timeout=20,
        headers={"Authorization": f"Bearer {settings.mga_api_key}"},
    ) as client:
        rows = client.get("/models").json().get("data") or []
    return latest_chat_models(rows)


class LLMClient:
    """Thin OpenAI-compatible client: local llama.cpp server or MGA."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.effective_base_url
        self.model = settings.effective_model
        self._client = httpx.Client(
            base_url=self.base_url.rstrip("/"),
            verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),  # macOS Keychain CAs: corporate TLS proxy for MGA
            timeout=settings.llm_timeout_s,
            headers={"Authorization": f"Bearer {settings.effective_api_key}"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def health(self) -> dict[str, Any]:
        try:
            r = self._client.get("/models")
            r.raise_for_status()
            rows = r.json().get("data") or []
            if self.settings.llm_provider == "mga":
                choices = latest_chat_models(rows)
                models = [c["id"] for c in choices]
                extra: dict[str, Any] = {"mga_models": choices}
            else:
                models = [short_model_id(str(m.get("id") or "")) for m in rows]
                extra = {}
            return {"ok": True, "models": models, "base_url": self.base_url, "model": self.model, **extra}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e), "base_url": self.base_url, "model": self.model}

    def _base_body(self, system: str, user: str, *, stream: bool = False) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": stream,
        }

    def chat(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int | None = None) -> str:
        sys = system
        if json_mode:
            sys = system + "\n\nReply with a single JSON object only. No markdown fences."
        body = self._base_body(sys, user, stream=False)
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            r = self._post_chat(body)
            if r.status_code >= 400:
                body.pop("response_format", None)
                body.pop("chat_template_kwargs", None)
                r = self._post_chat(body)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request failed: {e}") from e
        data = r.json()
        try:
            msg = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"Unexpected LLM response: {data!r}") from e
        text = _message_text(msg)
        if not text:
            raise LLMError(f"Empty model content: {data!r}")
        return text

    def chat_stream(self, system: str, user: str) -> Iterator[str]:
        body = self._base_body(system, user, stream=True)
        try:
            with self._client.stream("POST", "/chat/completions", json=body) as r:
                if r.status_code >= 400:
                    body.pop("chat_template_kwargs", None)
                    with self._client.stream("POST", "/chat/completions", json=body) as r2:
                        r2.raise_for_status()
                        yield from self._iter_sse(r2)
                    return
                r.raise_for_status()
                yield from self._iter_sse(r)
        except httpx.HTTPError as e:
            raise LLMError(f"LLM stream failed: {e}") from e

    def _iter_sse(self, r: httpx.Response) -> Iterator[str]:
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("data: "):
                payload = line[6:].strip()
            elif line.startswith("data:"):
                payload = line[5:].strip()
            else:
                continue
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                continue
            try:
                delta = data["choices"][0].get("delta") or {}
            except (KeyError, IndexError, TypeError):
                continue
            piece = delta.get("content") or delta.get("reasoning_content") or ""
            if piece:
                yield piece

    def _post_chat(self, body: dict[str, Any]) -> httpx.Response:
        return self._client.post("/chat/completions", json=body)

    def chat_json(self, system: str, user: str, *, max_tokens: int | None = None) -> dict[str, Any]:
        try:
            return _parse_json(self.chat(system, user, json_mode=True, max_tokens=max_tokens))
        except LLMError:
            # One retry with double room: big models occasionally emit a stray quote or run out of tokens.
            return _parse_json(self.chat(system, user, json_mode=True, max_tokens=max_tokens and max_tokens * 2))


def _parse_json(raw: str) -> dict[str, Any]:
    text = _strip_fences(raw)
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        try:
            out = json.loads(m.group(0) if m else text)
        except json.JSONDecodeError as e:
            raise LLMError(f"Model did not return valid JSON ({e}): {raw[:300]}") from e
    if not isinstance(out, dict):
        raise LLMError(f"Expected JSON object, got {type(out).__name__}")
    return out
