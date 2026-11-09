"""Thin REST adapters for Anthropic, OpenAI and Google Gemini.

Bring your own key: the key is passed in per call and is never written to
disk, logged or put in a URL. Plain `requests` is used instead of the vendor
SDKs to keep the install small. An `http` object can be injected for tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import requests

TIMEOUT = 120

PROVIDERS = {
    "anthropic": "Anthropic (Claude)",
    "openai": "OpenAI",
    "gemini": "Google (Gemini)",
}


class ProviderError(Exception):
    """A readable error with the API key already removed."""


@dataclass
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


def _scrub(message: str, key: str) -> str:
    if key:
        message = message.replace(key, "[key hidden]")
    return re.sub(r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{10,})", "[key hidden]", message)


def _check(resp, key: str) -> dict:
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400:
        detail = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                detail = err.get("message", "")
            elif isinstance(err, str):
                detail = err
        hint = {
            401: "The key was rejected. Check that it is correct and active.",
            403: "The key is valid but not allowed to do this.",
            404: "That model was not found for this key.",
            429: "Rate limit or quota reached. Wait a moment or check billing.",
        }.get(resp.status_code, "")
        raise ProviderError(_scrub(f"HTTP {resp.status_code}. {hint} {detail}".strip(), key))
    return data


# ---------- model listing ----------

_NOT_CHAT = re.compile(
    r"(embed|whisper|tts|dall-e|moderation|image|audio|realtime|transcribe|"
    r"search|similarity|davinci|babbage|computer-use|codex-mini-latest)",
    re.IGNORECASE,
)


def list_models(provider: str, key: str, http=requests) -> list[str]:
    """Return model ids this key can use for text generation."""
    if provider == "anthropic":
        r = http.get(
            "https://api.anthropic.com/v1/models",
            params={"limit": 100},
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
            timeout=TIMEOUT,
        )
        return [m["id"] for m in _check(r, key).get("data", [])]
    if provider == "openai":
        r = http.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=TIMEOUT,
        )
        ids = [m["id"] for m in _check(r, key).get("data", [])]
        return sorted(i for i in ids if not _NOT_CHAT.search(i))
    if provider == "gemini":
        r = http.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params={"pageSize": 200},
            headers={"x-goog-api-key": key},
            timeout=TIMEOUT,
        )
        out = []
        for m in _check(r, key).get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append(m["name"].removeprefix("models/"))
        return out
    raise ProviderError(f"Unknown provider: {provider}")


# ---------- generation ----------

def complete(provider: str, key: str, model: str, system: str, user: str, http=requests) -> Completion:
    """Send one system + user message and return the text reply."""
    if not key:
        raise ProviderError("Enter an API key first.")
    if not model:
        raise ProviderError("Choose a model first.")

    if provider == "anthropic":
        r = http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 4096,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=TIMEOUT,
        )
        d = _check(r, key)
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return Completion(text, u.get("input_tokens"), u.get("output_tokens"))

    if provider == "openai":
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        r = http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json={"model": model, "messages": messages},
            timeout=TIMEOUT,
        )
        d = _check(r, key)
        choices = d.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        u = d.get("usage", {})
        return Completion(text, u.get("prompt_tokens"), u.get("completion_tokens"))

    if provider == "gemini":
        body: dict = {"contents": [{"role": "user", "parts": [{"text": user}]}]}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = http.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "content-type": "application/json"},
            json=body,
            timeout=TIMEOUT,
        )
        d = _check(r, key)
        cands = d.get("candidates") or []
        parts = ((cands[0].get("content") or {}).get("parts") or []) if cands else []
        text = "".join(p.get("text", "") for p in parts)
        if not text and d.get("promptFeedback", {}).get("blockReason"):
            raise ProviderError("The request was blocked: " + d["promptFeedback"]["blockReason"])
        u = d.get("usageMetadata", {})
        return Completion(text, u.get("promptTokenCount"), u.get("candidatesTokenCount"))

    raise ProviderError(f"Unknown provider: {provider}")
