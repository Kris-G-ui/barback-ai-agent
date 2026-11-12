"""LLM Integration Module.

Provider-agnostic client for Anthropic, OpenAI and Google Gemini, using plain
HTTP (``requests``) so there are no SDK version headaches on deploy.

Features
--------
* One ``chat(system, messages)`` call regardless of provider.
* Timeouts on every request.
* Automatic retry with exponential backoff + jitter for rate limits (429)
  and transient server errors (5xx / connection problems).
* Honors ``Retry-After`` headers when the provider sends them.
* Typed errors so the UI can show a helpful message instead of a traceback.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import requests


class LLMError(Exception):
    """Base error for anything that went wrong talking to the model."""


class LLMAuthError(LLMError):
    """Bad / missing API key."""


class LLMRateLimitError(LLMError):
    """Still rate limited after all retries."""


class LLMResponseError(LLMError):
    """Provider returned something we could not use."""


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_s: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    retries: int = 0


DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}

KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


@dataclass
class LLMClient:
    provider: str = "anthropic"
    model: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1500
    timeout_s: float = 45.0
    max_retries: int = 4
    base_backoff_s: float = 1.0
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self.provider = self.provider.lower()
        if self.provider not in DEFAULT_MODELS:
            raise LLMError(f"Unknown provider '{self.provider}'. Use one of {list(DEFAULT_MODELS)}")
        self.model = self.model or DEFAULT_MODELS[self.provider]
        self.api_key = (self.api_key or os.getenv(KEY_ENV[self.provider]) or "").strip().strip('"').strip("'")
        if not self.api_key:
            raise LLMAuthError(
                f"No API key for {self.provider}. Set {KEY_ENV[self.provider]} "
                "in your environment or Streamlit secrets."
            )

    # ------------------------------------------------------------------ public
    def chat(self, system: str, messages: list[dict]) -> LLMResponse:
        """Send a conversation and return the assistant's text.

        ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
        """
        build = getattr(self, f"_build_{self.provider}")
        parse = getattr(self, f"_parse_{self.provider}")
        url, headers, payload = build(system, messages)

        attempt = 0
        start = time.time()
        while True:
            try:
                resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout_s)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt >= self.max_retries:
                    raise LLMError(f"Network error talking to {self.provider}: {exc}") from exc
                self._sleep(attempt, None)
                attempt += 1
                continue

            if resp.status_code in (401, 403):
                raise LLMAuthError(f"{self.provider} rejected the API key ({resp.status_code}).")
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt >= self.max_retries:
                    if resp.status_code == 429:
                        raise LLMRateLimitError(
                            f"{self.provider} rate limit hit; gave up after {attempt} retries."
                        )
                    raise LLMError(f"{self.provider} server error {resp.status_code}: {resp.text[:300]}")
                self._sleep(attempt, resp.headers.get("retry-after"))
                attempt += 1
                continue
            if resp.status_code == 400 and ("API_KEY_INVALID" in resp.text or "API key not valid" in resp.text):
                raise LLMAuthError(
                    f"{self.provider} says the API key is not valid. Check {KEY_ENV[self.provider]} "
                    "in your Streamlit secrets (full key, no extra spaces)."
                )
            if resp.status_code >= 400:
                raise LLMResponseError(f"{self.provider} error {resp.status_code}: {resp.text[:500]}")

            try:
                data = resp.json()
            except ValueError as exc:
                raise LLMResponseError("Provider returned non-JSON response.") from exc
            text, tin, tout = parse(data)
            if not text:
                raise LLMResponseError("Model returned an empty response.")
            return LLMResponse(
                text=text,
                provider=self.provider,
                model=self.model,
                latency_s=round(time.time() - start, 2),
                input_tokens=tin,
                output_tokens=tout,
                retries=attempt,
            )

    # --------------------------------------------------------------- helpers
    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                time.sleep(min(float(retry_after), 30))
                return
            except ValueError:
                pass
        delay = self.base_backoff_s * (2**attempt) + random.uniform(0, 0.5)
        time.sleep(min(delay, 30))

    # Anthropic -------------------------------------------------------------
    def _build_anthropic(self, system, messages):
        return (
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            {
                "model": self.model,
                "system": system,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

    @staticmethod
    def _parse_anthropic(data):
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        usage = data.get("usage", {})
        return "".join(parts).strip(), usage.get("input_tokens"), usage.get("output_tokens")

    # OpenAI ----------------------------------------------------------------
    def _build_openai(self, system, messages):
        return (
            "https://api.openai.com/v1/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, *messages],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            },
        )

    @staticmethod
    def _parse_openai(data):
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMResponseError(f"Unexpected OpenAI response shape: {data}") from exc
        usage = data.get("usage", {})
        return text.strip(), usage.get("prompt_tokens"), usage.get("completion_tokens")

    # Gemini ----------------------------------------------------------------
    def _build_gemini(self, system, messages):
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
            for m in messages
        ]
        return (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_tokens,
                    "responseMimeType": "application/json",
                },
            },
        )

    @staticmethod
    def _parse_gemini(data):
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError) as exc:
            reason = data.get("promptFeedback", {}).get("blockReason")
            raise LLMResponseError(f"Gemini returned no content (block reason: {reason}).") from exc
        usage = data.get("usageMetadata", {})
        text = "".join(p.get("text", "") for p in parts).strip()
        return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
