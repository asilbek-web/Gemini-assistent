"""Free-tier Google Gemini generator (generativelanguage.googleapis.com).

Same interface as ClaudeCLI.generate(user_prompt, system_prompt), so the
assistant service works unchanged. Uses the free API key from AI Studio
(aistudio.google.com/apikey); no subscription required.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

import aiohttp

from .errors import (
    ClaudeAuthError,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
)

log = logging.getLogger(__name__)

REDACTED = "[REDACTED]"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiGenerator:
    """One REST call per reply; the key never appears in argv or logs."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = DEFAULT_MODEL,
        timeout: int = 120,
        auth_label: Optional[str] = None,
        transport: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._api_key = api_key
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self._auth_label = auth_label
        # Tests inject a fake transport; production uses aiohttp.
        self._transport = transport

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"GeminiGenerator(model={self.model!r}, api_key={'set' if self._api_key else 'unset'})"

    @property
    def auth_mode(self) -> str:
        if self._auth_label:
            return self._auth_label
        return "Gemini API key"

    def _redact(self, text: str) -> str:
        if self._api_key and self._api_key in text:
            return text.replace(self._api_key, REDACTED)
        return text

    def _build_payload(self, user_prompt: str, system_prompt: str) -> Dict[str, Any]:
        return {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7},
        }

    async def _post(self, url: str, payload: Dict[str, Any]):
        if self._transport is not None:
            return await self._transport(url, payload, self._headers(), self.timeout)
        headers = self._headers()
        client_timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(url, json=payload, headers=headers) as response:
                body = await response.text()
                return response.status, body

    def _headers(self) -> Dict[str, str]:
        return {"x-goog-api-key": self._api_key}

    async def generate(self, message_text: str, system_prompt: str) -> str:
        """Send one message; return the model's reply text."""
        url = f"{API_ROOT}/{self.model}:generateContent"
        payload = self._build_payload(message_text, system_prompt)
        try:
            status, body = await self._post(url, payload)
        except ClaudeError:
            raise
        except Exception as exc:  # network / timeout at the HTTP layer
            raise ClaudeTimeout(
                f"Gemini request failed or timed out after {self.timeout}s: {exc}"
            ) from None

        body = self._redact(body or "")
        if status == 429:
            raise ClaudeLimitError(body[:500])
        if status in (400, 401, 403):
            if "api key" in body.lower() or status in (401, 403):
                raise ClaudeAuthError(
                    "Gemini API key rejected or expired: create a new key at "
                    "aistudio.google.com/apikey and update GEMINI_API_KEY. "
                    f"Provider said: {body[:300]} ({REDACTED})",
                    detail=body[:300],
                )
            raise ClaudeError(body[:500])
        if status >= 500:
            raise ClaudeError(f"Gemini server error {status}: {body[:300]}")
        if status != 200:
            raise ClaudeError(f"Gemini error {status}: {body[:300]}")

        try:
            data = json.loads(body)
        except ValueError:
            raise ClaudeError(f"Gemini returned non-JSON: {body[:300]}") from None

        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback") or {}
            reason = feedback.get("blockReason") or "no candidates returned"
            # Blocked prompts need no reply in the chat.
            log.warning("Gemini returned no candidates: %s", reason)
            return ""

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            return ""
        return text.strip()

    async def inspect(self) -> Dict[str, Any]:
        """Tiny probe used by --check: auth mode plus a ping reply."""
        reply = await self.generate("ping", "Reply with exactly: OK")
        return {
            "auth": self.auth_mode,
            "engine": "gemini",
            "model": self.model,
            "reply": reply,
        }
