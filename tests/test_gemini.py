"""Gemini generator tests with a fake HTTP transport (offline)."""
from __future__ import annotations

import json

import pytest

from assistant.config import load_settings
from assistant.errors import (
    ClaudeAuthError,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
)
from assistant.gemini import GeminiGenerator
from assistant.service import make_generator

KEY = "AIzaSyFAKEKEY123456789"


def make_generator_client(**kwargs) -> GeminiGenerator:
    defaults = dict(api_key=KEY, model="gemini-2.5-flash", timeout=10)
    defaults.update(kwargs)
    return GeminiGenerator(**defaults)


def ok_transport(reply="OK"):
    async def transport(url, payload, headers, timeout):
        body = json.dumps(
            {
                "candidates": [
                    {"content": {"parts": [{"text": reply}]}}
                ]
            }
        )
        return 200, body

    return transport


class TestGenerate:
    async def test_success(self):
        gen = make_generator_client(transport=ok_transport("Salom!"))
        assert await gen.generate("user msg", "sys prompt") == "Salom!"

    async def test_payload_structure(self):
        seen = {}

        async def transport(url, payload, headers, timeout):
            seen["url"] = url
            seen["payload"] = payload
            seen["headers"] = headers
            return 200, json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

        gen = make_generator_client(transport=transport)
        await gen.generate("matn", "tizim matni")
        assert "gemini-2.5-flash:generateContent" in seen["url"]
        assert seen["payload"]["systemInstruction"]["parts"][0]["text"] == "tizim matni"
        assert seen["payload"]["contents"][0]["parts"][0]["text"] == "matn"
        assert seen["headers"]["x-goog-api-key"] == KEY

    async def test_key_not_in_argv_anywhere(self):
        # REST call: the key travels only in a header.
        gen = make_generator_client(transport=ok_transport())
        await gen.generate("msg", "sys")
        assert gen._api_key == KEY
        assert "AIza" not in repr(gen)

    async def test_rate_limit(self):
        async def transport(url, payload, headers, timeout):
            return 429, '{"error": {"status": "RESOURCE_EXHAUSTED"}}'

        gen = make_generator_client(transport=transport)
        with pytest.raises(ClaudeLimitError):
            await gen.generate("msg", "sys")

    async def test_bad_key(self):
        async def transport(url, payload, headers, timeout):
            return 400, '{"error": {"message": "API key not valid"}}'

        gen = make_generator_client(transport=transport)
        with pytest.raises(ClaudeAuthError) as excinfo:
            await gen.generate("msg", "sys")
        assert "aistudio.google.com" in str(excinfo.value)

    async def test_unauthorized(self):
        async def transport(url, payload, headers, timeout):
            return 403, "permission denied"

        gen = make_generator_client(transport=transport)
        with pytest.raises(ClaudeAuthError):
            await gen.generate("msg", "sys")

    async def test_server_error(self):
        async def transport(url, payload, headers, timeout):
            return 503, "upstream unavailable"

        gen = make_generator_client(transport=transport)
        with pytest.raises(ClaudeError):
            await gen.generate("msg", "sys")

    async def test_network_timeout(self):
        async def transport(url, payload, headers, timeout):
            raise TimeoutError("too slow")

        gen = make_generator_client(transport=transport)
        with pytest.raises(ClaudeTimeout):
            await gen.generate("msg", "sys")

    async def test_key_redacted_in_errors(self):
        async def transport(url, payload, headers, timeout):
            return 400, f"API key not valid: {KEY}"

        gen = make_generator_client(transport=transport)
        with pytest.raises(ClaudeAuthError) as excinfo:
            await gen.generate("msg", "sys")
        assert KEY not in str(excinfo.value)

    async def test_blocked_prompt_returns_empty(self):
        async def transport(url, payload, headers, timeout):
            return 200, json.dumps({"promptFeedback": {"blockReason": "SAFETY"}})

        gen = make_generator_client(transport=transport)
        assert await gen.generate("msg", "sys") == ""

    async def test_inspect(self):
        gen = make_generator_client(transport=ok_transport("OK"), auth_label="Gemini API key from .env")
        report = await gen.inspect()
        assert report["engine"] == "gemini"
        assert report["reply"] == "OK"
        assert report["auth"] == "Gemini API key from .env"


class TestEngineSelection:
    def test_gemini_key_wins(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "TELEGRAM_BOT_TOKEN=1:x\nOWNER_ID=1\nGEMINI_API_KEY=" + KEY + "\n"
            "CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-also-set\n",
            encoding="utf-8",
        )
        settings = load_settings(env)
        assert settings.engine == "gemini"
        generator = make_generator(settings)
        assert isinstance(generator, GeminiGenerator)
        assert settings.claude_auth_mode == "Gemini API key from .env"

    def test_claude_without_key(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "TELEGRAM_BOT_TOKEN=1:x\nOWNER_ID=1\nCLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-x\n",
            encoding="utf-8",
        )
        settings = load_settings(env)
        assert settings.engine == "claude"
        from assistant.claude_cli import ClaudeCLI

        assert isinstance(make_generator(settings), ClaudeCLI)

    def test_key_hidden_from_repr(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "TELEGRAM_BOT_TOKEN=1:x\nOWNER_ID=1\nGEMINI_API_KEY=" + KEY + "\n",
            encoding="utf-8",
        )
        settings = load_settings(env)
        assert KEY not in repr(settings)

    def test_spaces_rejected(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "TELEGRAM_BOT_TOKEN=1:x\nOWNER_ID=1\nGEMINI_API_KEY=AIza bad key\n",
            encoding="utf-8",
        )
        from assistant.config import ConfigError

        with pytest.raises(ConfigError):
            load_settings(env)
