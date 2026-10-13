"""End-to-end message flow tests over the recorded Telegram session."""
from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
from aiogram.exceptions import TelegramBadRequest

from assistant.claude_cli import (
    ClaudeAuthError,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
)
from assistant.prompts import SKIP_TOKEN
from assistant.service import Outcome, read_profile

from tests.conftest import (
    OWNER_ID,
    STRANGER_ID,
    make_connection,
    make_user,
)


def request_names(harness):
    return [type(m).__name__ for m in harness.session.requests]


class TestPreviewMode:
    async def test_preview_shows_reply_and_sends_nothing(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = ["Salom! Yaxshi, rahmat."]
        await h.business_message("salom qalesan")
        assert h.assistant.outcomes[-1] == Outcome.previewed
        assert len(h.previews) == 1
        assert h.previews[0].reply == "Salom! Yaxshi, rahmat."
        assert h.session.sent == []

    async def test_previews_file_written_private(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello")
        previews = Path(h.settings.data_dir) / "previews.jsonl"
        assert previews.exists()
        mode = stat.S_IMODE(os.stat(previews).st_mode)
        assert mode == 0o600
        content = previews.read_text(encoding="utf-8")
        assert "ok" in content

    async def test_unsent_reply_not_in_history(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello there")
        items = h.assistant.history.get("conn-1", STRANGER_ID)
        roles = [item["role"] for item in items]
        assert roles == ["other"]  # only the incoming message

    async def test_known_connection_no_extra_requests(self, make_harness):
        h = make_harness()
        await h.connect()
        h.session.requests.clear()
        await h.business_message("hello", message_id=2)
        assert request_names(h) == []

    async def test_unknown_connection_lookup_only(self, make_harness):
        h = make_harness()
        await h.business_message("hello", connection_id="ghost")
        assert request_names(h) == ["GetBusinessConnection"]
        assert h.assistant.outcomes[-1] == Outcome.no_connection

    async def test_lookup_error_is_no_connection(self, make_harness):
        h = make_harness()
        h.session.lookup_errors["conn-1"] = TelegramBadRequest(
            method=None, message="bad"
        )
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.no_connection


class TestConnectionGuards:
    async def test_foreign_connection_ignored(self, make_harness):
        h = make_harness()
        await h.connect(make_connection(user_id=999888))
        h.generator.results = ["should not be used"]
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.foreign_connection
        assert h.generator.calls == []

    async def test_disabled_connection_ignored(self, make_harness):
        h = make_harness()
        await h.connect(make_connection(enabled=False))
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.connection_disabled
        assert h.generator.calls == []

    async def test_any_chat_gets_reply(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", chat_id=111, message_id=10)
        await h.business_message("hello again", chat_id=222, message_id=11)
        assert list(h.assistant.outcomes)[-2:] == [Outcome.previewed, Outcome.previewed]


class TestOwnerMessages:
    async def test_owner_message_recorded_no_reply(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message(
            "o'zim yozaman", from_id=OWNER_ID, from_name="Me", chat_id=STRANGER_ID
        )
        assert h.assistant.outcomes[-1] == Outcome.owner_message
        assert h.generator.calls == []
        items = h.assistant.history.get("conn-1", STRANGER_ID)
        assert items and items[-1]["role"] == "owner"

    async def test_takeover_window(self, make_harness):
        h = make_harness(OWNER_TAKEOVER_MINUTES=10)
        await h.connect()
        await h.business_message(
            "o'zim yozaman", from_id=OWNER_ID, chat_id=STRANGER_ID
        )
        await h.business_message("hello", message_id=2)
        assert h.assistant.outcomes[-1] == Outcome.owner_active
        assert h.generator.calls == []
        h.clock.advance(11 * 60)
        await h.business_message("hello again", message_id=3)
        assert h.assistant.outcomes[-1] == Outcome.previewed

    async def test_takeover_zero_disabled(self, make_harness):
        h = make_harness(OWNER_TAKEOVER_MINUTES=0)
        await h.connect()
        await h.business_message(
            "o'zim yozaman", from_id=OWNER_ID, chat_id=STRANGER_ID
        )
        await h.business_message("hello", message_id=2)
        assert h.assistant.outcomes[-1] == Outcome.previewed


class TestBotsAndDuplicates:
    async def test_sender_business_bot_ignored(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", via_business_bot=True)
        assert h.assistant.outcomes[-1] == Outcome.ignored_bot
        assert h.generator.calls == []

    async def test_bot_from_user_ignored(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", is_bot=True)
        assert h.assistant.outcomes[-1] == Outcome.ignored_bot

    async def test_duplicate_parallel_and_sequential(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", message_id=5)
        await h.business_message("hello", message_id=5)
        assert h.assistant.outcomes[-1] == Outcome.duplicate
        # Parallel redelivery
        await asyncio.gather(
            h.business_message("yo", message_id=6),
            h.business_message("yo", message_id=6),
        )
        processed = [
            o
            for o in h.assistant.outcomes
            if o in (Outcome.previewed, Outcome.duplicate)
        ]
        assert processed.count(Outcome.previewed) == 2  # one per message id

    async def test_duplicate_survives_restart(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", message_id=7)
        h.restart()
        await h.business_message("hello", message_id=7)
        assert h.assistant.outcomes[-1] == Outcome.duplicate


class TestUnsupported:
    async def test_service_message_unsupported(self, make_harness):
        h = make_harness()
        await h.connect()
        from tests.conftest import make_message
        import json

        message = make_message(message_id=3, chat_id=STRANGER_ID, connection_id="conn-1")
        data = json.loads(message.model_dump_json(by_alias=True, exclude_none=True))
        await h.feed({"business_message": data})
        assert h.assistant.outcomes[-1] == Outcome.unsupported

    async def test_sticker_described(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = ["😄"]
        await h.business_message(None, message_id=4, sticker_emoji="😀")
        assert h.assistant.outcomes[-1] == Outcome.previewed
        assert h.previews[0].incoming == "[sticker] 😀"

    async def test_voice_described(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message(None, message_id=5, voice=True)
        assert h.previews[0].incoming == "[voice message]"


class TestHistoryIsolation:
    async def test_chats_are_separate(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("alpha secret", chat_id=1, message_id=1)
        await h.business_message("beta question", chat_id=2, message_id=2)
        prompt_1 = h.generator.calls[0]["user"]
        prompt_2 = h.generator.calls[1]["user"]
        assert "alpha secret" in prompt_1
        assert "beta question" not in prompt_1
        assert "beta question" in prompt_2
        assert "alpha secret" not in prompt_2


class TestLanguageHints:
    async def test_russian_hint(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("Привет, как дела?")
        assert "Russian" in h.generator.calls[0]["user"]

    async def test_english_hint(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("Hi, how are you?")
        assert "English" in h.generator.calls[0]["user"]

    async def test_uzbek_hint(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("Salom, qalesan?")
        assert "Uzbek" in h.generator.calls[0]["user"]

    async def test_emoji_falls_back_to_last_language(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("Привет!", message_id=1)
        await h.business_message("👍", message_id=2)
        assert "Russian" in h.generator.calls[1]["user"]


class TestSkip:
    async def test_skip_is_not_shown(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = [SKIP_TOKEN]
        await h.business_message("ok")
        assert h.assistant.outcomes[-1] == Outcome.skipped
        assert h.previews == []

    async def test_empty_reply_skipped(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = ["   "]
        await h.business_message("ok")
        assert h.assistant.outcomes[-1] == Outcome.skipped


class TestAutoSend:
    async def test_lookup_then_send(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = ["Javob beraman."]
        h.session.requests.clear()
        await h.business_message("salom", message_id=1)
        assert h.assistant.outcomes[-1] == Outcome.sent
        names = request_names(h)
        assert names.index("GetBusinessConnection") < names.index("SendMessage")
        assert h.session.sent[0]["business_connection_id"] == "conn-1"
        assert h.session.sent[0]["chat_id"] == STRANGER_ID

    async def test_sent_reply_in_history(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = ["mana javob"]
        await h.business_message("salom")
        items = h.assistant.history.get("conn-1", STRANGER_ID)
        assert [i["role"] for i in items] == ["other", "owner"]
        assert items[-1]["text"] == "mana javob"

    async def test_long_reply_split(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = ["x" * 9000]
        await h.business_message("salom")
        assert len(h.session.sent) == 3
        assert all(len(s["text"]) <= 4000 for s in h.session.sent)
        assert "".join(s["text"] for s in h.session.sent) == "x" * 9000

    async def test_can_reply_removed(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect(make_connection(can_reply=False))
        h.generator.results = ["javob"]
        h.session.requests.clear()
        await h.business_message("salom")
        assert h.assistant.outcomes[-1] == Outcome.permission_denied
        assert request_names(h) == ["GetBusinessConnection"]
        assert h.session.sent == []
        assert h.previews[0].note

    async def test_connection_disabled_midway(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        connection = await h.connect()
        h.generator.results = ["javob"]
        # Owner disables the connection while the reply is being generated.
        h.session.business_connections["conn-1"] = make_connection(enabled=False)
        await h.business_message("salom")
        assert h.assistant.outcomes[-1] == Outcome.permission_denied
        assert h.session.sent == []

    async def test_owner_writes_during_generation(self, make_harness):
        h = make_harness(AUTO_SEND="true")

        def hook():
            # The owner's message arrives while Claude is thinking.
            h.assistant.state  # noqa: B018
            h.assistant._takeover[("conn-1", STRANGER_ID)] = h.clock.now + 900

        from tests.conftest import FakeGenerator
        h.generator = FakeGenerator(["javob"], hook=hook)
        h.assistant.generator = h.generator
        await h.connect()
        await h.business_message("salom")
        assert h.assistant.outcomes[-1] == Outcome.owner_active
        assert h.session.sent == []

    async def test_send_error(self, make_harness):
        from aiogram.exceptions import TelegramRetryAfter

        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = ["javob"]
        h.session.send_error = TelegramRetryAfter(
            method=None, message="flood", retry_after=1
        )
        await h.business_message("salom")
        assert h.assistant.outcomes[-1] == Outcome.error
        assert h.session.sent == []


class TestPauseResume:
    async def test_pause_resume_owner_only(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.direct_message("/pause")
        assert h.assistant.state.paused is True
        await h.business_message("hello", message_id=1)
        assert h.assistant.outcomes[-1] == Outcome.paused
        await h.direct_message("/resume")
        assert h.assistant.state.paused is False
        await h.business_message("hello again", message_id=2)
        assert h.assistant.outcomes[-1] == Outcome.previewed

    async def test_pause_survives_restart(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.direct_message("/pause")
        h.restart()
        assert h.assistant.state.paused is True

    async def test_stranger_commands_ignored(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", message_id=1)
        sent_before = len(h.session.sent)
        await h.business_message("/pause", message_id=2)
        assert h.assistant.state.paused is False
        assert len(h.session.sent) == sent_before

    async def test_stranger_direct_messages_silent(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.direct_message("salom", from_id=STRANGER_ID, message_id=51)
        await h.direct_message("/pause", from_id=STRANGER_ID, message_id=52)
        assert h.session.sent == []
        assert h.assistant.state.paused is False

    async def test_stranger_start_has_no_profile(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.direct_message("/start", from_id=STRANGER_ID, message_id=60)
        assert h.generator.calls == []  # Claude never called for strangers


class TestStartStatus:
    async def test_owner_start_uzbek(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.direct_message("/start")
        assert len(h.session.sent) == 1
        text = h.session.sent[0]["text"]
        assert "PREVIEW" in text
        assert "Telegram Business" in text

    async def test_start_autosend_mode(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        await h.direct_message("/start")
        assert "AUTO-SEND" in h.session.sent[0]["text"]

    async def test_start_when_paused(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.direct_message("/pause")
        await h.direct_message("/start", message_id=101)
        assert "pauzada" in h.session.sent[-1]["text"]

    async def test_status_text(self, make_harness):
        h = make_harness()
        await h.connect()
        await h.business_message("hello", message_id=1)
        await h.direct_message("/status")
        text = h.session.sent[-1]["text"]
        assert "PREVIEW" in text
        assert "Chats in history: 1" in text


class TestLimitHandling:
    async def test_limit_blocks_until_reset(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = [
            ClaudeLimitError("usage limit", resets_at=h.clock.now + 600, reset_text="5pm")
        ]
        await h.business_message("hello", message_id=1)
        assert h.assistant.outcomes[-1] == Outcome.limited
        assert len(h.session.sent) == 1  # owner notification
        h.session.sent.clear()
        # Blocked before reset
        await h.business_message("hello", message_id=2)
        assert h.assistant.outcomes[-1] == Outcome.limited
        assert h.generator.calls.__len__() == 1  # no new generation
        # After reset it continues
        h.clock.advance(601)
        h.generator.results = ["ok now"]
        await h.business_message("hello", message_id=3)
        assert h.assistant.outcomes[-1] == Outcome.sent

    async def test_limit_without_reset_uses_cooldown(self, make_harness):
        h = make_harness(LIMIT_COOLDOWN_MINUTES=30)
        await h.connect()
        h.generator.results = [ClaudeLimitError("usage limit")]
        await h.business_message("hello", message_id=1)
        assert h.assistant.state.limit_until == h.clock.now + 30 * 60

    async def test_limit_survives_restart(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = [ClaudeLimitError("usage limit")]
        await h.business_message("hello", message_id=1)
        h.restart()
        assert h.assistant.state.limit_until > 0
        await h.business_message("hello", message_id=2)
        assert h.assistant.outcomes[-1] == Outcome.limited

    async def test_preview_mode_limit_no_owner_message(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = [ClaudeLimitError("usage limit")]
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.limited
        assert h.session.sent == []

    async def test_resume_clears_limit(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = [ClaudeLimitError("usage limit")]
        await h.business_message("hello", message_id=1)
        await h.direct_message("/resume")
        assert h.assistant.state.limit_until == 0.0


class TestErrorHandling:
    async def test_timeout_sends_nothing_to_chat(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = [ClaudeTimeout("timed out")]
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.error
        assert all(s["chat_id"] == OWNER_ID for s in h.session.sent)

    async def test_auth_error_notified_once(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = [ClaudeAuthError("login problem", detail="d")]
        await h.business_message("hello", message_id=1)
        await h.business_message("hello", message_id=2)
        assert list(h.assistant.outcomes)[-2:] == [Outcome.error, Outcome.error]
        assert len(h.session.sent) == 1  # rate-limited to one per 10 min
        h.clock.advance(601)
        await h.business_message("hello", message_id=3)
        assert len(h.session.sent) == 2

    async def test_generic_error(self, make_harness):
        h = make_harness(AUTO_SEND="true")
        await h.connect()
        h.generator.results = [ClaudeError("boom")]
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.error

    async def test_preview_mode_errors_not_sent(self, make_harness):
        h = make_harness()
        await h.connect()
        h.generator.results = [ClaudeError("boom")]
        await h.business_message("hello")
        assert h.assistant.outcomes[-1] == Outcome.error
        assert h.session.sent == []


class TestProfile:
    async def test_profile_comments_stripped(self, tmp_path):
        profile = tmp_path / "profile.md"
        profile.write_text(
            "# Men haqimda\n<!-- maxfiy -->\nIsm: Asilbek\n", encoding="utf-8"
        )
        text = read_profile(profile)
        assert "maxfiy" not in text
        assert "Asilbek" in text

    async def test_missing_profile_empty(self, tmp_path):
        assert read_profile(tmp_path / "nope.md") == ""

    async def test_profile_in_system_prompt(self, make_harness, tmp_path):
        profile = Path(make_harness().settings.profile_path)
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.write_text("Ism: Asilbek. Shahar: Toshkent.", encoding="utf-8")
        h = make_harness()
        await h.connect()
        await h.business_message("salom")
        assert "Asilbek" in h.generator.calls[0]["system"]
