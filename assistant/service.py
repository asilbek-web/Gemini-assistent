"""Core assistant logic: decide, generate, and (optionally) send replies."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

from aiogram.exceptions import TelegramAPIError
from aiogram.types import BusinessConnection, Message

from .claude_cli import (
    ClaudeCLI,
    ClaudeAuthError,
    ClaudeError,
    ClaudeLimitError,
)
from .config import Settings
from .language import detect_language
from .prompts import SKIP_TOKEN, build_system_prompt, build_user_prompt
from .store import HistoryStore, RuntimeState, SeenMessages, ensure_private_dir

log = logging.getLogger(__name__)

MAX_CHUNK = 4000
ERROR_NOTIFY_INTERVAL = 600.0  # at most one owner error message per 10 min
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


class Outcome(str, Enum):
    previewed = "previewed"
    sent = "sent"
    duplicate = "duplicate"
    ignored_bot = "ignored_bot"
    no_connection = "no_connection"
    foreign_connection = "foreign_connection"
    connection_disabled = "connection_disabled"
    unsupported = "unsupported"
    owner_message = "owner_message"
    owner_active = "owner_active"
    paused = "paused"
    limited = "limited"
    skipped = "skipped"
    permission_denied = "permission_denied"
    error = "error"


@dataclass
class Preview:
    connection_id: str
    chat_id: int
    chat_name: str
    language: Optional[str]
    incoming: str
    reply: str
    note: str = ""


def print_preview(preview: Preview) -> None:
    note = f"  note: {preview.note}" if preview.note else ""
    log.warning(
        "PREVIEW, not sent\n"
        "  chat: %s (%s)\n"
        "  lang: %s\n"
        "  incoming: %s\n"
        "  reply: %s%s",
        preview.chat_name,
        preview.chat_id,
        preview.language or "?",
        preview.incoming,
        preview.reply,
        note,
    )


def describe(message: Message) -> Optional[str]:
    """Human-readable text for a message, or None for service messages."""
    if message.text:
        return message.text
    if message.voice:
        return "[voice message]"
    if message.photo:
        caption = message.caption or ""
        return f"[photo] {caption}".strip()
    if message.sticker:
        emoji = message.sticker.emoji or ""
        return f"[sticker] {emoji}".strip()
    if message.video_note:
        return "[video note]"
    if message.video:
        caption = message.caption or ""
        return f"[video] {caption}".strip()
    if message.audio:
        caption = message.caption or ""
        return f"[audio] {caption}".strip()
    if message.document:
        caption = message.caption or ""
        return f"[document] {caption}".strip()
    if message.animation:
        return "[animation]"
    if message.contact:
        return "[contact]"
    if message.location:
        return "[location]"
    if message.poll:
        return "[poll]"
    return None


def can_reply(connection: BusinessConnection) -> bool:
    """Reply permission; supports both new rights block and legacy field."""
    rights = getattr(connection, "rights", None)
    if rights is not None:
        return bool(rights.can_reply)
    return bool(getattr(connection, "can_reply", False))


def read_profile(path: Path) -> str:
    """Read profile.md on every reply; strip <!-- --> comments."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError:
        return ""
    return _COMMENT_RE.sub("", raw).strip()


def _chat_name(message: Message) -> str:
    chat = message.chat
    if chat.full_name:
        return chat.full_name
    if chat.username:
        return f"@{chat.username}"
    return str(chat.id)


def _sender_name(message: Message) -> str:
    user = message.from_user
    if user is None:
        return "Someone"
    if user.full_name:
        return user.full_name
    if user.username:
        return f"@{user.username}"
    return str(user.id)


def _split_chunks(text: str, size: int = MAX_CHUNK):
    text = text.strip()
    while text:
        yield text[:size]
        text = text[size:]


class Assistant:
    """Handles business messages, generates replies, honors blocks."""

    def __init__(
        self,
        settings: Settings,
        generator: ClaudeCLI,
        *,
        clock=time.time,
        preview_sink=print_preview,
    ) -> None:
        self.settings = settings
        self.generator = generator
        self.clock = clock
        self.preview_sink = preview_sink

        data_dir = Path(settings.data_dir)
        ensure_private_dir(data_dir)
        self.data_dir = data_dir
        self.history = HistoryStore(data_dir / "history.json", settings.history_limit)
        self.seen = SeenMessages(data_dir / "seen.json")
        self.state = RuntimeState(data_dir / "state.json")
        self._previews_path = data_dir / "previews.jsonl"

        self._connections: Dict[str, BusinessConnection] = {}
        self._locks: Dict[Tuple[str, int], asyncio.Lock] = {}
        self._takeover: Dict[Tuple[str, int], float] = {}
        self._last_error_notify = 0.0
        self.outcomes: Deque[Outcome] = deque(maxlen=100)

    # ------------------------------------------------------------------ #

    def _record(self, outcome: Outcome) -> Outcome:
        self.outcomes.append(outcome)
        return outcome

    def _lock(self, connection_id: str, chat_id: int) -> asyncio.Lock:
        key = (connection_id, chat_id)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _takeover_active(self, connection_id: str, chat_id: int) -> bool:
        until = self._takeover.get((connection_id, chat_id), 0.0)
        return until > self.clock()

    def _write_preview(self, preview: Preview) -> None:
        record = {
            "connection_id": preview.connection_id,
            "chat_id": preview.chat_id,
            "chat_name": preview.chat_name,
            "language": preview.language,
            "incoming": preview.incoming,
            "reply": preview.reply,
            "note": preview.note,
            "ts": self.clock(),
        }
        try:
            ensure_private_dir(self.data_dir)
            fd = os.open(
                str(self._previews_path),
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:  # pragma: no cover - disk issues
            log.warning("could not write previews.jsonl")

    def _emit_preview(self, preview: Preview) -> None:
        """Persist a preview and show it through the preview sink."""
        self._write_preview(preview)
        if self.preview_sink:
            self.preview_sink(preview)

    # ------------------------------------------------------------------ #

    async def on_business_connection(self, connection: BusinessConnection) -> None:
        """Cache a business connection; never sends messages."""
        self._connections[connection.id] = connection
        owner_id = self.settings.owner_id
        if connection.user.id != owner_id:
            log.warning(
                "Business connection %s from foreign account user=%s "
                "(owner=%s) - ignored",
                connection.id,
                connection.user.id,
                owner_id,
            )
            return
        log.info(
            "Business connection %s: user=%s enabled=%s can_reply=%s",
            connection.id,
            connection.user.id,
            connection.is_enabled,
            can_reply(connection),
        )

    async def on_business_message(self, bot: Any, message: Message) -> Outcome:
        connection_id = message.business_connection_id
        if not connection_id:
            return self._record(Outcome.no_connection)

        chat_id = message.chat.id
        dedupe_key = f"{connection_id}:{chat_id}:{message.message_id}"
        # Mark before any await so parallel deliveries collapse.
        if not self.seen.first_time(dedupe_key):
            return self._record(Outcome.duplicate)

        if message.sender_business_bot is not None or (
            message.from_user is not None and message.from_user.is_bot
        ):
            return self._record(Outcome.ignored_bot)

        connection = self._connections.get(connection_id)
        if connection is None:
            try:
                connection = await bot.get_business_connection(connection_id)
            except TelegramAPIError:
                return self._record(Outcome.no_connection)
            self._connections[connection_id] = connection

        owner_id = self.settings.owner_id
        if connection.user.id != owner_id:
            return self._record(Outcome.foreign_connection)
        if not connection.is_enabled:
            return self._record(Outcome.connection_disabled)

        incoming = describe(message)
        if incoming is None:
            return self._record(Outcome.unsupported)

        ts = self.clock()
        chat_key = (connection_id, chat_id)

        if message.from_user is not None and message.from_user.id == owner_id:
            self.history.add(
                connection_id,
                chat_id,
                {"role": "owner", "name": "You", "text": incoming, "lang": None, "ts": ts},
            )
            takeover = self.settings.owner_takeover_minutes
            if takeover > 0:
                self._takeover[chat_key] = ts + takeover * 60
            return self._record(Outcome.owner_message)

        language = detect_language(incoming) or self.history.last_language(
            connection_id, chat_id
        )
        self.history.add(
            connection_id,
            chat_id,
            {
                "role": "other",
                "name": _sender_name(message),
                "text": incoming,
                "lang": language,
                "ts": ts,
            },
        )

        async with self._lock(connection_id, chat_id):
            blocked = self._check_blocks(chat_key, ts)
            if blocked is not None:
                return self._record(blocked)

            reply, failure = await self._generate(message, incoming, language, chat_key, ts)
            if failure is not None:
                return self._record(failure)
            if reply is None:
                return self._record(Outcome.skipped)

            if not self.settings.auto_send:
                preview = Preview(
                    connection_id=connection_id,
                    chat_id=chat_id,
                    chat_name=_chat_name(message),
                    language=language,
                    incoming=incoming,
                    reply=reply,
                )
                self._write_preview(preview)
                if self.preview_sink:
                    self.preview_sink(preview)
                return self._record(Outcome.previewed)

            return await self._send_reply(
                bot, message, connection_id, chat_id, chat_key, reply, language, incoming
            )

    def _check_blocks(self, chat_key: Tuple[str, int], ts: float) -> Optional[Outcome]:
        if self.state.paused:
            return Outcome.paused
        if self.state.limit_until > ts:
            return Outcome.limited
        if self._takeover_active(*chat_key):
            return Outcome.owner_active
        return None

    async def _generate(
        self,
        message: Message,
        incoming: str,
        language: Optional[str],
        chat_key: Tuple[str, int],
        ts: float,
    ) -> Tuple[Optional[str], Optional[Outcome]]:
        profile = read_profile(self.settings.profile_path)
        system_prompt = build_system_prompt(profile, datetime.fromtimestamp(ts))
        items = self.history.get(*chat_key)
        user_prompt = build_user_prompt(items, language)
        try:
            raw = await self.generator.generate(user_prompt, system_prompt)
        except ClaudeLimitError as exc:
            resets_at = exc.resets_at
            if resets_at is None or resets_at <= ts:
                resets_at = ts + self.settings.limit_cooldown_minutes * 60
            self.state.set_limit_until(resets_at)
            log.error(
                "Claude usage limit hit; sleeping until %s (%s)",
                resets_at,
                exc.detail[:200],
            )
            if self.settings.auto_send:
                await self._notify_owner(
                    message.bot,
                    "Claude obuna limiti tugadi, birozdan keyin qayta ishlayman. "
                    f"({exc})",
                )
            return None, Outcome.limited
        except ClaudeAuthError as exc:
            log.error("Claude auth error: %s", exc)
            if self.settings.auto_send:
                await self._notify_owner(
                    message.bot,
                    "Claude bilan aloqa uzildi (login/token). Eshit: " + str(exc),
                )
            return None, Outcome.error
        except ClaudeError as exc:
            log.error("Claude error: %s", exc)
            if self.settings.auto_send:
                await self._notify_owner(
                    message.bot,
                    "Javob yaratishda xatolik bo'ldi. (" + str(exc)[:200] + ")",
                )
            return None, Outcome.error

        reply = (raw or "").strip()
        if not reply or reply == SKIP_TOKEN:
            return None, None
        return reply, None

    async def _notify_owner(self, bot: Any, text: str) -> None:
        """Owner DM, rate-limited to one per 10 minutes."""
        now = self.clock()
        if now - self._last_error_notify < ERROR_NOTIFY_INTERVAL:
            return
        self._last_error_notify = now
        owner_id = self.settings.owner_id
        if bot is None or owner_id is None:
            return
        try:
            await bot.send_message(owner_id, text)
        except TelegramAPIError as exc:  # pragma: no cover - network
            log.warning("could not notify owner: %s", exc)

    async def _send_reply(
        self,
        bot: Any,
        message: Message,
        connection_id: str,
        chat_id: int,
        chat_key: Tuple[str, int],
        reply: str,
        language: Optional[str],
        incoming: str,
    ) -> Outcome:
        ts = self.clock()
        # Re-check blocks: the owner may have written during generation.
        blocked = self._check_blocks(chat_key, ts)
        if blocked is not None:
            note = "owner wrote during generation; not sent" if blocked is Outcome.owner_active else str(blocked.value)
            self._emit_preview(
                Preview(connection_id, chat_id, _chat_name(message), language, incoming, reply, note=note)
            )
            return self._record(blocked)

        # Re-fetch the connection from Telegram (not the cache).
        try:
            fresh = await bot.get_business_connection(connection_id)
        except TelegramAPIError as exc:
            self._emit_preview(
                Preview(connection_id, chat_id, _chat_name(message), language, incoming, reply, note=f"lookup failed: {exc}")
            )
            return self._record(Outcome.permission_denied)

        owner_id = self.settings.owner_id
        if (
            fresh.user.id != owner_id
            or not fresh.is_enabled
            or not can_reply(fresh)
        ):
            self._emit_preview(
                Preview(
                    connection_id,
                    chat_id,
                    _chat_name(message),
                    language,
                    incoming,
                    reply,
                    note="reply permission missing (can_reply=False or connection disabled)",
                )
            )
            return self._record(Outcome.permission_denied)

        try:
            for chunk in _split_chunks(reply):
                await bot.send_message(
                    chat_id,
                    chunk,
                    business_connection_id=connection_id,
                )
        except TelegramAPIError as exc:
            log.error("send_message failed: %s", exc)
            return self._record(Outcome.error)

        self.history.add(
            connection_id,
            chat_id,
            {"role": "owner", "name": "You", "text": reply, "lang": language, "ts": ts},
        )
        return self._record(Outcome.sent)

    # ------------------------------------------------------------------ #

    def pause(self) -> None:
        self.state.set_paused(True)

    def resume(self) -> None:
        self.state.set_paused(False)
        self.state.clear_limit()
        self._takeover.clear()

    def status_text(self) -> str:
        mode = (
            "AUTO-SEND"
            if self.settings.auto_send
            else "PREVIEW (replies are only shown in the logs)"
        )
        paused = self.state.paused
        limit_until = self.state.limit_until
        lines = [
            f"Mode: {mode}",
            f"State: {'paused' if paused else 'running'}",
        ]
        if limit_until > self.clock():
            lines.append(
                "Limit: active until "
                + datetime.fromtimestamp(limit_until).strftime("%Y-%m-%d %H:%M")
            )
        else:
            lines.append("Limit: none")
        lines.append(f"Chats in history: {self.history.chat_count()}")
        return "\n".join(lines)

    def start_text(self) -> str:
        if self.settings.auto_send:
            mode = "AUTO-SEND: replies are sent automatically"
        else:
            mode = "PREVIEW: replies are only shown in the logs (set AUTO_SEND=true to send)"
        state = "pauzada" if self.state.paused else "ishlayapti"
        return (
            "Salom! Men sizning shaxsiy Telegram Business yordamchingizman.\n\n"
            "Men nima qilaman: menga kimdir yozsa, sizning uslubingizda javob "
            "tayyorlayman. Ma'lumotlar profile.md faylidan olinadi.\n\n"
            f"Holat: {mode}. Hozir: {state}.\n\n"
            "Qanday ulanadi: Telegram → Settings → Telegram Business → "
            "Chatbots → shu botni tanlang → qaysi chatlarni ko'rishini "
            "tanlang → javob berish ruxsatini yoqing (Telegram Premium kerak).\n\n"
            "Buyruqlar: /pause — to'xtatish, /resume — qayta ishga tushirish, "
            "/status — holat.\n\n"
            "Bilib qo'ying: o'zingiz yozsangiz bot bir necha daqiqa jim "
            "turadi; pul, qarz va uchrashuvlarga rozilik bermaydi; profile.md "
            "ni to'ldirish kerak."
        )


def make_claude_cli(settings: Settings) -> ClaudeCLI:
    """Build the Claude CLI wrapper from settings (token and label included)."""
    workdir = Path(settings.data_dir) / "claude-run"
    return ClaudeCLI(
        settings.claude_bin,
        workdir=workdir,
        model=settings.claude_model,
        effort=settings.claude_effort,
        timeout=settings.claude_timeout_seconds,
        oauth_token=settings.claude_oauth_token,
        auth_label=settings.claude_auth_mode,
    )


def make_generator(settings: Settings):
    """Free Gemini key wins; otherwise the Claude CLI subscription."""
    if settings.gemini_api_key:
        from .gemini import GeminiGenerator

        return GeminiGenerator(
            settings.gemini_api_key,
            model=settings.gemini_model,
            timeout=settings.claude_timeout_seconds,
            auth_label=settings.claude_auth_mode,
        )
    return make_claude_cli(settings)


def build_assistant(settings: Settings, **kwargs: Any) -> Assistant:
    return Assistant(settings, make_generator(settings), **kwargs)
