"""Offline test harness: recorded Telegram session, fake generator, fake clock."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pytest
from aiogram import Dispatcher
from aiogram.client.bot import Bot
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import (
    BusinessBotRights,
    BusinessConnection,
    Message,
    Update,
    User,
)

from assistant.config import Settings, load_settings
from assistant.handlers import build_router
from assistant.service import Assistant

BOT_TOKEN = "123456:TEST-TOKEN-FOR-TESTS"
OWNER_ID = 777000
OWNER_NAME = "Test Owner"
STRANGER_ID = 555001


def make_user(user_id: int, name: str, is_bot: bool = False) -> User:
    return User(
        id=user_id,
        is_bot=is_bot,
        first_name=name,
        username=None,
    )


def make_connection(
    connection_id: str = "conn-1",
    user_id: int = OWNER_ID,
    enabled: bool = True,
    can_reply: bool = True,
    use_rights: bool = True,
) -> BusinessConnection:
    """Connection payload carries both rights.can_reply and legacy can_reply."""
    payload: Dict[str, Any] = {
        "id": connection_id,
        "user": make_user(user_id, OWNER_NAME),
        "user_chat_id": user_id,
        "date": datetime.now(timezone.utc),
        "is_enabled": enabled,
    }
    if use_rights:
        payload["rights"] = BusinessBotRights(can_reply=can_reply)
        payload["can_reply"] = can_reply
    else:
        payload["can_reply"] = can_reply
    return BusinessConnection(**payload)


class RecordingSession(BaseSession):
    """Records every Telegram request; answers from fixtures."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: List[Any] = []
        self.business_connections: Dict[str, BusinessConnection] = {}
        self.lookup_errors: Dict[str, Exception] = {}
        self.send_error: Optional[Exception] = None
        self.sent: List[Dict[str, Any]] = []

    async def close(self) -> None:
        return None

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):  # type: ignore[override]
        raise AssertionError("streaming is not expected in tests")

    async def make_request(self, bot, method, timeout=None):  # type: ignore[override]
        self.requests.append(method)
        name = type(method).__name__
        if name == "GetBusinessConnection":
            connection = self.business_connections.get(method.business_connection_id)
            error = self.lookup_errors.get(method.business_connection_id)
            if error is not None:
                raise error
            if connection is None:
                raise TelegramBadRequest(
                    method=method, message="Bad Request: connection not found"
                )
            return connection
        if name == "SendMessage":
            if self.send_error is not None:
                raise self.send_error
            self.sent.append(
                {
                    "chat_id": method.chat_id,
                    "text": method.text,
                    "business_connection_id": getattr(
                        method, "business_connection_id", None
                    ),
                }
            )
            return make_message(
                message_id=len(self.sent),
                chat_id=method.chat_id,
                text=method.text,
            )
        if name == "GetMe":
            return make_user(1, "test_bot", is_bot=True)
        return True


def make_message(
    *,
    message_id: int,
    chat_id: int,
    text: Optional[str] = None,
    from_user: Optional[User] = None,
    connection_id: Optional[str] = None,
    is_bot: bool = False,
    via_business_bot: bool = False,
    sticker_emoji: Optional[str] = None,
    voice: bool = False,
) -> Message:
    payload: Dict[str, Any] = {
        "message_id": message_id,
        "date": datetime.now(timezone.utc),
        "chat": {"id": chat_id, "type": "private", "first_name": "Chat"},
    }
    if text is not None:
        payload["text"] = text
    if from_user is not None:
        payload["from"] = from_user
    elif is_bot:
        payload["from"] = make_user(999, "other_bot", is_bot=True)
    if connection_id is not None:
        payload["business_connection_id"] = connection_id
    if via_business_bot:
        payload["sender_business_bot"] = make_user(999, "relay_bot", is_bot=True)
    if sticker_emoji is not None:
        payload["sticker"] = {
            "file_id": "sticker-file",
            "file_unique_id": "sticker-unique",
            "type": "regular",
            "width": 100,
            "height": 100,
            "is_animated": False,
            "is_video": False,
            "emoji": sticker_emoji,
        }
    if voice:
        payload["voice"] = {
            "file_id": "voice-file",
            "file_unique_id": "voice-unique",
            "duration": 3,
        }
    return Message.model_validate(payload)


class FakeGenerator:
    """Returns canned replies or raises canned exceptions, in order."""

    def __init__(
        self,
        results: Optional[List[Any]] = None,
        hook: Optional[Callable[[], None]] = None,
    ) -> None:
        self.results = list(results or ["ok"])
        self.hook = hook
        self.calls: List[Dict[str, Any]] = []

    async def generate(self, user_prompt: str, system_prompt: str) -> str:
        self.calls.append({"user": user_prompt, "system": system_prompt})
        if self.hook is not None:
            self.hook()
        if not self.results:
            return "ok"
        item = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    """Deterministic clock."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Harness:
    """Real Dispatcher + Assistant over the recorded session."""

    def __init__(self, tmp_path, **overrides: Any) -> None:
        self.tmp_path = tmp_path
        self.clock = Clock()
        self.session = RecordingSession()
        self.bot = Bot(token=BOT_TOKEN, session=self.session)
        self.dp = Dispatcher()
        self.generator = FakeGenerator()
        self.previews: List[Any] = []

        raw = {
            "TELEGRAM_BOT_TOKEN": BOT_TOKEN,
            "OWNER_ID": str(OWNER_ID),
            "AUTO_SEND": "false",
        }
        raw.update({k: str(v) for k, v in overrides.items()})
        self.settings = load_settings(
            tmp_path / ".env.missing",
            require_telegram=True,
            overrides=raw,
        )
        self._spawn_assistant()
        self.dp.include_router(build_router(self.assistant))
        self._update_id = 0

    def _spawn_assistant(self) -> None:
        self.assistant = Assistant(
            self.settings,
            self.generator,
            clock=self.clock,
            preview_sink=self.previews.append,
        )

    def restart(self) -> None:
        """Simulate a redeploy: new Assistant, same data directory."""
        self._spawn_assistant()
        self.dp = Dispatcher()
        self.dp.include_router(build_router(self.assistant))

    async def feed(self, update: Dict[str, Any]) -> None:
        self._update_id += 1
        update["update_id"] = self._update_id
        model = Update.model_validate(update, context={"bot": self.bot})
        await self.dp.feed_update(self.bot, model)

    def set_connection(self, connection: BusinessConnection) -> None:
        self.session.business_connections[connection.id] = connection
        self.assistant._connections[connection.id] = connection

    async def connect(self, connection: Optional[BusinessConnection] = None) -> BusinessConnection:
        connection = connection or make_connection()
        self.session.business_connections[connection.id] = connection
        await self.feed(
            {
                "business_connection": json.loads(
                    connection.model_dump_json(by_alias=True, exclude_none=True)
                )
            }
        )
        return connection

    async def business_message(
        self,
        text: str,
        *,
        from_id: int = STRANGER_ID,
        from_name: str = "Stranger",
        chat_id: Optional[int] = None,
        message_id: int = 1,
        connection_id: str = "conn-1",
        is_bot: bool = False,
        via_business_bot: bool = False,
        sticker_emoji: Optional[str] = None,
        voice: bool = False,
    ) -> None:
        message = make_message(
            message_id=message_id,
            chat_id=chat_id if chat_id is not None else from_id,
            text=text,
            from_user=make_user(from_id, from_name, is_bot=is_bot),
            connection_id=connection_id,
            is_bot=is_bot,
            via_business_bot=via_business_bot,
            sticker_emoji=sticker_emoji,
            voice=voice,
        )
        await self.feed(
            {
                "business_message": json.loads(
                    message.model_dump_json(by_alias=True, exclude_none=True)
                )
            }
        )

    async def direct_message(
        self,
        text: str,
        *,
        from_id: int = OWNER_ID,
        message_id: int = 100,
    ) -> None:
        message = make_message(
            message_id=message_id,
            chat_id=from_id,
            text=text,
            from_user=make_user(from_id, "Owner"),
        )
        data = json.loads(message.model_dump_json(by_alias=True, exclude_none=True))
        if text.startswith("/"):
            data["entities"] = [
                {"type": "bot_command", "offset": 0, "length": len(text)}
            ]
        await self.feed({"message": data})


@pytest.fixture
def make_harness(tmp_path):
    def _make(**overrides: Any) -> Harness:
        return Harness(tmp_path, **overrides)

    return _make
