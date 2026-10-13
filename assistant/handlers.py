"""Update routing: business events plus the owner's direct commands."""
from __future__ import annotations

import logging
from typing import Any, List

from aiogram import F, Router
from aiogram.types import BusinessConnection, Message

from .service import Assistant

log = logging.getLogger(__name__)

ALLOWED_UPDATES: List[str] = ["message", "business_connection", "business_message"]

_STRANGER_NOTE = (
    "Bu shaxsiy yordamchi bot. U faqat egasi uchun ishlaydi — bu yerda "
    "suhbatlashib bo'lmaydi. Iltimos, egasiga to'g'ridan-to'g'ri yozing."
)


def build_router(assistant: Assistant) -> Router:
    router = Router(name="assistant")
    owner_id = assistant.settings.owner_id

    @router.business_connection()
    async def on_business_connection(event: BusinessConnection) -> None:
        await assistant.on_business_connection(event)

    @router.business_message()
    async def on_business_message(event: Message) -> None:
        await assistant.on_business_message(event.bot, event)

    @router.message(F.chat.type == "private", F.from_user.id == owner_id, F.text == "/start")
    async def owner_start(event: Message) -> None:
        await event.answer(assistant.start_text())

    @router.message(F.chat.type == "private", F.from_user.id == owner_id, F.text == "/pause")
    async def owner_pause(event: Message) -> None:
        assistant.pause()
        await event.answer("Pauza qo'yildi. Davom etish uchun /resume yozing.")

    @router.message(F.chat.type == "private", F.from_user.id == owner_id, F.text == "/resume")
    async def owner_resume(event: Message) -> None:
        assistant.resume()
        await event.answer("Davom etyapti.")

    @router.message(F.chat.type == "private", F.from_user.id == owner_id, F.text == "/status")
    async def owner_status(event: Message) -> None:
        await event.answer(assistant.status_text())

    @router.message(F.chat.type == "private", F.from_user.id == owner_id)
    async def owner_other(event: Message) -> None:
        # Any other private message from the owner: stay silent.
        return None

    @router.message(F.chat.type == "private", F.from_user.id != owner_id, F.text == "/start")
    async def stranger_start(event: Message) -> None:
        await event.answer(_STRANGER_NOTE)

    @router.message(F.chat.type == "private", F.from_user.id != owner_id)
    async def stranger_other(event: Message) -> None:
        # Strangers get nothing; the bot never reveals the owner's profile.
        return None

    return router
