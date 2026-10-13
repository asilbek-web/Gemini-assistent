#!/usr/bin/env python3
"""Telegram Business assistant bot entry point.

Usage:
  python bot.py                     run from .env (local development)
  python bot.py --settings-from-env run from process env (Railway container)
  python bot.py --check-claude      verify Claude CLI auth, no Telegram
  python bot.py --try "text"        generate one test reply, no Telegram
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from aiogram import Dispatcher
from aiogram.client.bot import Bot
from aiogram.exceptions import TelegramUnauthorizedError
from aiogram.filters import Command
from aiogram.types import Message

from assistant.claude_cli import ClaudeError
from assistant.config import (
    ConfigError,
    Settings,
    env_file_warning,
    load_settings,
    settings_from_process_env,
)
from assistant.handlers import ALLOWED_UPDATES, build_router
from assistant.language import detect_language
from assistant.prompts import build_system_prompt, build_user_prompt
from assistant.service import make_claude_cli, make_generator, build_assistant, read_profile
from assistant.store import ensure_private_dir

log = logging.getLogger("bot")


def _start_health_server(port: int) -> None:
    """Serve /health on $PORT so PaaS routers (Render) see an open port."""
    from aiohttp import web

    async def health(_request: web.Request) -> web.Response:
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)

    async def _run() -> None:
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", port)
        await site.start()
        log.info("Health server listening on 0.0.0.0:%d", port)

    asyncio.create_task(_run())

ENV_PATH = Path(".env")


def _load(args: argparse.Namespace) -> Settings:
    source: Optional[Dict[str, str]] = None
    if args.settings_from_env:
        # Reads SETTING_KEYS from os.environ and removes them right away, so
        # the Claude child process never inherits the bot token.
        source = settings_from_process_env()
        return load_settings(ENV_PATH, require_telegram=not args.lenient, source=source)
    return load_settings(ENV_PATH, require_telegram=not args.lenient)


async def _check_claude(settings: Settings) -> int:
    generator = make_generator(settings)

    if settings.engine == "gemini":
        try:
            report = await generator.inspect()
        except ClaudeError as exc:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, ensure_ascii=False))
            return 1
        report["ok"] = True
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    cli = generator
    try:
        report = await cli.inspect()
    except ClaudeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        return 1

    init = report.get("init") or {}
    problems = []
    if not report.get("init_seen"):
        problems.append("init event missing")
    reply = report.get("reply") or ""
    if "not logged in" in reply.lower():
        problems.append(f"Claude is not logged in: {reply}")
    for key in ("tools", "mcp_servers", "plugins", "skills"):
        value = init.get(key)
        if value:
            problems.append(f"{key} not empty: {value}")
    api_key_source = init.get("apiKeySource")
    if api_key_source in ("ANTHROPIC_API_KEY", "apiKeyHelper", "/login managed key"):
        problems.append(f"apiKeySource is {api_key_source}, not the subscription")
    api_provider = init.get("apiProvider")
    if api_provider not in (None, "firstParty"):
        problems.append(f"apiProvider is {api_provider}, not firstParty")

    report["ok"] = not problems
    if problems:
        report["problems"] = problems
    print(json.dumps(report, indent=2))
    if problems:
        return 1
    print(f"auth: {settings.claude_auth_mode}")
    return 0


async def _try_reply(settings: Settings, text: str) -> int:
    generator = make_generator(settings)
    profile = read_profile(settings.profile_path)
    language = detect_language(text)
    system_prompt = build_system_prompt(profile)
    user_prompt = build_user_prompt(
        [{"role": "other", "name": "Someone", "text": text, "lang": language}], language
    )
    try:
        reply = await generator.generate(user_prompt, system_prompt)
    except ClaudeError as exc:
        print(json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False))
        return 1
    print(
        json.dumps(
            {"language": language, "message": text, "reply": reply},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


async def _run_bot(settings: Settings) -> int:
    data_dir = Path(settings.data_dir)
    ensure_private_dir(data_dir)
    ensure_private_dir(data_dir / "claude-run")

    if not settings.from_environment:
        warning = env_file_warning(ENV_PATH)
        if warning:
            log.warning(warning)

    bot = Bot(token=settings.bot_token or "")
    dp = Dispatcher()
    assistant = build_assistant(settings)
    dp.include_router(build_router(assistant))

    try:
        me = await bot.get_me()
    except TelegramUnauthorizedError:
        log.error(
            "Telegram rejected the bot token. Check TELEGRAM_BOT_TOKEN "
            "or rotate it with @BotFather (/revoke)."
        )
        await bot.session.close()
        return 2

    mode = "AUTO-SEND" if settings.auto_send else "PREVIEW"
    log.info("Bot @%s is running", me.username)
    log.info("Mode: %s", mode)
    if not settings.auto_send:
        log.info("Mode: PREVIEW (replies are only shown in the logs)")
    log.info("Engine: %s", settings.engine)
    log.info("Claude auth: %s", settings.claude_auth_mode)
    if settings.engine == "claude":
        log.info("Claude binary: %s", settings.claude_bin)
    if assistant.state.paused:
        log.warning("Bot starts paused; send /resume to enable replies")
    if not read_profile(settings.profile_path):
        log.warning("profile.md is empty or missing; fill it in before going live")

    port = os.environ.get("PORT")
    if port and port.isdigit():
        _start_health_server(int(port))

    try:
        await dp.start_polling(bot, allowed_updates=ALLOWED_UPDATES)
    finally:
        await bot.session.close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Telegram Business assistant bot")
    parser.add_argument(
        "--settings-from-env",
        action="store_true",
        help="read settings from process environment (container mode)",
    )
    parser.add_argument(
        "--check-claude",
        action="store_true",
        help="verify Claude CLI auth and isolation, then exit",
    )
    parser.add_argument(
        "--try",
        dest="try_text",
        metavar="TEXT",
        help="generate one test reply without Telegram",
    )
    args = parser.parse_args(argv)
    # Telegram settings are only required for the real bot loop.
    args.lenient = bool(args.check_claude or args.try_text)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = _load(args)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        if not args.settings_from_env and not ENV_PATH.exists():
            print("cp .env.example .env && chmod 600 .env", file=sys.stderr)
        return 2

    if args.check_claude:
        return asyncio.run(_check_claude(settings))
    if args.try_text is not None:
        return asyncio.run(_try_reply(settings, args.try_text))

    if settings.bot_token is None or settings.owner_id is None:
        print(
            "Configuration error: TELEGRAM_BOT_TOKEN and OWNER_ID are required "
            "to run the bot",
            file=sys.stderr,
        )
        return 2

    try:
        return asyncio.run(_run_bot(settings))
    except TelegramUnauthorizedError:
        print(
            "Telegram rejected the bot token; rotate it with @BotFather (/revoke)",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
