"""Settings loading for the Telegram Business assistant bot.

Settings come from exactly one source: a local .env file (development) or
process environment variables (Railway container). The two are never mixed.
Secrets are never written to os.environ and never echoed in errors or reprs.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from dotenv import dotenv_values

ENV_FILE_NAME = ".env"
# Plain-text fallback holding only the Gemini key (git-ignored).
GEMINI_KEY_FILE_NAME = "gemini.key"

# Every variable the bot understands. bot.py reads and cleans the process
# environment by this list.
SETTING_KEYS = (
    "TELEGRAM_BOT_TOKEN",
    "OWNER_ID",
    "AUTO_SEND",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_BIN",
    "CLAUDE_MODEL",
    "CLAUDE_EFFORT",
    "CLAUDE_TIMEOUT_SECONDS",
    "LIMIT_COOLDOWN_MINUTES",
    "OWNER_TAKEOVER_MINUTES",
    "HISTORY_LIMIT",
    "DATA_DIR",
    "PROFILE_PATH",
)

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_EFFORTS = {"low", "medium", "high", "xhigh", "max"}

DEFAULT_CLAUDE_TIMEOUT_SECONDS = 120
DEFAULT_LIMIT_COOLDOWN_MINUTES = 30
DEFAULT_OWNER_TAKEOVER_MINUTES = 15
DEFAULT_HISTORY_LIMIT = 30


class ConfigError(Exception):
    """Invalid or missing configuration. The message never contains secret values."""


def _source_label(env_file: Optional[Path], source: Optional[Mapping[str, str]]) -> str:
    if source is not None:
        return "the environment variables"
    if env_file is not None:
        return str(env_file)
    return ENV_FILE_NAME


def _fail(label: str, message: str) -> "ConfigError":
    return ConfigError(f"{message} (check {label})")


def _parse_bool(key: str, raw: Optional[str], label: str) -> bool:
    if raw is None or raw.strip() == "":
        return False
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise _fail(label, f"{key} must be true/false (got an invalid value)")


def _parse_int(
    key: str,
    raw: Optional[str],
    label: str,
    default: int,
    minimum: Optional[int] = None,
) -> int:
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        raise _fail(label, f"{key} must be an integer") from None
    if minimum is not None and value < minimum:
        raise _fail(label, f"{key} must be >= {minimum}")
    return value


def _validate_bot_token(raw: Optional[str], label: str) -> str:
    if raw is None or raw.strip() == "":
        raise _fail(label, "TELEGRAM_BOT_TOKEN is required")
    token = raw.strip()
    if ":" not in token or not token.split(":", 1)[0].isdigit() or not token.split(":", 1)[1]:
        raise _fail(label, "TELEGRAM_BOT_TOKEN must look like <digits>:<text>")
    return token


def _validate_owner_id(raw: Optional[str], label: str) -> int:
    if raw is None or raw.strip() == "":
        raise _fail(label, "OWNER_ID is required")
    try:
        return int(raw.strip())
    except ValueError:
        raise _fail(label, "OWNER_ID must be a number")


def _validate_gemini_key(raw: Optional[str], label: str) -> Optional[str]:
    if raw is None or raw.strip() == "":
        return None
    key = raw.strip()
    if any(ch.isspace() for ch in key):
        raise _fail(
            label,
            "GEMINI_API_KEY must be a single line without spaces",
        )
    return key


def _validate_claude_token(raw: Optional[str], label: str) -> Optional[str]:
    if raw is None or raw.strip() == "":
        return None
    token = raw.strip()
    if token.startswith("sk-ant-api"):
        raise _fail(
            label,
            "CLAUDE_CODE_OAUTH_TOKEN looks like an Anthropic API key, not a "
            "subscription token; run `claude setup-token` instead",
        )
    if not token.startswith("sk-ant-oat"):
        raise _fail(
            label,
            "CLAUDE_CODE_OAUTH_TOKEN must start with sk-ant-oat "
            "(result of `claude setup-token`)",
        )
    if any(ch.isspace() for ch in token):
        raise _fail(
            label,
            "CLAUDE_CODE_OAUTH_TOKEN must be a single line without spaces",
        )
    return token


def _resolve_claude_bin(raw: Optional[str]) -> str:
    if raw is not None and raw.strip():
        return raw.strip()
    found = shutil.which("claude")
    if found:
        return found
    return str(Path.home() / ".local" / "bin" / "claude")


@dataclass(frozen=True)
class Settings:
    bot_token: Optional[str] = field(default=None, repr=False)
    owner_id: Optional[int] = None
    auto_send: bool = False
    gemini_api_key: Optional[str] = field(default=None, repr=False)
    gemini_model: str = ""
    claude_oauth_token: Optional[str] = field(default=None, repr=False)
    claude_bin: str = "claude"
    claude_model: Optional[str] = None
    claude_effort: str = "low"
    claude_timeout_seconds: int = DEFAULT_CLAUDE_TIMEOUT_SECONDS
    limit_cooldown_minutes: int = DEFAULT_LIMIT_COOLDOWN_MINUTES
    owner_takeover_minutes: int = DEFAULT_OWNER_TAKEOVER_MINUTES
    history_limit: int = DEFAULT_HISTORY_LIMIT
    data_dir: Path = Path("data")
    profile_path: Path = Path("profile.md")
    from_environment: bool = False
    env_file: Optional[Path] = None

    @property
    def claude_auth_mode(self) -> str:
        if self.gemini_api_key:
            if self.from_environment:
                return "Gemini API key from environment variables"
            return "Gemini API key from .env"
        if not self.claude_oauth_token:
            return "this machine's claude login"
        if self.from_environment:
            return "subscription token from environment variables"
        return "subscription token from .env"

    @property
    def engine(self) -> str:
        """Which reply generator wins: free Gemini key, else Claude CLI."""
        return "gemini" if self.gemini_api_key else "claude"


def load_settings(
    env_file: Optional[Path] = None,
    *,
    require_telegram: bool = True,
    overrides: Optional[Dict[str, str]] = None,
    source: Optional[Mapping[str, str]] = None,
) -> Settings:
    """Load settings from exactly one source.

    source: mapping of raw env values (container mode). When given, the env
    file is not read at all and env_file only anchors relative paths.
    overrides: test-only raw values applied on top of the chosen source.
    """
    if env_file is None:
        env_file = Path(ENV_FILE_NAME)

    if source is not None:
        raw_values: Dict[str, Optional[str]] = dict(source)
        from_environment = True
    else:
        loaded = dotenv_values(env_file)
        raw_values = {key: loaded.get(key) for key in SETTING_KEYS}
        from_environment = False
        # The editor may lock .env as read-only; fall back to a plain
        # one-line file so the key can still be pasted.
        if not (raw_values.get("GEMINI_API_KEY") or "").strip():
            key_file = env_file.parent / GEMINI_KEY_FILE_NAME
            try:
                pasted = key_file.read_text(encoding="utf-8").strip()
            except OSError:
                pasted = ""
            if pasted:
                raw_values["GEMINI_API_KEY"] = pasted

    if overrides:
        raw_values.update(overrides)

    label = _source_label(env_file, source)

    def get(key: str) -> Optional[str]:
        value = raw_values.get(key)
        if value is None:
            return None
        return value

    bot_token = _validate_bot_token(get("TELEGRAM_BOT_TOKEN"), label) if require_telegram else (
        _validate_bot_token(get("TELEGRAM_BOT_TOKEN"), label) if get("TELEGRAM_BOT_TOKEN") else None
    )
    owner_id = _validate_owner_id(get("OWNER_ID"), label) if require_telegram else (
        _validate_owner_id(get("OWNER_ID"), label) if get("OWNER_ID") else None
    )

    gemini_key = _validate_gemini_key(get("GEMINI_API_KEY"), label)
    claude_token = _validate_claude_token(get("CLAUDE_CODE_OAUTH_TOKEN"), label)

    effort_raw = (get("CLAUDE_EFFORT") or "").strip().lower()
    if effort_raw and effort_raw not in _EFFORTS:
        raise _fail(label, "CLAUDE_EFFORT must be one of low, medium, high, xhigh, max")

    base_dir = env_file.parent if env_file.name != ENV_FILE_NAME or env_file.parent != Path("") else Path.cwd()
    if source is not None and env_file == Path(ENV_FILE_NAME):
        base_dir = Path.cwd()

    def resolve_path(raw: Optional[str], default: str) -> Path:
        text = (raw or "").strip() or default
        path = Path(text)
        if not path.is_absolute():
            path = base_dir / path
        return path

    return Settings(
        bot_token=bot_token,
        owner_id=owner_id,
        auto_send=_parse_bool("AUTO_SEND", get("AUTO_SEND"), label),
        gemini_api_key=gemini_key,
        gemini_model=(get("GEMINI_MODEL") or "").strip(),
        claude_oauth_token=claude_token,
        claude_bin=_resolve_claude_bin(get("CLAUDE_BIN")),
        claude_model=(get("CLAUDE_MODEL") or "").strip() or None,
        claude_effort=effort_raw or "low",
        claude_timeout_seconds=_parse_int(
            "CLAUDE_TIMEOUT_SECONDS",
            get("CLAUDE_TIMEOUT_SECONDS"),
            label,
            DEFAULT_CLAUDE_TIMEOUT_SECONDS,
            minimum=1,
        ),
        limit_cooldown_minutes=_parse_int(
            "LIMIT_COOLDOWN_MINUTES",
            get("LIMIT_COOLDOWN_MINUTES"),
            label,
            DEFAULT_LIMIT_COOLDOWN_MINUTES,
            minimum=1,
        ),
        owner_takeover_minutes=_parse_int(
            "OWNER_TAKEOVER_MINUTES",
            get("OWNER_TAKEOVER_MINUTES"),
            label,
            DEFAULT_OWNER_TAKEOVER_MINUTES,
            minimum=0,
        ),
        history_limit=_parse_int(
            "HISTORY_LIMIT", get("HISTORY_LIMIT"), label, DEFAULT_HISTORY_LIMIT, minimum=1
        ),
        data_dir=resolve_path(get("DATA_DIR"), "data"),
        profile_path=resolve_path(get("PROFILE_PATH"), "profile.md"),
        from_environment=from_environment,
        env_file=env_file,
    )


def env_file_warning(path: Path) -> Optional[str]:
    """Return a warning when the env file is readable by group or others."""
    try:
        mode = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        return None
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        return (
            f"{path} is readable by other users (mode {oct(mode)}); "
            f"run: chmod 600 {path}"
        )
    return None


def settings_from_process_env(
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Read SETTING_KEYS from the process environment and remove them.

    Removal happens even if a value is invalid, so secrets never leak into
    child processes (Claude CLI). Other variables are left untouched.
    """
    env = os.environ if environ is None else environ
    collected: Dict[str, str] = {}
    sink = os.environ if environ is None else None
    for key in SETTING_KEYS:
        if key in env:
            collected[key] = env[key]
            if sink is not None:
                sink.pop(key, None)
    return collected
