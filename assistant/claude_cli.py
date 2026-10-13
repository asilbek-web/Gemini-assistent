"""Call the installed Claude Code CLI as an isolated, subscription-backed engine.

Every reply is a single `claude -p` run. The message text goes over stdin,
the system prompt (profile included) through a temporary 0600 file, and the
child environment is scrubbed so the bot's own secrets never reach Claude.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

# Builtin plugins known to ship with Claude Code. They are all disabled via
# ISOLATION_SETTINGS; inspect() discovers the real list at runtime.
KNOWN_BUILTIN_PLUGINS = (
    "cc-plugin-agents-md@builtin",
    "cc-plugin-telemetry@builtin",
    "cc-plugin-plugin-authoring@builtin",
)

ISOLATION_SETTINGS: Dict[str, Any] = {
    "disableAllHooks": True,
    "enabledPlugins": {name: False for name in KNOWN_BUILTIN_PLUGINS},
}

# Environment variables stripped from the child process.
_STRIP_PREFIXES = ("ANTHROPIC_",)
_STRIP_NAMES = (
    "TELEGRAM_BOT_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SIMPLE",
)

_LIMIT_RE = re.compile(
    r"usage limit|hit your (usage )?limit|limit reached|rate[ _-]?limit|"
    r"out of (extra )?usage|quota",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"not logged in|/login|invalid api key|authentication|unauthori[sz]ed|oauth token",
    re.IGNORECASE,
)
_EPOCH_RE = re.compile(r"\|(\d{10,})")
_RESET_TEXT_RE = re.compile(r"resets [^|\n]*", re.IGNORECASE)


# Errors live in errors.py so every generator (Claude CLI, Gemini) shares
# one set; they are re-exported here for backward compatibility.
from .errors import (  # noqa: E402
    ClaudeAuthError,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
)


def build_env(
    base: Optional[Dict[str, str]] = None,
    oauth_token: Optional[str] = None,
) -> Dict[str, str]:
    """Child environment: parent env minus secrets, plus isolation flags."""
    source = os.environ if base is None else base
    env = {}
    for key, value in source.items():
        if key.startswith(_STRIP_PREFIXES):
            continue
        if key in _STRIP_NAMES:
            continue
        env[key] = value
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    if oauth_token:
        env["CLAUDE_CODE_OAUTH_TOKEN"] = oauth_token
    return env


def classify_failure(message: str, status: Optional[int] = None) -> ClaudeError:
    """Map CLI output to a typed exception (returns, never raises)."""
    text = message or ""
    if status == 429 or _LIMIT_RE.search(text):
        resets_at: Optional[float] = None
        epoch_match = _EPOCH_RE.search(text)
        reset_text = ""
        reset_match = _RESET_TEXT_RE.search(text)
        if reset_match:
            reset_text = reset_match.group(0).strip()
        detail = text
        if epoch_match:
            resets_at = float(epoch_match.group(1))
            detail = text.replace("|" + epoch_match.group(1), "")
        return ClaudeLimitError(detail.strip()[:500], resets_at=resets_at, reset_text=reset_text)
    if status in (401, 403) or _AUTH_RE.search(text):
        detail = text.strip()[:300]
        return ClaudeAuthError(
            "Claude login problem: run the Claude CLI interactively and /login, "
            "or refresh the subscription token (CLAUDE_CODE_OAUTH_TOKEN). "
            f"CLI said: {detail or 'no details'} (token redacted)",
            detail=detail,
        )
    return ClaudeError(text.strip()[:500] or "claude CLI failed")


def parse_result(
    returncode: int,
    stdout: str,
    stderr: str,
):
    """Parse the CLI's JSON envelope.

    Returns the envelope dict on success, or a typed ClaudeError instance
    on failure (returned, never raised).
    """
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
    envelope: Optional[Dict[str, Any]] = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict):
            envelope = candidate
            break

    if envelope is None:
        return classify_failure(stderr or stdout or f"exit code {returncode}")

    if envelope.get("is_error") or envelope.get("subtype") != "success":
        parts = [str(envelope.get("result") or "")]
        if stderr:
            parts.append(stderr)
        status = envelope.get("api_error_status")
        return classify_failure("\n".join(p for p in parts if p), status)

    result = envelope.get("result")
    if result is None or str(result).strip() == "":
        return ClaudeError("claude CLI returned an empty result")
    return envelope


class ClaudeCLI:
    """Runs one isolated `claude -p` per reply."""

    def __init__(
        self,
        claude_bin: str,
        *,
        workdir: Path,
        model: Optional[str] = None,
        effort: str = "low",
        timeout: int = 120,
        oauth_token: Optional[str] = None,
        auth_label: Optional[str] = None,
    ) -> None:
        self.claude_bin = claude_bin
        self.workdir = Path(workdir)
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self._oauth_token = oauth_token
        self._auth_label = auth_label

    def __repr__(self) -> str:  # pragma: no cover - defensive
        return (
            f"ClaudeCLI(bin={self.claude_bin!r}, model={self.model!r}, "
            f"effort={self.effort!r}, timeout={self.timeout}, "
            f"oauth_token={'set' if self._oauth_token else 'unset'})"
        )

    @property
    def auth_mode(self) -> str:
        if self._auth_label:
            return self._auth_label
        if self._oauth_token:
            return "subscription token"
        return "this machine's claude login"

    def _redact(self, text: str) -> str:
        if self._oauth_token and self._oauth_token in text:
            return text.replace(self._oauth_token, REDACTED)
        return text

    def _build_args(self, system_prompt_path: str, output_format: str) -> List[str]:
        args: List[str] = [
            self.claude_bin,
            "-p",
            "--safe-mode",
            "--settings",
            json.dumps(ISOLATION_SETTINGS),
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt-file",
            system_prompt_path,
            "--output-format",
            output_format,
            "--effort",
            self.effort,
        ]
        if output_format == "stream-json":
            args.append("--verbose")
        if self.model:
            args.extend(["--model", self.model])
        return args

    async def generate(self, message_text: str, system_prompt: str) -> str:
        """Send one message; return Claude's reply text."""
        if not Path(self.claude_bin).exists() and not _which(self.claude_bin):
            raise ClaudeError(f"claude binary not found: {self.claude_bin}")
        self.workdir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.workdir, 0o700)
        except OSError:  # pragma: no cover - platform specific
            pass

        fd, prompt_path = tempfile.mkstemp(prefix="sysprompt-", dir=str(self.workdir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(system_prompt)
            os.chmod(prompt_path, 0o600)
            args = self._build_args(prompt_path, "json")
            stdout, stderr, returncode = await self._run(args, message_text)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(prompt_path)

        stdout = self._redact(stdout or "")
        stderr = self._redact(stderr or "")
        outcome = parse_result(returncode, stdout, stderr)
        if isinstance(outcome, ClaudeAuthError) and self._oauth_token:
            raise ClaudeAuthError(
                "the subscription token was rejected or expired: run "
                "`claude setup-token` on a browser machine and update "
                f"CLAUDE_CODE_OAUTH_TOKEN. CLI said: {outcome.detail} "
                f"({REDACTED})",
                detail=outcome.detail,
            )
        if isinstance(outcome, ClaudeError):
            raise outcome
        return str(outcome.get("result") or "").strip()

    async def _run(self, args: Sequence[str], stdin_text: str):
        env = build_env(None, self._oauth_token)
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workdir),
            env=env,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(stdin_text.encode("utf-8")), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            await _kill_process_group(process)
            raise ClaudeTimeout(
                f"claude CLI timed out after {self.timeout}s"
            ) from None
        except asyncio.CancelledError:
            await _kill_process_group(process)
            raise
        return stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace"), process.returncode

    async def inspect(self) -> Dict[str, Any]:
        """Small stream-json probe: auth mode, init metadata, reply."""
        self.workdir.mkdir(parents=True, exist_ok=True)
        fd, prompt_path = tempfile.mkstemp(prefix="sysprompt-", dir=str(self.workdir))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("You are a connectivity probe. Reply with exactly: OK")
            os.chmod(prompt_path, 0o600)
            args = self._build_args(prompt_path, "stream-json")
            stdout, stderr, _returncode = await self._run(args, "ping")
        finally:
            with contextlib.suppress(OSError):
                os.unlink(prompt_path)

        stdout = self._redact(stdout or "")
        events: List[Dict[str, Any]] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)

        init: Dict[str, Any] = {}
        reply = ""
        for event in events:
            kind = event.get("type")
            if kind == "init":
                init = event
            elif kind == "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text":
                        reply += block.get("text", "")
            elif kind == "result":
                reply = str(event.get("result") or reply)

        if not init and not reply:
            raise classify_failure(stderr or stdout or "no stream-json events")

        return {
            "auth": self.auth_mode,
            "init_seen": bool(init),
            "init": {
                "tools": init.get("tools"),
                "mcp_servers": init.get("mcp_servers"),
                "plugins": init.get("plugins"),
                "skills": init.get("skills"),
                "apiKeySource": init.get("apiKeySource"),
                "apiProvider": init.get("apiProvider"),
                "model": init.get("model"),
            },
            "reply": reply.strip(),
        }


def _which(binary: str) -> Optional[str]:
    import shutil

    return shutil.which(binary)



async def _kill_process_group(process: "asyncio.subprocess.Process") -> None:
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    with contextlib.suppress(Exception):
        await process.wait()
