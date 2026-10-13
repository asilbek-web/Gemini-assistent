"""Claude CLI runner tests using a fake `claude` executable (offline)."""
from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from assistant.claude_cli import (
    ClaudeAuthError,
    ClaudeCLI,
    ClaudeError,
    ClaudeLimitError,
    ClaudeTimeout,
    classify_failure,
    parse_result,
)

FAKE_SOURCE = r'''
import json, os, sys, time

def record(extra=None):
    argv = sys.argv[1:]
    try:
        stdin_text = sys.stdin.read()
    except Exception:
        stdin_text = ""
    system_prompt = ""
    prompt_path = None
    for i, arg in enumerate(argv):
        if arg == "--system-prompt-file":
            prompt_path = argv[i + 1]
            try:
                with open(prompt_path, "r", encoding="utf-8") as fh:
                    system_prompt = fh.read()
            except OSError:
                system_prompt = "<unreadable>"
    payload = {
        "argv": argv,
        "stdin": stdin_text,
        "env": {k: v for k, v in os.environ.items()},
        "cwd": os.getcwd(),
        "cwd_mode": oct(os.stat(os.getcwd()).st_mode & 0o777),
        "pid": os.getpid(),
        "system_prompt": system_prompt,
        "system_prompt_mode": None,
        "system_prompt_path_exists_after": None,
    }
    if prompt_path:
        try:
            payload["system_prompt_mode"] = oct(os.stat(prompt_path).st_mode & 0o777)
        except OSError:
            payload["system_prompt_mode"] = "gone"
        payload["system_prompt_path"] = prompt_path
    if extra:
        payload.update(extra)
    with open(os.environ["FAKE_RECORD"], "w", encoding="utf-8") as fh:
        json.dump(payload, fh)

mode = os.environ.get("FAKE_MODE", "ok")
stream = "--stream-json" if "--verbose" in sys.argv else None
is_stream = any(a == "stream-json" for a in sys.argv)

if mode == "sleep":
    record()
    child = os.fork()
    if child == 0:
        time.sleep(60)
        os._exit(0)
    with open(os.environ["FAKE_RECORD"] + ".child", "w") as fh:
        fh.write(str(child))
    time.sleep(60)
    os._exit(0)

record()
token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")

def emit(obj):
    print(json.dumps(obj))

if mode == "ok":
    if is_stream:
        emit({"type": "init", "tools": [], "mcp_servers": {}, "plugins": [],
              "skills": [], "apiKeySource": "subscription",
              "apiProvider": "firstParty", "model": "claude-test"})
        emit({"type": "result", "subtype": "success", "is_error": False,
              "result": "OK", "api_error_status": None})
    else:
        emit({"type": "result", "subtype": "success", "is_error": False,
              "result": "  pong  ", "api_error_status": None})
elif mode == "limit_epoch":
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "You have reached your usage limit |1770000000. resets 3pm",
          "api_error_status": 429})
elif mode == "limit_text":
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "hit your usage limit, resets 3pm PST", "api_error_status": None})
elif mode == "auth":
    emit({"type": "result", "subtype": "error", "is_error": True,
          "result": "Not logged in. Token=" + token, "api_error_status": 401})
elif mode == "echo_token":
    emit({"type": "result", "subtype": "success", "is_error": False,
          "result": "token seen: " + token, "api_error_status": None})
elif mode == "crash":
    sys.stderr.write("claude exploded\n")
    sys.exit(1)
elif mode == "empty":
    emit({"type": "result", "subtype": "success", "is_error": False,
          "result": "", "api_error_status": None})
'''


@pytest.fixture
def fake_claude(tmp_path):
    script = tmp_path / "claude"
    script.write_text("#!" + sys.executable + "\n" + FAKE_SOURCE, encoding="utf-8")
    os.chmod(script, 0o755)
    record = tmp_path / "record.json"
    return script, record


def make_cli(fake_claude, tmp_path, **kwargs) -> ClaudeCLI:
    script, record = fake_claude
    os.environ["FAKE_RECORD"] = str(record)
    defaults = dict(
        workdir=tmp_path / "run",
        effort="low",
        timeout=30,
        oauth_token="sk-ant-oat01-testtoken",
        auth_label="subscription token from .env",
    )
    defaults.update(kwargs)
    return ClaudeCLI(str(script), **defaults)


def read_record(fake_claude):
    _, record = fake_claude
    return json.loads(record.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def clean_fake_env():
    saved = {k: os.environ.get(k) for k in ("FAKE_MODE", "FAKE_RECORD")}
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class TestGenerate:
    async def test_result_is_stripped(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        cli = make_cli(fake_claude, tmp_path)
        assert await cli.generate("hi", "sys") == "pong"

    async def test_message_via_stdin_not_argv(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        cli = make_cli(fake_claude, tmp_path)
        await cli.generate("secret message $(touch pwned)", "system prompt text")
        record = read_record(fake_claude)
        assert record["stdin"] == "secret message $(touch pwned)"
        assert "secret message" not in " ".join(record["argv"])
        assert "system prompt text" not in " ".join(record["argv"])
        assert record["system_prompt"] == "system prompt text"
        assert record["system_prompt_mode"] == "0o600"
        assert not (tmp_path / "pwned").exists()

    async def test_tmp_prompt_removed_on_error(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "crash"
        cli = make_cli(fake_claude, tmp_path)
        with pytest.raises(ClaudeError):
            await cli.generate("hi", "sys")
        record = read_record(fake_claude)
        assert not Path(record["system_prompt_path"]).exists()

    async def test_isolation_flags_in_order(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        cli = make_cli(fake_claude, tmp_path, model="sonnet")
        await cli.generate("hi", "sys")
        argv = read_record(fake_claude)["argv"]
        expected_tail_base = ["-p", "--safe-mode"]
        assert argv[:2] == expected_tail_base
        assert "--bare" not in argv
        assert "--mcp-config" not in argv
        for flag in (
            "--settings",
            "--tools",
            "--strict-mcp-config",
            "--setting-sources",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--system-prompt-file",
            "--output-format",
            "--effort",
            "--model",
        ):
            assert flag in argv
        # --settings carries the isolation JSON
        settings_json = json.loads(argv[argv.index("--settings") + 1])
        assert settings_json["disableAllHooks"] is True
        assert all(value is False for value in settings_json["enabledPlugins"].values())
        assert argv[argv.index("--effort") + 1] == "low"
        assert argv[argv.index("--model") + 1] == "sonnet"

    async def test_env_scrubbed(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        os.environ["ANTHROPIC_API_KEY"] = "leak-me"
        os.environ["ANTHROPIC_BASE_URL"] = "leak"
        os.environ["TELEGRAM_BOT_TOKEN"] = "1:leak"
        os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
        os.environ["CLAUDE_CODE_SIMPLE"] = "1"
        os.environ["CLAUDECODE"] = "1"
        os.environ["HOME"] = str(tmp_path)
        try:
            cli = make_cli(fake_claude, tmp_path)
            await cli.generate("hi", "sys")
            env = read_record(fake_claude)["env"]
            assert "ANTHROPIC_API_KEY" not in env
            assert "ANTHROPIC_BASE_URL" not in env
            assert "TELEGRAM_BOT_TOKEN" not in env
            assert "CLAUDE_CODE_USE_BEDROCK" not in env
            assert "CLAUDE_CODE_SIMPLE" not in env
            assert "CLAUDECODE" not in env
            assert env["HOME"] == str(tmp_path)
            assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
            assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
            assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-testtoken"
        finally:
            for key in (
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_BASE_URL",
                "TELEGRAM_BOT_TOKEN",
                "CLAUDE_CODE_USE_BEDROCK",
                "CLAUDE_CODE_SIMPLE",
                "CLAUDECODE",
            ):
                os.environ.pop(key, None)

    async def test_no_token_when_unset(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        cli = make_cli(fake_claude, tmp_path, oauth_token=None, auth_label=None)
        await cli.generate("hi", "sys")
        env = read_record(fake_claude)["env"]
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
        assert cli.auth_mode == "this machine's claude login"

    async def test_token_redacted_in_errors(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "auth"
        cli = make_cli(fake_claude, tmp_path)
        with pytest.raises(ClaudeAuthError) as excinfo:
            await cli.generate("hi", "sys")
        text = str(excinfo.value)
        assert "sk-ant-oat01-testtoken" not in text
        assert "setup-token" in text
        assert "CLI said:" in text

    async def test_token_redacted_in_success_path(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "echo_token"
        cli = make_cli(fake_claude, tmp_path)
        reply = await cli.generate("hi", "sys")
        assert "sk-ant-oat01-testtoken" not in reply

    async def test_workdir_private(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        cli = make_cli(fake_claude, tmp_path)
        await cli.generate("hi", "sys")
        record = read_record(fake_claude)
        assert record["cwd"] == str(cli.workdir)
        assert record["cwd_mode"] == "0o700"

    async def test_timeout_kills_process_group(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "sleep"
        cli = make_cli(fake_claude, tmp_path, timeout=1)
        started = time.monotonic()
        with pytest.raises(ClaudeTimeout):
            await cli.generate("hi", "sys")
        elapsed = time.monotonic() - started
        assert elapsed < 5
        child_pid = int((tmp_path / "record.json.child").read_text().strip())
        await asyncio.sleep(0.1)
        with pytest.raises(OSError):
            os.kill(child_pid, 0)

    async def test_binary_missing(self, tmp_path):
        cli = ClaudeCLI(
            str(tmp_path / "no-such-claude"),
            workdir=tmp_path / "run",
            timeout=5,
        )
        with pytest.raises(ClaudeError) as excinfo:
            await cli.generate("hi", "sys")
        assert "not found" in str(excinfo.value)

    async def test_empty_result(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "empty"
        cli = make_cli(fake_claude, tmp_path)
        with pytest.raises(ClaudeError):
            await cli.generate("hi", "sys")

    async def test_inspect_reports_auth(self, fake_claude, tmp_path):
        os.environ["FAKE_MODE"] = "ok"
        cli = make_cli(fake_claude, tmp_path)
        report = await cli.inspect()
        assert report["auth"] == "subscription token from .env"
        assert report["init"]["tools"] == []
        assert report["init"]["apiKeySource"] == "subscription"
        assert report["init"]["apiProvider"] == "firstParty"
        assert report["reply"] == "OK"
        argv = read_record(fake_claude)["argv"]
        assert "--verbose" in argv
        assert argv[argv.index("--output-format") + 1] == "stream-json"


class TestParseResult:
    def test_success(self):
        envelope = parse_result(0, '{"result": " hi ", "is_error": false, "subtype": "success"}', "")
        assert envelope["result"] == " hi "

    def test_limit_epoch(self):
        err = parse_result(
            0,
            '{"result": "usage limit |1770000000", "is_error": true, "subtype": "error", "api_error_status": 429}',
            "",
        )
        assert isinstance(err, ClaudeLimitError)
        assert err.resets_at == 1770000000.0
        assert "1770000000" not in err.detail

    def test_limit_text(self):
        err = parse_result(
            0,
            '{"result": "hit your usage limit, resets 3pm", "is_error": true, "subtype": "error"}',
            "",
        )
        assert isinstance(err, ClaudeLimitError)
        assert "3pm" in err.reset_text

    def test_auth(self):
        err = parse_result(
            0,
            '{"result": "not logged in, run /login", "is_error": true, "subtype": "error", "api_error_status": 401}',
            "",
        )
        assert isinstance(err, ClaudeAuthError)

    def test_crash_falls_back_to_stderr(self):
        err = parse_result(1, "", "something broke")
        assert isinstance(err, ClaudeError)
        assert "something broke" in str(err)

    def test_plain_text_classified(self):
        err = parse_result(1, "", "rate limit exceeded, try later")
        assert isinstance(err, ClaudeLimitError)

    def test_generic(self):
        err = classify_failure("weird failure", None)
        assert type(err) is ClaudeError

    def test_api_status_403(self):
        err = classify_failure("nope", 403)
        assert isinstance(err, ClaudeAuthError)
