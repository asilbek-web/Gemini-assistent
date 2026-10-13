"""Engine-neutral errors for the reply generator (Claude CLI or Gemini)."""
from __future__ import annotations


class ClaudeError(Exception):
    """Any generator failure visible to the owner."""


class ClaudeTimeout(ClaudeError):
    """The generator run exceeded the configured timeout."""


class ClaudeAuthError(ClaudeError):
    """Login/token problem. detail carries the (redacted) provider text."""

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


class ClaudeLimitError(ClaudeError):
    """Provider usage/rate limit hit."""

    def __init__(
        self,
        detail: str,
        resets_at: float = None,
        reset_text: str = "",
    ) -> None:
        when = f"resets {reset_text}" if reset_text else "resets at an unknown time"
        super().__init__(f"AI usage limit reached ({when})")
        self.detail = detail
        self.resets_at = resets_at
        self.reset_text = reset_text
