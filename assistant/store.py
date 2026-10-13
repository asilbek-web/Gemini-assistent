"""Durable local state: per-chat history, seen messages, runtime flags.

Everything lives in the data directory (a Railway Volume in production) so
redeploys keep conversation memory, pause state, and limit cooldowns.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def ensure_private_dir(path: Path) -> None:
    """Create the directory with 0700 permissions."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:  # pragma: no cover - platform specific
        pass


class JsonFile:
    """Atomically-written JSON file. Unreadable files fall back to the default."""

    def __init__(self, path: Path, default: Any) -> None:
        self.path = Path(path)
        self.default = default

    def read(self) -> Any:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return self.default

    def write(self, data: Any) -> None:
        ensure_private_dir(self.path.parent)
        tmp = self.path.with_name(self.path.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(tmp), flags, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
        except Exception:
            tmp.unlink(missing_ok=True) if hasattr(tmp, "unlink") else None
            raise
        os.replace(str(tmp), str(self.path))
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - platform specific
            pass


class HistoryStore:
    """Short rolling conversation history per business connection + chat."""

    def __init__(self, path: Path, limit: int) -> None:
        self._file = JsonFile(path, {})
        self.limit = max(1, int(limit))

    @staticmethod
    def _key(connection_id: str, chat_id: int) -> str:
        return f"{connection_id}:{chat_id}"

    def get(self, connection_id: str, chat_id: int) -> List[Dict[str, Any]]:
        data = self._file.read()
        if not isinstance(data, dict):
            return []
        items = data.get(self._key(connection_id, chat_id))
        return items if isinstance(items, list) else []

    def add(self, connection_id: str, chat_id: int, item: Dict[str, Any]) -> None:
        data = self._file.read()
        if not isinstance(data, dict):
            data = {}
        key = self._key(connection_id, chat_id)
        items = data.get(key)
        if not isinstance(items, list):
            items = []
        items.append(item)
        data[key] = items[-self.limit:]
        self._file.write(data)

    def last_language(self, connection_id: str, chat_id: int) -> Optional[str]:
        for item in reversed(self.get(connection_id, chat_id)):
            lang = item.get("lang")
            if lang:
                return str(lang)
        return None

    def chat_count(self) -> int:
        data = self._file.read()
        return len(data) if isinstance(data, dict) else 0


class SeenMessages:
    """Bounded set of processed message keys; survives restarts."""

    def __init__(self, path: Path, capacity: int = 5000) -> None:
        self._file = JsonFile(path, [])
        self.capacity = capacity

    def first_time(self, key: str) -> bool:
        seen: "OrderedDict[str, None]" = OrderedDict()
        stored = self._file.read()
        if isinstance(stored, list):
            for entry in stored[: self.capacity]:
                seen[str(entry)] = None
        if key in seen:
            return False
        seen[key] = None
        while len(seen) > self.capacity:
            seen.popitem(last=False)
        self._file.write(list(seen.keys()))
        return True


class RuntimeState:
    """Pause flag and limit cooldown, persisted across restarts."""

    def __init__(self, path: Path) -> None:
        self._file = JsonFile(path, {})

    @property
    def paused(self) -> bool:
        data = self._file.read()
        return bool(data.get("paused")) if isinstance(data, dict) else False

    @property
    def limit_until(self) -> float:
        data = self._file.read()
        if not isinstance(data, dict):
            return 0.0
        try:
            return float(data.get("limit_until") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _update(self, **changes: Any) -> None:
        data = self._file.read()
        if not isinstance(data, dict):
            data = {}
        data.update(changes)
        self._file.write(data)

    def set_paused(self, paused: bool) -> None:
        self._update(paused=bool(paused))

    def set_limit_until(self, timestamp: float) -> None:
        self._update(limit_until=float(timestamp))

    def clear_limit(self) -> None:
        self.set_limit_until(0.0)

    def clear_all(self) -> None:
        self._update(paused=False, limit_until=0.0)
