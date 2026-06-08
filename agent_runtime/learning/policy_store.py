from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ToolStats:
    tool_name: str
    count: int = 0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0
    last_used_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolPolicyStore:
    """JSON-backed policy statistics store for tool selection rewards."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state = self._load()

    def get_tool_stats(self, tool_name: str) -> ToolStats:
        name = str(tool_name or "").strip()
        with self._lock:
            raw = self._state["tools"].get(name)
            if not isinstance(raw, dict):
                return ToolStats(tool_name=name)
            return _stats_from_dict(name, raw)

    def update(
        self,
        tool_name: str,
        reward: float,
        *,
        success: bool | None = None,
        created_at: str | None = None,
    ) -> ToolStats:
        name = str(tool_name or "").strip()
        if not name:
            raise ValueError("tool_name is required")
        with self._lock:
            old = self.get_tool_stats(name)
            count = old.count + 1
            total_reward = old.total_reward + float(reward)
            success_count = old.success_count + (1 if success is True else 0)
            failure_count = old.failure_count + (1 if success is False else 0)
            stats = ToolStats(
                tool_name=name,
                count=count,
                total_reward=round(total_reward, 6),
                avg_reward=round(total_reward / max(1, count), 6),
                success_count=success_count,
                failure_count=failure_count,
                success_rate=round(success_count / max(1, success_count + failure_count), 6),
                last_used_at=created_at or _now_iso(),
            )
            self._state["tools"][name] = stats.to_dict()
            self._save()
            return stats

    def list_stats(self) -> list[ToolStats]:
        with self._lock:
            items = [
                _stats_from_dict(name, raw)
                for name, raw in self._state["tools"].items()
                if isinstance(raw, dict)
            ]
        return sorted(items, key=lambda item: (-item.count, item.tool_name))

    def reset(self) -> None:
        with self._lock:
            self._state = {"version": 1, "tools": {}}
            self._save()

    def to_dict(self) -> dict[str, Any]:
        return {"tools": [item.to_dict() for item in self.list_stats()]}

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "tools": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            tools = data.get("tools") if isinstance(data, dict) else {}
            return {"version": 1, "tools": tools if isinstance(tools, dict) else {}}
        except Exception:
            return {"version": 1, "tools": {}}

    def _save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp.replace(self.path)


def _stats_from_dict(tool_name: str, raw: dict[str, Any]) -> ToolStats:
    count = max(0, int(raw.get("count") or 0))
    total_reward = float(raw.get("total_reward") or 0.0)
    success_count = max(0, int(raw.get("success_count") or 0))
    failure_count = max(0, int(raw.get("failure_count") or 0))
    return ToolStats(
        tool_name=str(raw.get("tool_name") or tool_name),
        count=count,
        total_reward=round(total_reward, 6),
        avg_reward=round(float(raw.get("avg_reward") or (total_reward / max(1, count))), 6),
        success_count=success_count,
        failure_count=failure_count,
        success_rate=round(
            float(raw.get("success_rate") or (success_count / max(1, success_count + failure_count))),
            6,
        ),
        last_used_at=str(raw.get("last_used_at") or ""),
    )
