"""Append-only logs: write-ahead operations and durable review decisions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from mindpalace.ids import new_id
from mindpalace.models import Decision


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _append_line(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class OpLog:
    """Write-ahead log. An op with a begin and no commit is replayed on startup."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def begin(self, intent: dict) -> str:
        op_id = new_id("op_")
        _append_line(
            self.path, {"kind": "op.begin", "op": op_id, "ts": _now(), "intent": intent}
        )
        return op_id

    def commit(self, op_id: str) -> None:
        _append_line(self.path, {"kind": "op.commit", "op": op_id, "ts": _now()})

    def pending(self) -> list[dict]:
        begun: dict[str, dict] = {}
        for record in _read_lines(self.path):
            if record["kind"] == "op.begin":
                begun[record["op"]] = record
            elif record["kind"] == "op.commit":
                begun.pop(record["op"], None)
        return list(begun.values())


class DecisionLog:
    """Durable review events. Assertion status is the fold of this log."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(
        self,
        assertion_id: str,
        action: str,
        via: str,
        op_id: str,
        reason: str | None = None,
    ) -> None:
        _append_line(
            self.path,
            {
                "op": op_id,
                "ts": _now(),
                "assertion": assertion_id,
                "action": action,
                "via": via,
                "reason": reason,
            },
        )

    def entries(self) -> list[Decision]:
        return [
            Decision(
                op=record["op"],
                ts=record["ts"],
                assertion=record["assertion"],
                action=record["action"],
                via=record["via"],
                reason=record.get("reason"),
            )
            for record in _read_lines(self.path)
        ]

    def status_map(self) -> dict[str, str]:
        statuses: dict[str, str] = {}
        for decision in self.entries():
            statuses[decision.assertion] = decision.action
        return statuses

    def dismissal_reasons(self) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for decision in self.entries():
            if decision.action == "dismiss":
                reasons[decision.assertion] = decision.reason or ""
            else:
                reasons.pop(decision.assertion, None)
        return reasons
