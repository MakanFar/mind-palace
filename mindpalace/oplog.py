"""Append-only logs: write-ahead operations and durable review decisions."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from mindpalace.ids import new_id, slugify
from mindpalace.models import Decision


def now_iso() -> str:
    """UTC, second precision is not promised, `Z` suffix: the `ts` shape every
    log line and the graph export carry."""
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

    lines = path.read_text(encoding="utf-8").splitlines()
    records = []

    for i, line in enumerate(lines):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Malformed final line is a crash artifact from incomplete write/flush/fsync.
            # Skip it; the record was never durably committed anyway.
            if i == len(lines) - 1:
                continue
            # Malformed line in the middle indicates real corruption, not a torn write.
            # Raise loudly with context so it doesn't fail silently.
            raise ValueError(
                f"Malformed JSON in {path} at line {i + 1}: {line!r}"
            ) from None

    return records


class OpLog:
    """Write-ahead log. An op with a begin and no commit is replayed on startup."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def begin(self, intent: dict) -> str:
        op_id = new_id("op_")
        _append_line(
            self.path, {"kind": "op.begin", "op": op_id, "ts": now_iso(), "intent": intent}
        )
        return op_id

    def commit(self, op_id: str) -> None:
        _append_line(self.path, {"kind": "op.commit", "op": op_id, "ts": now_iso()})

    def pending(self) -> list[dict]:
        begun: dict[str, dict] = {}
        for record in _read_lines(self.path):
            if record["kind"] == "op.begin":
                begun[record["op"]] = record
            elif record["kind"] == "op.commit":
                begun.pop(record["op"], None)
        return list(begun.values())


# "reopen" puts an assertion back under review: the Obsidian window's undo.
VALID_ACTIONS = {"confirm", "dismiss", "reopen"}


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
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"invalid decision action {action!r} for assertion "
                f"{assertion_id!r} (expected one of {sorted(VALID_ACTIONS)}). "
                f"The decision log is one shared file for the whole vault -- "
                f"an invalid action written here would block every future "
                f"rebuild with no way to quarantine just this line, so it is "
                f"rejected before it can be durably written."
            )
        _append_line(
            self.path,
            {
                "op": op_id,
                "ts": now_iso(),
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


VOCABULARY_KINDS = {"edge", "entity"}
VOCABULARY_ACTIONS = {"adopt", "revoke"}


class VocabularyLog:
    """Adoptions of proposed types into the vocabulary (docs/decisions/0001 §1).

    Its own file, not a record shape inside `decisions.jsonl`: forgetting to
    read this log leaves proposals unadopted, which is visible in
    `review_queue`; a malformed record in the decisions file would block
    every rebuild. Utopia's 0015 makes the same argument for `pending_facts`.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(
        self,
        kind: str,
        proposed: str,
        adopted: str,
        action: str,
        via: str,
        op_id: str,
    ) -> None:
        if kind not in VOCABULARY_KINDS:
            raise ValueError(
                f"invalid vocabulary kind {kind!r} (expected one of "
                f"{sorted(VOCABULARY_KINDS)})"
            )
        if action not in VOCABULARY_ACTIONS:
            raise ValueError(
                f"invalid vocabulary action {action!r} (expected one of "
                f"{sorted(VOCABULARY_ACTIONS)})"
            )
        _append_line(
            self.path,
            {
                "op": op_id,
                "ts": now_iso(),
                "kind": kind,
                "proposed": slugify(proposed),
                "adopted": adopted,
                "action": action,
                "via": via,
            },
        )

    def adoptions(self) -> dict[str, dict[str, str]]:
        """kind -> {slugified proposal -> adopted type name}. Last write wins."""
        result: dict[str, dict[str, str]] = {kind: {} for kind in VOCABULARY_KINDS}
        for record in _read_lines(self.path):
            table = result.setdefault(record["kind"], {})
            if record["action"] == "adopt":
                table[record["proposed"]] = record["adopted"]
            else:
                table.pop(record["proposed"], None)
        return result


MERGE_ACTIONS = {"merge", "unmerge", "keep"}


class MergeLog:
    """Durable identity decisions (docs/decisions/0001 §5).

    `merge` folds `duplicate` into `canonical` on every subsequent rebuild;
    `unmerge` reverses it; `keep` records that the two are different so the
    duplicate lint stops proposing them. All three are the same shape so a
    reader of the file sees the whole history of a pair in one place.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(
        self,
        duplicate: str,
        canonical: str,
        action: str,
        via: str,
        op_id: str,
        reason: str | None = None,
    ) -> None:
        if action not in MERGE_ACTIONS:
            raise ValueError(
                f"invalid merge action {action!r} (expected one of "
                f"{sorted(MERGE_ACTIONS)})"
            )
        _append_line(
            self.path,
            {
                "op": op_id,
                "ts": now_iso(),
                "duplicate": slugify(duplicate),
                "canonical": slugify(canonical),
                "action": action,
                "via": via,
                "reason": reason,
            },
        )

    def _fold(self) -> tuple[dict[str, str], set[tuple[str, str]]]:
        merges: dict[str, str] = {}
        kept: set[tuple[str, str]] = set()
        for record in _read_lines(self.path):
            duplicate, canonical = record["duplicate"], record["canonical"]
            pair = tuple(sorted((duplicate, canonical)))
            if record["action"] == "merge":
                merges[duplicate] = canonical
                kept.discard(pair)
            elif record["action"] == "unmerge":
                merges.pop(duplicate, None)
            else:
                kept.add(pair)
                # "These two are different" withdraws a merge *of these two*
                # only; a merge of either into some third entity stands.
                if merges.get(duplicate) == canonical:
                    merges.pop(duplicate)
                if merges.get(canonical) == duplicate:
                    merges.pop(canonical)
        return merges, kept

    def merges(self) -> dict[str, str]:
        """duplicate slug -> canonical slug, one hop; `fold` follows chains."""
        return self._fold()[0]

    def kept(self) -> set[tuple[str, str]]:
        return self._fold()[1]


RETIREMENT_ACTIONS = {"retire", "restore"}


class RetirementLog:
    """Durable "this is not an entity" decisions (docs/decisions/0004 §Part 2).

    `retire` drops the slug from every subsequent fold while nothing live
    names it; `restore` reverses it. Last write wins per slug, the same
    shape as `MergeLog`, and the Obsidian window appends lines of exactly
    this shape.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(
        self,
        slug: str,
        action: str,
        via: str,
        op_id: str,
        reason: str | None = None,
    ) -> None:
        if action not in RETIREMENT_ACTIONS:
            raise ValueError(
                f"invalid retirement action {action!r} (expected one of "
                f"{sorted(RETIREMENT_ACTIONS)})"
            )
        _append_line(
            self.path,
            {
                "op": op_id,
                "ts": now_iso(),
                "slug": slugify(slug),
                "action": action,
                "via": via,
                "reason": reason,
            },
        )

    def retired(self) -> set[str]:
        retired: set[str] = set()
        for record in _read_lines(self.path):
            if record["action"] == "retire":
                retired.add(record["slug"])
            else:
                retired.discard(record["slug"])
        return retired
