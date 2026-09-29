"""The human gate (docs/decisions/0005).

Structure-adding decisions -- confirm, adopt, merge, retire -- pass through
here, and only a person may answer the questions this module asks. No MCP
types: the chat surface (server.py) and the terminal (cli.py) each supply an
`Asker` and a `call` that runs a locked unit of work, so both share one flow
and neither holds a lock while a person is thinking.
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

from mindpalace import tools
from mindpalace.session import Session
from mindpalace.tools import ToolError

T = TypeVar("T")
Call = Callable[[Callable[[Session], T]], Awaitable[T]]

CANNOT_ASK = (
    "this client cannot ask the user directly, so nothing that needs their "
    "decision can happen here; they can confirm in the Obsidian window or with "
    "`mindpalace review`"
)
DECLINED = "declined by the user; nothing was written"
DECLINED_REVIEW_NEXT = (
    "The user chose not to review now. Confirmation waits for the Obsidian window "
    "or `mindpalace review`; you may still dismiss or reopen with resolve_assertion."
)


class Asker(Protocol):
    """Puts a question to a person. Never answered by the assistant."""

    def can_ask(self) -> bool: ...

    async def confirm(self, message: str) -> bool: ...

    async def choose(self, message: str) -> tuple[str, str | None]:
        """("confirm" | "dismiss" | "skip" | "stop", optional reason)."""
        ...


def render_proposal(proposal: dict, position: int, total: int) -> str:
    """One proposal as a person reads it: no extra `read` needed."""
    head = f"({position} of {total})"
    if proposal["kind"] == "relationship":
        label = proposal["type"] or f"untyped: {proposal['proposed_type']}"
        lines = [
            f"{head} Relationship: {proposal['source']} —[{label}]"
            f"{'→' if proposal['directed'] else '—'} {proposal['target']}",
            f"Why: {proposal['why']}",
            f"Strength: {proposal['strength']}/10"
            + (" (direction corrected to fit the edge type)" if proposal["direction_corrected"] else ""),
            f"{proposal['source']}: {proposal['source_snippet']}",
            f"{proposal['target']}: {proposal['target_snippet']}",
        ]
    else:
        lines = [f"{head} Claim about {proposal['subject']}: {proposal['text']}"]
        if proposal["valid_from"] or proposal["valid_to"]:
            lines.append(f"Valid: {proposal['valid_from'] or '?'} → {proposal['valid_to'] or 'now'}")
        if proposal["supersedes"]:
            lines.append(f"Supersedes: {proposal['supersedes']}")
        lines.append(f"{proposal['subject']}: {proposal['subject_snippet']}")
    lines.append(f"From note {proposal['note']}")
    return "\n".join(lines)


async def run_review(asker: Asker, call: Call, *, via: str, limit: int = 20) -> dict[str, Any]:
    """The chat review (docs/decisions/0005 §Part 1)."""
    if not asker.can_ask():
        raise ToolError(CANNOT_ASK)
    queue = await call(functools.partial(tools.review_queue, limit=limit))
    proposals = queue["proposals"]
    result: dict[str, Any] = {
        "reviewed": False,
        "pending": len(proposals),
        "confirmed": 0,
        "dismissed": 0,
        "skipped": 0,
        "already_decided": [],
        "stopped_early": False,
        "next": "Nothing is waiting for review.",
    }
    if not proposals:
        return result
    relationships = sum(1 for p in proposals if p["kind"] == "relationship")
    claims = len(proposals) - relationships
    opening = (
        f"{tools._plural(len(proposals), 'proposal')} "
        f"{'is' if len(proposals) == 1 else 'are'} waiting "
        f"({tools._plural(relationships, 'relationship')}, {tools._plural(claims, 'claim')}). "
        f"Review them now?"
    )
    if not await asker.confirm(opening):
        result["next"] = DECLINED_REVIEW_NEXT
        return result

    result["reviewed"] = True
    for position, proposal in enumerate(proposals, start=1):
        decision, reason = await asker.choose(render_proposal(proposal, position, len(proposals)))
        if decision == "stop":
            result["stopped_early"] = True
            break
        if decision == "skip":
            result["skipped"] += 1
            continue
        try:
            written = await call(functools.partial(
                tools.decide, identifier=proposal["id"], action=decision, via=via,
                reason=reason, rebuild=False, require_proposed=True,
            ))
        except ToolError:
            # The item left the vault while the person was reading it.
            written = False
        if written:
            result["confirmed" if decision == "confirm" else "dismissed"] += 1
        else:
            result["already_decided"].append(proposal["id"])
    if result["confirmed"] or result["dismissed"]:
        await call(tools.rebuild_tool)
    result["next"] = (
        "Tell the user what they decided: "
        f"{result['confirmed']} confirmed, {result['dismissed']} dismissed, "
        f"{result['skipped']} skipped."
    )
    return result


async def approve(
    asker: Asker,
    call: Call,
    describe: Callable[[Session], str],
    act: Callable[[Session], dict],
) -> dict:
    """One gated action (docs/decisions/0005 §Part 2): validate and describe,
    ask with no lock held, then act -- which validates again, because the
    vault may have moved while the person was reading."""
    if not asker.can_ask():
        raise ToolError(CANNOT_ASK)
    summary = await call(describe)
    if not await asker.confirm(summary):
        raise ToolError(DECLINED)
    return await call(act)
