"""MCP wiring. All logic lives in tools.py; this module only exposes it."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Literal

import anyio.to_thread

# The installed `mcp` package (2.x) no longer ships `mcp.server.fastmcp`; the
# server class now lives at `mcp.server.mcpserver.MCPServer`. Its decorator
# and `list_tools()` behave the same way the brief assumed FastMCP's would
# (docstring becomes the description, `list_tools()` is async), so only the
# import path changes here.
from mcp.server.mcpserver import Context, MCPServer
from mcp_types import ClientCapabilities, ElicitationCapability
from pydantic import BaseModel

from mindpalace import gate, tools
from mindpalace.session import Session

TOOL_NAMES = (
    "save_capture",
    "ingest_file",
    "write_note",
    "local_search",
    "global_search",
    "read",
    "neighbors",
    "get_entity",
    "graph_stats",
    "propose_relationship",
    "resolve_assertion",
    "adopt_type",
    "merge_entities",
    "retire_entity",
    "review_queue",
    "review",
    "cluster",
    "write_community_report",
    "write_entity_description",
    "rebuild",
)


def _run(session: Session, fn, *args, **kwargs):
    """Serialise one tool dispatch against the session's shared connection.

    `server.py` is the only place a transport-originated call enters the
    vault, which makes it the right (and only necessary) place to take
    `session.lock`: the MCP runtime dispatches every synchronous tool body
    onto an anyio worker thread, and `tools/call` requests are not processed
    inline (see `session.lock`'s docstring and `mindpalace/index/db.py`), so
    two pipelined calls can otherwise run concurrently against one
    `sqlite3.Connection` mid-transaction. This is concurrency plumbing, not
    business logic, so it belongs here rather than in `tools.py`.
    """
    with session.lock:
        return fn(session, *args, **kwargs)


class _Proceed(BaseModel):
    """Accept means yes; the form has nothing to fill in."""


class _Decision(BaseModel):
    decision: Literal["confirm", "dismiss", "skip", "stop"]
    reason: str | None = None


class McpAsker:
    """An `Asker` over MCP elicitation (docs/decisions/0005): the client puts
    the question to the user, and the assistant never sees it until it is
    answered. An error from the client -- a timeout, a closed dialog --
    counts as no."""

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx

    def can_ask(self) -> bool:
        try:
            session = self.ctx.request_context.session
        except ValueError:
            return False  # no live request: nobody to ask
        return session.check_client_capability(
            ClientCapabilities(elicitation=ElicitationCapability())
        )

    async def confirm(self, message: str) -> bool:
        try:
            answer = await self.ctx.elicit(message, _Proceed)
        except Exception:
            return False
        return answer.action == "accept"

    async def choose(self, message: str) -> tuple[str, str | None]:
        try:
            answer = await self.ctx.elicit(message, _Decision)
        except Exception:
            return "stop", None
        if answer.action != "accept":
            return "stop", None
        return answer.data.decision, answer.data.reason


def build_server(
    session: Session, asker_for: Callable[[Context], gate.Asker] = McpAsker
) -> MCPServer:
    server = MCPServer("mindpalace")

    async def call(fn):
        """Run one locked unit of gated work off the event loop, so no lock is
        held while `gate` waits on a person (docs/decisions/0005 §Part 2)."""
        return await anyio.to_thread.run_sync(_run, session, fn)

    @server.tool(name="save_capture")
    def _save_capture(text: str, why: str | None = None, source: str = "manual") -> dict:
        """Save a thought verbatim and return the nearest existing material,
        the known entities, and what to extract next."""
        return _run(session, tools.save_capture, text, why, source)

    @server.tool(name="ingest_file")
    def _ingest_file(path: str, why: str | None = None, source: str = "file") -> dict:
        """Ingest a local file (PDF, DOCX, PPTX, XLSX/XLS/ODS, CSV/TSV, HTML,
        Markdown, text). Stores the original, renders it to a capture, splits
        it into text units, and returns a preview plus what to extract next.
        A file already in the vault returns its existing capture."""
        return _run(session, tools.ingest_file, path, why, source)

    @server.tool(name="write_note")
    def _write_note(
        derived_from: str,
        content: str,
        entities: list[dict] | None = None,
        relationship_assertions: list[dict] | None = None,
        claim_assertions: list[dict] | None = None,
    ) -> dict:
        """Record your analysis of a capture plus the entities, relationships,
        and claims you extracted. All assertions enter as proposed. A type
        outside the vocabulary is kept as a proposal, not refused; a bad item
        is dropped with a reason and the rest is written. A claim may carry
        valid_from / valid_to (YYYY, YYYY-MM, YYYY-MM-DD; valid_to may be
        "unknown") and supersedes (a claim id it corrects). Show the user the
        returned `landed` line, not what you intended."""
        return _run(
            session,
            tools.write_note,
            derived_from,
            content,
            entities or [],
            relationship_assertions or [],
            claim_assertions or [],
        )

    @server.tool(name="local_search")
    def _local_search(query: str, k: int = 8, expand_graph: bool = False) -> dict:
        """Hybrid keyword and semantic search over captures, notes, and entities.
        Abstains when the vault holds nothing relevant."""
        return _run(session, tools.search_local, query, k, expand_graph)

    @server.tool(name="global_search")
    def _global_search(query: str) -> dict:
        """Answer a question about themes using community reports. Explains
        itself instead of failing when clustering is not yet active."""
        return _run(session, tools.search_global, query)

    @server.tool(name="read")
    def _read(identifier: str) -> dict:
        """Read any capture, note, entity page, or community report by id."""
        return _run(session, tools.read, identifier)

    @server.tool(name="neighbors")
    def _neighbors(
        identifier: str, depth: int = 1, edge_types: list[str] | None = None
    ) -> dict:
        """Traverse confirmed relationships out from a node, grouped by type."""
        return _run(session, tools.neighbors, identifier, depth, edge_types)

    @server.tool(name="get_entity")
    def _get_entity(name: str, as_of: str | None = None) -> dict:
        """Fetch an entity by name, alias, or merged-away name, with its claims
        and neighbours. Pass as_of (YYYY, YYYY-MM, YYYY-MM-DD) to see only the
        claims that held at that date."""
        return _run(session, tools.get_entity, name, as_of)

    @server.tool(name="graph_stats")
    def _graph_stats() -> dict:
        """Counts, orphans, stale prose, clustering distance to threshold, and
        which embedder is active."""
        return _run(session, tools.graph_stats)

    @server.tool(name="propose_relationship")
    def _propose_relationship(
        source: str, target: str, type: str, description: str, strength: int = 5
    ) -> dict:
        """Propose a link spotted outside extraction. Enters as proposed. An
        unknown type is kept as a proposal; a signature violation is swapped
        (direction_corrected) or untyped, never silently accepted."""
        return _run(
            session,
            tools.propose_relationship,
            source,
            target,
            type,
            description,
            strength,
        )

    @server.tool(name="resolve_assertion")
    def _resolve_assertion(
        identifier: str, action: str, reason: str | None = None
    ) -> dict:
        """Dismiss a proposed relationship or claim, or reopen a decided one so
        it is proposed again. You cannot confirm: call `review` so the user
        decides, or they confirm in Obsidian or with `mindpalace review`."""
        return _run(session, tools.resolve_assertion, identifier, action, reason)

    @server.tool(name="adopt_type")
    async def _adopt_type(
        kind: str,
        proposed: str,
        name: str,
        ctx: Context,
        directed: bool | None = None,
        cluster_weight: float = 1.0,
        domain: list[str] | None = None,
        range: list[str] | None = None,
        action: str = "adopt",
    ) -> dict:
        """Adopt a proposed wording (see review_queue's `vocabulary`) into the
        vocabulary as edge or entity type `name`, adding it to MINDPALACE.md
        if new (a new edge type needs `directed`). The user is asked to
        approve first; nothing is written if they decline. action="revoke"
        withdraws the mapping."""
        args = dict(kind=kind, proposed=proposed, name=name, directed=directed,
                    cluster_weight=cluster_weight, domain=domain, range=range, action=action)
        return await gate.approve(
            asker_for(ctx), call,
            functools.partial(tools.describe_adopt_type, **args),
            functools.partial(tools.adopt_type, **args, via="chat_approval"),
        )

    @server.tool(name="merge_entities")
    async def _merge_entities(
        duplicate: str,
        canonical: str,
        ctx: Context,
        action: str = "merge",
        reason: str | None = None,
    ) -> dict:
        """Record that two entities are one (action="merge", reversible with
        "unmerge") or that a flagged pair is genuinely different ("keep").
        The user is asked to approve first; nothing is written if they
        decline. Logged, never applied to note files."""
        args = dict(duplicate=duplicate, canonical=canonical, action=action, reason=reason)
        return await gate.approve(
            asker_for(ctx), call,
            functools.partial(tools.describe_merge_entities, **args),
            functools.partial(tools.merge_entities, **args, via="chat_approval"),
        )

    @server.tool(name="retire_entity")
    async def _retire_entity(
        slug: str, ctx: Context, action: str = "retire", reason: str | None = None
    ) -> dict:
        """Record that a slug is not an entity (action="retire", reversible
        with "restore"). Refused while any proposed or confirmed relationship
        or claim names it. The user is asked to approve first; nothing is
        written if they decline."""
        args = dict(slug=slug, action=action, reason=reason)
        return await gate.approve(
            asker_for(ctx), call,
            functools.partial(tools.describe_retire_entity, **args),
            functools.partial(tools.retire_entity, **args, via="chat_approval"),
        )

    @server.tool(name="review_queue")
    def _review_queue(limit: int = 20) -> dict:
        """Pending proposals, vocabulary proposals (wordings with no type yet),
        recent extraction drops, and vault issues, in separate sections."""
        return _run(session, tools.review_queue, limit)

    @server.tool(name="review")
    async def _review(ctx: Context, limit: int = 20) -> dict:
        """Ask the user whether to review pending proposals now and, if they
        agree, put each one to them to confirm, dismiss, or skip. The user
        answers, not you; you receive only the outcome. Call this when the
        user wants to review or when proposals need confirming."""
        return await gate.run_review(asker_for(ctx), call, via="chat_review", limit=limit)

    @server.tool(name="cluster")
    def _cluster(force: bool = False) -> dict:
        """Partition the graph into communities and return those needing
        reports. Refuses below the activation threshold unless forced."""
        return _run(session, tools.cluster_tool, force)

    @server.tool(name="write_community_report")
    def _write_community_report(
        lineage_id: str,
        title: str,
        summary: str,
        rank: float,
        findings: list[dict],
        cites: list[str],
    ) -> dict:
        """Store a community report. Rejected if any citation does not resolve."""
        return _run(
            session,
            tools.write_community_report,
            lineage_id,
            title,
            summary,
            rank,
            findings,
            cites,
        )

    @server.tool(name="write_entity_description")
    def _write_entity_description(slug: str, description: str) -> dict:
        """Store an entity's aggregated description and clear its stale flag."""
        return _run(session, tools.write_entity_description, slug, description)

    @server.tool(name="rebuild")
    def _rebuild(scope: str = "all") -> dict:
        """Regenerate the cache and mark generated prose stale where its inputs
        moved. Never deletes or rewrites prose."""
        return _run(session, tools.rebuild_tool, scope)

    return server


def main(argv: list[str] | None = None) -> int:
    """The `mindpalace` console script; see mindpalace.cli."""
    from mindpalace.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
