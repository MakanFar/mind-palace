"""MCP wiring. All logic lives in tools.py; this module only exposes it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The installed `mcp` package (2.x) no longer ships `mcp.server.fastmcp`; the
# server class now lives at `mcp.server.mcpserver.MCPServer`. Its decorator
# and `list_tools()` behave the same way the brief assumed FastMCP's would
# (docstring becomes the description, `list_tools()` is async), so only the
# import path changes here.
from mcp.server.mcpserver import MCPServer

from mindpalace import tools
from mindpalace.config import ConfigError
from mindpalace.embed import EmbedderError
from mindpalace.session import Session, VaultLockedError

TOOL_NAMES = (
    "save_capture",
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
    "review_queue",
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


def build_server(session: Session) -> MCPServer:
    server = MCPServer("mindpalace")

    @server.tool(name="save_capture")
    def _save_capture(text: str, why: str | None = None, source: str = "manual") -> dict:
        """Save a thought verbatim and return the nearest existing material,
        the known entities, and what to extract next."""
        return _run(session, tools.save_capture, text, why, source)

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
        """Confirm or dismiss a proposed relationship or claim. Call this only
        on the user's explicit instruction; the decision is logged with your
        stated reason."""
        return _run(session, tools.resolve_assertion, identifier, action, reason)

    @server.tool(name="adopt_type")
    def _adopt_type(
        kind: str,
        proposed: str,
        name: str,
        directed: bool | None = None,
        cluster_weight: float = 1.0,
        domain: list[str] | None = None,
        range: list[str] | None = None,
        action: str = "adopt",
    ) -> dict:
        """Adopt a proposed wording (see review_queue's `vocabulary`) into the
        vocabulary as edge or entity type `name`, adding it to MINDPALACE.md
        if new (a new edge type needs `directed`). Every assertion carrying
        that wording is retyped on the next fold; note files are untouched.
        action="revoke" withdraws the mapping. Call on the user's instruction."""
        return _run(
            session,
            tools.adopt_type,
            kind,
            proposed,
            name,
            directed,
            cluster_weight,
            domain,
            range,
            action,
        )

    @server.tool(name="merge_entities")
    def _merge_entities(
        duplicate: str, canonical: str, action: str = "merge", reason: str | None = None
    ) -> dict:
        """Record that two entities are one (action="merge": duplicate folds
        into canonical on every rebuild, reversible with "unmerge") or that a
        flagged pair is genuinely different ("keep", so the similarity lint
        stops asking). Logged, never applied to note files. Call on the
        user's instruction."""
        return _run(session, tools.merge_entities, duplicate, canonical, action, reason)

    @server.tool(name="review_queue")
    def _review_queue(limit: int = 20) -> dict:
        """Pending proposals, vocabulary proposals (wordings with no type yet),
        recent extraction drops, and vault issues, in separate sections."""
        return _run(session, tools.review_queue, limit)

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
    parser = argparse.ArgumentParser(prog="mindpalace")
    parser.add_argument("--vault", required=True, help="path to the vault directory")
    parser.add_argument("--init", action="store_true", help="scaffold a new vault")
    parser.add_argument(
        "--check", action="store_true", help="open and validate, then exit"
    )
    args = parser.parse_args(argv)

    try:
        session = Session(Path(args.vault), init=args.init).open()
    except (ConfigError, VaultLockedError, EmbedderError) as exc:
        # EmbedderError included so a misconfigured embedder is a legible message
        # rather than a traceback out of the MCP entry point.
        print(str(exc), file=sys.stderr)
        return 2

    try:
        if args.check:
            return 0
        build_server(session).run()
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    raise SystemExit(main())
