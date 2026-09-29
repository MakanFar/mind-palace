import asyncio
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import mindpalace.server as server_module
from mindpalace import tools
from mindpalace.embed import StubEmbedder
from mindpalace.server import TOOL_NAMES, build_server, main
from mindpalace.session import Session


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def registered_names(session):
    server = build_server(session)
    return {tool.name for tool in asyncio.run(server.list_tools())}


def _server(session, asker=None):
    if asker is None:
        return build_server(session)
    return build_server(session, asker_for=lambda ctx: asker)


def call(session, name: str, arguments: dict | None = None, asker=None) -> dict:
    """Round-trip a call through the real MCPServer object, not tools.py directly.

    `list_tools()` alone only proves a name and a docstring got registered. Actually
    invoking `call_tool` exercises the decorator, the argument binding, and the
    delegation into `tools.py` in one pass. `asker` stands in for the person a gated
    tool asks (docs/decisions/0005).
    """
    result = asyncio.run(_server(session, asker).call_tool(name, arguments or {}))
    assert result.is_error is not True, result.content
    return json.loads(result.content[0].text)


def call_error(session, name: str, arguments: dict | None = None, asker=None) -> str:
    """The refusal a tool call produces. In-process, `MCPServer.call_tool`
    raises the error that the protocol layer would wrap as `isError`."""
    from mcp.server.mcpserver.exceptions import ToolError as McpToolError

    with pytest.raises(McpToolError) as refused:
        asyncio.run(_server(session, asker).call_tool(name, arguments or {}))
    return str(refused.value)


def test_all_tools_are_registered(session):
    assert registered_names(session) == set(TOOL_NAMES)
    assert len(TOOL_NAMES) == 20


def test_every_tool_has_a_description(session):
    server = build_server(session)
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description"


def test_importing_the_server_does_not_import_graspologic():
    """Startup cost is a correctness problem, not a nicety: an MCP client
    gives the server a fixed window (30s in Claude Code) to answer
    `initialize`, and `import mindpalace.server` spent ~12s of it, ~7s of
    that in graspologic -> umap -> pynndescent -> numba. Clustering is one
    tool out of fifteen and most sessions never call it, so that import
    belongs at first use. A subprocess, because pytest has already imported
    half the world.
    """
    probe = "import sys, mindpalace.server; print('graspologic' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_main_requires_a_vault_argument(capsys):
    with pytest.raises(SystemExit):
        main([])


def test_main_reports_a_refused_vault_without_traceback(tmp_path, capsys):
    (tmp_path / "unrelated.txt").write_text("hello")
    assert main(["--vault", str(tmp_path), "--init", "--check"]) == 2
    assert "not a Mind Palace vault" in capsys.readouterr().err


def test_main_check_succeeds_on_a_fresh_vault(tmp_path):
    assert main(["--vault", str(tmp_path), "--init", "--check"]) == 0


def test_graph_stats_round_trips_through_the_real_server(session):
    payload = call(session, "graph_stats")
    assert payload["entities"] == 0
    assert payload["clustering"]["active"] is False


def test_review_queue_round_trips_through_the_real_server(session):
    payload = call(session, "review_queue", {"limit": 5})
    assert payload == {
        "proposals": [],
        "vocabulary": [],
        "drops": [],
        "vault_issues": [],
    }


def test_local_search_round_trips_through_the_real_server(session):
    payload = call(session, "local_search", {"query": "anything", "k": 3})
    assert payload["hits"] == []
    assert "note" in payload


def test_concurrent_tool_calls_do_not_corrupt_the_session(session):
    """Two `tools/call` requests pipelined by one client are not processed
    inline (see `session.lock`'s docstring) -- `anyio.to_thread.run_sync`
    can genuinely run two tool bodies on two different worker threads at the
    same time, both sharing the one `sqlite3.Connection` on `session.conn`.

    This does not try to catch the race in the act (that would be flaky by
    construction); it drives real contention through real threads via the
    real `MCPServer` object and then checks the vault landed in a fully
    consistent state -- every concurrent write present exactly once, cache
    and filesystem in agreement. Without `session.lock` serialising access in
    `server.py`'s `_run`, `save_capture`'s unwrapped `write_capture` +
    `resync()` sequence (see `tools.save_capture`) is exactly the kind of
    multi-statement operation the review flagged as vulnerable to another
    thread's implicit transaction committing it early or interleaving with
    it -- so a lost or duplicated capture here is a real signal, not noise.
    """
    server = build_server(session)

    def write(i: int):
        return asyncio.run(
            server.call_tool("save_capture", {"text": f"concurrent capture {i}"})
        )

    def read():
        return asyncio.run(server.call_tool("graph_stats", {}))

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write, i) for i in range(12)]
        futures += [pool.submit(read) for _ in range(12)]
        results = [future.result(timeout=30) for future in futures]

    assert all(result.is_error is not True for result in results), results

    capture_files = list(session.paths.captures.glob("*.md"))
    assert len(capture_files) == 12
    doc_count = session.conn.execute(
        "SELECT COUNT(*) FROM docs WHERE kind = 'capture'"
    ).fetchone()[0]
    assert doc_count == 12
    assert session.oplog.pending() == []


def test_the_session_lock_is_held_for_a_tool_calls_full_duration(session, monkeypatch):
    """Assert the lock's actual behaviour -- held while a tool body runs,
    released once it returns -- using `threading.Event`s to make the timing
    deterministic instead of racing on `sleep`.
    """
    started = threading.Event()
    release = threading.Event()
    original = tools.graph_stats

    def slow_graph_stats(session_arg):
        started.set()
        assert release.wait(timeout=5), "test deadlocked waiting to be released"
        return original(session_arg)

    monkeypatch.setattr(tools, "graph_stats", slow_graph_stats)
    server = build_server(session)

    def call_in_background():
        return asyncio.run(server.call_tool("graph_stats", {}))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(call_in_background)
        assert started.wait(timeout=5), "tool body never started"

        # The tool body is mid-flight in the pool's worker thread; the lock
        # it took in `_run` must still be held, so a non-blocking acquire
        # attempt from this thread must fail.
        assert session.lock.acquire(blocking=False) is False

        release.set()
        result = future.result(timeout=5)

    assert result.is_error is not True

    # And now that the call has returned, the lock must be free again.
    assert session.lock.acquire(blocking=False) is True
    session.lock.release()


def test_adopt_and_merge_round_trip_through_the_real_server(session, scripted):
    capture = call(session, "save_capture", {"text": "Star Wars is on GeForce Now."})
    note = call(
        session,
        "write_note",
        {
            "derived_from": capture["id"],
            "content": "Availability note.",
            "entities": [
                {"name": "star-wars", "type": "concept", "description": "game"},
                {"name": "starwars", "type": "concept", "description": "game"},
            ],
            "relationship_assertions": [
                {"source": "star-wars", "target": "geforce-now", "type": "available on",
                 "description": "playable there"}
            ],
        },
    )
    assert note["landed"].startswith("2 entities, 1 relationship (1 untyped)")
    queue = call(session, "review_queue", {"limit": 5})
    assert queue["vocabulary"][0]["proposed"] == "available-on"
    adopted = call(
        session,
        "adopt_type",
        {"kind": "edge", "proposed": "available on", "name": "available-on", "directed": True},
        asker=scripted(confirms=[True]),
    )
    assert adopted["retyped"] == [note["relationship_assertions"][0]["id"]]
    merged = call(
        session, "merge_entities", {"duplicate": "starwars", "canonical": "star-wars"},
        asker=scripted(confirms=[True]),
    )
    assert merged["status"] == "merged"
    entity = call(session, "get_entity", {"name": "starwars", "as_of": "2026"})
    assert entity["slug"] == "star-wars"
    assert entity["merged_from"] == ["starwars"]


def test_ingest_file_round_trips_through_the_real_server(session, tmp_path):
    src = tmp_path / "doc.txt"
    src.write_text("hello from a file")
    payload = call(session, "ingest_file", {"path": str(src)})
    assert payload["id"].startswith("c_") and payload["units"] == 1


# ---- docs/decisions/0005: the gate on the chat surface --------------------

from types import SimpleNamespace  # noqa: E402

from mindpalace.server import McpAsker  # noqa: E402


def _proposal(session):
    capture = call(session, "save_capture", {"text": "Scaling hits a data wall."})
    note = call(session, "write_note", {
        "derived_from": capture["id"], "content": "n",
        "relationship_assertions": [{"source": "a", "target": "b", "type": "supports",
                                     "description": "because"}],
    })
    return note["relationship_assertions"][0]["id"]


def test_resolve_assertion_refuses_confirm_through_the_server(session):
    assertion = _proposal(session)
    assert "review" in call_error(session, "resolve_assertion",
                                  {"identifier": assertion, "action": "confirm"})


def test_review_is_refused_when_the_client_cannot_ask(session):
    _proposal(session)
    # The default asker over a Context with no live request: no elicitation capability.
    assert "cannot ask" in call_error(session, "review")


def test_review_confirms_through_the_server(session, scripted):
    assertion = _proposal(session)
    result = call(session, "review", {}, asker=scripted(confirms=[True], choices=[("confirm", None)]))
    assert result["confirmed"] == 1
    [entry] = session.decisions.entries()
    assert (entry.assertion, entry.via) == (assertion, "chat_review")


@pytest.mark.parametrize("tool, arguments", [
    ("merge_entities", {"duplicate": "a", "canonical": "b"}),
    ("retire_entity", {"slug": "a"}),
    ("adopt_type", {"kind": "edge", "proposed": "inspires", "name": "supports"}),
])
def test_gated_tools_write_nothing_when_declined(session, scripted, tool, arguments):
    capture = call(session, "save_capture", {"text": "x"})
    call(session, "write_note", {
        "derived_from": capture["id"], "content": "n",
        "entities": [{"name": "a", "type": "concept", "description": "d"},
                     {"name": "b", "type": "concept", "description": "d"}],
    })
    assert "declined" in call_error(session, tool, arguments, asker=scripted(confirms=[False]))
    assert session.merges.merges() == {}
    assert session.retirements.retired() == set()
    assert session.vocabulary.adoptions()["edge"] == {}


def test_a_gated_tool_records_chat_approval(session, scripted):
    capture = call(session, "save_capture", {"text": "x"})
    call(session, "write_note", {
        "derived_from": capture["id"], "content": "n",
        "entities": [{"name": "loose", "type": "concept", "description": "d"}],
    })
    asker = scripted(confirms=[True])
    call(session, "retire_entity", {"slug": "loose", "reason": "noise"}, asker=asker)
    assert asker.messages == ["Retire `loose`: it leaves the graph; its page stays. Reason: noise"]
    line = json.loads(session.paths.retirements_log.read_text().splitlines()[-1])
    assert line["via"] == "chat_approval"


def test_no_lock_is_held_while_the_user_is_asked(session, scripted):
    _proposal(session)
    free = []

    def probe(message):
        # A worker thread stands in for a second tool call arriving mid-prompt.
        # It must release what it grabs itself: an RLock belongs to its taker.
        grabbed = []

        def worker():
            got = session.lock.acquire(timeout=1)
            grabbed.append(got)
            if got:
                session.lock.release()

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        free.append(grabbed[0])

    call(session, "review", {}, asker=scripted(confirms=[True], choices=[("stop", None)], on_ask=probe))
    assert free == [True, True]


def test_an_elicitation_error_counts_as_a_decline():
    async def broken(message, schema):
        raise RuntimeError("client timed out")

    ctx = SimpleNamespace(
        elicit=broken,
        request_context=SimpleNamespace(
            session=SimpleNamespace(check_client_capability=lambda capability: True)
        ),
    )
    asker = McpAsker(ctx)
    assert asker.can_ask() is True
    assert asyncio.run(asker.confirm("Proceed?")) is False
    assert asyncio.run(asker.choose("Decide")) == ("stop", None)


def test_mcp_asker_maps_the_answers():
    from mcp.server.elicitation import AcceptedElicitation, DeclinedElicitation

    answers = []

    async def elicit(message, schema):
        return answers.pop(0)

    asker = McpAsker(SimpleNamespace(elicit=elicit))
    answers[:] = [AcceptedElicitation(data=server_module._Proceed()), DeclinedElicitation()]
    assert asyncio.run(asker.confirm("?")) is True
    assert asyncio.run(asker.confirm("?")) is False
    answers[:] = [AcceptedElicitation(data=server_module._Decision(decision="dismiss", reason="no"))]
    assert asyncio.run(asker.choose("?")) == ("dismiss", "no")

