import asyncio
import json

import pytest

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


def call(session, name: str, arguments: dict | None = None) -> dict:
    """Round-trip a call through the real MCPServer object, not tools.py directly.

    `list_tools()` alone only proves a name and a docstring got registered — it
    would pass just as happily if a tool body called the wrong function or
    passed arguments in the wrong order. Actually invoking `call_tool` exercises
    the decorator, the argument binding, and the delegation into `tools.py` in
    one pass, and decodes the MCP JSON content back into a plain dict so the
    assertions read like a normal tool-return check.
    """
    server = build_server(session)
    result = asyncio.run(server.call_tool(name, arguments or {}))
    assert result.is_error is not True, result.content
    return json.loads(result.content[0].text)


def test_all_fifteen_tools_are_registered(session):
    assert registered_names(session) == set(TOOL_NAMES)
    assert len(TOOL_NAMES) == 15


def test_every_tool_has_a_description(session):
    server = build_server(session)
    for tool in asyncio.run(server.list_tools()):
        assert tool.description, f"{tool.name} has no description"


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
    assert payload == {"proposals": [], "vault_issues": []}


def test_local_search_round_trips_through_the_real_server(session):
    payload = call(session, "local_search", {"query": "anything", "k": 3})
    assert payload["hits"] == []
    assert "note" in payload
