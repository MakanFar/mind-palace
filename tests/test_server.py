import asyncio

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
