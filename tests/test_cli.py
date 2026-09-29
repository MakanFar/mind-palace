import io

import pytest

from mindpalace import cli
from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import save_capture, write_note


class TTYInput(io.StringIO):
    def isatty(self):
        return True


def terminal(text: str, tty: bool = True):
    stdin = TTYInput(text) if tty else io.StringIO(text)
    return cli.Terminal(stdin=stdin, stdout=io.StringIO())


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


@pytest.fixture
def two_links(session):
    capture = save_capture(session, "x")
    note = write_note(session, derived_from=capture["id"], content="n", relationship_assertions=[
        {"source": "a", "target": "b", "type": "supports", "strength": 9, "description": "one"},
        {"source": "c", "target": "d", "type": "supports", "strength": 3, "description": "two"},
    ])
    return [a["id"] for a in note["relationship_assertions"]]


def test_cli_review_confirms_with_a_reason_and_dismisses(session, two_links):
    first, second = two_links
    term = terminal("y\nc\nsame thing\nd\n\n")
    assert cli.review_command(session, term) == 0
    assert [(d.assertion, d.action, d.via, d.reason) for d in session.decisions.entries()] == [
        (first, "confirm", "cli", "same thing"), (second, "dismiss", "cli", None),
    ]
    assert "1 confirmed, 1 dismissed, 0 skipped" in term.stdout.getvalue()


def test_cli_review_reasks_on_an_unknown_key(session, two_links):
    term = terminal("y\nx\ns\nq\n")
    assert cli.review_command(session, term) == 0
    assert session.decisions.entries() == []
    assert term.stdout.getvalue().count("[c]onfirm") == 3


def test_cli_review_stops_on_eof_and_keeps_earlier_decisions(session, two_links):
    first, _ = two_links
    term = terminal("y\nc\n\n")  # EOF at the second proposal
    assert cli.review_command(session, term) == 0
    assert [d.assertion for d in session.decisions.entries()] == [first]
    status = session.conn.execute("SELECT status FROM assertions WHERE id = ?", (first,)).fetchone()[0]
    assert status == "confirmed"  # the one rebuild at the end ran


def test_cli_review_refuses_without_a_terminal(session, two_links):
    term = terminal("y\nc\n\nc\n\n", tty=False)
    assert cli.review_command(session, term) == 2
    assert session.decisions.entries() == []


def test_cli_merge_asks_and_records_cli(session):
    capture = save_capture(session, "x")
    write_note(session, derived_from=capture["id"], content="n", entities=[
        {"name": "a", "type": "concept", "description": "d"},
        {"name": "b", "type": "concept", "description": "d"},
    ])
    term = terminal("y\n")
    assert cli.merge_command(session, term, duplicate="a", canonical="b") == 0
    assert "Merge `a` into `b`" in term.stdout.getvalue()
    assert session.merges.merges() == {"a": "b"}


def test_cli_retire_declined_writes_nothing(session):
    capture = save_capture(session, "x")
    write_note(session, derived_from=capture["id"], content="n",
               entities=[{"name": "a", "type": "concept", "description": "d"}])
    term = terminal("n\n")
    assert cli.retire_command(session, term, slug="a") == 1
    assert "declined" in term.stdout.getvalue()
    assert session.retirements.retired() == set()


def test_cli_adopt_refuses_piped_input(session):
    term = terminal("y\n", tty=False)
    assert cli.adopt_command(session, term, kind="edge", proposed="x", name="supports") == 2


def test_bare_vault_flag_still_means_serve(tmp_path, capsys):
    (tmp_path / "unrelated.txt").write_text("hello")
    assert cli.main(["--vault", str(tmp_path), "--init", "--check"]) == 2
    assert "not a Mind Palace vault" in capsys.readouterr().err


def test_subcommands_require_a_vault():
    for command in ("review", "merge", "retire", "adopt", "serve"):
        with pytest.raises(SystemExit):
            cli.main([command])
