import pytest

from mindpalace.embed import StubEmbedder
from mindpalace.session import Session
from mindpalace.tools import ToolError, save_capture, write_note


@pytest.fixture
def session(tmp_path):
    with Session(tmp_path, init=True, embedder=StubEmbedder()) as opened:
        yield opened


def test_save_capture_writes_a_file_and_returns_its_id(session):
    result = save_capture(session, "The plateau is about data exhaustion.")
    assert result["id"].startswith("c_")
    assert (session.paths.root / result["path"]).exists()


def test_save_capture_payload_carries_the_vocabularies(session):
    result = save_capture(session, "A thought.")
    assert result["entity_types"] == session.config.entity_types
    assert result["edge_vocabulary"]["contradicts"] == "symmetric"
    assert result["edge_vocabulary"]["supports"] == "directed"


def test_save_capture_payload_carries_the_extraction_instruction(session):
    result = save_capture(session, "A thought.")
    assert "Proposing nothing is a valid outcome" in result["next"]


def test_save_capture_finds_nearest_existing_material(session):
    save_capture(session, "sourdough starter hydration ratios")
    write_note(
        session,
        derived_from=save_capture(session, "the plateau is data exhaustion")["id"],
        content="Scaling limits come from data supply, not architecture.",
    )
    result = save_capture(session, "data exhaustion and scaling limits")
    assert any("scaling" in hit["snippet"].lower() for hit in result["nearest"])


def test_two_captures_in_the_same_minute_both_survive(session):
    first = save_capture(session, "first thought")
    second = save_capture(session, "second thought")
    assert first["path"] != second["path"]
    assert (session.paths.root / first["path"]).exists()
    assert (session.paths.root / second["path"]).exists()


def test_write_note_assigns_assertion_ids_and_leaves_them_proposed(session):
    capture = save_capture(session, "The plateau is data exhaustion.")
    result = write_note(
        session,
        derived_from=capture["id"],
        content="Data supply is the binding constraint.",
        entities=[{"name": "scaling-laws", "type": "concept", "description": "…"}],
        relationship_assertions=[
            {
                "source": "scaling-laws",
                "target": "data-exhaustion",
                "type": "contradicts",
                "strength": 8,
                "description": "because.",
            }
        ],
    )
    [assertion] = result["relationship_assertions"]
    assert assertion["id"].startswith("x_")
    assert assertion["status"] == "proposed"


def test_write_note_rejects_an_unknown_edge_type(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="invented"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            relationship_assertions=[
                {
                    "source": "a",
                    "target": "b",
                    "type": "invented",
                    "strength": 5,
                    "description": "x",
                }
            ],
        )


def test_write_note_rejects_an_unknown_entity_type(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="teapot"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            entities=[{"name": "a", "type": "teapot", "description": "x"}],
        )


def test_write_note_rejects_a_missing_capture(session):
    with pytest.raises(ToolError, match="c_nope"):
        write_note(session, derived_from="c_nope", content="Body.")


def test_write_note_creates_entity_pages(session):
    capture = save_capture(session, "A thought.")
    write_note(
        session,
        derived_from=capture["id"],
        content="Body.",
        entities=[{"name": "Scaling Laws", "type": "concept", "description": "…"}],
    )
    assert session.store.read_entity_page("scaling-laws") is not None


def test_save_capture_rejects_empty_text(session):
    with pytest.raises(ToolError, match="empty"):
        save_capture(session, "   ")


def test_write_note_rejects_empty_content(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="content is empty"):
        write_note(session, derived_from=capture["id"], content="  ")


def test_write_note_rejects_a_name_that_normalises_to_nothing(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="empty slug"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            entities=[{"name": "!!!", "type": "concept", "description": "x"}],
        )


def test_write_note_rejects_out_of_range_strength(session):
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="1-10"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            relationship_assertions=[
                {
                    "source": "a",
                    "target": "b",
                    "type": "contradicts",
                    "strength": 99,
                    "description": "x",
                }
            ],
        )


def test_write_note_reports_a_missing_field_as_a_tool_error(session):
    """A KeyError traceback tells the assistant nothing it can act on."""
    capture = save_capture(session, "A thought.")
    with pytest.raises(ToolError, match="missing 'description'"):
        write_note(
            session,
            derived_from=capture["id"],
            content="Body.",
            relationship_assertions=[
                {"source": "a", "target": "b", "type": "contradicts"}
            ],
        )
