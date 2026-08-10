from mindpalace.models import (
    Capture,
    ClaimAssertion,
    EntityInstance,
    Note,
    RelationshipAssertion,
    capture_from_markdown,
    capture_to_markdown,
    note_from_markdown,
    note_to_markdown,
)

NOTE = Note(
    id="n_01hq",
    derived_from="c_01hq",
    created="2026-08-08T14:22:00Z",
    author="llm",
    body="The plateau is a data-supply constraint.",
    entities=(
        EntityInstance(
            name="scaling-laws",
            type="concept",
            description="Relationship between compute, data, and loss.",
        ),
    ),
    relationship_assertions=(
        RelationshipAssertion(
            id="x_01ab",
            source="scaling-laws",
            target="data-exhaustion",
            type="contradicts",
            strength=8,
            description="Argues plateau is data-driven, not architectural.",
        ),
    ),
    claim_assertions=(
        ClaimAssertion(
            id="k_01cd",
            subject="scaling-laws",
            text="The plateau reflects data exhaustion.",
        ),
    ),
)


def test_note_round_trips_through_markdown():
    assert note_from_markdown(note_to_markdown(NOTE)) == NOTE


def test_note_markdown_omits_status_fields():
    """Status lives in the decision log, never in the immutable note."""
    assert "status" not in note_to_markdown(NOTE)


def test_note_round_trips_with_no_assertions():
    bare = Note(
        id="n_02",
        derived_from="c_02",
        created="2026-08-08T15:00:00Z",
        author="user",
        body="A thought with nothing extracted.",
    )
    assert note_from_markdown(note_to_markdown(bare)) == bare


def test_capture_round_trips_through_markdown():
    capture = Capture(
        id="c_01hq",
        created="2026-08-08T14:22:00Z",
        source="manual",
        why="might matter for the essay",
        text="The plateau talk is mostly about data exhaustion.",
    )
    assert capture_from_markdown(capture_to_markdown(capture)) == capture


def test_capture_round_trips_without_optional_why():
    capture = Capture(
        id="c_02",
        created="2026-08-08T16:00:00Z",
        source="manual",
        why=None,
        text="A bare thought.",
    )
    assert capture_from_markdown(capture_to_markdown(capture)) == capture
