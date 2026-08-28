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


def test_note_round_trips_without_a_source_capture():
    synthesis = Note(
        id="n_03",
        derived_from=None,
        created="2026-08-08T17:00:00Z",
        author="user",
        body="A link I spotted myself.",
    )
    assert note_from_markdown(note_to_markdown(synthesis)) == synthesis


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


def test_a_note_whose_created_is_unquoted_yaml_parses_to_the_canonical_string():
    """`created: 2026-08-02T00:00:00Z` without quotes is valid YAML for a
    datetime, and a human hand-editing in Obsidian writes it that way. It
    used to reach `fold` as a datetime and take the sort down with a
    TypeError -- not a FoldError, so quarantine never saw it and the whole
    vault failed to sync.
    """
    note = note_from_markdown(
        "---\n"
        "id: n_zz\n"
        "created: 2026-08-02T00:00:00Z\n"
        "author: human\n"
        "---\n"
        "\n"
        "Hand-written.\n"
    )

    assert note.created == "2026-08-02T00:00:00Z"


def test_a_capture_whose_created_is_unquoted_yaml_parses_to_the_canonical_string():
    """Captures carry the same `created` field written by the same hands."""
    capture = capture_from_markdown(
        "---\n"
        "id: c_zz\n"
        "created: 2026-08-02T00:00:00Z\n"
        "source: chat\n"
        "---\n"
        "\n"
        "Pasted.\n"
    )

    assert capture.created == "2026-08-02T00:00:00Z"


def test_a_timestamp_in_another_offset_is_normalised_to_utc():
    """These strings are ordered lexicographically against each other, so a
    surviving `+05:00` would sort by wall-clock rather than by instant.
    """
    note = note_from_markdown(
        "---\nid: n_zz\ncreated: 2026-08-02T05:00:00+05:00\nauthor: human\n---\n\nX.\n"
    )

    assert note.created == "2026-08-02T00:00:00Z"


def test_a_created_value_that_is_not_a_timestamp_is_rejected():
    """Raising is right here where coercing was right above: there is no
    correct reading of the value. `_load_notes` catches it and records a
    `malformed_note` issue, so the file degrades rather than the vault.
    """
    import pytest

    with pytest.raises(TypeError, match="must be a timestamp"):
        note_from_markdown(
            "---\nid: n_zz\ncreated: 12345\nauthor: human\n---\n\nX.\n"
        )
