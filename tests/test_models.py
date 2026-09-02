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


# ---- fields borrowed from Utopia (docs/decisions/0001) --------------------


def test_untyped_assertion_and_entity_keep_the_proposed_wording():
    note = Note(
        id="n_02",
        derived_from="c_02",
        created="2026-09-02T10:00:00Z",
        author="llm",
        body="Body.",
        entities=(
            EntityInstance(name="acme", type=None, description="A company.", proposed_type="organisation"),
        ),
        relationship_assertions=(
            RelationshipAssertion(
                id="x_02",
                source="star-wars",
                target="geforce-now",
                type=None,
                strength=6,
                description="playable there",
                proposed_type="available on",
            ),
        ),
    )
    text = note_to_markdown(note)
    assert "proposed_type: available on" in text
    assert "proposed_type: organisation" in text
    back = note_from_markdown(text)
    assert back == note
    assert back.relationship_assertions[0].type is None
    assert back.entities[0].type is None


def test_direction_corrected_round_trips_and_defaults_false():
    plain = RelationshipAssertion(
        id="x_1", source="a", target="b", type="supports", strength=5, description="d"
    )
    assert plain.direction_corrected is False
    swapped = RelationshipAssertion(
        id="x_2", source="a", target="b", type="supports", strength=5,
        description="d", direction_corrected=True,
    )
    note = Note(id="n_3", derived_from=None, created="2026-09-02T10:00:00Z",
                author="llm", body="B.", relationship_assertions=(plain, swapped))
    text = note_to_markdown(note)
    assert text.count("direction_corrected") == 1
    assert note_from_markdown(text) == note


def test_claim_validity_and_supersession_round_trip():
    claim = ClaimAssertion(
        id="k_2", subject="acme", text="HQ is in Shenzhen.",
        valid_from="2026-03-15", valid_to="unknown", supersedes="k_1",
    )
    assert claim.valid_from_precision == "day"
    assert claim.valid_to_precision == "unknown"
    bare = ClaimAssertion(id="k_3", subject="acme", text="Founded.")
    assert bare.valid_from is None and bare.valid_from_precision is None
    note = Note(id="n_4", derived_from=None, created="2026-09-02T10:00:00Z",
                author="llm", body="B.", claim_assertions=(claim, bare))
    text = note_to_markdown(note)
    assert "valid_from: '2026-03-15'" in text or "valid_from: 2026-03-15" in text
    back = note_from_markdown(text)
    assert back == note
    assert back.claim_assertions[0].valid_from == "2026-03-15"


def test_precision_is_read_off_the_shape_of_the_date():
    assert ClaimAssertion(id="k", subject="s", text="t", valid_from="2023").valid_from_precision == "year"
    assert ClaimAssertion(id="k", subject="s", text="t", valid_from="2023-05").valid_from_precision == "month"
    assert ClaimAssertion(id="k", subject="s", text="t", valid_to="2023-05-01").valid_to_precision == "day"


def test_note_drop_ledger_round_trips():
    from mindpalace.models import Drop

    note = Note(
        id="n_5", derived_from="c_5", created="2026-09-02T10:00:00Z", author="llm",
        body="B.",
        drops=(Drop(kind="relationship", reason="missing_field", detail="target",
                    example="a -> ?"),),
    )
    text = note_to_markdown(note)
    assert "reason: missing_field" in text
    assert note_from_markdown(text) == note
    assert note_from_markdown(note_to_markdown(NOTE)).drops == ()
