"""Entity-type integrity findings derived from a folded source set.

`fold` never validates a declared entity type -- only `write_note` does, and
only for notes it writes itself. A hand-edited note can therefore put any
string into `entities.type`, and two notes can disagree about one entity with
the disagreement resolved silently by `(created, id)` ordering. These checks
report that; they never change what `fold` produced.
"""

import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.graph.fold import fold
from mindpalace.graph.integrity import check_entity_types
from mindpalace.models import ClaimAssertion, EntityInstance, Note, RelationshipAssertion


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept", "person"],
        edge_types={"supports": EdgeType("supports", directed=True, cluster_weight=1.0)},
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )


def note(note_id, created, *, entities=(), relationships=(), claims=()):
    return Note(
        id=note_id,
        derived_from=None,
        created=created,
        author="llm",
        body="Body.",
        entities=tuple(EntityInstance(n, t, "d.") for n, t in entities),
        relationship_assertions=tuple(relationships),
        claim_assertions=tuple(claims),
    )


def rel(assertion_id, source, target):
    return RelationshipAssertion(
        id=assertion_id,
        source=source,
        target=target,
        type="supports",
        strength=5,
        description="d.",
    )


def check(notes, config):
    """Run the checks over exactly what `fold` made of the same notes."""
    return check_entity_types(notes, fold(notes, {}, config), config)


def test_an_entity_type_absent_from_config_is_reported(config):
    notes = [note("n_1", "2026-01-01", entities=[("Scaling", "concpet")])]

    found = check(notes, config)

    assert [(f.kind, f.slug) for f in found] == [("unknown_entity_type", "scaling")]
    assert "concpet" in found[0].detail
    assert found[0].note_ids == ("n_1",)


def test_two_notes_declaring_different_types_for_one_entity_conflict(config):
    """`fold`'s `touch` resolves this last-write-wins by `(created, id)` and
    says nothing. The finding must name the winner, because the winner is
    what is actually in the graph and the user needs to know which of the
    two files is the one to edit.
    """
    # `person` is chronologically last but alphabetically second, so a
    # finding that named the alphabetically-first type would read as the
    # winner here and send the user to edit the wrong file.
    notes = [
        note("n_1", "2026-01-01", entities=[("Ada", "concept")]),
        note("n_2", "2026-01-02", entities=[("Ada", "person")]),
    ]

    found = check(notes, config)

    assert [(f.kind, f.slug) for f in found] == [("conflicting_entity_type", "ada")]
    assert "kept 'person'" in found[0].detail
    assert "ignored 'concept'" in found[0].detail
    assert found[0].note_ids == ("n_1", "n_2")


def test_a_typo_alongside_the_right_type_is_reported_once_not_twice(config):
    """`concpet` vs `concept` is one mistake with one fix. Reporting it as
    both an unknown type and a conflict would make the review queue read as
    if there were two problems, and only one of the two is actionable.
    """
    notes = [
        note("n_1", "2026-01-01", entities=[("Scaling", "concpet")]),
        note("n_2", "2026-01-02", entities=[("Scaling", "concept")]),
    ]

    assert [f.kind for f in check(notes, config)] == ["unknown_entity_type"]


def test_an_entity_that_only_appears_as_an_assertion_endpoint_is_reported(config):
    """`touch(slug, note_id, None)` gives such a slug the `unknown` sentinel
    type and moves on, so it enters the graph as a real node with no type at
    all. The extraction template asks for a relationship only between
    entities that were extracted, so this means one of them was not.
    """
    notes = [
        note(
            "n_1",
            "2026-01-01",
            entities=[("Ada", "person")],
            relationships=[rel("x_1", "Ada", "Bert")],
        )
    ]

    found = check(notes, config)

    assert [(f.kind, f.slug) for f in found] == [("reference_only_entity", "bert")]
    assert found[0].note_ids == ("n_1",)


def test_a_claim_subject_that_was_never_extracted_is_reported(config):
    notes = [
        note("n_1", "2026-01-01", claims=[ClaimAssertion("x_1", "Bert", "Fast.")])
    ]

    assert [(f.kind, f.slug) for f in check(notes, config)] == [
        ("reference_only_entity", "bert")
    ]


def test_findings_come_back_in_a_deterministic_order(config):
    """`sync` writes these straight into `vault_issues`. Order that follows
    note-iteration order would churn the table on every run.
    """
    notes = [
        note(
            "n_1",
            "2026-01-01",
            entities=[("Zeta", "concpet"), ("Ada", "person")],
            relationships=[rel("x_1", "Ada", "Bert")],
        ),
        note("n_2", "2026-01-02", entities=[("Ada", "concept")]),
    ]

    assert [(f.kind, f.slug) for f in check(notes, config)] == [
        ("conflicting_entity_type", "ada"),
        ("reference_only_entity", "bert"),
        ("unknown_entity_type", "zeta"),
    ]


def test_a_clean_source_set_produces_no_findings(config):
    notes = [
        note(
            "n_1",
            "2026-01-01",
            entities=[("Ada", "person"), ("Scaling", "concept")],
            relationships=[rel("x_1", "Ada", "Scaling")],
            claims=[ClaimAssertion("x_2", "Scaling", "Holds.")],
        )
    ]

    assert check(notes, config) == []


def test_an_entity_both_extracted_and_referenced_is_not_reference_only(config):
    """The normal case: every endpoint was also extracted. If this fired
    here the check would flag most of the vault and be worth nothing.
    """
    notes = [
        note("n_1", "2026-01-01", entities=[("Ada", "person")]),
        note(
            "n_2",
            "2026-01-02",
            entities=[("Bert", "person")],
            relationships=[rel("x_1", "Ada", "Bert")],
        ),
    ]

    assert check(notes, config) == []


def test_one_bad_type_repeated_across_notes_is_a_single_finding(config):
    """Both files need the same edit, so it is one finding listing both --
    not one finding per occurrence.
    """
    notes = [
        note("n_1", "2026-01-01", entities=[("Scaling", "concpet")]),
        note("n_2", "2026-01-02", entities=[("Scaling", "concpet")]),
    ]

    found = check(notes, config)

    assert len(found) == 1
    assert found[0].note_ids == ("n_1", "n_2")


def test_an_entity_listed_twice_in_one_note_names_that_note_once(config):
    """A hand-edited note can repeat an entity. The finding is about files
    to go edit, so the same file must not be listed twice.
    """
    notes = [
        note(
            "n_1",
            "2026-01-01",
            entities=[("Scaling", "concpet"), ("Scaling", "concpet")],
        )
    ]

    assert check(notes, config)[0].note_ids == ("n_1",)


def test_each_kind_names_its_notes_with_the_right_verb(config):
    """The note list means something different per kind -- files that
    declared a type, versus files that only mentioned a name they never
    extracted. Rendering all three as "declared in" tells the reader to go
    look for a declaration that is not there.
    """
    notes = [
        note(
            "n_1",
            "2026-01-01",
            entities=[("Zeta", "concpet"), ("Ada", "concept")],
            relationships=[rel("x_1", "Ada", "Bert")],
        ),
        note("n_2", "2026-01-02", entities=[("Ada", "person")]),
    ]

    detail = {f.kind: f.detail for f in check(notes, config)}

    assert "declared as 'concpet' in n_1" in detail["unknown_entity_type"]
    assert "declared in n_1, n_2" in detail["conflicting_entity_type"]
    assert "referenced by n_1" in detail["reference_only_entity"]
    assert "declared" not in detail["reference_only_entity"]


def test_an_entity_declared_with_a_configured_unknown_type_is_not_reference_only():
    """`touch` stores the string `unknown` for an entity no note extracted.

    Nothing stops a vault from configuring `unknown` as a real entity type,
    and then that sentinel is indistinguishable from a deliberate
    declaration. So the reference-only check must be derived from what the
    notes declare; reading the stored type would report an entity the user
    extracted on purpose.
    """
    config = Config(
        schema_version=1,
        entity_types=["unknown"],
        edge_types={"supports": EdgeType("supports", directed=True, cluster_weight=1.0)},
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )
    notes = [note("n_1", "2026-01-01", entities=[("Scaling", "unknown")])]

    assert check(notes, config) == []


def test_an_untyped_instance_with_a_proposal_is_neither_unknown_nor_reference_only(config):
    from mindpalace.graph.fold import fold
    from mindpalace.graph.integrity import check_entity_types
    from mindpalace.models import EntityInstance, Note

    note = Note(
        id="n_1", derived_from="c_1", created="2026-01-01", author="llm", body="B.",
        entities=(EntityInstance("acme", None, "d", proposed_type="organisation"),),
    )
    tables = fold([note], {}, config)
    assert check_entity_types([note], tables, config) == []
