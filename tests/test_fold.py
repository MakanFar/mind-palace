import pytest

from mindpalace.config import Config, EdgeType, Thresholds
from mindpalace.graph.fold import (
    DuplicateAssertionIdError,
    FoldError,
    UnknownDecisionActionError,
    UnknownEdgeTypeError,
    fold,
)
from mindpalace.models import ClaimAssertion, EntityInstance, Note, RelationshipAssertion


@pytest.fixture
def config():
    return Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "contradicts": EdgeType("contradicts", directed=False, cluster_weight=1.0),
            "supports": EdgeType("supports", directed=True, cluster_weight=1.0),
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )


def make_note(note_id, created, *, entities=(), relationships=(), claims=()):
    return Note(
        id=note_id,
        derived_from=f"c_{note_id[2:]}",
        created=created,
        author="llm",
        body="Body.",
        entities=tuple(entities),
        relationship_assertions=tuple(relationships),
        claim_assertions=tuple(claims),
    )


def rel(assertion_id, source, target, edge_type="contradicts", strength=5):
    return RelationshipAssertion(
        id=assertion_id,
        source=source,
        target=target,
        type=edge_type,
        strength=strength,
        description="because.",
    )


def test_empty_input_yields_empty_tables(config):
    tables = fold([], {}, config)
    assert tables.entities == {}
    assert tables.aggregates == {}


def test_entities_come_from_declared_instances(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        entities=[EntityInstance("Scaling Laws", "concept", "…")],
    )
    tables = fold([note], {}, config)
    assert tables.entities["scaling-laws"].type == "concept"
    assert tables.entities["scaling-laws"].note_ids == ("n_01",)


def test_assertion_endpoints_create_entities_with_unknown_type(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    tables = fold([note], {}, config)
    assert tables.entities["a"].type == "unknown"
    assert tables.entities["b"].type == "unknown"


def test_most_recent_note_wins_the_entity_type(config):
    older = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        entities=[EntityInstance("thing", "concept", "…")],
    )
    newer = make_note(
        "n_02",
        "2026-08-05T00:00:00Z",
        entities=[EntityInstance("thing", "paper", "…")],
    )
    assert fold([older, newer], {}, config).entities["thing"].type == "paper"
    assert fold([newer, older], {}, config).entities["thing"].type == "paper"


def test_symmetric_assertions_merge_into_one_aggregate(config):
    forward = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    reverse = make_note(
        "n_02", "2026-08-02T00:00:00Z", relationships=[rel("x_2", "b", "a")]
    )
    tables = fold([forward, reverse], {}, config)
    assert len(tables.aggregates) == 1
    [aggregate] = tables.aggregates.values()
    assert set(aggregate.assertion_ids) == {"x_1", "x_2"}


def test_directed_assertions_do_not_merge(config):
    forward = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", edge_type="supports")],
    )
    reverse = make_note(
        "n_02",
        "2026-08-02T00:00:00Z",
        relationships=[rel("x_2", "b", "a", edge_type="supports")],
    )
    assert len(fold([forward, reverse], {}, config).aggregates) == 2


def test_aggregate_is_not_traversable_while_only_proposed(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    [aggregate] = fold([note], {}, config).aggregates.values()
    assert aggregate.traversable is False
    assert aggregate.weight == 0


def test_one_confirmation_makes_the_aggregate_traversable(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b", strength=8)]
    )
    [aggregate] = fold([note], {"x_1": "confirm"}, config).aggregates.values()
    assert aggregate.traversable is True
    assert aggregate.weight == 1
    assert aggregate.mean_strength == 8.0


def test_dismissing_one_of_two_leaves_the_aggregate_alive(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", strength=4), rel("x_2", "a", "b", strength=8)],
    )
    statuses = {"x_1": "dismiss", "x_2": "confirm"}
    [aggregate] = fold([note], statuses, config).aggregates.values()
    assert aggregate.traversable is True
    assert aggregate.weight == 1
    assert aggregate.mean_strength == 8.0


def test_rank_counts_only_traversable_degree(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[
            rel("x_1", "hub", "spoke-one"),
            rel("x_2", "hub", "spoke-two"),
        ],
    )
    tables = fold([note], {"x_1": "confirm"}, config)
    assert tables.entities["hub"].rank == 1
    assert tables.entities["spoke-one"].rank == 1
    assert tables.entities["spoke-two"].rank == 0


def test_claims_take_status_from_the_decision_log(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        claims=[ClaimAssertion("k_1", "a", "A claim."), ClaimAssertion("k_2", "a", "B.")],
    )
    tables = fold([note], {"k_1": "confirm"}, config)
    assert tables.claims["k_1"].status == "confirmed"
    assert tables.claims["k_2"].status == "proposed"


def test_fold_is_idempotent(config):
    notes = [
        make_note(
            "n_01",
            "2026-08-01T00:00:00Z",
            entities=[EntityInstance("a", "concept", "…")],
            relationships=[rel("x_1", "a", "b")],
        )
    ]
    assert fold(notes, {"x_1": "confirm"}, config) == fold(
        notes, {"x_1": "confirm"}, config
    )


def test_editing_a_note_retracts_its_old_assertions(config):
    """The guard against double-counting: fold is over the current set, not a delta."""
    version_one = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    version_two = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_2", "a", "c")]
    )
    after = fold([version_two], {"x_2": "confirm"}, config)
    assert "x_1" not in after.assertions
    assert len(after.aggregates) == 1
    assert "b" not in after.entities


def test_unknown_edge_type_is_rejected(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", edge_type="invented")],
    )
    with pytest.raises(Exception, match="invented"):
        fold([note], {}, config)


def test_dismissing_one_of_three_and_leaving_one_proposed_still_traverses(config):
    """Mixed statuses on the same pair: one confirmed, one dismissed, one
    still proposed. traversable/weight/mean_strength must reflect only the
    confirmed assertion -- proposed and dismissed both contribute nothing."""
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[
            rel("x_1", "a", "b", strength=4),  # dismissed
            rel("x_2", "a", "b", strength=8),  # confirmed
            rel("x_3", "a", "b", strength=2),  # left proposed
        ],
    )
    statuses = {"x_1": "dismiss", "x_2": "confirm"}
    [aggregate] = fold([note], statuses, config).aggregates.values()
    assert aggregate.traversable is True
    assert aggregate.weight == 1
    assert aggregate.mean_strength == 8.0
    assert set(aggregate.assertion_ids) == {"x_1", "x_2", "x_3"}


# --- Additional adversarial tests (not in the brief) ---
#
# The brief's task description explicitly asks: "Can any input cause an
# assertion or claim to be silently dropped rather than surfacing? Consider
# two notes sharing an assertion id." The sample implementation in the brief
# does exactly this: `assertions[raw.id] = folded` silently overwrites on a
# duplicate id, and the duplicate is independently appended into `grouped`
# for aggregation, meaning a re-used id can be double-counted into an
# aggregate's weight while the `assertions` table silently keeps only one
# of the two. These tests pin the corrected behaviour: surface loudly, with
# structured attributes a caller (the vault-wide `sync`) can act on rather
# than parse out of a message string.


def test_duplicate_relationship_assertion_id_across_notes_is_rejected(config):
    first = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    second = make_note(
        "n_02", "2026-08-02T00:00:00Z", relationships=[rel("x_1", "c", "d")]
    )
    with pytest.raises(DuplicateAssertionIdError) as excinfo:
        fold([first, second], {}, config)
    assert isinstance(excinfo.value, FoldError)
    assert excinfo.value.assertion_id == "x_1"
    assert excinfo.value.note_ids == ("n_01", "n_02")


def test_duplicate_claim_assertion_id_across_notes_is_rejected(config):
    first = make_note(
        "n_01", "2026-08-01T00:00:00Z", claims=[ClaimAssertion("k_1", "a", "First.")]
    )
    second = make_note(
        "n_02", "2026-08-02T00:00:00Z", claims=[ClaimAssertion("k_1", "b", "Second.")]
    )
    with pytest.raises(DuplicateAssertionIdError) as excinfo:
        fold([first, second], {}, config)
    assert isinstance(excinfo.value, FoldError)
    assert excinfo.value.assertion_id == "k_1"
    assert excinfo.value.note_ids == ("n_01", "n_02")


def test_unknown_edge_type_error_carries_structured_attributes(config):
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", edge_type="invented")],
    )
    with pytest.raises(UnknownEdgeTypeError) as excinfo:
        fold([note], {}, config)
    assert isinstance(excinfo.value, FoldError)
    assert excinfo.value.edge_type == "invented"
    assert excinfo.value.note_id == "n_01"
    assert excinfo.value.assertion_id == "x_1"


def test_unrecognized_decision_action_is_rejected(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    with pytest.raises(UnknownDecisionActionError) as excinfo:
        fold([note], {"x_1": "bogus"}, config)
    assert isinstance(excinfo.value, FoldError)
    assert excinfo.value.action == "bogus"
    assert excinfo.value.assertion_id == "x_1"


def test_edge_type_name_containing_pipe_does_not_corrupt_the_aggregate(config):
    """`aggregate_key` joins source/type/target with "|" and config places no
    character restriction on edge-type names. Fold must capture
    (source, type, target) directly rather than re-parsing the key string,
    or an edge type like "relates|to" would mis-split into the wrong
    source/type/target -- corrupting Aggregate fields and, downstream, the
    `degree` dict that computes rank."""
    piped_config = Config(
        schema_version=1,
        entity_types=["concept"],
        edge_types={
            "relates|to": EdgeType("relates|to", directed=False, cluster_weight=1.0),
        },
        thresholds=Thresholds(150, 2.0, 0.35, 0.5),
        embedder={"kind": "stub"},
        templates={},
    )
    note = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        relationships=[rel("x_1", "a", "b", edge_type="relates|to")],
    )
    [aggregate] = fold([note], {"x_1": "confirm"}, piped_config).aggregates.values()
    assert aggregate.source == "a"
    assert aggregate.type == "relates|to"
    assert aggregate.target == "b"
    assert aggregate.traversable is True
    assert aggregate.weight == 1


def test_fold_output_is_independent_of_input_note_ordering(config):
    """Notes are re-sorted by (created, id) internally, so iteration order of
    every produced table is a function of note content, not caller order."""
    note_a = make_note(
        "n_01",
        "2026-08-01T00:00:00Z",
        entities=[EntityInstance("a", "concept", "…")],
        relationships=[rel("x_1", "a", "b")],
    )
    note_b = make_note(
        "n_02",
        "2026-08-02T00:00:00Z",
        entities=[EntityInstance("c", "concept", "…")],
        relationships=[rel("x_2", "c", "d")],
    )
    statuses = {"x_1": "confirm", "x_2": "confirm"}
    forward = fold([note_a, note_b], statuses, config)
    backward = fold([note_b, note_a], statuses, config)
    assert forward == backward
    assert list(forward.entities) == list(backward.entities)
    assert list(forward.aggregates) == list(backward.aggregates)
    assert list(forward.assertions) == list(backward.assertions)
