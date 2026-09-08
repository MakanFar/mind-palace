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


def test_a_reopened_assertion_folds_as_proposed_again(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")]
    )
    tables = fold([note], {"x_1": "reopen"}, config)
    assert tables.assertions["x_1"].status == "proposed"
    [aggregate] = tables.aggregates.values()
    assert aggregate.traversable is False


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


# ---- borrowed from Utopia (docs/decisions/0001) --------------------------


def untyped_rel(assertion_id, source, target, proposed):
    return RelationshipAssertion(
        id=assertion_id, source=source, target=target, type=None,
        strength=5, description="because.", proposed_type=proposed,
    )


def test_an_untyped_assertion_keeps_its_endpoints_but_forms_no_aggregate(config):
    note = make_note("n_1", "2026-01-01", relationships=[untyped_rel("x_1", "a", "b", "available on")])
    tables = fold([note], {"x_1": "confirm"}, config)
    assert set(tables.entities) == {"a", "b"}
    assert tables.aggregates == {}
    assert tables.entities["a"].rank == 0
    folded = tables.assertions["x_1"]
    assert folded.type is None
    assert folded.proposed_type == "available on"
    assert folded.status == "confirmed"


def test_an_adopted_proposal_folds_as_the_adopted_type(config):
    note = make_note("n_1", "2026-01-01", relationships=[untyped_rel("x_1", "a", "b", "Supports!")])
    tables = fold([note], {"x_1": "confirm"}, config, adoptions={"edge": {"supports": "supports"}})
    [aggregate] = tables.aggregates.values()
    assert aggregate.type == "supports"
    assert aggregate.traversable
    assert tables.assertions["x_1"].type == "supports"
    assert tables.assertions["x_1"].proposed_type == "Supports!"


def test_an_adoption_naming_a_type_no_longer_configured_stays_untyped(config):
    note = make_note("n_1", "2026-01-01", relationships=[untyped_rel("x_1", "a", "b", "gone")])
    tables = fold([note], {}, config, adoptions={"edge": {"gone": "removed-type"}})
    assert tables.aggregates == {}
    assert tables.assertions["x_1"].type is None


def test_an_untyped_entity_instance_can_be_adopted_too(config):
    note = make_note(
        "n_1", "2026-01-01",
        entities=[EntityInstance(name="acme", type=None, description="d", proposed_type="Organisation")],
    )
    assert fold([note], {}, config).entities["acme"].type == "unknown"
    tables = fold([note], {}, config, adoptions={"entity": {"organisation": "concept"}})
    assert tables.entities["acme"].type == "concept"


def test_a_typed_string_absent_from_config_still_raises(config):
    note = make_note("n_1", "2026-01-01", relationships=[rel("x_1", "a", "b", edge_type="invented")])
    with pytest.raises(UnknownEdgeTypeError):
        fold([note], {}, config)


def test_merges_rewrite_every_slug_before_anything_is_touched(config):
    notes = [
        make_note(
            "n_1", "2026-01-01",
            entities=[EntityInstance(name="open-ai", type="concept", description="d")],
            relationships=[rel("x_1", "open-ai", "b", edge_type="supports")],
            claims=[ClaimAssertion(id="k_1", subject="open-ai", text="t")],
        ),
        make_note(
            "n_2", "2026-01-02",
            entities=[EntityInstance(name="openai", type="concept", description="d")],
        ),
    ]
    tables = fold(notes, {"x_1": "confirm"}, config, merges={"open-ai": "openai"})
    assert "open-ai" not in tables.entities
    canonical = tables.entities["openai"]
    assert canonical.note_ids == ("n_1", "n_2")
    assert canonical.merged_from == ("open-ai",)
    assert canonical.rank == 1
    assert tables.assertions["x_1"].source == "openai"
    assert tables.claims["k_1"].subject == "openai"
    [aggregate] = tables.aggregates.values()
    assert aggregate.source == "openai"


def test_merge_chains_are_followed_and_cycles_raise(config):
    from mindpalace.graph.fold import MergeCycleError

    note = make_note("n_1", "2026-01-01", entities=[EntityInstance(name="a", type="concept", description="d")])
    tables = fold([note], {}, config, merges={"a": "b", "b": "c"})
    assert set(tables.entities) == {"c"}
    with pytest.raises(MergeCycleError) as excinfo:
        fold([note], {}, config, merges={"a": "b", "b": "a"})
    assert excinfo.value.slug in {"a", "b"}


def test_a_merge_collapsing_a_self_loop_drops_no_assertion(config):
    note = make_note("n_1", "2026-01-01", relationships=[rel("x_1", "a", "b", edge_type="supports")])
    tables = fold([note], {"x_1": "confirm"}, config, merges={"b": "a"})
    assert tables.assertions["x_1"].source == tables.assertions["x_1"].target == "a"
    [aggregate] = tables.aggregates.values()
    assert aggregate.source == aggregate.target == "a"


def test_a_confirmed_superseding_claim_marks_the_old_one_superseded(config):
    old = ClaimAssertion(id="k_1", subject="acme", text="HQ in Beijing.", valid_from="2015")
    new = ClaimAssertion(id="k_2", subject="acme", text="HQ in Shenzhen.", valid_from="2026-03-15", supersedes="k_1")
    notes = [
        make_note("n_1", "2026-01-01", claims=[old]),
        make_note("n_2", "2026-01-02", claims=[new]),
    ]
    proposed = fold(notes, {"k_1": "confirm"}, config)
    assert proposed.claims["k_1"].status == "confirmed"
    assert proposed.claims["k_2"].status == "proposed"
    assert proposed.claims["k_2"].supersedes == "k_1"
    assert proposed.claims["k_2"].valid_from == "2026-03-15"
    confirmed = fold(notes, {"k_1": "confirm", "k_2": "confirm"}, config)
    assert confirmed.claims["k_1"].status == "superseded"
    assert confirmed.claims["k_2"].status == "confirmed"


def test_fold_carries_unit_provenance_onto_entities_and_items(config):
    note = make_note(
        "n_1", "2026-01-01",
        entities=[EntityInstance("a", "concept", "d", text_unit_ids=("u_1_0000",))],
        relationships=[RelationshipAssertion("x_1", "a", "b", "supports", 5, "d", text_unit_ids=("u_1_0001",))],
        claims=[ClaimAssertion("k_1", "b", "t", text_unit_ids=("u_1_0002",))],
    )
    tables = fold([note], {}, config)
    assert tables.assertions["x_1"].text_unit_ids == ("u_1_0001",)
    assert tables.claims["k_1"].text_unit_ids == ("u_1_0002",)
    assert tables.entities["a"].text_unit_ids == ("u_1_0000", "u_1_0001")
    assert tables.entities["b"].text_unit_ids == ("u_1_0001", "u_1_0002")


# ---- retiring an entity (docs/decisions/0004) ------------------------------


def test_a_dismissed_only_endpoint_is_not_an_entity(config):
    """A wrong relationship, once dismissed, must not leave its endpoints
    behind; reopen brings them back because the fold reruns from statuses."""
    note = make_note("n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")])
    tables = fold([note], {"x_1": "dismiss"}, config)
    assert set(tables.entities) == set()
    assert "x_1" in tables.assertions
    assert set(fold([note], {"x_1": "reopen"}, config).entities) == {"a", "b"}


def test_a_dismissed_only_claim_subject_is_not_an_entity(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z",
        claims=[ClaimAssertion(id="k_1", subject="a", text="t")],
    )
    assert set(fold([note], {"k_1": "dismiss"}, config).entities) == set()
    assert set(fold([note], {}, config).entities) == {"a"}


def test_a_declared_entity_survives_its_only_assertion_being_dismissed(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z",
        entities=[EntityInstance("a", "concept", "…")],
        relationships=[rel("x_1", "a", "b")],
    )
    tables = fold([note], {"x_1": "dismiss"}, config)
    assert set(tables.entities) == {"a"}
    assert tables.entities["a"].declared is True


def test_an_endpoint_only_entity_is_not_declared(config):
    note = make_note("n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")])
    tables = fold([note], {}, config)
    assert tables.entities["a"].declared is False


def test_a_retired_entity_is_dropped_from_the_fold(config):
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z",
        entities=[EntityInstance("a", "concept", "…")],
    )
    tables = fold([note], {}, config, retired={"a"})
    assert "a" not in tables.entities
    assert tables.revived == ()


def test_a_live_assertion_revives_a_retired_entity_and_says_so(config):
    note = make_note("n_01", "2026-08-01T00:00:00Z", relationships=[rel("x_1", "a", "b")])
    tables = fold([note], {}, config, retired={"a"})
    assert "a" in tables.entities
    assert tables.revived == ("a",)


def test_a_retirement_is_resolved_through_merges(config):
    """Retire the canonical name: the merged-away alias goes with it."""
    note = make_note(
        "n_01", "2026-08-01T00:00:00Z",
        entities=[EntityInstance("old", "concept", "…")],
    )
    tables = fold([note], {}, config, merges={"old": "new"}, retired={"new"})
    assert tables.entities == {}
