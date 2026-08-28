import pytest

from mindpalace.ids import (
    UnknownIdError,
    aggregate_key,
    entity_id,
    id_kind,
    new_id,
    slugify,
)


def test_new_id_carries_prefix_and_is_unique():
    first = new_id("c_")
    second = new_id("c_")
    assert first.startswith("c_")
    assert first != second


def test_new_id_rejects_unknown_prefix():
    with pytest.raises(UnknownIdError):
        new_id("z_")


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("c_01J", "capture"),
        ("n_01J", "note"),
        ("x_01J", "relationship_assertion"),
        ("k_01J", "claim_assertion"),
        ("e_scaling-laws", "entity"),
        ("g_01J", "community"),
        ("op_01J", "operation"),
    ],
)
def test_id_kind_dispatches_on_prefix(identifier, expected):
    assert id_kind(identifier) == expected


def test_id_kind_rejects_unrecognised():
    with pytest.raises(UnknownIdError):
        id_kind("zzz")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Scaling Laws", "scaling-laws"),
        ("  LLM  ", "llm"),
        ("Café Society", "cafe-society"),
        ("data--exhaustion!!", "data-exhaustion"),
    ],
)
def test_slugify(raw, expected):
    assert slugify(raw) == expected


def test_entity_id_is_slug_based():
    assert entity_id("Scaling Laws") == "e_scaling-laws"


def test_symmetric_aggregate_key_sorts_endpoints():
    forward = aggregate_key("scaling-laws", "contradicts", "data-exhaustion", symmetric=True)
    reverse = aggregate_key("data-exhaustion", "contradicts", "scaling-laws", symmetric=True)
    assert forward == reverse
    assert forward == "r:data-exhaustion|contradicts|scaling-laws"


def test_directed_aggregate_key_preserves_order():
    forward = aggregate_key("chinchilla", "supports", "scaling-laws", symmetric=False)
    reverse = aggregate_key("scaling-laws", "supports", "chinchilla", symmetric=False)
    assert forward != reverse
    assert forward == "r:chinchilla|supports|scaling-laws"
