import numpy as np
import pytest

from mindpalace.embed import EmbedderError, StubEmbedder, get_embedder


def test_stub_is_deterministic():
    first = StubEmbedder().embed(["hello world"])
    second = StubEmbedder().embed(["hello world"])
    assert np.array_equal(first, second)


def test_stub_distinguishes_texts():
    embedder = StubEmbedder()
    vectors = embedder.embed(["hello world", "entirely different"])
    assert not np.array_equal(vectors[0], vectors[1])


def test_stub_shape_matches_declared_dim():
    embedder = StubEmbedder(dim=32)
    assert embedder.embed(["a", "b"]).shape == (2, 32)
    assert embedder.dim == 32


def test_stub_vectors_are_unit_length():
    vectors = StubEmbedder().embed(["hello"])
    assert np.linalg.norm(vectors[0]) == pytest.approx(1.0, abs=1e-6)


def test_stub_output_dtype_is_float32():
    vectors = StubEmbedder().embed(["hello"])
    assert vectors.dtype == np.float32


def test_stub_handles_empty_input():
    assert StubEmbedder().embed([]).shape == (0, 64)


def test_get_embedder_returns_stub_for_stub_kind():
    embedder = get_embedder({"kind": "stub", "dim": 16})
    assert embedder.model_id == "stub-16"


def test_get_embedder_rejects_cloud_kind():
    with pytest.raises(EmbedderError, match="cloud embedder is not implemented"):
        get_embedder({"kind": "cloud"})


def test_get_embedder_rejects_unknown_kind():
    with pytest.raises(EmbedderError, match="unknown embedder kind"):
        get_embedder({"kind": "telepathy"})


@pytest.mark.network
def test_local_embedder_produces_real_vectors():
    from mindpalace.embed import LocalEmbedder

    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    vectors = embedder.embed(["scaling laws", "banana bread"])
    assert vectors.shape == (2, embedder.dim)
    assert float(vectors[0] @ vectors[1]) < 0.9
