import numpy as np
import pytest

from mindpalace.index import db, vectors


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "cache.db")
    db.create_schema(connection)
    return connection


def test_search_ranks_by_cosine_similarity(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0, 0.0]))
    vectors.store(conn, "n_02", "note", "stub", np.array([0.0, 1.0, 0.0]))
    results = vectors.search(conn, np.array([1.0, 0.1, 0.0]), "stub", ["note"], 10)
    assert [doc_id for doc_id, _ in results] == ["n_01", "n_02"]
    assert results[0][1] > results[1][1]


def test_cosine_is_magnitude_invariant(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([3.0, 0.0, 0.0]))
    [(_, score)] = vectors.search(conn, np.array([9.0, 0.0, 0.0]), "stub", ["note"], 10)
    assert score == pytest.approx(1.0)


def test_search_filters_by_kind(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0]))
    vectors.store(conn, "e_x", "entity", "stub", np.array([1.0, 0.0]))
    results = vectors.search(conn, np.array([1.0, 0.0]), "stub", ["entity"], 10)
    assert [doc_id for doc_id, _ in results] == ["e_x"]


def test_search_ignores_other_models(conn):
    vectors.store(conn, "n_01", "note", "other-model", np.array([1.0, 0.0]))
    assert vectors.search(conn, np.array([1.0, 0.0]), "stub", ["note"], 10) == []


def test_store_replaces_existing_vector(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0]))
    vectors.store(conn, "n_01", "note", "stub", np.array([0.0, 1.0]))
    [(_, score)] = vectors.search(conn, np.array([0.0, 1.0]), "stub", ["note"], 10)
    assert score == pytest.approx(1.0)


def test_limit_is_respected(conn):
    for index in range(5):
        vectors.store(conn, f"n_{index}", "note", "stub", np.array([1.0, float(index)]))
    assert len(vectors.search(conn, np.array([1.0, 1.0]), "stub", ["note"], 2)) == 2


def test_clear_removes_everything(conn):
    vectors.store(conn, "n_01", "note", "stub", np.array([1.0, 0.0]))
    vectors.clear(conn)
    assert vectors.search(conn, np.array([1.0, 0.0]), "stub", ["note"], 10) == []
