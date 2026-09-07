import pytest

from mindpalace.oplog import DecisionLog, OpLog


def test_begin_marks_pending_and_commit_clears_it(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    op_id = log.begin({"tool": "save_capture", "capture": "c_01"})
    assert len(log.pending()) == 1
    log.commit(op_id)
    assert log.pending() == []


def test_begin_without_commit_is_pending(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    op_id = log.begin({"tool": "save_capture", "capture": "c_01"})
    reopened = OpLog(tmp_path / "log.jsonl")
    pending = reopened.pending()
    assert [entry["op"] for entry in pending] == [op_id]
    assert pending[0]["intent"]["tool"] == "save_capture"


def test_only_uncommitted_operations_are_pending(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    done = log.begin({"tool": "save_capture"})
    log.commit(done)
    stranded = log.begin({"tool": "write_note"})
    assert [entry["op"] for entry in OpLog(tmp_path / "log.jsonl").pending()] == [
        stranded
    ]


def test_commit_is_idempotent(tmp_path):
    log = OpLog(tmp_path / "log.jsonl")
    op_id = log.begin({"tool": "save_capture"})
    log.commit(op_id)
    log.commit(op_id)
    assert log.pending() == []


def test_status_map_is_proposed_until_a_decision_exists(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    assert decisions.status_map() == {}


def test_status_map_takes_the_latest_decision(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "wrong sense")
    decisions.append("x_01", "confirm", "resolve_assertion", "op_2", "new evidence")
    assert decisions.status_map() == {"x_01": "confirm"}


def test_decisions_survive_reopen(tmp_path):
    path = tmp_path / "decisions.jsonl"
    DecisionLog(path).append("x_01", "confirm", "resolve_assertion", "op_1")
    assert DecisionLog(path).status_map() == {"x_01": "confirm"}


def test_dismissal_reasons_are_recoverable(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "different sense")
    assert decisions.dismissal_reasons() == {"x_01": "different sense"}


def test_confirmation_clears_a_prior_dismissal_reason(tmp_path):
    decisions = DecisionLog(tmp_path / "decisions.jsonl")
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "different sense")
    decisions.append("x_01", "confirm", "resolve_assertion", "op_2")
    assert decisions.dismissal_reasons() == {}


def test_append_rejects_an_invalid_action(tmp_path):
    """The decision log is one shared file for the whole vault: a bad action
    line, once durably written, would block every future rebuild with no
    per-file quarantine available. Reject it before it can be written."""
    path = tmp_path / "decisions.jsonl"
    decisions = DecisionLog(path)
    with pytest.raises(ValueError, match="bogus"):
        decisions.append("x_01", "bogus", "resolve_assertion", "op_1")
    # Nothing durable was written -- the file must not even exist.
    assert not path.exists()


def test_oplog_survives_truncated_final_line(tmp_path):
    """Malformed final line from incomplete write is a crash artifact; skip it gracefully."""
    log = OpLog(tmp_path / "log.jsonl")
    op1 = log.begin({"tool": "save_capture"})
    log.commit(op1)
    op2 = log.begin({"tool": "write_note"})

    # Simulate a crash mid-write: remove the closing brace from the final line
    path = tmp_path / "log.jsonl"
    lines = path.read_text().splitlines(keepends=True)
    lines[-1] = lines[-1][:-3]  # Remove closing "}\n" to make malformed JSON
    path.write_text("".join(lines))

    # pending() should survive by skipping the malformed final line (crash artifact)
    # op1 was committed, so not pending; op2's line is malformed so skipped; result: []
    reopened = OpLog(path)
    pending = reopened.pending()
    assert pending == []


def test_oplog_raises_on_malformed_middle_line(tmp_path):
    """Malformed line in the middle is real corruption, not a torn write; raise loudly."""
    path = tmp_path / "log.jsonl"
    path.write_text('{"kind": "op.begin", "op": "op_001", "ts": "2024-01-01T00:00:00Z", "intent": {}}\n')
    path.write_text(
        path.read_text() + 'CORRUPTED JSON LINE\n',
    )
    path.write_text(
        path.read_text() + '{"kind": "op.commit", "op": "op_001", "ts": "2024-01-01T00:00:00Z"}\n',
    )

    log = OpLog(path)
    try:
        log.pending()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Malformed JSON" in str(e)
        assert str(path) in str(e)
        assert "line 2" in str(e)


def test_decision_log_survives_truncated_final_line(tmp_path):
    """Decision log degrades gracefully: truncated final line skipped, surviving decisions kept."""
    path = tmp_path / "decisions.jsonl"
    decisions = DecisionLog(path)
    decisions.append("x_01", "dismiss", "resolve_assertion", "op_1", "reason1")
    decisions.append("x_02", "confirm", "resolve_assertion", "op_2")

    # Simulate a crash mid-write: truncate the file
    content = path.read_text()
    truncated = content[:-10]
    path.write_text(truncated)

    # status_map() should recover and return only the surviving decision
    reopened = DecisionLog(path)
    status = reopened.status_map()
    assert status == {"x_01": "dismiss"}


# ---- vocabulary and merge logs (docs/decisions/0001 §1, §5) --------------


def test_vocabulary_log_folds_to_an_adoption_map_last_write_wins(tmp_path):
    from mindpalace.oplog import VocabularyLog

    log = VocabularyLog(tmp_path / "vocabulary.jsonl")
    assert log.adoptions() == {"edge": {}, "entity": {}}
    log.append("edge", "Available On", "available-on", "adopt", "adopt_type", "op_1")
    log.append("entity", "organisation", "organisation", "adopt", "adopt_type", "op_2")
    assert log.adoptions() == {
        "edge": {"available-on": "available-on"},
        "entity": {"organisation": "organisation"},
    }
    log.append("edge", "available on", "available-on", "revoke", "adopt_type", "op_3")
    assert log.adoptions()["edge"] == {}


def test_vocabulary_log_rejects_a_bad_kind_or_action(tmp_path):
    import pytest
    from mindpalace.oplog import VocabularyLog

    log = VocabularyLog(tmp_path / "vocabulary.jsonl")
    with pytest.raises(ValueError, match="kind"):
        log.append("verb", "x", "x", "adopt", "t", "op")
    with pytest.raises(ValueError, match="action"):
        log.append("edge", "x", "x", "forget", "t", "op")
    assert not (tmp_path / "vocabulary.jsonl").exists()


def test_merge_log_folds_to_a_map_and_a_keep_set(tmp_path):
    from mindpalace.oplog import MergeLog

    log = MergeLog(tmp_path / "merges.jsonl")
    assert log.merges() == {}
    assert log.kept() == set()
    log.append("open-ai", "openai", "merge", "merge_entities", "op_1", reason="same org")
    log.append("gpt-4", "gpt-4-turbo", "keep", "merge_entities", "op_2")
    assert log.merges() == {"open-ai": "openai"}
    assert log.kept() == {("gpt-4", "gpt-4-turbo")}
    log.append("open-ai", "openai", "unmerge", "merge_entities", "op_3")
    assert log.merges() == {}
    # A later merge of a kept pair overrides the keep, and vice versa.
    log.append("gpt-4", "gpt-4-turbo", "merge", "merge_entities", "op_4")
    assert log.merges() == {"gpt-4": "gpt-4-turbo"}
    assert log.kept() == set()


def test_merge_log_normalises_pair_order_for_keep(tmp_path):
    from mindpalace.oplog import MergeLog

    log = MergeLog(tmp_path / "merges.jsonl")
    log.append("zeta", "alpha", "keep", "merge_entities", "op_1")
    assert log.kept() == {("alpha", "zeta")}


def test_keep_only_undoes_a_merge_of_that_same_pair(tmp_path):
    from mindpalace.oplog import MergeLog

    log = MergeLog(tmp_path / "merges.jsonl")
    log.append("a", "c", "merge", "t", "op_1")
    log.append("a", "b", "keep", "t", "op_2")
    assert log.merges() == {"a": "c"}
    assert log.kept() == {("a", "b")}


def test_decision_log_conformance_fixture_folds_the_same_as_the_plugin():
    """tests/fixtures/decisions-conformance.jsonl is also read by the Obsidian
    plugin's Vitest suite; both must fold it to the same map and tolerate the
    torn final line the same way."""
    from pathlib import Path

    from mindpalace.oplog import DecisionLog

    fixture = Path(__file__).parent / "fixtures" / "decisions-conformance.jsonl"
    log = DecisionLog(fixture)
    assert log.status_map() == {"x_1": "confirm", "k_2": "confirm"}
    assert [d.via for d in log.entries()] == ["resolve_assertion", "obsidian", "resolve_assertion"]
