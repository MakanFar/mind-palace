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
