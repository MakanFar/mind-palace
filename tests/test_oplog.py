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
