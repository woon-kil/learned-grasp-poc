from __future__ import annotations

from dataclasses import replace

import pytest

from learned_grasp.arm import FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import FakeCamera
from learned_grasp.g0_runner import (
    J5_OFFSETS_DEG,
    LATERAL_OFFSETS_MM,
    Outcome,
    compute_tolerance,
    dryrun_sheet,
    offset_sheet,
    replay_sheet,
    run_dryrun,
    run_replay,
    summarize,
)
from learned_grasp.safety import GRASP, HOVER, Pose, SafetyGate
from learned_grasp.stats import percentile, wilson_interval


def make(ready_config, table, tmp_path):
    driver = FakeArm()
    arm = SafeArm(driver, SafetyGate(ready_config, table), sleep=lambda _: None)
    ticks = iter(range(100000))
    ledger = AttemptLedger(tmp_path / "attempts.jsonl", clock=lambda: float(next(ticks)))
    return driver, arm, ledger


def test_wilson_matches_plan_statement():
    low, high = wilson_interval(24, 30)
    assert low == pytest.approx(0.627, abs=0.001)
    assert high == pytest.approx(0.905, abs=0.001)


def test_percentile():
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0


def test_sheet_fixed_by_seed(ready_config, table):
    assert replay_sheet(ready_config, table, 1) == replay_sheet(ready_config, table, 1)
    assert replay_sheet(ready_config, table, 1) != replay_sheet(ready_config, table, 2)


def test_sheet_is_ten_cells_three_repeats(ready_config, table):
    sheet = replay_sheet(ready_config, table, 1)
    assert len(sheet) == 30
    cells = {t["cell"] for t in sheet}
    assert len(cells) == 10
    assert all(sum(t["cell"] == cell for t in sheet) == 3 for cell in cells)
    locations = {(t["r_index"], t["j1"], t["j5"]) for t in sheet}
    assert len(locations) == 10
    assert len({t["trial_id"] for t in sheet}) == 30


def test_sheet_covers_radius_bands(ready_config, table):
    count = len(table.entries)
    bands = {t["r_index"] * 3 // count for t in replay_sheet(ready_config, table, 3)}
    assert bands == {0, 1, 2}


def test_all_success_passes_gate(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    camera = FakeCamera()
    loads = []
    run_replay(arm, camera, ledger, sheet, lambda t: Outcome(True), table, "stored", loads.append)
    result = summarize(sheet, ledger)
    assert result["gate_pass"] and result["successes"] == 30 and result["complete"]
    assert len(loads) == 1
    assert len(camera.names) == 30


def test_replay_puts_block_down_after_last_success(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:1]
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    grasp = arm.gate.joints(Pose(GRASP, sheet[0]["r_index"], sheet[0]["j1"], sheet[0]["j5"]))
    assert driver.writes[-1][1] == ready_config.open_sdk and arm.pose.kind == "park"
    assert driver.writes[-3] == (grasp, ready_config.open_sdk, driver.writes[-3][2])
    assert summarize(sheet, ledger)["successes"] == 1


def test_replay_failure_does_not_put_down_again(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:1]
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(False, "미집기", True), table, "stored", lambda t: None)
    grasp = arm.gate.joints(Pose(GRASP, sheet[0]["r_index"], sheet[0]["j1"], sheet[0]["j5"]))
    assert grasp not in [write[0] for write in driver.writes[-3:]]
    assert arm.pose.kind == "park"


def test_replay_crash_does_not_put_down(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:2]
    seen = {}

    def crashing(trial):
        seen["writes"] = len(driver.writes)
        raise ConnectionError("연결 단절")

    with pytest.raises(ConnectionError):
        run_replay(arm, FakeCamera(), ledger, sheet, crashing, table, "stored", lambda t: None)
    assert len(driver.writes) == seen["writes"]


def test_replay_rerun_with_nothing_left_writes_nothing(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:1]
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    driver2, arm2, _ = make(ready_config, table, tmp_path / "other")
    run_replay(arm2, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    assert driver2.writes == []


def test_label_mode_runs(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "label", lambda t: None)
    assert summarize(sheet, ledger)["gate_pass"]


def test_failure_counts_and_human_reloads(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    failing = {t["trial_id"] for t in sheet[:7]}
    loads = []
    run_replay(
        arm, FakeCamera(), ledger, sheet,
        lambda t: Outcome(t["trial_id"] not in failing, "미집기", t["trial_id"] in failing),
        table, "stored", loads.append,
    )
    result = summarize(sheet, ledger)
    assert result["successes"] == 23 and not result["gate_pass"]
    assert result["human_interventions"] == 7
    assert len(loads) == 8  # 첫 적재 1 + 실패 뒤 다음 시도 7, 연속 실패 포함


def test_threshold_is_exactly_24(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    failing = {t["trial_id"] for t in sheet[:6]}
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(t["trial_id"] not in failing), table, "stored", lambda t: None)
    assert summarize(sheet, ledger)["gate_pass"]


def test_resume_after_crash_neither_loses_nor_repeats(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    calls = []

    def crashing(trial):
        calls.append(trial["trial_id"])
        if len(calls) == 5:
            raise ConnectionError("연결 단절")
        return Outcome(True)

    with pytest.raises(ConnectionError):
        run_replay(arm, FakeCamera(), ledger, sheet, crashing, table, "stored", lambda t: None)
    partial = summarize(sheet, ledger)
    assert not partial["complete"] and not partial["gate_pass"]

    driver2, arm2, _ = make(ready_config, table, tmp_path / "other")
    seen = []

    def second(trial):
        seen.append(trial["trial_id"])
        return Outcome(True)

    run_replay(arm2, FakeCamera(), ledger, sheet, second, table, "stored", lambda t: None)
    result = summarize(sheet, ledger)
    assert result["complete"] and result["interrupted"] == 1 and result["successes"] == 29
    assert len(seen) == 25
    assert calls[4] not in seen  # 중단된 시도를 다시 하지 않는다
    assert set(calls[:4]).isdisjoint(seen)


def test_incomplete_never_passes(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    run_replay(arm, FakeCamera(), ledger, sheet[:24], lambda t: Outcome(True), table, "stored", lambda t: None)
    result = summarize(sheet, ledger)
    assert result["successes"] == 24 and not result["complete"] and not result["gate_pass"]


def test_snapshot_failure_stops_session(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    camera = FakeCamera()
    camera.fail_next = True
    with pytest.raises(RuntimeError):
        run_replay(arm, camera, ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    assert not summarize(sheet, ledger)["gate_pass"]
    assert camera.names == []
    # 촬영 실패 뒤 다음 명령이 없다. 마지막 쓰기는 물러난 park 자세다.
    assert driver.writes[-1][0] == ready_config.park
    assert arm.pose.kind == "park"


def test_failed_pick_opens_low_not_at_park(ready_config, table, tmp_path):
    """물체를 쥔 채일 수 있으니 높은 park에서 열지 않는다."""
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:1]
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(False, "미집기", True), table, "stored", lambda t: None)
    opens = [i for i, (q5, grip, _) in enumerate(driver.writes) if grip == ready_config.open_sdk and i and driver.writes[i - 1][1] == ready_config.close_sdk]
    last_open = opens[-1]
    assert driver.writes[last_open][0] != ready_config.park
    assert driver.writes[-1][0] == ready_config.park and not arm.holding


def test_waiting_for_human_leaves_no_interrupted_trial(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)

    def no_human(trial):
        raise TimeoutError("사람 없음")

    with pytest.raises(TimeoutError):
        run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", no_human)
    assert ledger.state() == ({}, set())


def test_loader_at_hover_does_not_break_transition(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:2]

    def load(trial):
        arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))

    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", load)
    assert ledger.state()[0][sheet[1]["trial_id"]]["success"]


def test_all_commands_stay_in_table(ready_config, table, tmp_path):
    driver, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 5)
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    allowed = {ready_config.park[1:4]}
    for entry in table.entries:
        allowed |= {entry.grasp, entry.hover}
    for q5, grip, time_ms in driver.writes:
        assert q5[1:4] in allowed
        assert grip in (ready_config.open_sdk, ready_config.close_sdk)
        assert time_ms >= ready_config.motion.min_time_ms


def test_dryrun_covers_full_table_and_stops_on_report(ready_config, table, tmp_path):
    sheet = dryrun_sheet(ready_config, table)
    assert len(sheet) == len(table.entries) * 3 * 2
    driver, arm, ledger = make(ready_config, table, tmp_path)
    run_dryrun(arm, ledger, sheet, lambda t: Outcome(t["trial_id"] != sheet[2]["trial_id"], "간섭"), table)
    finished, _ = ledger.state()
    assert len(finished) == 3
    assert all(grip == ready_config.open_sdk for _, grip, _ in driver.writes)


def test_dryrun_refuses_to_resume_after_reported_problem(ready_config, table, tmp_path):
    """이상이 보고된 조합을 건너뛰고 계속하면 안 된다."""
    sheet = dryrun_sheet(ready_config, table)
    driver, arm, ledger = make(ready_config, table, tmp_path)
    run_dryrun(arm, ledger, sheet, lambda t: Outcome(False, "간섭"), table)
    driver2, arm2, _ = make(ready_config, table, tmp_path / "again")
    with pytest.raises(RuntimeError, match="재개하지 않는다"):
        run_dryrun(arm2, ledger, sheet, lambda t: Outcome(True), table)
    assert driver2.writes == []


def test_dryrun_refuses_resume_after_interruption(ready_config, table, tmp_path):
    """이동이 불확실하게 끊겼다. 사람이 확인하기 전에 이어서 움직이지 않는다."""
    sheet = dryrun_sheet(ready_config, table)
    _, arm, ledger = make(ready_config, table, tmp_path)
    calls = []

    def crash(trial):
        calls.append(trial["trial_id"])
        if len(calls) == 2:
            raise ConnectionError("단절")
        return Outcome(True)

    with pytest.raises(ConnectionError):
        run_dryrun(arm, ledger, sheet, crash, table)
    driver2, arm2, _ = make(ready_config, table, tmp_path / "again")
    with pytest.raises(RuntimeError, match="재개하지 않는다"):
        run_dryrun(arm2, ledger, sheet, lambda t: Outcome(True), table)
    assert driver2.writes == []
    assert ledger.state()[0][calls[1]]["outcome"] == "interrupted"


def test_finished_dryrun_sends_nothing(ready_config, table, tmp_path):
    sheet = dryrun_sheet(ready_config, table)[:2]
    _, arm, ledger = make(ready_config, table, tmp_path)
    run_dryrun(arm, ledger, sheet, lambda t: Outcome(True), table)
    driver2, arm2, _ = make(ready_config, table, tmp_path / "again")
    run_dryrun(arm2, ledger, sheet, lambda t: Outcome(True), table)
    assert driver2.writes == []


def test_ledger_rejects_other_configuration(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)
    run_replay(arm, FakeCamera(), ledger, sheet[:1], lambda t: Outcome(True), table, "stored", lambda t: None)
    changed = replace(ready_config, config_version="1")
    arm2 = SafeArm(FakeArm(), SafetyGate(changed, table), sleep=lambda _: None)
    with pytest.raises(RuntimeError, match="다른 구성"):
        run_replay(arm2, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)


def test_attempt_records_commands_and_configuration(ready_config, table, tmp_path):
    _, arm, ledger = make(ready_config, table, tmp_path)
    sheet = replay_sheet(ready_config, table, 1)[:1]
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    records = ledger.records()
    assert records[0]["event"] == "meta" and records[0]["table_sha256"] == table.sha256()
    finished, _ = ledger.state()
    record = finished[sheet[0]["trial_id"]]
    entry = table.entries[sheet[0]["r_index"]]
    assert record["place_q"] == [sheet[0]["j1"], *entry.grasp, sheet[0]["j5"]] == record["pick_q"]
    assert record["read_back"] is not None


def test_gate_pass_absent_for_gates_without_threshold(ready_config, table, tmp_path):
    sheet = dryrun_sheet(ready_config, table)[:2]
    _, arm, ledger = make(ready_config, table, tmp_path)
    run_dryrun(arm, ledger, sheet, lambda t: Outcome(True), table)
    assert summarize(sheet, ledger)["gate_pass"] is None


def _tol_records(cell_results):
    records = []
    for (cell, offset), successes in cell_results.items():
        records += [
            {"axis": "j5", "cell": cell, "offset": offset, "success": index < successes} for index in range(5)
        ]
    return records


def test_tolerance_takes_worst_cell_and_sign():
    results = {(c, s * v): 5 for c in range(3) for v in J5_OFFSETS_DEG for s in (1, -1)}
    results[(1, -20)] = 3
    assert compute_tolerance(_tol_records(results), "j5", J5_OFFSETS_DEG) == 10


def test_tolerance_ignores_success_after_small_failure():
    results = {(c, s * v): 5 for c in range(3) for v in J5_OFFSETS_DEG for s in (1, -1)}
    results[(0, 10)] = 0
    assert compute_tolerance(_tol_records(results), "j5", J5_OFFSETS_DEG) == 0


def test_offset_sheet_fixed_and_applied(ready_config, table, tmp_path):
    sheet = offset_sheet(ready_config, table, 1)
    assert sheet == offset_sheet(ready_config, table, 1)
    assert len(sheet) == 3 * (3 * 2 + 3 * 2 + 2 * 2) * 5
    assert {t["axis"] for t in sheet} == {"r", "lateral", "j5"}
    lateral = [t for t in sheet if t["axis"] == "lateral"]
    assert all(t["pick_j1"] != t["j1"] and t["pick_r_index"] == t["r_index"] for t in lateral)
    assert {t["pick_j1"] > t["j1"] for t in lateral} == {True, False}
    assert all(ready_config.j1_range[0] <= t["pick_j1"] <= ready_config.j1_range[1] for t in lateral)
    _, arm, ledger = make(ready_config, table, tmp_path)
    run_replay(arm, FakeCamera(), ledger, sheet, lambda t: Outcome(True), table, "stored", lambda t: None)
    finished, _ = ledger.state()
    assert compute_tolerance(list(finished.values()), "j5", J5_OFFSETS_DEG) == 20
    assert compute_tolerance(list(finished.values()), "lateral", LATERAL_OFFSETS_MM) == 10


def test_offset_sheet_records_actual_offset(ready_config, table):
    """표 간격 때문에 명목 오프셋과 실제가 다르다. 실제 값을 남기고 0이 되지 않아야 한다."""
    sheet = offset_sheet(ready_config, table, 1)
    assert all(t["actual_r_offset_mm"] != 0 and t["actual_r_offset_mm"] * t["offset"] > 0 for t in sheet if t["axis"] == "r")
    assert all(t["actual_lateral_offset_mm"] * t["offset"] > 0 for t in sheet if t["axis"] == "lateral")
    assert all(t["actual_r_offset_mm"] == 0 and t["actual_lateral_offset_mm"] == 0 for t in sheet if t["axis"] == "j5")


def test_offset_pick_uses_offset_command(ready_config, table, tmp_path):
    """놓은 자세와 집는 자세가 실제로 달라야 허용 오차 시험이다."""
    driver, arm, ledger = make(ready_config, table, tmp_path)
    trial = next(t for t in offset_sheet(ready_config, table, 1) if t["axis"] == "lateral")
    run_replay(arm, FakeCamera(), ledger, [trial], lambda t: Outcome(True), table, "stored", lambda t: None)
    j1_values = {q5[0] for q5, _, _ in driver.writes if q5 != ready_config.park}
    assert {trial["j1"], trial["pick_j1"]} <= j1_values
