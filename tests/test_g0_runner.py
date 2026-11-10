from __future__ import annotations

from dataclasses import replace

import pytest

from learned_grasp.arm import FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import FakeCamera
from learned_grasp.g0_runner import (
    Outcome,
    TOL_TRIALS,
    compute_tolerance,
    dryrun_sheet,
    tolerance_sheet,
    tolerance_variants,
    variant_verdict,
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


# 플랜 v9 238~246행에서 실측한 표 모델 값. 표가 바뀌면 이 시험이 먼저 깨진다.
PLAN_STEPS = {
    "c0-r-": (1, 0, 90, 90, -4.29), "c0-r+": (1, 2, 90, 90, 5.622),
    "c1-r-": (4, 3, 90, 90, -5.554), "c1-r+": (4, 5, 90, 90, 4.398),
    "c2-r-": (7, 6, 90, 90, -5.581), "c2-r+": (7, 8, 90, 90, 3.951),
    "c0-lateral+": (1, 1, 91, 90, 2.165), "c1-lateral+": (4, 4, 91, 90, 2.438), "c2-lateral+": (7, 7, 91, 90, 2.708),
}


def test_tolerance_variants_match_plan(ready_config, table):
    variants = {v["variant_id"]: v for v in tolerance_variants(ready_config, table)}
    assert len(variants) == 18
    for vid, (r_index, pick_r, pick_j1, pick_j5, offset) in PLAN_STEPS.items():
        v = variants[vid]
        assert (v["r_index"], v["pick_r_index"], v["pick_j1"], v["pick_j5"], v["model_offset"]) == (r_index, pick_r, pick_j1, pick_j5, offset)
    for cell, r_index in enumerate((1, 4, 7)):
        lat = variants[f"c{cell}-lateral-"]
        assert (lat["pick_r_index"], lat["pick_j1"], lat["pick_j5"]) == (r_index, 89, 90)
        assert lat["model_offset"] == -variants[f"c{cell}-lateral+"]["model_offset"]
        for sign, j5 in (("-", 80), ("+", 100)):
            v = variants[f"c{cell}-j5{sign}"]
            assert (v["pick_r_index"], v["pick_j1"], v["pick_j5"], v["model_offset"]) == (r_index, 90, j5, float(j5 - 90))
    poses = {(v["pick_r_index"], v["pick_j1"], v["pick_j5"]) for v in variants.values()}
    assert len(poses) == 18
    assert all((v["pick_r_index"], v["pick_j1"], v["pick_j5"]) != (v["r_index"], v["j1"], v["j5"]) for v in variants.values())


def test_tolerance_sheet_fixed_interleaved_rounds(ready_config, table):
    sheet = tolerance_sheet(ready_config, table, 1)
    assert sheet == tolerance_sheet(ready_config, table, 1)
    assert sheet != tolerance_sheet(ready_config, table, 2)
    assert len(sheet) == 18 * TOL_TRIALS and len({t["trial_id"] for t in sheet}) == len(sheet)
    for round_index in range(TOL_TRIALS):
        block = sheet[18 * round_index:18 * (round_index + 1)]
        assert {t["round"] for t in block} == {round_index}
        assert len({t["variant_id"] for t in block}) == 18


def test_variant_verdict_matches_fixed_five_for_all_sequences():
    """32개 결과열 전부에서 조기 종결이 5회 고정·4성공 기준과 같은 결론을 낸다."""
    from itertools import product

    for sequence in product((True, False), repeat=TOL_TRIALS):
        fixed = "pass" if sum(sequence) >= 4 else "fail"
        early = next(v for n in range(1, TOL_TRIALS + 1) if (v := variant_verdict(list(sequence[:n]))) is not None)
        assert early == fixed, sequence


@pytest.mark.parametrize(("outcomes", "verdict"), [([True] * 3, None), ([True] * 4, "pass"), ([False, True, False], "fail"), ([True, False, True, True], None)])
def test_variant_verdict_stops_early(outcomes, verdict):
    assert variant_verdict(outcomes) == verdict


def _states(variants, status="pass"):
    return {v["variant_id"]: {"status": status} for v in variants}


def test_tolerance_all_pass_takes_smallest_step(ready_config, table):
    variants = tolerance_variants(ready_config, table)
    result = compute_tolerance(variants, _states(variants), "verified")
    assert (result["r"]["tol_cmd"], result["lateral"]["tol_cmd"], result["j5"]["tol_cmd"]) == (3.951, 2.165, 10.0)
    assert all(axis["usable"] and axis["confirmed"] for axis in result.values())


def test_tolerance_failure_gives_zero_and_unusable(ready_config, table):
    variants = tolerance_variants(ready_config, table)
    states = _states(variants)
    states["c1-j5-"] = {"status": "fail"}
    result = compute_tolerance(variants, states, "verified")
    assert result["j5"]["tol_cmd"] == 0.0 and not result["j5"]["usable"] and result["j5"]["confirmed"]
    assert result["r"]["usable"]


def test_tolerance_incomplete_or_missing_is_unconfirmed(ready_config, table):
    variants = tolerance_variants(ready_config, table)
    states = _states(variants)
    states["c0-r+"] = {"status": "incomplete", "reason": "안전상 미측정"}
    del states["c2-lateral-"]
    result = compute_tolerance(variants, states, "verified")
    assert result["r"]["confirmed"] is False and result["r"]["tol_cmd"] is None and not result["r"]["usable"]
    assert result["r"]["incomplete"] == {"c0-r+": "안전상 미측정"}
    assert result["lateral"]["incomplete"] == {"c2-lateral-": "기록 없음"} and not result["lateral"]["confirmed"]
    assert result["j5"]["confirmed"]


def test_tolerance_unverified_transfer_is_unusable(ready_config, table):
    variants = tolerance_variants(ready_config, table)
    result = compute_tolerance(variants, _states(variants), "unverified")
    assert not any(axis["usable"] for axis in result.values())
    assert all(axis["confirmed"] for axis in result.values())
    with pytest.raises(ValueError):
        compute_tolerance(variants, _states(variants), "maybe")
