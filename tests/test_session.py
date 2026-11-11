from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from learned_grasp.arm import FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.kinematics import fk_batch
from learned_grasp.pose_table import generate
from learned_grasp.safety import GRASP, HOVER, PARK_POSE, Pose, SafetyGate, SafetyViolation
from learned_grasp.session import (
    LADDER_OFFSETS_MM,
    CountingDriver,
    FIT_Z_MM,
    NeedHuman,
    PICK_BOARD_ROI,
    PICK_CHANGE_MIN,
    PICK_YELLOW_MAX,
    descend_and_stop,
    run_dabai_view,
    run_grip_max,
    run_p1,
    P1_GRIP_READ_MAX,
    fit_config,
    run_fit,
    run_carry_test,
    run_close_check,
    run_grip_test,
    make_block_loader,
    make_pick_observer,
    ladder_config,
    read_start_joints,
    record_observation,
    require_site_folder,
    require_previous_observation,
    run_ladder,
)

import numpy as np


def test_counting_driver_persists_across_runs(tmp_path):
    path = tmp_path / "writes.count"
    first = CountingDriver(FakeArm(), path, 3)
    first.write6((90, 90, 90, 90, 90), 100, 1500)
    first.write6((90, 90, 90, 90, 90), 100, 1500)
    second_fake = FakeArm()
    second = CountingDriver(second_fake, path, 3)
    second.write6((90, 90, 90, 90, 90), 100, 1500)
    with pytest.raises(SafetyViolation, match="상한"):
        second.write6((90, 90, 90, 90, 90), 100, 1500)
    assert len(second_fake.writes) == 1


def test_counting_driver_counts_failed_send(tmp_path):
    class Broken(FakeArm):
        def write6(self, q5, grip, time_ms):
            raise ConnectionError("단절")

    path = tmp_path / "writes.count"
    with pytest.raises(ConnectionError):
        CountingDriver(Broken(), path, 3).write6((90, 90, 90, 90, 90), 100, 1500)
    assert path.read_text() == "1"


@pytest.mark.parametrize("offset", LADDER_OFFSETS_MM)
def test_every_ladder_step_has_a_table(ready_config, offset):
    config = ladder_config(ready_config, offset)
    table = generate(config.table)
    assert len(table.entries) >= 2
    entry = table.entries[len(table.entries) // 2]
    position, _ = fk_batch(np.array([[90, *entry.grasp, 90]]))
    assert position[0, 2] == pytest.approx(ready_config.table.z_grasp_mm + offset, abs=ready_config.table.z_tol_mm + 1e-6)


def test_ladder_starts_above_and_descends(ready_config):
    heights = []
    for offset in LADDER_OFFSETS_MM:
        table = generate(ladder_config(ready_config, offset).table)
        entry = table.entries[len(table.entries) // 2]
        heights.append(fk_batch(np.array([[90, *entry.grasp, 90]]))[0][0, 2])
    assert heights == sorted(heights, reverse=True)
    assert LADDER_OFFSETS_MM[-1] == 0


def test_ladder_commands_only_one_descent(ready_config, tmp_path):
    config = ladder_config(ready_config, 15)
    table = generate(config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(config, table), sleep=lambda _: None)
    run_ladder(arm, AttemptLedger(tmp_path / "l.jsonl", clock=lambda: 0.0), config, table)
    # 첫 park(개방 포함) 1회와 모서리 6곳의 hover, grasp, hover, park. 그리퍼를 다시 쓰지 않는다
    assert len(fake.writes) == 1 + 6 * 4
    assert {q5[0] for q5, _, _ in fake.writes} >= set(config.j1_range)
    assert {grip for _, grip, _ in fake.writes} == {config.open_sdk}
    assert fake.writes[-1][0] == config.park


def test_hold_adds_no_writes_and_waits_at_grasp(ready_config, tmp_path):
    config = ladder_config(ready_config, 15)
    table = generate(config.table)
    fake = FakeArm()
    held = []

    def sleep(seconds):
        held.append((seconds, fake.writes[-1][0]))

    arm = SafeArm(fake, SafetyGate(config, table), sleep=sleep)
    run_ladder(arm, AttemptLedger(tmp_path / "h.jsonl", clock=lambda: 0.0), config, table, hold_s=40.0)
    assert len(fake.writes) == 1 + 6 * 4
    entry = table.entries[(len(table.entries) - 1) // 2]
    assert (40.0, (config.j1_range[0], *entry.grasp, 90)) in held


def test_ladder_records_do_not_mix_between_offsets(ready_config):
    versions = {ladder_config(ready_config, offset).config_version for offset in LADDER_OFFSETS_MM}
    assert len(versions) == len(LADDER_OFFSETS_MM)


class ChangeCamera:
    """yellow는 첫 노란색 영역의 값이다. 영역별로 주려면 {roi: 값}을 넘긴다."""

    def __init__(self, value, yellow=0):
        self.value = value
        self.yellow = yellow if isinstance(yellow, dict) else {PICK_BOARD_ROI[0]: yellow}
        self.names = []
        self.rois = []

    def snapshot(self, name):
        self.names.append(name)

    def change(self, before, after, roi):
        return self.value

    def yellow_pixels(self, name, roi):
        assert name in self.names
        self.rois.append(roi)
        return self.yellow.get(roi, 0)


def test_pick_observer_uses_image_change():
    silent = lambda *a, **k: None
    camera = ChangeCamera(0.03)
    assert make_pick_observer(camera, log=silent)({"trial_id": "t"}).success
    assert camera.names == ["t-after"]
    failed = make_pick_observer(ChangeCamera(0.001), log=silent)({"trial_id": "t"})
    assert not failed.success and failed.human_intervention


@pytest.mark.parametrize(
    ("change", "yellow", "success"),
    [(0.01, 100, True), (0.0099, 0, False), (0.01, 101, False), (0.3, 2000, False)],
)
def test_pick_observer_needs_change_and_clear_board(change, yellow, success):
    outcome = make_pick_observer(ChangeCamera(change, yellow), log=lambda *a, **k: None)({"trial_id": "t"})
    assert outcome.success is success
    assert f"{change:.4f}" in outcome.reason and str(yellow) in outcome.reason
    assert outcome.human_intervention is not success


def test_pick_board_rois_are_fixed():
    assert PICK_CHANGE_MIN == 0.01 and PICK_YELLOW_MAX == 100
    assert PICK_BOARD_ROI == ((585, 825, 150, 395), (565, 585, 250, 395))


@pytest.mark.parametrize(("first", "second", "success"), [(60, 40, True), (60, 41, False), (0, 101, False), (0, 100, True)])
def test_pick_observer_sums_board_rois(first, second, success):
    camera = ChangeCamera(0.05, {PICK_BOARD_ROI[0]: first, PICK_BOARD_ROI[1]: second})
    outcome = make_pick_observer(camera, log=lambda *a, **k: None)({"trial_id": "t"})
    assert outcome.success is success
    assert camera.rois == list(PICK_BOARD_ROI)
    assert str(first + second) in outcome.reason


def _loader_arm(ready_config):
    table = generate(ready_config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    arm.goto(PARK_POSE)
    return arm, fake


TRIAL = {"trial_id": "t", "r_index": 0, "j1": 90, "j5": 90}


def test_block_loader_waits_at_hover_until_ready(ready_config, tmp_path):
    arm, _ = _loader_arm(ready_config)
    ready = tmp_path / "ready"
    calls = []

    def sleep(seconds):
        calls.append(seconds)
        ready.write_text("1")

    make_block_loader(arm, ready, sleep=sleep, clock=lambda: 0.0, log=lambda *a, **k: None)(TRIAL)
    assert arm.pose.kind == "hover" and calls and not ready.exists()


def test_block_loader_ignores_stale_ready_and_times_out(ready_config, tmp_path):
    arm, _ = _loader_arm(ready_config)
    ready = tmp_path / "ready"
    ready.write_text("이전 시도의 확인")
    ticks = iter([0.0, 400.0])
    with pytest.raises(NeedHuman):
        make_block_loader(arm, ready, wait_s=300.0, sleep=lambda _: None, clock=lambda: next(ticks), log=lambda *a, **k: None)(TRIAL)


def test_descend_stops_low_without_returning(ready_config):
    table = generate(ready_config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    descend_and_stop(arm, ready_config, table)
    assert len(fake.writes) == 3 and arm.pose.kind == "grasp"
    assert fake.writes[-1][0][0] == 90 and {g for _, g, _ in fake.writes} == {ready_config.open_sdk}


def _fit_setup(ready_config, z):
    config = fit_config(ready_config, z)
    table = generate(config.table)
    fake = FakeArm()
    return config, table, fake, SafeArm(fake, SafetyGate(config, table), sleep=lambda _: None)


@pytest.mark.parametrize("z", FIT_Z_MM)
def test_fit_never_closes_and_returns(ready_config, z):
    config, table, fake, arm = _fit_setup(ready_config, z)
    loaded = []

    def load(trial):
        arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))
        loaded.append(fake.writes[-1][0])

    run_fit(arm, config, table, load, 0.0)
    assert {g for _, g, _ in fake.writes} == {config.open_sdk}
    assert len(fake.writes) == 5 and arm.pose.kind == "park" and len(loaded) == 1
    grasp = fake.writes[2][0]
    assert fk_batch(np.array([grasp]))[0][0, 2] == pytest.approx(z, abs=config.table.z_tol_mm + 1e-6)


def test_fit_rejects_other_height(ready_config):
    with pytest.raises(SafetyViolation):
        fit_config(ready_config, 0.0)


def test_grip_test_closes_lifts_and_opens_low(ready_config):
    config, table, fake, arm = _fit_setup(ready_config, 27.5)

    def load(trial):
        arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))

    run_grip_test(arm, config, table, load, 0.0)
    grips = [g for _, g, _ in fake.writes]
    assert config.close_sdk in grips and grips[-1] == config.open_sdk and not arm.holding
    assert len(fake.writes) == 9 and arm.pose.kind == "park"
    opened_at = next(i for i, (_, g, _) in enumerate(fake.writes) if i > 2 and g == config.open_sdk)
    assert fake.writes[opened_at][0] == fake.writes[opened_at - 1][0] != config.park


def _carry(ready_config, tmp_path, sleep=None):
    table = generate(ready_config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=sleep or (lambda _: None))
    ledger = AttemptLedger(tmp_path / "carry.jsonl")

    def load(trial):
        arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))

    return fake, arm, ledger, table, load


def test_carry_test_goes_to_park_closed_and_opens_at_grasp(ready_config, tmp_path):
    seen = []
    holder = {}

    def sleep(seconds):
        seen.append((len(holder["fake"].writes), seconds))

    fake, arm, ledger, table, load = _carry(ready_config, tmp_path, sleep)
    holder["fake"] = fake
    run_carry_test(arm, ledger, ready_config, table, load, 123.0, 1)
    assert len(fake.writes) == 11 and arm.pose.kind == "park"
    q = [write[0] for write in fake.writes]
    grips = [write[1] for write in fake.writes]
    assert q[0] == q[5] == q[10] == ready_config.park
    assert q[1] == q[4] == q[6] == q[9] and q[2] == q[3] == q[7] == q[8]
    assert grips[:3] == [ready_config.open_sdk] * 3
    assert grips[3:8] == [ready_config.close_sdk] * 5
    assert grips[8:] == [ready_config.open_sdk] * 3
    assert [entry for entry in seen if entry[1] == 123.0] == [(6, 123.0)]
    finished, open_ids = ledger.state()
    record = finished[f"carry-{ready_config.config_version}-1"]
    assert not open_ids and record["outcome"] == "ok"
    assert ledger.records()[0]["table_sha256"] == table.sha256()


@pytest.mark.parametrize("attempt", [0, 4])
def test_carry_test_rejects_attempt_out_of_range(ready_config, tmp_path, attempt):
    fake, arm, ledger, table, load = _carry(ready_config, tmp_path)
    with pytest.raises(SafetyViolation):
        run_carry_test(arm, ledger, ready_config, table, load, 0.0, attempt)
    assert fake.writes == []


def test_carry_test_rejects_repeated_attempt(ready_config, tmp_path):
    fake, arm, ledger, table, load = _carry(ready_config, tmp_path)
    run_carry_test(arm, ledger, ready_config, table, load, 0.0, 1)
    fake2, arm2, _, _, load2 = _carry(ready_config, tmp_path)
    with pytest.raises(SafetyViolation, match="이미 수행"):
        run_carry_test(arm2, ledger, ready_config, table, load2, 0.0, 1)
    assert fake2.writes == []


def test_carry_test_stops_after_crash_and_refuses_next_attempt(ready_config, tmp_path):
    def crash(seconds):
        if seconds == 123.0:
            raise ConnectionError("연결 단절")

    fake, arm, ledger, table, load = _carry(ready_config, tmp_path, crash)
    with pytest.raises(ConnectionError):
        run_carry_test(arm, ledger, ready_config, table, load, 123.0, 1)
    assert len(fake.writes) == 6
    fake2, arm2, _, _, load2 = _carry(ready_config, tmp_path)
    with pytest.raises(SafetyViolation, match="중단"):
        run_carry_test(arm2, ledger, ready_config, table, load2, 0.0, 2)
    assert fake2.writes == []
    # 한 번 닫힌 중단 기록도 다음 실행을 계속 막는다
    fake3, arm3, _, _, load3 = _carry(ready_config, tmp_path)
    with pytest.raises(SafetyViolation, match="중단"):
        run_carry_test(arm3, ledger, ready_config, table, load3, 0.0, 3)
    assert fake3.writes == []


def test_carry_test_records_read_while_holding(ready_config, tmp_path):
    fake, arm, ledger, table, load = _carry(ready_config, tmp_path)
    run_carry_test(arm, ledger, ready_config, table, load, 0.0, 1)
    record = ledger.state()[0][f"carry-{ready_config.config_version}-1"]
    assert "read_at_park" in record


def test_carry_test_closed_time_stays_under_limit(draft_config, tmp_path):
    config = replace(draft_config, verified_g0a=True)
    seen = []
    holder = {}

    def sleep(seconds):
        seen.append((len(holder["fake"].writes), seconds))

    fake, arm, ledger, table, load = _carry(config, tmp_path, sleep)
    holder["fake"] = fake
    run_carry_test(arm, ledger, config, table, load, 5.0, 1)
    closed = sum(seconds for count, seconds in seen if 4 <= count <= 9)
    assert 20.0 < closed <= 30.0


def test_close_check_closes_and_opens_at_grasp_only(ready_config):
    seen = []
    holder = {}

    def sleep(seconds):
        seen.append((len(holder["fake"].writes), seconds))

    table = generate(ready_config.table)
    fake = FakeArm()
    holder["fake"] = fake
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=sleep)
    run_close_check(arm, ready_config, table, 123.0)
    assert len(fake.writes) == 7 and arm.pose.kind == "park"
    q = [write[0] for write in fake.writes]
    grips = [write[1] for write in fake.writes]
    assert q[2] == q[3] == q[4] and q[1] == q[5] and q[0] == q[6] == ready_config.park
    assert grips == [ready_config.open_sdk] * 3 + [ready_config.close_sdk] + [ready_config.open_sdk] * 3
    assert [entry for entry in seen if entry[1] == 123.0] == [(4, 123.0)]


def test_close_check_uses_given_position(ready_config):
    table = generate(ready_config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    run_close_check(arm, ready_config, table, 0.0, 2)
    assert len(fake.writes) == 7
    assert fake.writes[2][0] == (90, *table.entries[2].grasp, 90)
    assert fake.writes[3][1] == ready_config.close_sdk


@pytest.mark.parametrize("r_index", [-1, 10])
def test_close_check_rejects_position_out_of_table(ready_config, r_index):
    table = generate(ready_config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    with pytest.raises(SafetyViolation):
        run_close_check(arm, ready_config, table, 0.0, r_index)
    assert fake.writes == []


def test_site_folder_must_match_position(tmp_path):
    require_site_folder(tmp_path / "G0b-5-r2", 2)
    with pytest.raises(SafetyViolation):
        require_site_folder(tmp_path / "G0b-5-r2", 0)
    with pytest.raises(SafetyViolation):
        require_site_folder(tmp_path / "G0b-5", 2)


def test_counting_driver_counts_grip_writes(tmp_path):
    fake = FakeArm()
    fake.writes.append(((90, 90, 90, 90, 90), 100, 1500))
    driver = CountingDriver(fake, tmp_path / "n", 1)
    driver.write_grip(120, 1500)
    with pytest.raises(SafetyViolation):
        driver.write_grip(130, 1500)


def test_descent_requires_previous_observation(tmp_path):
    require_previous_observation(tmp_path, 15)
    with pytest.raises(SafetyViolation, match="z\\+15"):
        require_previous_observation(tmp_path, 5)
    record_observation(tmp_path, 15, True, "간격 충분")
    require_previous_observation(tmp_path, 5)
    with pytest.raises(SafetyViolation):
        require_previous_observation(tmp_path, 0)


def test_bad_observation_stops_ladder(tmp_path):
    record_observation(tmp_path, 15, True, "")
    record_observation(tmp_path, 5, False, "닿을 듯함")
    with pytest.raises(SafetyViolation, match="이상"):
        require_previous_observation(tmp_path, 5)


def test_start_read_failure_blocks_motion(tmp_path):
    class Failing:
        def read(self):
            return {"values": [None] * 6, "status": ["fail"] * 6}

    with pytest.raises(SafetyViolation, match="시작 조회"):
        read_start_joints(Failing(), tmp_path)
    assert not (tmp_path / "start_read.json").exists()


def test_start_read_is_recorded(tmp_path):
    class Good:
        def read(self):
            return {"values": [82, 85, 25, 0, 90, 125], "status": ["sdk_return"] * 6}

    assert read_start_joints(Good(), tmp_path) == (82, 85, 25, 0, 90)
    assert (tmp_path / "start_read.json").exists()


def test_start_read_none_refuses_before_any_write(tmp_path):
    class NoneRead:
        def read(self):
            return {"ok": True, "values": [90, 90, 90, None, 90, 100], "status": ["sdk_return"] * 3 + ["fail"] + ["sdk_return"] * 2}

    with pytest.raises(SafetyViolation, match="시작 조회 실패"):
        read_start_joints(NoneRead(), tmp_path)


@pytest.mark.parametrize(
    ("fields", "ok"),
    [({"close_limit_s": 30, "release_s": 8}, True), ({"release_s": 8}, False), ({"close_limit_s": 30, "release_s": 0}, False), ({"close_limit_s": 10, "release_s": 10}, False), ({"close_limit_s": "30", "release_s": 8}, False)],
)
def test_g0d_limits_have_no_default(tmp_path, fields, ok):
    import json

    from learned_grasp.session import g0d_limits

    path = tmp_path / "approval.json"
    path.write_text(json.dumps(fields))
    if ok:
        assert g0d_limits(path) == (30.0, 8.0)
    else:
        with pytest.raises(SystemExit):
            g0d_limits(path)


def _dabai_arm(ready_config):
    table = generate(ready_config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    return arm, fake


def _signaling_sleep(signal, arrivals):
    """도착 줄이 나오면 촬영이 끝났다고 보고 신호 파일을 쓰는 가짜 사람."""

    def sleep(_):
        if arrivals and not signal.exists():
            signal.write_text("1")

    return sleep


def test_dabai_view_runs_five_writes_and_returns_to_park(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    signal = tmp_path / "dabai_next"
    lines = []
    run_dabai_view(
        arm, signal, sleep=_signaling_sleep(signal, lines), clock=lambda: 0.0,
        log=lambda line, **k: lines.append(line),
    )
    kinds = [w[0] for w in fake.writes]
    assert len(fake.writes) == 5 and arm.pose == PARK_POSE
    assert [w[1] for w in fake.writes] == [ready_config.open_sdk] * 5  # 닫지 않는다
    assert kinds[0] == kinds[4] == tuple(ready_config.park) and kinds[1] == kinds[3] and kinds[1] != kinds[2]
    assert lines == ['{"arrived": "hover"}', '{"arrived": "grasp"}'] and not signal.exists()


def test_dabai_view_does_not_descend_without_signal(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    ticks = iter([0.0, 400.0])
    with pytest.raises(NeedHuman):
        run_dabai_view(arm, tmp_path / "dabai_next", sleep=lambda _: None, clock=lambda: next(ticks), log=lambda *a, **k: None)
    assert len(fake.writes) == 3 and arm.pose == PARK_POSE  # park, hover, 복귀 park. grasp 없음


def test_dabai_view_timeout_at_grasp_returns_through_hover(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    signal = tmp_path / "dabai_next"
    state = {"waits": 0}

    def sleep(_):
        if state["waits"] == 0:  # 첫 대기(hover)에서만 신호를 준다
            signal.write_text("1")
            state["waits"] = 1

    clock_values = iter([0.0, 0.0, 0.0, 400.0])  # hover 대기 시작·확인, grasp 대기 시작, grasp 상한 초과
    with pytest.raises(NeedHuman):
        run_dabai_view(arm, signal, sleep=sleep, clock=lambda: next(clock_values), log=lambda *a, **k: None)
    assert len(fake.writes) == 5 and arm.pose == PARK_POSE  # park, hover, grasp, hover, park


def test_dabai_view_ignores_stale_signal(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    signal = tmp_path / "dabai_next"
    signal.write_text("이전 신호")
    ticks = iter([0.0, 400.0])
    with pytest.raises(NeedHuman):
        run_dabai_view(arm, signal, sleep=lambda _: None, clock=lambda: next(ticks), log=lambda *a, **k: None)
    assert len(fake.writes) == 3 and not signal.exists()


def test_dabai_view_safety_violation_sends_no_recovery_writes(ready_config, tmp_path):
    class Stuck(FakeArm):
        def read(self):
            return {"values": [0, 0, 0, 0, 0, 100], "status": ["sdk_return"] * 6}

    stuck = Stuck()
    arm = SafeArm(stuck, SafetyGate(ready_config, generate(ready_config.table)), sleep=lambda _: None)
    with pytest.raises(SafetyViolation):
        run_dabai_view(arm, tmp_path / "dabai_next", sleep=lambda _: None, clock=lambda: 0.0, log=lambda *a, **k: None)
    assert len(stuck.writes) == 1  # 첫 park 쓰기 뒤 도착 검사 실패로 멈춘다. 자동 복귀 쓰기 없음


def test_dabai_view_hover_only_never_goes_to_grasp(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    signal = tmp_path / "dabai_next"
    lines = []
    run_dabai_view(
        arm, signal, sleep=_signaling_sleep(signal, lines), clock=lambda: 0.0,
        log=lambda line, **k: lines.append(line), hover_only=True,
    )
    assert len(fake.writes) == 3 and arm.pose == PARK_POSE
    assert lines == ['{"arrived": "hover"}'] and not signal.exists()
    assert fake.writes[1][0] != fake.writes[0][0]  # hover가 park와 다른 자세
    assert all(w[1] == ready_config.open_sdk for w in fake.writes)  # 닫지 않는다


def test_dabai_view_hover_only_waits_for_signal_and_returns_on_timeout(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    ticks = iter([0.0, 400.0])
    with pytest.raises(NeedHuman):
        run_dabai_view(arm, tmp_path / "dabai_next", sleep=lambda _: None, clock=lambda: next(ticks), log=lambda *a, **k: None, hover_only=True)
    assert len(fake.writes) == 3 and arm.pose == PARK_POSE  # park, hover, 복귀 park. grasp 없음


def test_dabai_view_uses_requested_r_index(ready_config, tmp_path):
    table = generate(ready_config.table)
    arm, fake = _dabai_arm(ready_config)
    signal = tmp_path / "dabai_next"
    run_dabai_view(arm, signal, sleep=_signaling_sleep(signal, [1]), clock=lambda: 0.0, log=lambda *a, **k: None, r_index=0, hover_only=True)
    q = tuple(fake.writes[1][0])
    assert q == (90,) + tuple(table.entries[0].hover)[:3] + (90,), (q, table.entries[0].hover)


def test_dabai_view_cli_rejects_hover_only_on_other_commands(monkeypatch, tmp_path):
    import sys

    from learned_grasp import session

    approval = tmp_path / "approval.json"
    approval.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["session", "park", "--session", str(approval), "--hover-only"])
    with pytest.raises(SystemExit, match="dabai-view에서만"):
        session.main()


# --- grip-max: close_sdk 위로 5씩 올리는 빈 손 단계 측정 ---------------------------------------------


class LaggingFake(FakeArm):
    """서보 6 판독이 limit을 넘지 못하는 가짜 드라이버(손가락이 막힌 상태)."""

    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def read(self):
        reading = super().read()
        values = reading.get("values")
        if self.limit is not None and values and len(values) > 5:
            values[5] = min(values[5], self.limit)
        return reading


def _grip_arm(ready_config, fake=None, sweep=True):
    table = generate(ready_config.table)
    fake = fake or FakeArm()
    arm = SafeArm(fake, SafetyGate(ready_config, table, grip_sweep=sweep), sleep=lambda _: None)
    return arm, fake


def _grip_run(ready_config, tmp_path, fake=None, max_sdk=None, signal_every_wait=True, **kwargs):
    arm, fake = _grip_arm(ready_config, fake)
    signal, stop = tmp_path / "dabai_next", tmp_path / "grip_stop"
    lines = []

    def sleep(_):
        if signal_every_wait and not signal.exists():
            signal.write_text("1")

    run_grip_max(
        arm, ready_config, signal, stop, tmp_path, 4,
        max_sdk=max_sdk if max_sdk is not None else ready_config.close_sdk + 10,
        sleep=sleep, clock=kwargs.pop("clock", lambda: 0.0), log=lambda line, **k: lines.append(line), **kwargs,
    )
    return arm, fake, lines, [json.loads(line) for line in (tmp_path / "grip-max.jsonl").read_text(encoding="utf-8").splitlines()]


def test_grip_max_steps_up_by_five_then_opens_and_returns(ready_config, tmp_path):
    arm, fake, lines, ledger = _grip_run(ready_config, tmp_path)
    grips = [w[1] for w in fake.writes]
    close = ready_config.close_sdk
    # park, hover, grasp(열림) | 닫기, +5, +10 | 열기 | hover, park
    assert grips == [ready_config.open_sdk] * 3 + [close, close + 5, close + 10] + [ready_config.open_sdk] * 3
    assert len(fake.writes) == 9 and arm.pose == PARK_POSE and not arm.holding
    assert [item["step"] for item in ledger] == ["open", "close", "close_settled", "level", "level", "end"]
    assert ledger[-1]["reason"] == "max_sdk 도달"
    assert lines[0] == '{"arrived": "open"}' and lines[-1] == '{"end": "max_sdk 도달"}'


def test_grip_max_auto_stops_when_servo_reading_lags(ready_config, tmp_path):
    close = ready_config.close_sdk
    fake = LaggingFake(limit=close + 1)  # 닫기 150은 판독 150, 155부터 판독이 막힌다
    arm, fake, lines, ledger = _grip_run(ready_config, tmp_path, fake=fake, max_sdk=close + 20)
    assert [w[1] for w in fake.writes][-3:] == [ready_config.open_sdk] * 3  # 열고 hover, park
    assert max(w[1] for w in fake.writes) == close + 5  # 한 단계 올린 뒤 막힘을 보고 더 올리지 않는다
    assert ledger[-1]["reason"] == "막힘(서보 판독이 명령에 못 미침)"
    assert arm.pose == PARK_POSE


def test_grip_max_stop_file_ends_sweep_and_opens(ready_config, tmp_path):
    arm, fake = _grip_arm(ready_config)
    signal, stop = tmp_path / "dabai_next", tmp_path / "grip_stop"
    waits = {"n": 0}

    def sleep(_):
        waits["n"] += 1
        if waits["n"] == 1:
            signal.write_text("1")  # open 대기 통과
        elif waits["n"] == 2:
            signal.write_text("1")  # close 대기 통과
        else:
            stop.write_text("1")  # 첫 레벨 대기에서 정지 요청

    run_grip_max(arm, ready_config, signal, stop, tmp_path, 4, max_sdk=ready_config.close_sdk + 10,
                 sleep=sleep, clock=lambda: 0.0, log=lambda *a, **k: None)
    ledger = [json.loads(line) for line in (tmp_path / "grip-max.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ledger[-1]["reason"] == "정지 파일"
    assert max(w[1] for w in fake.writes) == ready_config.close_sdk + 5
    assert [w[1] for w in fake.writes][-3:] == [ready_config.open_sdk] * 3 and arm.pose == PARK_POSE


def test_grip_max_timeout_opens_returns_then_raises(ready_config, tmp_path):
    arm, fake = _grip_arm(ready_config)
    signal, stop = tmp_path / "dabai_next", tmp_path / "grip_stop"
    ticks = iter([0.0, 0.0, 400.0])  # 첫 대기(open)에서 상한 초과
    with pytest.raises(NeedHuman):
        run_grip_max(arm, ready_config, signal, stop, tmp_path, 4, max_sdk=ready_config.close_sdk + 10,
                     sleep=lambda _: None, clock=lambda: next(ticks), log=lambda *a, **k: None)
    assert [w[1] for w in fake.writes] == [ready_config.open_sdk] * 5  # park, hover, grasp, hover, park. 닫지 않았다
    assert arm.pose == PARK_POSE and not arm.holding


def test_grip_max_rejects_out_of_range_max_sdk_before_any_write(ready_config, tmp_path):
    arm, fake = _grip_arm(ready_config)
    for bad in (ready_config.close_sdk, 181):
        with pytest.raises(SafetyViolation):
            run_grip_max(arm, ready_config, tmp_path / "s", tmp_path / "t", tmp_path, 4, max_sdk=bad,
                         sleep=lambda _: None, clock=lambda: 0.0, log=lambda *a, **k: None)
    assert fake.writes == []


def test_grip_max_requires_gate_flag(ready_config, tmp_path):
    arm, fake = _grip_arm(ready_config, sweep=False)
    signal = tmp_path / "dabai_next"
    with pytest.raises(SafetyViolation):  # level 동작은 grip_sweep 없이는 거절, 복구 쓰기 없음
        run_grip_max(arm, ready_config, signal, tmp_path / "grip_stop", tmp_path, 4, max_sdk=ready_config.close_sdk + 10,
                     sleep=lambda _: signal.write_text("1"), clock=lambda: 0.0, log=lambda *a, **k: None)
    assert max(w[1] for w in fake.writes) == ready_config.close_sdk  # 닫기까지만 나갔다


def test_grip_max_does_not_step_up_when_already_lagging_at_close_sdk(ready_config, tmp_path):
    close = ready_config.close_sdk
    fake = LaggingFake(limit=close - 5)  # 빈 손인데 close_sdk에서 이미 판독이 5도 뒤진다(손가락 사이에 무엇이 있다)
    arm, fake, lines, ledger = _grip_run(ready_config, tmp_path, fake=fake, max_sdk=close + 20)
    assert max(w[1] for w in fake.writes) == close  # 레벨을 올리지 않았다
    assert [w[1] for w in fake.writes][-3:] == [ready_config.open_sdk] * 3 and arm.pose == PARK_POSE
    assert ledger[-1]["reason"].startswith("close_sdk에서 이미 판독이 뒤짐")


def test_dabai_view_uses_requested_j1(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    signal = tmp_path / "dabai_next"
    lines = []
    run_dabai_view(arm, signal, sleep=_signaling_sleep(signal, lines), clock=lambda: 0.0, log=lambda line, **k: lines.append(line), j1=70)
    assert [w[0][0] for w in fake.writes] == [ready_config.park[0], 70, 70, 70, ready_config.park[0]]  # J1 70에서 hover, grasp, hover
    assert len(fake.writes) == 5 and arm.pose == PARK_POSE


def test_dabai_view_rejects_j1_outside_range(ready_config, tmp_path):
    arm, fake = _dabai_arm(ready_config)
    with pytest.raises(SafetyViolation):
        run_dabai_view(arm, tmp_path / "dabai_next", sleep=lambda _: None, clock=lambda: 0.0, log=lambda *a, **k: None, j1=60)
    assert len(fake.writes) == 1  # park만 나갔고 hover로 가기 전에 거절


def test_dabai_view_cli_rejects_j1_on_other_commands(monkeypatch, tmp_path):
    from learned_grasp import session

    monkeypatch.setattr("sys.argv", ["session", "read", "--session", str(tmp_path / "approval.json"), "--j1", "70"])
    with pytest.raises(SystemExit) as info:
        session.main()
    assert "--j1은 dabai-view, p1-pick, release-here에서만" in str(info.value)


# --- p1-pick: 사람이 놓은 블록을 L1이 정한 자세에서 집는 시도 ----------------------------------------


class _Cam:
    def __init__(self):
        self.snaps = []

    def snapshot(self, name):
        self.snaps.append(name)


class _GripFake(FakeArm):
    """close 뒤 서보 6 판독을 grip_after로 돌려주는 가짜 드라이버."""

    def __init__(self, grip_after):
        super().__init__()
        self.grip_after = grip_after

    def read(self):
        reading = super().read()
        values = reading.get("values")
        if values and len(values) > 5 and values[5] > 100:
            values[5] = self.grip_after
        return reading


def _p1_run(ready_config, tmp_path, grip_after=148, observer_success=True, signal_ok=True, r_index=4, j1=80):
    from learned_grasp.g0_runner import Outcome

    table = generate(ready_config.table)
    fake = _GripFake(grip_after)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    ledger = AttemptLedger(tmp_path / "p1.jsonl")
    cam = _Cam()
    lines = []

    def sleep(_):
        if signal_ok and not signal.exists():
            signal.write_text("1")

    ticks = iter([0.0, 0.0, 400.0])
    kwargs = dict(sleep=sleep, clock=lambda: 0.0 if signal_ok else next(ticks), log=lambda line, **k: lines.append(line))
    observer = lambda trial: Outcome(observer_success, "관찰기")
    try:
        run_p1(arm, cam, observer, ledger, ready_config, table, {"trial_id": "p1-01", "r_index": r_index, "j1": j1, "j5": 90}, signal, hold_s=0.0, **kwargs)
        error = None
    except NeedHuman as exc:
        error = exc
    return arm, fake, ledger, cam, error


def test_p1_success_requires_observer_and_servo_reading(ready_config, tmp_path):
    arm, fake, ledger, cam, error = _p1_run(ready_config, tmp_path, grip_after=P1_GRIP_READ_MAX - 3)
    assert error is None and cam.snaps == ["p1-01"]
    # park, hover | grasp, 닫기, hover, park | hover, grasp, 열기, hover, park
    assert len(fake.writes) == 11 and arm.pose == PARK_POSE and not arm.holding
    finish = [r for r in ledger.records() if r["event"] == "finish"][-1]
    assert finish["success"] is True and finish["outcome"] == "success"
    assert {w[0][0] for w in fake.writes if w[0] != tuple(ready_config.park)} == {80}  # J1 80에서 hover·grasp


def test_p1_fails_when_servo_reads_empty_hand_even_if_observer_passes(ready_config, tmp_path):
    arm, fake, ledger, cam, error = _p1_run(ready_config, tmp_path, grip_after=P1_GRIP_READ_MAX + 3)
    finish = [r for r in ledger.records() if r["event"] == "finish"][-1]
    assert finish["success"] is False and "빈 손" in finish["reason"]


def test_p1_fails_when_observer_fails(ready_config, tmp_path):
    arm, fake, ledger, cam, error = _p1_run(ready_config, tmp_path, observer_success=False)
    finish = [r for r in ledger.records() if r["event"] == "finish"][-1]
    assert finish["success"] is False and finish["outcome"] == "failed"


def test_p1_does_not_descend_without_confirmation(ready_config, tmp_path):
    arm, fake, ledger, cam, error = _p1_run(ready_config, tmp_path, signal_ok=False)
    assert isinstance(error, NeedHuman)
    assert len(fake.writes) == 3 and arm.pose == PARK_POSE  # park, hover, 복귀 park. grasp 없음, 닫지 않음
    finish = [r for r in ledger.records() if r["event"] == "finish"][-1]
    assert finish["outcome"] == "interrupted"


def test_p1_refuses_repeated_trial_id(ready_config, tmp_path):
    _p1_run(ready_config, tmp_path)
    with pytest.raises(SafetyViolation):
        _p1_run(ready_config, tmp_path)


# --- chain: L1이 고른 자리에서 집고 목록의 다음 자리에 내려놓는 연쇄 시험 ---------------------------------


class _ChainCam(_Cam):
    def fetch(self, name):
        return Path(name)


def _measured(j1c, r_mm, psi_offset=0.0):
    from learned_grasp.l1 import L1Map

    l1 = L1Map()
    du, dv = l1.forward(j1c, r_mm)
    return {"du": float(du), "dv": float(dv), "block_angle_deg": float(l1.radial_angle_deg(j1c, r_mm) % 180.0 + psi_offset), "area": 2300, "board": [691.0, 291.0]}


def _chain_run(ready_config, tmp_path, measures, cells, count, grip_after=148, observer_success=True, signal_ok=True, ledger=None, signals=None, **extra):
    from learned_grasp.g0_runner import Outcome
    from learned_grasp.session import run_chain

    table = generate(ready_config.table)
    fake = _GripFake(grip_after)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    ledger = ledger or AttemptLedger(tmp_path / "chain.jsonl")
    cam = _ChainCam()
    queue = iter(measures)
    given = [0]

    def sleep(_):
        if signal_ok and not signal.exists() and (signals is None or given[0] < signals):
            signal.write_text("1")
            given[0] += 1

    # 신호를 주지 않는 대기는 시계가 마감을 넘기게 한다: 대기 시작(clock 호출 1회) 뒤 첫 확인에서 400초
    clock_calls = [0]

    def clock():
        if signal_ok and (signals is None or given[0] < signals):
            return 0.0
        clock_calls[0] += 1
        return 0.0 if clock_calls[0] % 2 == 1 else 400.0

    kwargs = dict(sleep=sleep, clock=clock, log=lambda line, **k: None)
    observer = lambda trial: Outcome(observer_success, "관찰기")
    error = None
    try:
        run_chain(arm, cam, observer, ledger, ready_config, table, cells, count, signal, lambda path: next(queue), hold_s=0.0, **kwargs, **extra)
    except (NeedHuman, SafetyViolation) as exc:
        error = exc
    return arm, fake, ledger, cam, error, table


def test_chain_picks_at_l1_cell_and_places_at_next_cell(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0), _measured(99.8, 150.0)], cells, 2)
    assert error is None and cam.snaps == ["chain-001", "chain-002", "chain-003"]  # 마지막은 놓은 블록 사진
    finished, _ = ledger.state()
    assert [finished[t]["outcome"] for t in ("chain-001", "chain-002")] == ["success", "success"]
    starts = [r for r in ledger.records() if r.get("event") == "start"]
    assert starts[0]["plan"]["j1"] == 85 and starts[0]["place_at"] == {"r_index": 6, "j1": 100}
    gate = arm.gate
    assert tuple(fake.writes[-1][0]) == tuple(gate.joints(PARK_POSE))  # 마지막에 park
    place_hover = tuple(gate.joints(Pose(HOVER, 6, 100, 90)))
    assert place_hover in [tuple(w[0]) for w in fake.writes]  # 목록의 자리로 갔다
    assert arm.pose == PARK_POSE and not arm.holding


def test_chain_stops_without_moving_when_l1_rejects(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0, psi_offset=12.0)], cells, 1)
    assert isinstance(error, NeedHuman) and "L1이 시도하지 않는다" in str(error)
    assert fake.writes == []
    assert any(r.get("event") == "note" and r.get("kind") == "rejected" for r in ledger.records())


def test_chain_failure_puts_block_back_at_same_cell_and_stops(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 2, grip_after=P1_GRIP_READ_MAX + 3)
    assert isinstance(error, NeedHuman) and "실패" in str(error)
    finished, _ = ledger.state()
    assert finished["chain-001"]["outcome"] == "failed"
    assert tuple(arm.gate.joints(Pose(HOVER, 6, 100, 90))) not in [tuple(w[0]) for w in fake.writes]  # 다음 자리로 가지 않았다
    assert arm.pose == PARK_POSE
    # 실패한 기록이 있으면 다시 이어 가지 않는다
    _, _, _, _, again, _ = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, ledger=ledger)
    assert isinstance(again, SafetyViolation)


def test_chain_does_not_descend_without_confirmation(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, signal_ok=False)
    assert isinstance(error, NeedHuman)
    finished, _ = ledger.state()
    assert finished["chain-001"]["outcome"] == "interrupted"
    assert all(tuple(w[0]) != tuple(arm.gate.joints(Pose(GRASP, 4, 85, 90))) for w in fake.writes)
    assert arm.pose == PARK_POSE


def test_chain_refuses_when_cells_run_out(ready_config, tmp_path):
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [], [{"r_index": 6, "j1": 100}], 2)
    assert isinstance(error, SafetyViolation) and fake.writes == []


def test_chain_place_descent_needs_confirmation_too(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, signals=1, confirm_each=True)
    assert isinstance(error, NeedHuman) and "내려놓기 전 확인" in str(error)
    finished, _ = ledger.state()
    assert finished["chain-001"]["outcome"] == "interrupted" and finished["chain-001"]["picked"] is True
    writes = [tuple(w[0]) for w in fake.writes]
    assert tuple(arm.gate.joints(Pose(GRASP, 6, 100, 90))) not in writes  # 내려놓는 하강을 하지 않았다
    assert tuple(arm.gate.joints(Pose(HOVER, 6, 100, 90))) in writes and arm.pose == PARK_POSE and arm.holding


def test_chain_keeps_rejected_photo_and_records_same_cell(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0, psi_offset=12.0)], cells, 1)
    assert isinstance(error, NeedHuman) and cam.snaps == ["chain-001"]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, ledger=ledger)
    assert error is None and cam.snaps == ["chain-001-r1", "chain-002"]  # 거절된 사진 이름을 다시 쓰지 않는다
    start = [r for r in ledger.records() if r.get("event") == "start"][0]
    assert start["before_frame"] == "chain-001-r1" and start["same_cell"] is False
    # 두 번째 시도는 직전에 내려놓은 자리(6, 100)를 L1이 다시 고르면 같은 자리로 기록한다
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(100.0, 149.6)], cells, 1, ledger=ledger)
    second = [r for r in ledger.records() if r.get("event") == "start"][1]
    assert second["placed_at"] == {"r_index": 6, "j1": 100} and second["same_cell"] is True, second


def test_chain_default_confirms_only_the_first_pick(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}, {"r_index": 4, "j1": 90}]
    arm, fake, ledger, cam, error, table = _chain_run(
        ready_config, tmp_path, [_measured(85.2, 140.0), _measured(99.8, 150.0), _measured(76.1, 129.7)], cells, 3, signals=1)
    assert error is None  # 신호 1개로 3시도가 끝난다: 첫 집기만 확인을 기다렸다
    finished, _ = ledger.state()
    assert [finished[t]["outcome"] for t in ("chain-001", "chain-002", "chain-003")] == ["success"] * 3


def _rest_rig(ready_config, tmp_path, stop=None, stop_when_rest=False, rest_every=2):
    from learned_grasp.g0_runner import Outcome
    from learned_grasp.session import run_chain

    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}, {"r_index": 4, "j1": 90}]
    table = generate(ready_config.table)
    fake = _GripFake(148)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    queue = iter([_measured(85.2, 140.0), _measured(99.8, 150.0), _measured(76.1, 129.7)])
    now = [0.0]
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
        if seconds == 1.0 and stop_when_rest and now[0] > 5.0 and stop is not None:
            stop.write_text("1")
        if not signal.exists() and not stop_when_rest:
            signal.write_text("1")
        if not signal.exists() and stop_when_rest:
            signal.write_text("1")

    ledger = AttemptLedger(tmp_path / "chain.jsonl")
    error = None
    try:
        run_chain(arm, _ChainCam(), lambda t: Outcome(True, "관찰기"), ledger, ready_config, table, cells, 3, signal, lambda path: next(queue),
                  hold_s=0.0, sleep=sleep, clock=lambda: now[0], log=lambda *a, **k: None, rest_every=rest_every, rest_s=10.0, stop=stop)
    except NeedHuman as exc:
        error = exc
    return fake, ledger, sleeps, error


def test_chain_rests_every_n_closes_without_writes(ready_config, tmp_path):
    fake, ledger, sleeps, error = _rest_rig(ready_config, tmp_path)
    assert error is None
    assert sum(1 for s in sleeps if s == 1.0) >= 10  # 쉬는 동안 1초씩 기다렸다(마지막 시도 뒤에는 쉬지 않는다)
    finished, _ = ledger.state()
    assert len(finished) == 3 and len(fake.writes) == 31  # 첫 시도 park 1 + 시도당 10: 쉬는 동안 쓰기 없음


def test_chain_stop_file_stops_between_trials_without_writes(ready_config, tmp_path):
    stop = tmp_path / "chain_stop"
    stop.write_text("1")
    fake, ledger, sleeps, error = _rest_rig(ready_config, tmp_path, stop=stop)
    assert isinstance(error, NeedHuman) and "정지 파일" in str(error) and fake.writes == []
    assert any(r.get("kind") == "stop_requested" for r in ledger.records())


def test_chain_stop_file_during_rest_stops_at_park(ready_config, tmp_path):
    stop = tmp_path / "chain_stop"
    fake, ledger, sleeps, error = _rest_rig(ready_config, tmp_path, stop=stop, stop_when_rest=True)
    assert isinstance(error, NeedHuman) and "쉬는 중" in str(error)
    finished, _ = ledger.state()
    assert len(finished) == 2 and all(r["outcome"] == "success" for r in finished.values())  # 2번째 닫기 뒤 쉼에서 멈춤
    assert tuple(fake.writes[-1][0]) == tuple(generate(ready_config.table) and SafetyGate(ready_config, generate(ready_config.table)).joints(PARK_POSE))


def test_chain_resumes_after_failure_only_with_acknowledgement(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}, {"r_index": 4, "j1": 90}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, grip_after=P1_GRIP_READ_MAX + 3)
    assert isinstance(error, NeedHuman)
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, ledger=ledger)
    assert isinstance(error, SafetyViolation) and "--ack" in str(error)
    ledger.note("acknowledged", trial_id="chain-001")
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(76.1, 129.7)], cells, 1, ledger=ledger)
    assert error is None
    start = [r for r in ledger.records() if r.get("event") == "start"][1]
    assert start["trial_id"] == "chain-002" and start["placed_at"] is None and start["same_cell"] is False  # 실패 뒤라 직전 놓은 자리를 모른다


def _aligned(j1c, r_mm, r_index, j5):
    from learned_grasp.l1 import L1Map, YawModel

    l1, yaw = L1Map(), YawModel()
    du, dv = l1.forward(j1c, r_mm, j5)
    block = (l1.radial_angle_deg(j1c, r_mm) + yaw.hand_d(j5, round(j1c), r_index)) % 180.0
    return {"du": float(du), "dv": float(dv), "block_angle_deg": float(block), "area": 2300, "board": [691.0, 291.0]}


def test_chain_align_uses_j5_for_pick_and_place_and_confirms_j5_only_once(ready_config, tmp_path):
    from learned_grasp.l1 import YawModel

    yaw = YawModel()
    table = generate(ready_config.table)
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    first = _aligned(85.0, table.entries[4].r_mm, 4, 72)
    second = _aligned(100.0, table.entries[6].r_mm, 6, yaw.place_j5(100, 6))
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [first, second], cells, 2, signals=2, align=True)
    assert error is None, error
    starts = [r for r in ledger.records() if r.get("event") == "start"]
    assert abs(starts[0]["pick_j5"] - 72) <= 1 and starts[0]["place_j5"] == yaw.place_j5(100, 6) and starts[0]["place_j5"] != 90
    j5_written = {w[0][4] for w in fake.writes}
    assert starts[0]["pick_j5"] in j5_written and starts[0]["place_j5"] in j5_written  # 5번째 축(J5)이 실제로 바뀌어 나갔다
    finished, _ = ledger.state()
    assert [finished[t]["outcome"] for t in ("chain-001", "chain-002")] == ["success", "success"]  # 신호 2개: 첫 집기와 첫 J5≠90 내려놓기


def test_chain_align_rejects_block_that_needs_j5_outside_range(ready_config, tmp_path):
    table = generate(ready_config.table)
    m = _aligned(85.0, table.entries[4].r_mm, 4, 90)
    m["block_angle_deg"] = (m["block_angle_deg"] + 50.0) % 180.0
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [m], [{"r_index": 6, "j1": 100}], 1, align=True)
    assert isinstance(error, NeedHuman) and "J5 보정" in str(error) and fake.writes == []


def test_release_here_opens_at_grasp_then_hover_and_park(ready_config):
    from learned_grasp.session import release_here

    table = generate(ready_config.table)
    gate = SafetyGate(ready_config, table)
    fake = FakeArm()
    fake.write6(gate.joints(Pose(GRASP, 1, 76, 93)), ready_config.close_sdk, 100)  # 블록을 쥔 채 멈춘 상태
    arm = SafeArm(fake, gate, sleep=lambda _: None)
    release_here(arm, 1, 76, 93)
    after = fake.writes[1:]
    assert [w[1] for w in after] == [ready_config.open_sdk, ready_config.open_sdk, ready_config.open_sdk]  # 열고, 열린 채 hover, park
    assert tuple(after[1][0]) == tuple(gate.joints(Pose(HOVER, 1, 76, 93))) and tuple(after[2][0]) == tuple(gate.joints(PARK_POSE))
    assert arm.pose == PARK_POSE and not arm.holding


def test_release_here_refuses_when_reading_does_not_match(ready_config):
    from learned_grasp.session import release_here

    table = generate(ready_config.table)
    gate = SafetyGate(ready_config, table)
    fake = FakeArm()
    fake.write6(gate.joints(Pose(GRASP, 3, 90, 90)), ready_config.close_sdk, 100)  # 다른 자리에 서 있다
    arm = SafeArm(fake, gate, sleep=lambda _: None)
    with pytest.raises(SafetyViolation, match="맞지 않아"):
        release_here(arm, 1, 76, 93)
    assert len(fake.writes) == 1


# --- R1 반경 오프셋, 빈 손 실패 이어 가기, P2 J5 확인 상승·비틀림 사진 ------------------------------------------


def _rows(ready_config):
    return [e.r_mm for e in generate(ready_config.table).entries]


def test_chain_offsets_shift_pick_row_recompute_j5_and_compensate_place_row(ready_config, tmp_path):
    from learned_grasp.l1 import YawModel

    yaw = YawModel()
    rows = _rows(ready_config)
    cells = [{"r_index": 5, "j1": 90}, {"r_index": 4, "j1": 80}]
    first = _aligned(85.0, rows[4], 4, 90)
    second = _aligned(90.0, rows[5], 5, yaw.place_j5(90, 5 + 3 - 3 + 0))
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [first, second], cells, 2, signals=9, align=True, offsets=[3, -3], continue_on_miss=True)
    assert error is None, error
    s1, s2 = [r for r in ledger.records() if r.get("event") == "start"]
    assert (s1["r_l1"], s1["r_pick"], s1["offset"]) == (4, 7, 3) and s1["place_row"] == 5 + 3 and s1["place_at"] == {"r_index": 5, "j1": 90}
    assert s1["pick_j5"] == int(round(yaw.j5_for(s1["plan"]["d_deg"], 85, 7)))  # 실제 집는 행으로 J5를 다시 계산했다
    assert s1["place_j5"] == yaw.place_j5(90, 8)
    assert tuple(arm.gate.joints(Pose(HOVER, 8, 90, s1["place_j5"]))) in [tuple(w[0]) for w in fake.writes]  # 놓는 손 행 = 공칭 + 오프셋
    assert s2["offset"] == -3 and s2["r_pick"] == s2["r_l1"] - 3


def test_chain_continue_on_miss_goes_on_after_empty_hand_failure(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    arm, fake, ledger, cam, error, table = _chain_run(
        ready_config, tmp_path, [_measured(85.2, 140.0), _measured(85.2, 140.0)], cells, 2, grip_after=P1_GRIP_READ_MAX + 3, continue_on_miss=True)
    assert error is None, error  # 첫 실패(빈 손)에서 멈추지 않고 두 번째 시도까지 한다
    finished, _ = ledger.state()
    assert [finished[t]["outcome"] for t in ("chain-001", "chain-002")] == ["failed", "failed"]
    # 재시작 때도 빈 손 실패는 사람 확인을 요구하지 않는다
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [], cells, 0, ledger=ledger, continue_on_miss=True)
    assert error is None


def test_chain_continue_on_miss_still_stops_when_hand_holds_but_observer_fails(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 2, grip_after=148, observer_success=False, continue_on_miss=True)
    assert isinstance(error, NeedHuman) and "실패" in str(error)  # 서보가 쥔 값인데 관찰기 실패 = 오판정 의심이라 멈춘다


def _p1_aligned(ready_config, tmp_path, j5, confirm_file, name):
    from learned_grasp.g0_runner import Outcome

    table = generate(ready_config.table)
    fake = _GripFake(148)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    ledger = AttemptLedger(tmp_path / "p1.jsonl")
    cam = _Cam()
    asked = []

    def sleep(_):
        if not signal.exists():
            signal.write_text("1")

    run_p1(arm, cam, lambda t: Outcome(True, "관찰기"), ledger, ready_config, table, {"trial_id": name, "r_index": 4, "j1": 80, "j5": j5}, signal, hold_s=0.0,
           sleep=sleep, clock=lambda: 0.0, log=lambda line, **k: asked.append(line), confirm_dev_file=confirm_file, twist_photo=True)
    return fake, cam, asked


def test_p1_j5_confirmation_escalates_by_ten_degrees_and_takes_twist_photo(ready_config, tmp_path):
    confirm = tmp_path / "j5_dev_confirmed"
    results = {}
    for name, j5 in (("p2-01", 85), ("p2-02", 90), ("p2-03", 80), ("p2-04", 70), ("p2-05", 100)):
        fake, cam, asked = _p1_aligned(ready_config, tmp_path, j5, confirm, name)
        results[name] = any("arrived" in line for line in asked)
        assert cam.snaps == [name, f"{name}-putback"]  # 집기 전 사진과 도로 놓은 뒤 사진
        assert any(w[0][4] == j5 for w in fake.writes)  # J5가 실제 쓰기 5번째 축으로 나간다
    # 첫 시도(편차 5) 확인, 편차 0과 10(15 미만 증가)은 확인 없음, 편차 20(≥ 5+10)에서 다시 확인, 편차 10은 없음
    assert results == {"p2-01": True, "p2-02": False, "p2-03": False, "p2-04": True, "p2-05": False}


def _writes_after_close(fake, ready_config):
    """닫기(그리퍼 close_sdk) 쓰기 뒤의 쓰기들."""
    idx = max(i for i, w in enumerate(fake.writes) if w[1] == ready_config.close_sdk)
    return fake.writes[idx + 1:]


def test_empty_hand_failure_never_redescends_to_grasp_in_chain(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}]
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_measured(85.2, 140.0)], cells, 1, grip_after=P1_GRIP_READ_MAX + 3, continue_on_miss=True)
    assert error is None
    gate = arm.gate
    after = [tuple(w[0]) for w in _writes_after_close(fake, ready_config)]
    pick_grasp = tuple(gate.joints(Pose(GRASP, 4, 85, 90)))
    assert after and pick_grasp not in after  # 닫은 뒤에는 grasp 높이로 다시 내려가지 않았다
    assert tuple(gate.joints(PARK_POSE)) == after[-1] and fake.writes[-1][1] == ready_config.open_sdk  # hover에서 열고 park


def test_empty_hand_failure_never_redescends_to_grasp_in_p1(ready_config, tmp_path):
    arm, fake, ledger, cam, error = _p1_run(ready_config, tmp_path, grip_after=P1_GRIP_READ_MAX + 3)
    table = generate(ready_config.table)
    gate = SafetyGate(ready_config, table)
    after = [tuple(w[0]) for w in _writes_after_close(fake, ready_config)]
    assert tuple(gate.joints(Pose(GRASP, 4, 80, 90))) not in after


def test_chain_x1_place_j5_schedule_and_human_placed_flag(ready_config, tmp_path):
    from learned_grasp.l1 import YawModel

    yaw = YawModel()
    rows = _rows(ready_config)
    cells = [{"r_index": 5, "j1": 90}, {"r_index": 4, "j1": 80}]
    first = _aligned(85.0, rows[4], 4, 90)
    second = _aligned(90.0, rows[5], 5, 66)  # 첫 시도가 J5 66으로 놓은 블록
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [first, second], cells, 2, signals=9, align=True, place_j5_schedule=[66, None])
    assert error is None, error
    s1, s2 = [r for r in ledger.records() if r.get("event") == "start"]
    assert s1["place_j5"] == 66 and s2["place_j5"] == yaw.place_j5(80, 4)  # null이면 정렬 놓기
    assert 66 in {w[0][4] for w in fake.writes} and s1["human_placed"] is False
    assert abs(s2["pick_j5"] - 66) <= 1  # 극단으로 놓인 블록을 극단 J5로 집었다


def test_chain_marks_block_placed_by_human_after_rejection(ready_config, tmp_path):
    table = generate(ready_config.table)
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    bad = _aligned(85.0, table.entries[4].r_mm, 4, 90)
    bad["block_angle_deg"] = (bad["block_angle_deg"] + 50.0) % 180.0  # J5가 범위를 크게 넘는 방향 → 거절
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [bad], cells, 1, align=True)
    assert isinstance(error, NeedHuman)
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_aligned(85.0, table.entries[4].r_mm, 4, 90)], cells, 1, ledger=ledger, align=True)
    assert error is None
    start = [r for r in ledger.records() if r.get("event") == "start"][0]
    assert start["human_placed"] is False  # 사진만 다시 찍은 경우(--replaced 없음)는 데이터를 버리지 않는다
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path / "again", [bad], cells, 1, align=True)
    assert isinstance(error, NeedHuman)
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path / "again", [_aligned(85.0, table.entries[4].r_mm, 4, 90)], cells, 1, ledger=ledger, align=True, human_replaced=True)
    assert error is None
    start = [r for r in ledger.records() if r.get("event") == "start"][0]
    assert start["human_placed"] is True and start["placed_at"] is None


def test_chain_escalation_confirms_each_side_of_j5_separately(ready_config, tmp_path):
    from learned_grasp.g0_runner import Outcome
    from learned_grasp.l1 import YawModel
    from learned_grasp.session import run_chain

    yaw = YawModel()
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 5, "j1": 90}, {"r_index": 4, "j1": 80}, {"r_index": 5, "j1": 90}, {"r_index": 4, "j1": 80}]
    schedule = [112, None, 64, None]
    # 측정: 1번은 near-90 블록, 2번은 112로 놓인 블록, 3번은 정렬로 놓인 블록, 4번은 64로 놓인 블록(실제 위치·방향은 모델이 만든 값)
    queue = iter([_aligned(85.0, rows[4], 4, 90), _aligned(90.0, rows[5], 5, 112), _aligned(80.0, rows[4], 4, yaw.place_j5(80, 4)), _aligned(90.0, rows[5], 5, 64)])
    fake = _GripFake(148)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    arrivals = []

    def log(line, **k):
        if "arrived" in line:
            arrivals.append(json.loads(line)["arrived"])

    def sleep(_):
        if not signal.exists():
            signal.write_text("1")

    run_chain(arm, _ChainCam(), lambda t: Outcome(True, "관찰기"), AttemptLedger(tmp_path / "chain.jsonl"), ready_config, table, cells, 4, signal, lambda path: next(queue),
              hold_s=0.0, sleep=sleep, clock=lambda: 0.0, log=log, align=True, escalate=True, place_j5_schedule=schedule, j5_clamp_margin=4.0, rest_every=0)
    # 첫 집기, 높은 쪽 첫 극단 놓기(112), 낮은 쪽 첫 극단 놓기(64). 같은 쪽 두 번째 극단 집기는 확인 없음
    assert arrivals == ["descend", "place", "place"], arrivals


def test_chain_j1_offset_recomputes_j5_for_base_rotation_and_places_at_nominal(ready_config, tmp_path):
    from learned_grasp.g0_runner import Outcome
    from learned_grasp.l1 import L1Map, YawModel, signed_diff
    from learned_grasp.session import run_chain

    yaw, l1 = YawModel(), L1Map()
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 5, "j1": 92}, {"r_index": 4, "j1": 84}]
    first = _aligned(90.0, rows[4], 4, 90)
    second = _aligned(92.0, rows[5], 5, yaw.place_j5(92, 5))
    fake = _GripFake(148)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    arrivals = []

    def log(line, **k):
        if "arrived" in line:
            arrivals.append(json.loads(line)["arrived"])

    def sleep(_):
        if not signal.exists():
            signal.write_text("1")

    queue = iter([first, second])
    ledger = AttemptLedger(tmp_path / "chain.jsonl")
    run_chain(arm, _ChainCam(), lambda t: Outcome(True, "관찰기"), ledger, ready_config, table, cells, 2, signal, lambda path: next(queue), hold_s=0.0, sleep=sleep, clock=lambda: 0.0,
              log=log, align=True, escalate=True, continue_on_miss=True, j1_offsets=[2, -1], confirm_j1_abs_ge=2, rest_every=0)
    s1, s2 = [r for r in ledger.records() if r.get("event") == "start"]
    assert (s1["j1_offset"], s1["j1"], s1["j1_l1"]) == (2, 92, 90) and s1["lateral_mm"] > 0  # J1 +2° → 횡방향 약 +5 mm
    # 손의 세계 각도가 블록과 같다: 모델이 말하는 손 방향 = 집는 자리의 반경선 기준 블록 방향
    block_angle = s1["plan"]["block_angle_deg"]
    d_pick = signed_diff(block_angle, l1.radial_angle_deg(92, rows[4]) % 180.0)
    assert abs(yaw.hand_d(s1["pick_j5"], 92, 4) - d_pick) < 1.0
    assert s2["j1_offset"] == -1 and s2["j1"] == s2["j1_l1"] - 1
    # 놓는 손 J1은 공칭(오프셋 없음)
    gate = arm.gate
    assert tuple(gate.joints(Pose(HOVER, 5, 92, s1["place_j5"]))) in [tuple(w[0]) for w in fake.writes]
    # Δ=2 집기는 확인(|Δ| ≥ 2 모두), Δ=−1은 그 방향 첫 오프셋이라 확인, 첫 집기도 확인
    assert arrivals.count("descend") >= 2


def test_chain_j1_offset_rejects_when_pick_j1_leaves_range(ready_config, tmp_path):
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 5, "j1": 92}]
    edge = _aligned(109.0, rows[4], 4, 90)  # L1 J1 109에서 +2° → 111 범위 밖
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [edge], cells, 1, align=True, j1_offsets=[2])
    assert isinstance(error, NeedHuman) and "범위 밖" in str(error) and fake.writes == []


def test_chain_confirms_every_j1_offset_pick_with_abs_at_least_two_and_only_first_of_each_side_for_one(ready_config, tmp_path):
    from learned_grasp.g0_runner import Outcome
    from learned_grasp.l1 import YawModel
    from learned_grasp.session import run_chain

    yaw = YawModel()
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 5, "j1": 90}] * 4
    fake = _GripFake(148)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    arrivals = []

    def log(line, **k):
        if "arrived" in line:
            arrivals.append(json.loads(line)["arrived"])

    def sleep(_):
        if not signal.exists():
            signal.write_text("1")

    def run(offsets, folder):
        folder.mkdir(exist_ok=True)
        led = AttemptLedger(folder / "chain.jsonl")
        queue = iter([_aligned(90.0, rows[5], 5, yaw.place_j5(90, 5))] * len(offsets))
        arrivals.clear()
        run_chain(arm, _ChainCam(), lambda t: Outcome(True, "관찰기"), led, ready_config, table, cells[: len(offsets)], len(offsets), signal, lambda path: next(queue),
                  hold_s=0.0, sleep=sleep, clock=lambda: 0.0, log=log, align=True, escalate=True, continue_on_miss=True, j1_offsets=offsets, confirm_j1_abs_ge=2, rest_every=0)
        return list(arrivals)

    # Δ [0, 2, 2, -2]처럼 |Δ|≥2는 같은 방향이라도 매번 확인(첫 집기 포함 4가 아니라 Δ=0 첫 집기 1 + 2, 2, −2 = 4)
    a = run([0, 2, 2, -2], tmp_path / "a")
    assert a.count("descend") == 4, a
    # Δ=1이 같은 방향에서 두 번이면 첫 번째만 확인(첫 집기는 Δ=1 자체가 첫 집기), Δ=−1은 그 방향 첫 번이라 확인
    b = run([1, 1, -1], tmp_path / "b")
    assert b.count("descend") == 2, b


def test_chain_block_shape_gate_rejects_tipped_block(ready_config, tmp_path):
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    ok = dict(_aligned(85.0, rows[4], 4, 90), area=2400, aspect=2.0)
    tipped = dict(_aligned(100.0, rows[6], 6, 90), area=4200, aspect=1.1)  # 끝으로 세워진 블록: 면적이 크고 정사각에 가까움
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [ok, tipped], cells, 2, signals=9, align=True)
    assert isinstance(error, NeedHuman) and "모양이 이상" in str(error)
    finished, _ = ledger.state()
    assert finished["chain-001"]["outcome"] == "success"  # 첫 시도는 정상, 둘째는 쓰기 없이 멈춤


def test_all_rejection_notes_survive_the_reason_keyword(ready_config, tmp_path):
    # R1 반경 오프셋 J5 범위 밖 경로
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 6, "j1": 100}]
    m = _aligned(85.0, rows[4], 4, 90)
    m["block_angle_deg"] = (m["block_angle_deg"] + 34.0) % 180.0  # L1 J5는 범위 안(약 117)이지만 +3행 오프셋에서 J5가 범위를 넘는 방향
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [m], cells, 1, align=True, offsets=[-3])
    assert isinstance(error, NeedHuman)  # TypeError(reason 키워드 중복)가 아니라 NeedHuman이어야 한다


def test_chain_shape_gate_accepts_natural_area_variation(ready_config, tmp_path):
    table = generate(ready_config.table)
    rows = [e.r_mm for e in table.entries]
    cells = [{"r_index": 6, "j1": 100}, {"r_index": 2, "j1": 76}]
    small = dict(_aligned(85.0, rows[4], 4, 90), area=1888, aspect=2.3)  # X2-1 첫 측정
    large = dict(_aligned(100.0, rows[6], 6, 90), area=2986, aspect=1.75)  # X2-1 둘째: 같은 블록의 자연 변동(비 1.58)
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [small, large], cells, 2, signals=9, align=True)
    assert error is None, error


def test_chain_hover_photo_is_taken_before_descent(ready_config, tmp_path):
    cells = [{"r_index": 6, "j1": 100}]
    table = generate(ready_config.table)
    arm, fake, ledger, cam, error, table = _chain_run(ready_config, tmp_path, [_aligned(85.0, table.entries[4].r_mm, 4, 90)], cells, 1, align=True, hover_photo=True)
    assert error is None and "chain-001-hover" in cam.snaps and cam.snaps.index("chain-001-hover") > cam.snaps.index("chain-001")


def test_hover_sweep_visits_only_hover_poses_via_park_and_never_grasps(ready_config, tmp_path):
    from learned_grasp.session import hover_sweep_poses, run_hover_sweep

    table = generate(ready_config.table)
    gate = SafetyGate(ready_config, table)
    fake = _GripFake(148)
    arm = SafeArm(fake, gate, sleep=lambda _: None)
    signal = tmp_path / "dabai_next"

    def sleep(_):
        if not signal.exists():
            signal.write_text("1")

    cam = _ChainCam()
    run_hover_sweep(arm, cam, signal, 5, 90, folder=tmp_path, sleep=sleep, clock=lambda: 0.0, log=lambda *a, **k: None)
    poses = hover_sweep_poses(5, 90)
    assert len(poses) == 14 and all(p.kind == HOVER for _, p in poses)  # grasp 자세가 없다
    assert [n for n, _ in poses][0] == "ref-1" and [n for n, _ in poses][-1] == "ref-3" and "ref-2" in [n for n, _ in poses][1:-1]  # 기준 재방문이 시작·중간·끝
    assert cam.snaps == [f"sweep-{n}-{ab}" for n, _ in poses for ab in "ab"]  # 자세마다 2장
    grasp_joints = {tuple(gate.joints(Pose(GRASP, 5, j1, j5))) for j1 in range(70, 111) for j5 in range(60, 121)}
    assert not any(tuple(w[0]) in grasp_joints for w in fake.writes)
    assert len(fake.writes) == 1 + 2 * 14  # 처음 park + 자세마다 hover·park, 사진·읽기는 쓰기가 아니다
    assert all(w[1] == ready_config.open_sdk for w in fake.writes)
    assert arm.pose == PARK_POSE
    lines = [json.loads(l) for l in (tmp_path / "sweep.jsonl").read_text().splitlines()]
    assert len(lines) == 14 and lines[0]["pose"] == "ref-1" and lines[0]["read"] is not None  # 읽은 관절값 기록


def test_hover_sweep_stops_between_poses_when_stop_file_exists(ready_config, tmp_path):
    from learned_grasp.session import run_hover_sweep

    table = generate(ready_config.table)
    fake = _GripFake(148)
    arm = SafeArm(fake, SafetyGate(ready_config, table), sleep=lambda _: None)
    signal = tmp_path / "dabai_next"
    stop = tmp_path / "chain_stop"
    stop.write_text("1")

    def sleep(_):
        if not signal.exists():
            signal.write_text("1")

    cam = _ChainCam()
    run_hover_sweep(arm, cam, signal, 5, 90, folder=tmp_path, stop=stop, sleep=sleep, clock=lambda: 0.0, log=lambda *a, **k: None)
    assert cam.snaps == [] and len(fake.writes) == 1 and arm.pose == PARK_POSE  # 첫 park 뒤 자세 전에 멈춤
