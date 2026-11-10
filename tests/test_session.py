from __future__ import annotations

import json
from dataclasses import replace

import pytest

from learned_grasp.arm import FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.kinematics import fk_batch
from learned_grasp.pose_table import generate
from learned_grasp.safety import HOVER, PARK_POSE, Pose, SafetyGate, SafetyViolation
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
    assert [item["step"] for item in ledger] == ["open", "close", "level", "level", "end"]
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
