from __future__ import annotations

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
    descend_and_stop,
    fit_config,
    run_fit,
    run_grip_test,
    make_block_loader,
    make_pick_observer,
    ladder_config,
    read_start_joints,
    record_observation,
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
    def __init__(self, value, yellow=0):
        self.value = value
        self.yellow = yellow
        self.names = []

    def snapshot(self, name):
        self.names.append(name)

    def change(self, before, after, roi):
        return self.value

    def yellow_pixels(self, name, roi):
        assert name in self.names
        return self.yellow


def test_pick_observer_uses_image_change():
    silent = lambda *a, **k: None
    camera = ChangeCamera(0.03)
    assert make_pick_observer(camera, log=silent)({"trial_id": "t"}).success
    assert camera.names == ["t-after"]
    failed = make_pick_observer(ChangeCamera(0.001), log=silent)({"trial_id": "t"})
    assert not failed.success and failed.human_intervention


@pytest.mark.parametrize(
    ("change", "yellow", "success"),
    [(0.02, 100, True), (0.0199, 0, False), (0.02, 101, False), (0.3, 2000, False)],
)
def test_pick_observer_needs_change_and_clear_board(change, yellow, success):
    outcome = make_pick_observer(ChangeCamera(change, yellow), log=lambda *a, **k: None)({"trial_id": "t"})
    assert outcome.success is success
    assert f"{change:.4f}" in outcome.reason and str(yellow) in outcome.reason
    assert outcome.human_intervention is not success


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
