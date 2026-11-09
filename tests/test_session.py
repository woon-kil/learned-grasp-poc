from __future__ import annotations

import pytest

from learned_grasp.arm import FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.kinematics import fk_batch
from learned_grasp.pose_table import generate
from learned_grasp.safety import SafetyGate, SafetyViolation
from learned_grasp.session import (
    LADDER_OFFSETS_MM,
    CountingDriver,
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
    config = ladder_config(ready_config, 60)
    table = generate(config.table)
    fake = FakeArm()
    arm = SafeArm(fake, SafetyGate(config, table), sleep=lambda _: None)
    run_ladder(arm, AttemptLedger(tmp_path / "l.jsonl", clock=lambda: 0.0), config, table)
    # park(개방 포함), hover, grasp, hover, park. 그리퍼를 다시 쓰지 않는다
    assert len(fake.writes) == 5
    assert {grip for _, grip, _ in fake.writes} == {config.open_sdk}
    assert fake.writes[-1][0] == config.park


def test_hold_adds_no_writes_and_waits_at_grasp(ready_config, tmp_path):
    config = ladder_config(ready_config, 60)
    table = generate(config.table)
    fake = FakeArm()
    held = []

    def sleep(seconds):
        held.append((seconds, fake.writes[-1][0]))

    arm = SafeArm(fake, SafetyGate(config, table), sleep=sleep)
    run_ladder(arm, AttemptLedger(tmp_path / "h.jsonl", clock=lambda: 0.0), config, table, hold_s=40.0)
    assert len(fake.writes) == 5
    entry = table.entries[len(table.entries) // 2]
    assert (40.0, (90, *entry.grasp, 90)) in held


def test_ladder_records_do_not_mix_between_offsets(ready_config):
    versions = {ladder_config(ready_config, offset).config_version for offset in LADDER_OFFSETS_MM}
    assert len(versions) == len(LADDER_OFFSETS_MM)


def test_counting_driver_counts_grip_writes(tmp_path):
    fake = FakeArm()
    fake.writes.append(((90, 90, 90, 90, 90), 100, 1500))
    driver = CountingDriver(fake, tmp_path / "n", 1)
    driver.write_grip(120, 1500)
    with pytest.raises(SafetyViolation):
        driver.write_grip(130, 1500)


def test_descent_requires_previous_observation(tmp_path):
    require_previous_observation(tmp_path, 60)
    with pytest.raises(SafetyViolation, match="z\\+60"):
        require_previous_observation(tmp_path, 40)
    record_observation(tmp_path, 60, True, "간격 충분")
    require_previous_observation(tmp_path, 40)
    with pytest.raises(SafetyViolation):
        require_previous_observation(tmp_path, 25)


def test_bad_observation_stops_ladder(tmp_path):
    record_observation(tmp_path, 60, True, "")
    record_observation(tmp_path, 40, False, "닿을 듯함")
    with pytest.raises(SafetyViolation, match="이상"):
        require_previous_observation(tmp_path, 40)


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
