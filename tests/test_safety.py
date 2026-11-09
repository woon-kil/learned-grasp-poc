from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from learned_grasp.arm import ApprovalError, FakeArm, SafeArm, load_approval
from learned_grasp.config import ConfigError
from learned_grasp.safety import GRASP, HOVER, PARK_POSE, Pose, SafetyGate, SafetyViolation


@pytest.fixture
def driver():
    return FakeArm()


@pytest.fixture
def arm(ready_config, table, driver):
    return SafeArm(driver, SafetyGate(ready_config, table), sleep=lambda _: None)


def hover(index=0, j1=90, j5=90):
    return Pose(HOVER, index, j1, j5)


def grasp(index=0, j1=90, j5=90):
    return Pose(GRASP, index, j1, j5)


@pytest.mark.parametrize("missing", ["park", "open_sdk", "close_sdk"])
def test_config_without_motion_values_cannot_build_gate(draft_config, table, missing):
    with pytest.raises(ConfigError):
        SafetyGate(replace(draft_config, **{missing: None}), table)


def test_first_move_must_be_park(arm, driver):
    with pytest.raises(SafetyViolation):
        arm.goto(hover())
    assert driver.writes == []


@pytest.mark.parametrize(
    "target",
    [
        Pose(HOVER, 999, 90, 90),
        Pose(HOVER, -1, 90, 90),
        Pose(HOVER, 0, 10, 90),
        Pose(HOVER, 0, 90, 200),
        Pose(HOVER, None, 90, 90),
        Pose("free", 0, 90, 90),
    ],
)
def test_out_of_table_never_reaches_driver(arm, driver, target):
    arm.goto(PARK_POSE)
    before = len(driver.writes)
    with pytest.raises((SafetyViolation, KeyError)):
        arm.goto(target)
    assert len(driver.writes) == before


def test_direct_park_to_grasp_rejected(arm, driver):
    arm.goto(PARK_POSE)
    before = len(driver.writes)
    with pytest.raises(SafetyViolation):
        arm.goto(grasp())
    assert len(driver.writes) == before


def test_hover_grasp_must_share_cell(arm, driver):
    arm.goto(PARK_POSE)
    arm.goto(hover(0))
    before = len(driver.writes)
    for target in (grasp(1), grasp(0, j1=80), grasp(0, j5=80)):
        with pytest.raises(SafetyViolation):
            arm.goto(target)
    assert len(driver.writes) == before


def test_grasp_cannot_jump_to_park(arm, driver):
    arm.goto(PARK_POSE)
    arm.goto(hover())
    arm.goto(grasp())
    before = len(driver.writes)
    with pytest.raises(SafetyViolation):
        arm.goto(PARK_POSE)
    assert len(driver.writes) == before


def test_close_only_at_grasp(arm, driver):
    arm.goto(PARK_POSE)
    arm.goto(hover())
    before = len(driver.writes)
    with pytest.raises(SafetyViolation):
        arm.set_grip("close")
    assert len(driver.writes) == before
    arm.goto(grasp())
    arm.set_grip("close")
    assert arm.holding
    assert driver.writes[-1][1] == arm.gate.config.close_sdk


def test_open_while_holding_at_park_rejected(arm, driver):
    arm.goto(PARK_POSE)
    arm.goto(hover())
    arm.goto(grasp())
    arm.set_grip("close")
    arm.goto(hover())
    arm.goto(PARK_POSE)
    before = len(driver.writes)
    with pytest.raises(SafetyViolation):
        arm.set_grip("open")
    assert len(driver.writes) == before


def test_motion_time_respects_floor(arm, driver, ready_config):
    motion = ready_config.motion
    arm.goto(PARK_POSE)
    assert driver.writes[-1][2] == motion.first_move_time_ms
    arm.goto(hover())
    previous, _, time_ms = driver.writes[-1]
    delta = max(abs(a - b) for a, b in zip(previous, ready_config.park))
    assert time_ms == max(motion.min_time_ms, delta * motion.ms_per_deg)


def test_move_angle_cap(ready_config, table, driver):
    tight = replace(ready_config, motion=replace(ready_config.motion, max_move_deg=1))
    arm = SafeArm(driver, SafetyGate(tight, table), sleep=lambda _: None)
    arm.goto(PARK_POSE)
    before = len(driver.writes)
    with pytest.raises(SafetyViolation):
        arm.goto(hover())
    assert len(driver.writes) == before


def _approval(tmp_path, **overrides):
    raw = {
        "session_id": "s1", "gate": "G0a", "target": "t", "scope": "s", "max_writes": 10,
        "observation": "o", "stop_condition": "c", "power_cut_operator": "사용자", "approved_by": "user",
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
    }
    raw.update(overrides)
    path = tmp_path / "a.json"
    path.write_text(json.dumps(raw))
    return path


def test_approval_valid(tmp_path, draft_config):
    approval = load_approval(_approval(tmp_path), draft_config)
    assert approval.max_writes == 10 and len(approval.token) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"approved_by": ""},
        {"max_writes": 0},
        {"max_writes": True},
        {"power_cut_operator": ""},
        {"expires_at": "2020-01-01T00:00:00+00:00"},
        {"gate": "G0b"},
    ],
)
def test_approval_rejected(tmp_path, draft_config, overrides):
    with pytest.raises(ApprovalError):
        load_approval(_approval(tmp_path, **overrides), replace(draft_config, verified_g0a=False))


def test_example_session_file_is_not_approved(draft_config):
    from learned_grasp.config import ROOT

    with pytest.raises(ApprovalError):
        load_approval(ROOT / "config" / "session.example.json", draft_config)


def test_first_move_uses_measured_distance(ready_config, table):
    gate = SafetyGate(ready_config, table)
    far = tuple(a + 60 for a in ready_config.park[:4]) + (ready_config.park[4],)
    assert gate.move_time_ms(None, PARK_POSE, far) == max(ready_config.motion.first_move_time_ms, 60 * ready_config.motion.ms_per_deg)


def test_first_move_too_far_is_refused_before_write(ready_config, table, driver):
    start = (*ready_config.park[:4], ready_config.park[4] + ready_config.motion.max_move_deg + 1)
    arm = SafeArm(driver, SafetyGate(ready_config, table), sleep=lambda _: None, start_joints=start)
    with pytest.raises(SafetyViolation, match="첫 이동각"):
        arm.goto(PARK_POSE)
    assert driver.writes == []


class LaggingArm(FakeArm):
    """쓰기는 받지만 도착하지 않는 팔."""

    def read(self):
        return {"values": [0, 0, 0, 0, 0, 0], "status": ["sdk_return"] * 6}


def test_no_further_write_when_arm_did_not_arrive(ready_config, table):
    lagging = LaggingArm()
    arm = SafeArm(lagging, SafetyGate(ready_config, table), sleep=lambda _: None)
    with pytest.raises(SafetyViolation, match="미도달"):
        arm.goto(PARK_POSE)
    assert len(lagging.writes) == 1 and arm.pose is None


def test_failed_arrival_read_blocks_next_write(ready_config, table):
    class FailingRead(FakeArm):
        def read(self):
            return {"values": [None] * 6, "status": ["fail"] * 6}

    failing = FailingRead()
    arm = SafeArm(failing, SafetyGate(ready_config, table), sleep=lambda _: None)
    with pytest.raises(SafetyViolation, match="조회 실패"):
        arm.goto(PARK_POSE)
    assert len(failing.writes) == 1


def test_grip_does_not_resend_arm_and_skips_when_unchanged(ready_config, table):
    class Recording(FakeArm):
        def __init__(self):
            super().__init__()
            self.arm_writes = 0

        def write6(self, q5, grip, time_ms):
            self.arm_writes += 1
            super().write6(q5, grip, time_ms)

    recording = Recording()
    arm = SafeArm(recording, SafetyGate(ready_config, table), sleep=lambda _: None)
    arm.goto(PARK_POSE)
    arm.set_grip("open")
    assert len(recording.writes) == 1  # 첫 쓰기에 이미 개방이 포함됐다
    arm.goto(hover())
    arm.goto(grasp())
    before = recording.arm_writes
    arm.set_grip("close")
    assert recording.arm_writes == before  # 팔 5축을 재전송하지 않는다
    assert recording.writes[-1][1] == ready_config.close_sdk
