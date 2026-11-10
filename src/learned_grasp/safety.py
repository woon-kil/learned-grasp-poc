"""모든 쓰기가 통과하는 안전 게이트.

허용 표와 전이 밖의 명령은 전송 전에 거절한다. 이 검사는 표 안의 무간섭을 증명하지 않는다.
무간섭은 G0a 드라이런으로 확인하고, 정지 수단은 감독자의 전원 차단이다.
"""

from __future__ import annotations

from dataclasses import dataclass

from learned_grasp.config import Config
from learned_grasp.pose_table import PoseTable

PARK, HOVER, GRASP = "park", "hover", "grasp"

GRIP_SWEEP_STEP = 5
GRIP_SWEEP_MAX_SDK = 180  # jetson_arm_server SDK_MAX[6]. 이보다 위는 서버도 거절한다


class SafetyViolation(RuntimeError):
    """명령이 허용 범위 밖이라 전송하지 않았다."""


@dataclass(frozen=True)
class Pose:
    kind: str
    r_index: int | None = None
    j1: int | None = None
    j5: int | None = None


PARK_POSE = Pose(PARK)


class SafetyGate:
    def __init__(self, config: Config, table: PoseTable, grip_sweep: bool = False):
        config.require_motion_values()
        self.config = config
        self.table = table
        self.grip_sweep = grip_sweep  # grip-max 명령만 켠다. close_sdk 위로 5씩 올리는 level 동작의 허가

    def joints(self, pose: Pose) -> tuple[int, int, int, int, int]:
        if pose.kind == PARK:
            return self.config.park
        if pose.kind not in (HOVER, GRASP):
            raise SafetyViolation(f"알 수 없는 자세 {pose.kind}")
        if pose.r_index is None or not 0 <= pose.r_index < len(self.table.entries):
            raise SafetyViolation(f"표 밖의 r 색인 {pose.r_index}")
        low, high = self.config.j1_range
        if pose.j1 is None or not low <= pose.j1 <= high:
            raise SafetyViolation(f"J1 범위 밖 {pose.j1}")
        low, high = self.config.j5_range
        if pose.j5 is None or not low <= pose.j5 <= high:
            raise SafetyViolation(f"J5 범위 밖 {pose.j5}")
        entry = self.table.entries[pose.r_index]
        middle = entry.grasp if pose.kind == GRASP else entry.hover
        return (pose.j1, *middle, pose.j5)

    def check_transition(self, current: Pose | None, target: Pose) -> None:
        if current is None:
            if target.kind != PARK:
                raise SafetyViolation("첫 이동은 park만 허용한다")
            return
        allowed = {PARK: {HOVER}, HOVER: {GRASP, PARK}, GRASP: {HOVER}}
        if target.kind not in allowed[current.kind]:
            raise SafetyViolation(f"허용되지 않은 전이 {current.kind}->{target.kind}")
        if {current.kind, target.kind} == {HOVER, GRASP} and (
            current.r_index != target.r_index or current.j1 != target.j1 or current.j5 != target.j5
        ):
            raise SafetyViolation("hover와 grasp 사이에는 r, J1, J5를 바꿀 수 없다")

    def move_time_ms(self, current: Pose | None, target: Pose, start_joints: tuple[int, ...] | None = None) -> int:
        """이동시간 하한. 첫 이동은 조회한 시작값이 있으면 그 거리로도 검사한다. 시작값은 목표로 쓰지 않는다."""
        motion = self.config.motion
        if current is None:
            if start_joints is None:
                return motion.first_move_time_ms
            delta = max(abs(a - b) for a, b in zip(start_joints, self.joints(target)))
            if delta > motion.max_move_deg:
                raise SafetyViolation(f"시작 자세에서 첫 이동각 {delta}도가 상한 {motion.max_move_deg}도를 넘는다")
            return max(motion.first_move_time_ms, delta * motion.ms_per_deg)
        delta = max(abs(a - b) for a, b in zip(self.joints(current), self.joints(target)))
        if delta > motion.max_move_deg:
            raise SafetyViolation(f"1회 이동각 {delta}도가 상한 {motion.max_move_deg}도를 넘는다")
        return max(motion.min_time_ms, delta * motion.ms_per_deg)

    def check_grip(self, current: Pose | None, action: str, holding: bool, grip: int | None = None) -> int:
        """close는 grasp에서만, open은 grasp·hover와 빈 손 park에서만 허용한다.

        level:<sdk>는 grip-max 명령이 grasp에서 이미 닫은 손을 close_sdk 위로 5씩 올릴 때만 허용한다(상한 GRIP_SWEEP_MAX_SDK).
        """
        kind = current.kind if current is not None else None
        if isinstance(action, str) and action.startswith("level:"):
            if not self.grip_sweep:
                raise SafetyViolation("level 동작은 grip-max 명령에서만 허용한다")
            if kind != GRASP:
                raise SafetyViolation("level은 grasp 자세에서만 허용한다")
            if not holding or grip is None or grip < self.config.close_sdk:
                raise SafetyViolation("level은 close_sdk로 닫은 뒤에만 허용한다")
            try:
                level = int(action.split(":", 1)[1])
            except ValueError:
                raise SafetyViolation(f"level 값을 읽을 수 없다 {action}") from None
            if not grip < level <= grip + GRIP_SWEEP_STEP:
                raise SafetyViolation(f"level {level}은 현재 {grip}에서 {GRIP_SWEEP_STEP} 이하로만 올린다")
            if level > GRIP_SWEEP_MAX_SDK:
                raise SafetyViolation(f"level {level}은 상한 {GRIP_SWEEP_MAX_SDK}을 넘는다")
            return level
        if action == "close":
            if kind != GRASP:
                raise SafetyViolation("close는 grasp 자세에서만 허용한다")
            return self.config.close_sdk
        if action == "open":
            if kind in (GRASP, HOVER) or (kind == PARK and not holding):
                return self.config.open_sdk
            raise SafetyViolation("이 자세에서는 open을 허용하지 않는다")
        raise SafetyViolation(f"알 수 없는 그리퍼 동작 {action}")
