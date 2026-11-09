"""팔 인터페이스. SafeArm만 드라이버에 쓰기를 보낸다."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from learned_grasp.config import Config
from learned_grasp.safety import PARK_POSE, Pose, SafetyGate, SafetyViolation


class ArmDriver(Protocol):
    def write6(self, q5: tuple[int, ...], grip: int, time_ms: int) -> None: ...
    def read(self) -> dict: ...
    def close(self) -> None: ...


class ApprovalError(RuntimeError):
    """세션 승인이 없거나 유효하지 않다."""


@dataclass(frozen=True)
class SessionApproval:
    session_id: str
    gate: str
    max_writes: int
    token: str


def load_approval(path: Path, config: Config, now: datetime | None = None) -> SessionApproval:
    """세션 승인 파일을 검사한다. G0a 드라이런이 끝나기 전에는 G0a 세션만 허용한다."""
    data = Path(path).read_bytes()
    raw = json.loads(data)
    for field in ("session_id", "gate", "target", "scope", "observation", "stop_condition", "power_cut_operator", "expires_at"):
        if not raw.get(field):
            raise ApprovalError(f"승인 파일에 {field}가 없다")
    if raw.get("approved_by") != "user":
        raise ApprovalError("사용자 승인이 없다")
    if type(raw.get("max_writes")) is not int or raw["max_writes"] <= 0:
        raise ApprovalError("max_writes는 양의 정수다")
    expires = datetime.fromisoformat(raw["expires_at"])
    if (now or datetime.now(expires.tzinfo)) >= expires:
        raise ApprovalError("승인 기간이 지났다")
    if not config.verified_g0a and raw["gate"] != "G0a":
        raise ApprovalError("G0a 통과 전에는 G0a 세션만 허용한다")
    return SessionApproval(raw["session_id"], raw["gate"], raw["max_writes"], hashlib.sha256(data).hexdigest())


class SafeArm:
    """게이트 통과 후에만 쓰기를 보낸다. 조회값을 목표로 쓰지 않는다."""

    def __init__(self, driver: ArmDriver, gate: SafetyGate, sleep: Callable[[float], None] = time.sleep):
        self.driver = driver
        self.gate = gate
        self.sleep = sleep
        self.pose: Pose | None = None
        self.grip: int = gate.config.open_sdk
        self.holding = False

    def goto(self, target: Pose) -> None:
        self.gate.check_transition(self.pose, target)
        q5 = self.gate.joints(target)
        time_ms = self.gate.move_time_ms(self.pose, target)
        self.driver.write6(q5, self.grip, time_ms)
        self.sleep(time_ms / 1000.0)
        self.pose = target

    def set_grip(self, action: str) -> None:
        grip = self.gate.check_grip(self.pose, action, self.holding)
        if self.pose is None:
            raise SafetyViolation("자세를 모르는 상태에서 그리퍼를 움직이지 않는다")
        time_ms = self.gate.config.motion.grip_time_ms
        self.driver.write6(self.gate.joints(self.pose), grip, time_ms)
        self.sleep(time_ms / 1000.0)
        self.grip = grip
        self.holding = action == "close"

    def read(self) -> dict:
        return self.driver.read()


class FakeArm:
    """장치 없는 드라이버. 쓰기를 기록한다."""

    def __init__(self):
        self.writes: list[tuple[tuple[int, ...], int, int]] = []
        self.closed = False

    def write6(self, q5, grip, time_ms):
        self.writes.append((tuple(q5), grip, time_ms))

    def read(self):
        return {"values": list(self.writes[-1][0]) + [self.writes[-1][1]] if self.writes else None}

    def close(self):
        self.closed = True


class RemoteArm:
    """SSH 표준입출력으로 Jetson 서버를 구동한다. 세션 승인 없이 만들 수 없다."""

    def __init__(
        self,
        approval: SessionApproval,
        host: str | None = None,
        server_path: str = "learned_grasp/jetson_arm_server.py",
        command: list[str] | None = None,
    ):
        host = host or os.environ.get("LG_JETSON_HOST")
        if command is None and not host:
            raise ApprovalError("Jetson 주소가 없다. .env의 LG_JETSON_HOST를 설정한다")
        command = command or [
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2", host,
            f"python3 -u {server_path} --token {approval.token} --max-writes {approval.max_writes}",
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self._call({"op": "hello", "token": approval.token})

    def _call(self, request: dict) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise ConnectionError("Jetson 서버 연결이 끊겼다")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"서버 거절: {response.get('error')}")
        return response

    def write6(self, q5, grip, time_ms):
        self._call({"op": "write6", "q": [*q5, grip], "time_ms": time_ms})

    def read(self):
        return self._call({"op": "read"})

    def close(self):
        try:
            self._call({"op": "bye"})
        finally:
            self.process.stdin.close()
            self.process.wait(timeout=5)


__all__ = ["ApprovalError", "ArmDriver", "FakeArm", "PARK_POSE", "RemoteArm", "SafeArm", "SessionApproval", "load_approval"]
