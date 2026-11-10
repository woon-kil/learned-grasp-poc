"""팔 인터페이스. SafeArm만 드라이버에 쓰기를 보낸다."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

from learned_grasp.config import Config
from learned_grasp.lineio import LineReader
from learned_grasp.safety import PARK_POSE, Pose, SafetyGate, SafetyViolation


class ArmDriver(Protocol):
    def write6(self, q5: tuple[int, ...], grip: int, time_ms: int) -> None: ...
    def write_grip(self, grip: int, time_ms: int) -> None: ...
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


class CloseWatch:
    """닫기 명령 전송부터 단조 시계로 재는 닫힘 한도. 입력·원격 호출 대기와 독립된 스레드로 만료를 표시한다.

    한도는 운영 값이고 안전 보증이 아니다. 만료는 SafeArm의 추가 쓰기만 막는다. 그리퍼는 닫힌 채 팔이 서 있다.
    값은 기본값 없이 실행 계약에서 받는다(G0d 플랜 124~126행).
    """

    def __init__(self, limit_s: float, on_expire: Callable[[str], None] = print, clock: Callable[[], float] = time.monotonic):
        if not limit_s > 0:
            raise ValueError("닫힘 한도는 양수다")
        self.limit_s = limit_s
        self.on_expire = on_expire
        self.clock = clock
        self.started_at: float | None = None
        self.expired = False
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop()
        self._cancel = threading.Event()
        self.started_at = self.clock()
        cancel = self._cancel
        self._thread = threading.Thread(target=self._run, args=(cancel,), daemon=True)
        self._thread.start()

    def _run(self, cancel: threading.Event) -> None:
        if not cancel.wait(self.limit_s):
            self.expired = True
            self.on_expire(f"닫힘 {self.limit_s:g}초 한도 만료. 추가 쓰기를 막는다. 그리퍼는 닫힌 채다")

    def stop(self) -> None:
        self._cancel.set()
        self.started_at = None

    def remaining(self) -> float | None:
        if self.started_at is None:
            return None
        return self.limit_s - (self.clock() - self.started_at)


class SafeArm:
    """게이트 통과 후에만 쓰기를 보낸다. 조회값을 목표로 쓰지 않는다."""

    def __init__(
        self,
        driver: ArmDriver,
        gate: SafetyGate,
        sleep: Callable[[float], None] = time.sleep,
        start_joints: tuple[int, ...] | None = None,
        close_watch: CloseWatch | None = None,
    ):
        self.driver = driver
        self.gate = gate
        self.sleep = sleep
        self.start_joints = start_joints
        self.close_watch = close_watch
        self.pose: Pose | None = None
        self.grip: int = gate.config.open_sdk
        self.holding = False

    def _check_watch(self) -> None:
        if self.close_watch is not None and self.close_watch.expired:
            raise SafetyViolation("닫힘 한도가 만료돼 추가 쓰기를 보내지 않는다. 합의한 절차를 따른다")

    def goto(self, target: Pose) -> None:
        self._check_watch()
        self.gate.check_transition(self.pose, target)
        q5 = self.gate.joints(target)
        time_ms = self.gate.move_time_ms(self.pose, target, self.start_joints)
        self.driver.write6(q5, self.grip, time_ms)
        self.sleep((time_ms + self.gate.config.motion.settle_ms) / 1000.0)
        self._check_arrival(q5)
        self.pose = target

    def _check_arrival(self, q5: tuple[int, ...]) -> None:
        """시간이 지났다고 도착으로 보지 않는다. SDK 반환을 비교하고 어긋나면 다음 쓰기를 막는다.

        반환은 축 실측이 아니며 비교에만 쓴다. 목표로 재사용하지 않는다.
        """
        reading = self.driver.read()
        if any(item != "sdk_return" for item in reading.get("status", [])):
            raise SafetyViolation(f"도착 확인 조회 실패: {reading}")
        values = reading.get("values")
        tolerance = self.gate.config.motion.arrival_tol_deg
        if not values or any(abs(value - goal) > tolerance for value, goal in zip(values[:5], q5)):
            raise SafetyViolation(f"목표 미도달로 멈춘다. 목표 {q5}, SDK 반환 {values}. 감독자가 팔 상태를 확인한다")

    def set_grip(self, action: str) -> None:
        grip = self.gate.check_grip(self.pose, action, self.holding)
        if self.pose is None:
            raise SafetyViolation("자세를 모르는 상태에서 그리퍼를 움직이지 않는다")
        if grip != self.grip:
            self._check_watch()
            # 팔 5축을 더 짧은 시간으로 재전송하지 않도록 6번 서보만 쓴다.
            time_ms = self.gate.config.motion.grip_time_ms
            if self.close_watch is not None and action == "close":
                self.close_watch.start()
            self.driver.write_grip(grip, time_ms)
            self.sleep((time_ms + self.gate.config.motion.settle_ms) / 1000.0)
            if self.close_watch is not None and action == "open":
                self.close_watch.stop()
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

    def write_grip(self, grip, time_ms):
        self.writes.append((self.writes[-1][0], grip, time_ms))

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

    def _call(self, request: dict, timeout_s: float = 10.0) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not hasattr(self, "reader"):
            self.reader = LineReader(self.process.stdout)
        try:
            line = self.reader.readline(timeout_s)
        except TimeoutError:
            self.process.kill()
            raise ConnectionError(f"Jetson 서버가 {timeout_s:g}초 안에 응답하지 않았다. 이미 보낸 이동은 멈추지 않으니 감독자가 확인한다") from None
        if not line:
            raise ConnectionError("Jetson 서버 연결이 끊겼다")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"서버 거절: {response.get('error')}")
        return response

    def write6(self, q5, grip, time_ms):
        self._call({"op": "write6", "q": [*q5, grip], "time_ms": time_ms})

    def write_grip(self, grip, time_ms):
        self._call({"op": "grip", "angle": grip, "time_ms": time_ms})

    def read(self):
        return self._call({"op": "read"})

    def close(self):
        """정리 실패가 원래 오류를 가리지 않게 한다."""
        try:
            if self.process.poll() is None:
                self._call({"op": "bye"}, timeout_s=3.0)
        except (OSError, ValueError, RuntimeError):
            pass
        finally:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


__all__ = ["ApprovalError", "ArmDriver", "CloseWatch", "FakeArm", "PARK_POSE", "RemoteArm", "SafeArm", "SessionApproval", "load_approval"]
