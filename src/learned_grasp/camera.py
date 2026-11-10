"""카메라 인터페이스. 실장치 스냅샷 경로는 G0b 전에 구현한다."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from learned_grasp.lineio import LineReader


@dataclass(frozen=True)
class Snapshot:
    frame_id: str
    captured_at: float


class Camera(Protocol):
    def snapshot(self, name: str) -> Snapshot: ...
    def change(self, before: str, after: str, roi: tuple[int, int, int, int]) -> float: ...
    def yellow_pixels(self, name: str, roi: tuple[int, int, int, int]) -> int: ...
    def fetch(self, name: str) -> Path: ...


class FakeCamera:
    def __init__(self, clock=lambda: 0.0):
        self.clock = clock
        self.names: list[str] = []
        self.fail_next = False
        self.photos: dict[str, Path] = {}

    def snapshot(self, name: str) -> Snapshot:
        if self.fail_next:
            raise RuntimeError("스냅샷 실패")
        self.names.append(name)
        return Snapshot(name, self.clock())

    def change(self, before, after, roi):
        """시험용. 집기 성공을 뜻하는 큰 변화를 돌려준다."""
        return self.change_value

    def yellow_pixels(self, name, roi):
        """시험용. 보드에 노란색이 없음을 돌려준다."""
        return self.yellow_value

    def fetch(self, name: str) -> Path:
        """시험용. photos[name]에 넣어 둔 경로를 돌려준다. 없으면 가져오기 실패다."""
        if name not in self.photos:
            raise FileNotFoundError(name)
        return self.photos[name]

    change_value = 1.0
    yellow_value = 0


class P620Camera:
    """p620 상시 카메라 서버. 드라이버 시작이 느리고 노출 수렴이 필요해 켜 둔 채 요청한다."""

    def __init__(self, output: str, host: str | None = None, command: list[str] | None = None):
        host = host or os.environ.get("LG_CAMERA_HOST", "p620")
        command = command or [
            "ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2", host,
            f'cd projects/learned-grasp-poc && bash scripts/p620_camera.sh serve "$PWD/{output}"',
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.host, self.output = host, output
        self._read(timeout_s=90.0)

    def _read(self, timeout_s: float) -> dict:
        if not hasattr(self, "reader"):
            self.reader = LineReader(self.process.stdout)
        try:
            line = self.reader.readline(timeout_s)
        except TimeoutError:
            self.process.kill()
            raise ConnectionError("카메라 서버가 응답하지 않았다") from None
        if not line:
            raise ConnectionError("카메라 서버 연결이 끊겼다")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"카메라 서버 거절: {response.get('error')}")
        return response

    def _call(self, request: dict, timeout_s: float = 10.0) -> dict:
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._read(timeout_s)

    def snapshot(self, name: str) -> Snapshot:
        response = self._call({"op": "snap", "name": name})
        return Snapshot(name, response["stamp"])

    def change(self, before: str, after: str, roi: tuple[int, int, int, int]) -> float:
        return self._call({"op": "diff", "a": before, "b": after, "roi": list(roi)})["fraction"]

    def yellow_pixels(self, name: str, roi: tuple[int, int, int, int]) -> int:
        return self._call({"op": "yellow", "name": name, "roi": list(roi)})["pixels"]

    def fetch(self, name: str, timeout_s: float = 10.0) -> Path:
        """찍은 사진을 Mac의 같은 상대 경로로 가져온다. 측정은 Mac 고정 OpenCV로만 한다."""
        local = Path(self.output) / f"{name}.color.png"
        local.parent.mkdir(parents=True, exist_ok=True)
        source = f"{self.host}:projects/learned-grasp-poc/{self.output}/{name}.color.png"
        try:
            done = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", source, str(local)], timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            raise FileNotFoundError(f"사진 가져오기 {timeout_s:g}초 초과: {name}") from error
        if done.returncode != 0:
            raise FileNotFoundError(f"사진 가져오기 실패: {name}")
        return local

    def close(self) -> None:
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
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
