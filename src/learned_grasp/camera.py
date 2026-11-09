"""카메라 인터페이스. 실장치 스냅샷 경로는 G0b 전에 구현한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Snapshot:
    frame_id: str
    captured_at: float


class Camera(Protocol):
    def snapshot(self, name: str) -> Snapshot: ...


class FakeCamera:
    def __init__(self, clock=lambda: 0.0):
        self.clock = clock
        self.names: list[str] = []
        self.fail_next = False

    def snapshot(self, name: str) -> Snapshot:
        if self.fail_next:
            raise RuntimeError("스냅샷 실패")
        self.names.append(name)
        return Snapshot(name, self.clock())


class P620Camera:
    """p620 컨테이너의 Gemini 335L 스냅샷. 경로가 실측되기 전에는 명시적으로 실패한다."""

    def snapshot(self, name: str) -> Snapshot:
        raise NotImplementedError("p620 스냅샷 경로는 G0b 전 작업이다. PROJECT-STATUS.md 참조")
