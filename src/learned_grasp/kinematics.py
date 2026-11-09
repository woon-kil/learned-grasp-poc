"""모델 FK. 결과는 모델 값이고 실측 TCP가 아니다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learned_grasp.config import CHAIN_PATH


def _rpy(rpy: np.ndarray) -> np.ndarray:
    cr, sr = np.cos(rpy[0]), np.sin(rpy[0])
    cp, sp = np.cos(rpy[1]), np.sin(rpy[1])
    cy, sy = np.cos(rpy[2]), np.sin(rpy[2])
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _load_chain(path: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
    joints = json.loads(Path(path).read_text(encoding="utf-8"))["joints"]
    return [(np.array(j["xyz"]), np.array(j["rpy"]), np.array(j["axis"]), j["type"]) for j in joints]


_CHAIN = _load_chain(CHAIN_PATH)


def _axis_rotations(axis: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """축 고정 Rodrigues 회전을 N개 각도에 대해 계산한다."""
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angles), np.sin(angles)
    t = 1.0 - c
    out = np.empty((len(angles), 3, 3))
    out[:, 0, 0] = c + x * x * t
    out[:, 0, 1] = x * y * t - z * s
    out[:, 0, 2] = x * z * t + y * s
    out[:, 1, 0] = y * x * t + z * s
    out[:, 1, 1] = c + y * y * t
    out[:, 1, 2] = y * z * t - x * s
    out[:, 2, 0] = z * x * t - y * s
    out[:, 2, 1] = z * y * t + x * s
    out[:, 2, 2] = c + z * z * t
    return out


def fk_batch(sdk_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SDK 도(N×5) → 모델 TCP mm(N×3), 접근축과 수직 아래의 각도(N)."""
    q = np.radians(np.atleast_2d(np.asarray(sdk_degrees, dtype=float)) - 90.0)
    if q.shape[1] != 5:
        raise ValueError("arm joints are five")
    count = len(q)
    rotation = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    position = np.zeros((count, 3))
    for index, (xyz, rpy, axis, joint_type) in enumerate(_CHAIN):
        position = position + rotation @ xyz
        rotation = rotation @ _rpy(rpy)
        if joint_type != "fixed":
            rotation = rotation @ _axis_rotations(axis, q[:, index])
    approach = rotation[:, :3, 1]
    tilt = np.degrees(np.arccos(np.clip(-approach[:, 2], -1.0, 1.0)))
    return position * 1000.0, tilt


def fk(sdk_degrees) -> tuple[np.ndarray, float]:
    """한 자세의 모델 TCP mm와 수직 기울기."""
    position, tilt = fk_batch(np.asarray(sdk_degrees, dtype=float)[None, :])
    return position[0], float(tilt[0])
