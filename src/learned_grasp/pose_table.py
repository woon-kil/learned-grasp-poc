"""허용 자세 표. 실행 가능한 (J2, J3, J4)는 이 표의 항목뿐이다.

표는 J1=90, J5=90 평면에서 만든 정수 SDK 해이고 r로 색인한다. J1, J5는 구성 범위 안에서 따로 정한다.
r은 모델 값이다. 실물 거리나 작업대 높이로 해석하지 않는다.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_left
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from learned_grasp.config import Config, TableSpec
from learned_grasp.kinematics import fk_batch

_J2 = np.arange(40, 136)  # 작은 반경의 대기 자세는 J2 123~133이 필요하다
_J3 = np.arange(0, 121)
_J4 = np.arange(0, 121)
_HOVER_R_TOL_MM = 2.0
_HOVER_Z_TOL_MM = 3.0
_HOVER_MAX_TILT = 50.0


class TableError(ValueError):
    """표를 만들거나 쓸 수 없다."""


@dataclass(frozen=True)
class TableEntry:
    r_mm: float
    grasp: tuple[int, int, int]
    hover: tuple[int, int, int]


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.stack(np.meshgrid(_J2, _J3, _J4, indexing="ij"), axis=-1).reshape(-1, 3)
    q = np.column_stack([np.full(len(grid), 90), grid, np.full(len(grid), 90)])
    position, tilt = fk_batch(q)
    return grid, position, tilt


def generate(spec: TableSpec) -> "PoseTable":
    """r 구간마다 이전 항목과 가장 가까운 정수 해를 고른다. 같은 IK 분기를 유지하기 위해서다."""
    grid, position, tilt = _grid()
    r_axis, z_axis = position[:, 1], position[:, 2]
    grasp_mask = (np.abs(z_axis - spec.z_grasp_mm) <= spec.z_tol_mm) & (np.abs(tilt - spec.tilt_deg) <= spec.tilt_tol_deg)
    hover_mask = (np.abs(z_axis - (spec.z_grasp_mm + spec.hover_lift_mm)) <= _HOVER_Z_TOL_MM) & (tilt <= _HOVER_MAX_TILT)
    entries: list[TableEntry] = []
    previous: np.ndarray | None = None
    for nominal in np.arange(spec.r_min_mm, spec.r_max_mm + 1e-6, spec.r_step_mm):
        near = grasp_mask & (np.abs(r_axis - nominal) <= spec.z_tol_mm)
        indices = np.flatnonzero(near)
        if not len(indices):
            continue
        if previous is None:
            error = np.hypot(r_axis[indices] - nominal, z_axis[indices] - spec.z_grasp_mm)
            chosen = indices[int(np.argmin(error))]
        else:
            jump = np.abs(grid[indices] - previous).max(axis=1)
            chosen = indices[int(np.argmin(jump))]
        grasp = grid[chosen]
        hover_indices = np.flatnonzero(hover_mask & (np.abs(r_axis - r_axis[chosen]) <= _HOVER_R_TOL_MM))
        if not len(hover_indices):
            continue
        hover = grid[hover_indices[int(np.argmin(np.abs(grid[hover_indices] - grasp).max(axis=1)))]]
        entries.append(TableEntry(round(float(r_axis[chosen]), 3), tuple(int(v) for v in grasp), tuple(int(v) for v in hover)))
        previous = grasp
    table = PoseTable(entries)
    if len(table.entries) < 2:
        raise TableError("표 항목이 2개 미만이다. 높이·기울기 구성을 확인한다")
    return table


class PoseTable:
    def __init__(self, entries: list[TableEntry]):
        unique = {entry.r_mm: entry for entry in entries}
        self.entries = [unique[r] for r in sorted(unique)]
        self._radii = [entry.r_mm for entry in self.entries]

    def nearest_index(self, r_mm: float) -> int:
        position = bisect_left(self._radii, r_mm)
        candidates = [i for i in (position - 1, position) if 0 <= i < len(self.entries)]
        return min(candidates, key=lambda i: abs(self._radii[i] - r_mm))

    def model_r(self, grasp: tuple[int, int, int]) -> float:
        position, _ = fk_batch(np.array([[90, *grasp, 90]]))
        return float(position[0, 1])

    def recover(self, grasp: tuple[int, int, int]) -> tuple[int, int, int]:
        """정수 해 → 라벨 r → 표 항목. 라벨에서 명령이 복원되는지 검사하는 경로다."""
        return self.entries[self.nearest_index(self.model_r(grasp))].grasp

    def to_json(self) -> str:
        return json.dumps({"entries": [asdict(entry) for entry in self.entries]}, ensure_ascii=False, indent=1) + "\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PoseTable":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            [TableEntry(item["r_mm"], tuple(item["grasp"]), tuple(item["hover"])) for item in raw["entries"]]
        )


def table_for(config: Config) -> PoseTable:
    return generate(config.table)
