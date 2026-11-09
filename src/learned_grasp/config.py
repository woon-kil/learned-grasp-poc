"""구성 로드. 실장치 실행에 필요한 값이 비어 있으면 거부한다."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CHAIN_PATH = ROOT / "assets" / "kinematic_chain.json"

# 닫으면 손끝이 약 16~20 mm 돌출하는 그리퍼 구조 때문에 그 아래 높이를 막는다. 모델 TCP 명목값이다.
# 이 높이에서 닫아도 된다는 보증이 아니다. 17.5 mm 닫기는 시험하지 않았다.
MIN_Z_GRASP_MM = 17.5


class ConfigError(ValueError):
    """구성이 해당 용도에 쓸 수 없다."""


@dataclass(frozen=True)
class TableSpec:
    z_grasp_mm: float
    tilt_deg: float
    tilt_tol_deg: float
    z_tol_mm: float
    r_min_mm: float
    r_max_mm: float
    r_step_mm: float
    hover_lift_mm: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.z_grasp_mm) or self.z_grasp_mm < MIN_Z_GRASP_MM:
            raise ConfigError(f"z_grasp_mm {self.z_grasp_mm}는 하한 {MIN_Z_GRASP_MM} mm 미만이거나 유한값이 아니다")


@dataclass(frozen=True)
class Motion:
    min_time_ms: int
    ms_per_deg: int
    max_move_deg: int
    first_move_time_ms: int
    grip_time_ms: int
    settle_ms: int
    arrival_tol_deg: int


@dataclass(frozen=True)
class Config:
    config_version: str
    verified_g0a: bool
    table: TableSpec
    j1_range: tuple[int, int]
    j5_range: tuple[int, int]
    park: tuple[int, int, int, int, int] | None
    open_sdk: int | None
    close_sdk: int | None
    motion: Motion

    def require_motion_values(self) -> None:
        """이동에 필요한 값이 G0a에서 정해졌는지 확인한다."""
        missing = [
            name
            for name, value in (("park", self.park), ("open_sdk", self.open_sdk), ("close_sdk", self.close_sdk))
            if value is None
        ]
        if missing:
            raise ConfigError(f"구성에 값이 없다: {', '.join(missing)}")


def parse_config(raw: dict) -> Config:
    park = raw.get("park")
    if park is not None and len(park) != 5:
        raise ConfigError("park는 관절 5개다")
    return Config(
        config_version=str(raw["config_version"]),
        verified_g0a=bool(raw["verified_g0a"]),
        table=TableSpec(**raw["table"]),
        j1_range=tuple(raw["ranges"]["j1_deg"]),
        j5_range=tuple(raw["ranges"]["j5_deg"]),
        park=tuple(park) if park is not None else None,
        open_sdk=raw["gripper"]["open_sdk"],
        close_sdk=raw["gripper"]["close_sdk"],
        motion=Motion(**raw["motion"]),
    )


def load_config(path: Path) -> Config:
    return parse_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
