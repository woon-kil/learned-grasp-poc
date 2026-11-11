"""L1: 놓은 블록의 Gemini 사진 → 명령 (r_index, J1). 모델 값이고 정확도는 P1이 처음 잰다.

블록 위치 = 노란 영역(HSV H20~35, S>200, V>200) 볼록 껍질의 무게중심, 보드 마커 id 1 대비 (du, dv) 화소.
맵 = config/l1_map.json(scripts/l1_fit_map.py): (du, dv) = 2차 다항식(J1, r) + J5 선형항. J5 90에서 수치로 역변환한다.
L2 = J1 정수 반올림, r은 표에서 가장 가까운 행. 구역 밖이거나 역변환 잔차가 크면 시도하지 않는다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from learned_grasp.config import ROOT
from learned_grasp.pose_table import PoseTable

MAP_PATH = ROOT / "config" / "l1_map.json"
J1_RANGE = (70.0, 110.0)
INVERSE_TOL_PX = 2.0
PSI_MAX_DEG = 5.0  # 블록 방향과 반경 방향의 차 허용(플랜 §2.2: 창 6.5 mm)


@dataclass(frozen=True)
class Plan:
    accept: bool
    reason: str
    r_index: int | None = None
    j1: int | None = None
    j1_cont: float | None = None
    r_cont: float | None = None
    residual_px: float | None = None
    du: float | None = None
    dv: float | None = None
    psi_deg: float | None = None
    block_angle_deg: float | None = None

    def asdict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


class L1Map:
    def __init__(self, path: Path = MAP_PATH):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.cu = np.array(data["coef_du"], float)
        self.cv = np.array(data["coef_dv"], float)
        self.data = data

    @staticmethod
    def _f(j1: float, r: float, j5: float) -> np.ndarray:
        a, b = j1 - 90.0, r - 140.0
        return np.array([1.0, a, b, a * a, a * b, b * b, j5 - 90.0])

    def forward(self, j1: float, r_mm: float, j5: float = 90.0) -> np.ndarray:
        f = self._f(j1, r_mm, j5)
        return np.array([f @ self.cu, f @ self.cv])

    def inverse(self, du: float, dv: float, r_bounds: tuple[float, float], j5: float = 90.0):
        """(du, dv) → (J1, r_mm, 잔차 화소). 여러 시작점의 Gauss-Newton, 경계로 자른다."""
        target = np.array([du, dv], float)
        best = None
        for j1_0 in (80.0, 90.0, 100.0):
            for r_0 in (130.0, 142.0, 155.0):
                x = np.array([j1_0, r_0])
                for _ in range(40):
                    f = self.forward(x[0], x[1], j5) - target
                    J = np.empty((2, 2))
                    for k, h in enumerate((1e-3, 1e-3)):
                        xp = x.copy()
                        xp[k] += h
                        J[:, k] = (self.forward(xp[0], xp[1], j5) - self.forward(x[0], x[1], j5)) / h
                    try:
                        step = np.linalg.solve(J, -f)
                    except np.linalg.LinAlgError:
                        break
                    x = x + np.clip(step, -10.0, 10.0)
                    x[0] = min(max(x[0], J1_RANGE[0] - 15.0), J1_RANGE[1] + 15.0)
                    x[1] = min(max(x[1], r_bounds[0] - 25.0), r_bounds[1] + 25.0)
                    if np.linalg.norm(step) < 1e-6:
                        break
                res = float(np.linalg.norm(self.forward(x[0], x[1], j5) - target))
                if best is None or res < best[2]:
                    best = (float(x[0]), float(x[1]), res)
        return best

    def radial_angle_deg(self, j1: float, r_mm: float) -> float:
        """이 자리에서 반경이 늘 때 영상 위치가 움직이는 방향(영상 각도, 도)."""
        d = (self.forward(j1, r_mm + 0.5) - self.forward(j1, r_mm - 0.5))
        return math.degrees(math.atan2(d[1], d[0]))


def block_from_image(img: np.ndarray, board_ref=None) -> dict | None:
    """BGR 영상에서 블록 위치(보드 마커 대비 화소)와 긴 변 각도를 구한다. 못 찾으면 None."""
    import cv2
    from cv2 import aruco

    params = aruco.DetectorParameters()
    params.errorCorrectionRate = 1.0  # 보드 마커 id 1은 기본 설정에서 조명에 따라 놓치는 사진이 있다(P1 p1-02 빈 보드 사진)
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), params)
    corners, ids, _ = det.detectMarkers(img[:, :, 2])
    board = None
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if int(i) == 1:
                board = c.reshape(4, 2).mean(0)
    if board is None:
        board = None if board_ref is None else np.asarray(board_ref, float)
    if board is None:
        return None
    x0, y0, x1, y1 = 480, 100, 960, 470
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 200) & (hsv[:, :, 2] > 200)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < 800:
        return None
    ys, xs = np.nonzero(lab == i)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    m = cv2.moments(hull)
    center = np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0])
    (cx, cy), (w, h), ang = cv2.minAreaRect(hull)
    long_angle = ang if w >= h else ang + 90.0
    return {"du": float(center[0] - board[0]), "dv": float(center[1] - board[1]), "block_angle_deg": float(long_angle % 180.0), "area": int(stats[i, cv2.CC_STAT_AREA]), "board": board.tolist()}


def plan(measured: dict | None, table: PoseTable, l1: L1Map | None = None) -> Plan:
    """측정한 블록 위치로 (r_index, J1)과 ψ를 계산한다. 시도 불가면 accept=False와 이유."""
    if measured is None:
        return Plan(False, "블록 또는 보드 마커 id 1을 찾지 못했다")
    l1 = l1 or L1Map()
    r_lo, r_hi = table.entries[0].r_mm, table.entries[-1].r_mm
    inv = l1.inverse(measured["du"], measured["dv"], (r_lo, r_hi))
    j1c, rc, res = inv
    base = dict(j1_cont=j1c, r_cont=rc, residual_px=res, du=measured["du"], dv=measured["dv"], block_angle_deg=measured["block_angle_deg"])
    if res > INVERSE_TOL_PX:
        return Plan(False, f"맵 구역 밖(역변환 잔차 {res:.1f}화소)", **base)
    if not (J1_RANGE[0] - 0.5 <= j1c <= J1_RANGE[1] + 0.5) or not (r_lo - 2.5 <= rc <= r_hi + 2.5):
        return Plan(False, f"자세 범위 밖(J1 {j1c:.1f}, r {rc:.1f} mm)", **base)
    j1 = int(min(max(round(j1c), J1_RANGE[0]), J1_RANGE[1]))
    r_index = int(np.argmin([abs(e.r_mm - rc) for e in table.entries]))
    radial = l1.radial_angle_deg(j1c, rc) % 180.0
    d = abs(measured["block_angle_deg"] - radial) % 180.0
    psi = min(d, 180.0 - d)
    base["psi_deg"] = psi
    if psi > PSI_MAX_DEG:
        return Plan(False, f"블록 방향이 반경 방향과 {psi:.1f}° 어긋남(허용 {PSI_MAX_DEG:g}°)", r_index=r_index, j1=j1, **base)
    return Plan(True, "ok", r_index=r_index, j1=j1, **base)
