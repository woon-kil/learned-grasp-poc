#!/usr/bin/env python3
"""L2 반올림 오차 분포. 장치 없음.

팔이 격자 자리에 놓은 블록의 실제 위치는 놓기 흩어짐(횡방향 1σ 2.8 mm, 반경 3.2 mm, 명령 셀→블록 영상 위치 맵 분석)만큼
연속값으로 흩어진다. L1이 그 위치를 (J1, r)로 재서 정수 J1·가장 가까운 표 행으로 반올림하면 집는 자리와 실제 블록 사이에
남는 오차(횡방향·반경)를 모의한다. L1 측정 오차(교차 검증 du 2.03, dv 4.16 화소)를 선택적으로 더한다.
사용: uv run python scripts/l2_rounding_sim.py
"""
import math

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import J1_RANGE, L1Map
from learned_grasp.pose_table import generate

SIGMA_LAT, SIGMA_RAD = 2.8, 3.2  # mm, 같은 셀 반복 놓기 흩어짐
LOO_DU, LOO_DV = 2.03, 4.16  # 화소, L1 맵 교차 검증
SCALE_TOP = 1.14  # 화소/mm 블록 윗면 높이
N = 20000


def main() -> None:
    table = generate(load_config("config/workspace.yaml").table)
    rows = np.array([e.r_mm for e in table.entries])
    rng = np.random.default_rng(20261112)
    l1 = L1Map()
    out = {}
    for label, with_l1 in (("L2만(L1 완벽)", False), ("L2 + L1 오차", True)):
        lat, rad = [], []
        for _ in range(N):
            j1 = int(rng.integers(72, 109))  # 놓은 격자 자리
            r_idx = int(rng.integers(1, 9))
            r0 = rows[r_idx]
            # 실제 블록 위치(연속): 격자 + 흩어짐
            j1_true = j1 + rng.normal(0, SIGMA_LAT) / (r0 * math.pi / 180.0)
            r_true = r0 + rng.normal(0, SIGMA_RAD)
            if with_l1:  # L1이 읽은 위치 = 실제 위치의 영상 좌표 + 맵 오차
                du, dv = l1.forward(j1_true, r_true)
                du += rng.normal(0, LOO_DU)
                dv += rng.normal(0, LOO_DV)
                j1_est, r_est, _ = l1.inverse(du, dv, (rows[0] - 10, rows[-1] + 10))
            else:
                j1_est, r_est = j1_true, r_true
            j1_pick = min(max(round(j1_est), J1_RANGE[0]), J1_RANGE[1])
            r_pick = rows[int(np.argmin(abs(rows - r_est)))]
            lat.append((j1_pick - j1_true) * r_true * math.pi / 180.0)
            rad.append(r_pick - r_true)
        lat, rad = np.array(lat), np.array(rad)
        out[label] = (lat, rad)
        print(f"{label}: 횡방향 평균 {lat.mean():+.2f} 표준편차 {lat.std():.2f} |p95| {np.percentile(abs(lat), 95):.2f} mm, "
              f"반경 평균 {rad.mean():+.2f} 표준편차 {rad.std():.2f} |p95| {np.percentile(abs(rad), 95):.2f} mm")
        for m in (6.48, 9.05):  # 창: ψ 5° 6.48 mm, ψ 0° 9.05 mm(열린 손 개방 간격 48 mm 기준)
            print(f"  횡방향 |오차| > {m} mm 비율 {np.mean(abs(lat) > m):.4f}")
    print("표 행 간격(mm):", np.round(np.diff(rows), 2).tolist())


if __name__ == "__main__":
    main()
