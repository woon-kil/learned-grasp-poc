#!/usr/bin/env python3
"""J5 정렬 모의. 장치 없음, 쓰기 0.

횡방향 창 m(ψ) = (48 − (29.89 cosψ + 60.33 sinψ))/2 mm(열린 손 개방 간격 48 mm, 블록 폭 29.89·윗면 길이 60.33 mm)에 대해,
집는 순간 손과 블록의 실제 어긋남 ψ와 횡방향 위치 오차(L1+L2 모의 σ 2.13 mm)로 성공 확률을 모의한다. 모의이고 실측이 아니다.
정렬 없음(J5 90): ψ = 손 yaw 모델 d_hand(J5 90, J1, r) + 모델 오차. 정렬 후: ψ = 모델 오차 + 측정 오차 + J5 반올림.
사용: uv run python scripts/j5_align_sim.py
"""
import json
import math
from pathlib import Path

import numpy as np

from learned_grasp.l1 import YawModel

LAT_SIGMA = 2.13  # mm, docs/*연쇄-시험-구현과-L2-반올림-계산-결과.md
MEAS_SIGMA = 1.5  # ° 블록 각도 측정 잡음(같은 상태 4.5°↔5.5° 관찰에서 가정)
ROUND = 0.5 / math.sqrt(3)  # J5 정수 반올림(균일 ±0.5°)
N = 200000


def window(psi_deg):
    p = np.radians(np.abs(psi_deg))
    return (48.0 - (29.89 * np.cos(p) + 60.33 * np.sin(p))) / 2.0


def success(psi, rng):
    lat = rng.normal(0.0, LAT_SIGMA, len(psi))
    return float(np.mean(np.abs(lat) <= window(psi)))


def main() -> None:
    yaw = YawModel()
    cells = json.loads(Path("config/chain_cells.json").read_text())["cells"]
    rng = np.random.default_rng(20261112)
    mean_hand = np.array([yaw.hand_d(90, c["j1"], c["r_index"]) for c in cells])
    print(f"창 m(ψ): ψ0 {window(0):.2f}, 3° {window(3):.2f}, 5° {window(5):.2f}, 8° {window(8):.2f}, 10° {window(10):.2f}, 14° {window(14):.2f} mm. 횡방향 σ {LAT_SIGMA} mm")
    for model_err in (2.5, 3.9):
        idx = rng.integers(0, len(cells), N)
        base = mean_hand[idx] + rng.normal(0, model_err, N)  # J5 90: 손이 반경선과 어긋남 + 모델 오차(손 yaw 불확실)
        aligned = rng.normal(0, math.sqrt(model_err**2 + MEAS_SIGMA**2 + ROUND**2), N)
        print(f"손 yaw 모델 오차 σ {model_err}°: J5 90 정렬 없음 성공 {success(base, rng):.3f}, 정렬 후 {success(aligned, rng):.3f}")
    print(f"(J5 90, 정렬 없음에서 반경선 기준 |d|>5° 비율 {np.mean(abs(mean_hand + rng.normal(0, 2.52, len(mean_hand))) > 5):.2f}: 목록 40자리 평균)")
    # 정렬 뒤 목록 자리를 놓을 때 J5와 다음 집기 J5 범위
    place = np.array([yaw.place_j5(c["j1"], c["r_index"]) for c in cells])
    print(f"놓기 J5(목록 40자리): 최소 {place.min()}, 최대 {place.max()}, J5≠90 자리 {int((place != 90).sum())}개")


if __name__ == "__main__":
    main()
