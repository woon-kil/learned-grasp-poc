#!/usr/bin/env python3
"""J5 대 블록 방향(ψ 부호 있음) 분석. 장치 없음, 쓰기 0.

runs/G0b/block-map.json의 70장(G0b·G0c·G0b-9, 팔이 명령 자세로 놓은 블록)에서 블록 긴 변 각도를 재고, 명령 자리의 반경 방향 각도(L1 맵)와의
부호 있는 차 d를 낸다. J5 그룹별 평균·표준편차와 d = a + k·(J5 − 90) 적합을 출력한다.
사용: uv run python scripts/g0_j5_yaw.py runs/G0b/block-map.json
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, block_from_image
from learned_grasp.pose_table import generate


def signed_diff(block_angle: float, radial: float) -> float:
    return ((block_angle - radial + 90.0) % 180.0) - 90.0


def main() -> None:
    rows = [r for r in json.loads(Path(sys.argv[1]).read_text()) if "du" in r]
    table = generate(load_config("config/workspace.yaml").table)
    l1 = L1Map()
    out = []
    for r in rows:
        img = cv2.imread(r["file"])
        if img is None:
            continue
        m = block_from_image(img, board_ref=r["board"])
        if m is None:
            continue
        rad = l1.radial_angle_deg(r["j1"], table.entries[r["r_index"]].r_mm) % 180.0
        out.append({"src": r["src"], "cell": r["cell"], "j1": r["j1"], "r_index": r["r_index"], "j5": r["j5"], "block": m["block_angle_deg"], "radial": rad, "d": signed_diff(m["block_angle_deg"], rad), "area": m["area"]})
    print(f"사진 {len(out)}/{len(rows)}장")
    for j5 in (60, 90, 120):
        d = np.array([o["d"] for o in out if o["j5"] == j5])
        if len(d):
            print(f"J5 {j5}: n {len(d)}, d 평균 {d.mean():+.1f}°, 표준편차 {d.std():.1f}°, 범위 {d.min():+.1f}~{d.max():+.1f}°")
    x = np.array([o["j5"] - 90.0 for o in out]); y = np.array([o["d"] for o in out])
    k, a = np.polyfit(x, y, 1)
    res = y - (a + k * x)
    print(f"적합 d = {a:+.1f}° + {k:+.3f}·(J5−90): 잔차 표준편차 {res.std():.1f}°, 최대 |잔차| {abs(res).max():.1f}°")
    for j1 in (70, 90, 110):
        d = np.array([o["d"] for o in out if o["j5"] == 90 and o["j1"] == j1])
        if len(d):
            print(f"  J5 90, J1 {j1}: n {len(d)}, d 평균 {d.mean():+.1f}°, 표준편차 {d.std():.1f}°")
    json.dump(out, open("runs/G0b/j5-yaw.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
