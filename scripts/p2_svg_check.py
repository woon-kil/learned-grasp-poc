#!/usr/bin/env python3
"""P2 목표 SVG 사전 점검. 장치 없음. 사용자에게 보이기 전에 목표 사각형이 놓을 수 있는 자리인지 사진(빈 보드)에서 확인한다.

검사: (1) 위 끝이 받침대(영상 y 195)보다 아래 (2) 사각형 안 화소가 모두 흰 종이색(채도 낮고 밝음: 마커·바깥 4X4 카드·나무·받침대 제외) (3) L1 허용 J1 70~110 안쪽 여유 (4) 보드 마커 id 1에서 거리.
사용: uv run python scripts/p2_svg_check.py data/p2/blank.png p2-16 [p2-17 ...]
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

L, W = 60.33 * 1.14 / 2, 29.89 * 1.14 / 2
BOARD = (691.25, 291.5)


def polygon(t):
    cx, cy = BOARD[0] + t["du"], BOARD[1] + t["dv"]
    a = math.radians(t["block_angle_deg"])
    ux, uy = math.cos(a), math.sin(a)
    return np.array([(cx + sl * L * ux - sw * W * uy, cy + sl * L * uy + sw * W * ux) for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def main() -> int:
    img = cv2.imread(sys.argv[1])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    targets = {x["id"]: x for x in json.loads(Path("config/p2_targets.json").read_text())["targets"]}
    bad = 0
    for tid in sys.argv[2:]:
        t = targets[tid]
        pts = polygon(t)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        inside = mask.astype(bool)
        s, v = hsv[:, :, 1][inside], hsv[:, :, 2][inside]
        paper = float(np.mean((s < 70) & (v > 150)))
        top = float(pts[:, 1].min())
        d_marker = float(np.hypot(*(pts.mean(0) - np.array([691.25, 291.0]))))
        problems = []
        if top < 195:
            problems.append(f"받침대와 겹침(위 끝 y {top:.0f} < 195)")
        if paper < 0.97:
            problems.append(f"흰 종이 밖/마커·나무 포함(종이색 비율 {paper:.2f})")
        if not (73 <= t["j1"] <= 104):
            problems.append(f"J1 {t['j1']}이 허용 70~110의 가장자리")
        if d_marker < 35:
            problems.append(f"보드 마커에 가까움({d_marker:.0f} 화소)")
        print(f"{tid}: 위 끝 y {top:.0f}, 종이색 {paper:.2f}, 마커 거리 {d_marker:.0f}px, J1 {t['j1']}, 목표 J5 {t['target_j5']} → " + ("통과" if not problems else "문제: " + "; ".join(problems)))
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
