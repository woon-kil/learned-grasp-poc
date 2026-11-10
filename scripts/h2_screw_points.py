#!/usr/bin/env python3
"""H2-1: DaBai 브래킷의 나사 머리 두 점(밝은 점)을 Gemini 컬러 영상에서 찾아 위치와 잡음을 낸다. 장치 없음.

절차(실행 전 고정): 회색조 → DoG(σ1 − σ4) → 임계(최대의 0.6) 연결 성분 → 면적 ≥ 3 → 면적이 큰 두 성분 중
행 차이 ≤ 4 화소, 간격 10~25 화소인 쌍 → 각 점은 DoG 양수 가중 무게중심(하위 화소).
사용: uv run python scripts/h2_screw_points.py <hover|grasp> <png...>
"""
import json
import sys

import cv2
import numpy as np

ROI = {"hover": (700, 0, 780, 70), "grasp": (690, 70, 770, 140)}  # x0, y0, x1, y1 (H2-0 사진 기준 대략 위치)


def find_pair(path: str, pose: str):
    x0, y0, x1, y1 = ROI[pose]
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1].astype(np.float32)
    d = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 4.0)
    mask = (d > 0.6 * d.max()).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 3:
            continue
        ys, xs = np.nonzero(lab == i)
        w = np.clip(d[ys, xs], 0, None)
        comps.append((int(stats[i, cv2.CC_STAT_AREA]), float((xs * w).sum() / w.sum() + x0), float((ys * w).sum() / w.sum() + y0)))
    comps.sort(key=lambda c: -c[0])
    best = None
    for a in range(len(comps)):
        for b in range(a + 1, len(comps)):
            (_, ax, ay), (_, bx, by) = comps[a], comps[b]
            if abs(ay - by) <= 4 and 10 <= abs(ax - bx) <= 25:
                best = (comps[a], comps[b]) if ax < bx else (comps[b], comps[a])
                return best
    return None


def main() -> None:
    pose, paths = sys.argv[1], sys.argv[2:]
    rows, miss = [], 0
    for path in paths:
        pair = find_pair(path, pose)
        if pair is None:
            miss += 1
            continue
        (_, lx, ly), (_, rx, ry) = pair
        rows.append((lx, ly, rx, ry))
    a = np.array(rows)
    out = {"pose": pose, "frames": len(paths), "detected": len(rows), "missed": miss}
    if len(rows):
        mid = np.c_[(a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2]
        out |= {"mid_mean": mid.mean(0).round(2).tolist(), "mid_std": mid.std(0, ddof=1).round(3).tolist() if len(rows) > 1 else None,
                "spacing_mean": float((a[:, 2] - a[:, 0]).mean().round(2)), "left": a[:, :2].mean(0).round(2).tolist(), "right": a[:, 2:].mean(0).round(2).tolist()}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
