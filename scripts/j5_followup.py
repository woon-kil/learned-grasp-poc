#!/usr/bin/env python3
"""J5 정렬 분석 보강. 장치 없음, 쓰기 0. (advisor 지적 반영)

1. d 모델을 포즈(셀) 단위 leave-one-pose-out으로 다시 평가(사진 70장은 20개 포즈의 반복).
2. 연쇄 시험에서 팔이 놓은 4개 읽기의 부호 있는 d를 모델 예측과 비교(표본 밖 자료), 목록 40자리의 예측 d와 |d|>5° 비율.
3. L1 맵 위치 오차(leave-one-pose-out)를 J5 그룹(60·120 대 90)별로 횡방향·반경으로 환산.
사용: uv run python scripts/j5_followup.py
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, block_from_image
from learned_grasp.pose_table import generate

PX_PER_MM = 1.14


def signed(a, rad):
    return ((a - rad + 90.0) % 180.0) - 90.0


def main() -> None:
    table = generate(load_config("config/workspace.yaml").table)
    rows_mm = [e.r_mm for e in table.entries]
    l1 = L1Map()
    o = json.loads(Path("runs/G0b/j5-yaw.json").read_text())
    # 포즈 = (src가 G0b/G0c면 cell, G0b-9면 사진 하나)
    def pose_id(r):
        return r["cell"]
    poses = sorted({pose_id(r) for r in o})
    print(f"사진 {len(o)}장, 서로 다른 포즈 {len(poses)}개, J5 90 포즈 {len({pose_id(r) for r in o if r['j5'] == 90})}개: {sorted({pose_id(r) for r in o if r['j5'] == 90})}")
    X = np.array([[1, r["j5"] - 90, r["j1"] - 90, r["r_index"] - 4.5] for r in o], float)
    y = np.array([r["d"] for r in o])
    g = np.array([pose_id(r) for r in o])
    models = {"J5": [0, 1], "J5+J1": [0, 1, 2], "J5+J1+r": [0, 1, 2, 3]}
    full = {}
    print("-- 1. d 모델, 포즈 단위 leave-one-pose-out")
    for name, cols in models.items():
        res = np.zeros(len(y))
        for p in poses:
            tr, te = g != p, g == p
            W, *_ = np.linalg.lstsq(X[tr][:, cols], y[tr], rcond=None)
            res[te] = X[te][:, cols] @ W - y[te]
        W, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        full[name] = W
        print(f"  {name}: 계수 {np.round(W, 3).tolist()}, 포즈 단위 LOO 표준편차 {res.std():.1f}°, |p95| {np.percentile(abs(res), 95):.1f}°")

    def predict(name, j5, j1, r_index):
        cols = models[name]
        return float(np.array([1, j5 - 90, j1 - 90, r_index - 4.5])[cols] @ full[name])

    print("-- 2. 연쇄 시험에서 팔이 놓은 블록(표본 밖)")
    reads = [("CH-0 chain-002", "runs/CH-0/frames/chain-002.color.png", 8, 104), ("CH-1 chain-001", "runs/CH-1/frames/chain-001.color.png", 4, 75),
             ("CH-1 chain-002", "runs/CH-1/frames/chain-002.color.png", 8, 104), ("CH-1 chain-003", "runs/CH-1/frames/chain-003.color.png", 4, 75)]
    for label, path, r_index, j1 in reads:
        img = cv2.imread(path)
        if img is None:
            print(f"  {label}: 사진 없음 {path}")
            continue
        m = block_from_image(img, board_ref=(691.0, 291.0))
        rad = l1.radial_angle_deg(j1, rows_mm[r_index]) % 180.0
        d = signed(m["block_angle_deg"], rad)
        pred = predict("J5+J1+r", 90, j1, r_index)
        print(f"  {label}: 놓은 자리 (r_index {r_index}, J1 {j1}), 측정 d {d:+.1f}°, 모델 예측 {pred:+.1f}°, 잔차 {d - pred:+.1f}°")
    cells = json.loads(Path("config/chain_cells.json").read_text())["cells"]
    pred = np.array([predict("J5+J1+r", 90, c["j1"], c["r_index"]) for c in cells])
    within = 2.52
    from math import erf, sqrt

    def over5(mu):
        cdf = lambda x: 0.5 * (1 + erf((x - mu) / (within * sqrt(2))))
        return 1 - (cdf(5) - cdf(-5))

    share = np.mean([over5(m) for m in pred])
    print(f"  목록 40자리: 예측 d 범위 {pred.min():+.1f}~{pred.max():+.1f}°, 평균 |d| {abs(pred).mean():.1f}°, 예측 |d|>5° 자리 {int((abs(pred) > 5).sum())}개, "
          f"셀 안 표준편차 {within}°를 더한 |d|>5° 기대 비율 {share:.2f}")

    print("-- 3. L1 맵 위치 오차(leave-one-pose-out, 20 포즈)를 J5 그룹별로")
    m = json.loads(Path("config/l1_map.json").read_text())
    pts = m["points"]
    def feat(a, b, j5):
        return np.array([1.0, a, b, a * a, a * b, b * b, (j5 - 90.0)])
    Xp = np.array([feat(p["j1"] - 90.0, p["r_mm"] - 140.0, p["j5"]) for p in pts])
    U = np.array([[p["du"], p["dv"]] for p in pts])
    out = {}
    for i, p in enumerate(pts):
        keep = [k for k in range(len(pts)) if k != i]
        W, *_ = np.linalg.lstsq(Xp[keep], U[keep], rcond=None)
        e = Xp[i] @ W - U[i]
        rad = math.radians(l1.radial_angle_deg(p["j1"], p["r_mm"]))
        ur = np.array([math.cos(rad), math.sin(rad)])
        un = np.array([-math.sin(rad), math.cos(rad)])
        out.setdefault(p["j5"], []).append((float(e @ un) / PX_PER_MM, float(e @ ur) / PX_PER_MM))
    for j5 in (60, 90, 120):
        v = np.array(out.get(j5, []))
        if len(v):
            print(f"  J5 {j5}: 포즈 {len(v)}개, 횡방향 RMS {np.sqrt((v[:, 0] ** 2).mean()):.1f} mm, 반경 RMS {np.sqrt((v[:, 1] ** 2).mean()):.1f} mm, 최대 |횡방향| {abs(v[:, 0]).max():.1f} mm")
    allv = np.array([x for v in out.values() for x in v])
    print(f"  전체: 횡방향 RMS {np.sqrt((allv[:, 0] ** 2).mean()):.1f} mm, 반경 RMS {np.sqrt((allv[:, 1] ** 2).mean()):.1f} mm")
    off = np.array([x for k, v in out.items() if k != 90 for x in v])
    print(f"  J5≠90: 횡방향 RMS {np.sqrt((off[:, 0] ** 2).mean()):.1f} mm (창 6.48 mm, ψ 5° 기준)")


if __name__ == "__main__":
    main()
