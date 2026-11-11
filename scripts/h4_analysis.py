#!/usr/bin/env python3
"""H4 hover 스윕 분석(장치 없음). 손끝 검출 정의 후보를 비교한다. 각도는 마커 id 1 호모그래피로 보정한 좌표에서 잰다.
사용: uv run python scripts/h4_analysis.py runs/H4-0
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

DARK = 90


def marker_h(img):
    p = aruco.DetectorParameters()
    p.errorCorrectionRate = 1.0
    d = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), p)
    c, i, _ = d.detectMarkers(img[:, :, 2])
    if i is None:
        return None
    for cc, k in zip(c, i.ravel()):
        if int(k) == 1:
            src = cc.reshape(4, 2).astype(np.float32)
            return cv2.getPerspectiveTransform(src, np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32)), src.mean(0)
    return None


def tf(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def finger_blobs(img, center):
    """예측 손끝 중심 둘레 창에서 어두운 화소의 아래쪽 부분으로 두 손가락을 가른다. (x0, y0)과 마스크를 돌려준다."""
    cx, cy = center
    x0, y0, x1, y1 = int(cx - 110), int(cy - 75), int(cx + 110), int(cy + 5)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    ys, xs = np.nonzero(g < DARK)
    if len(ys) == 0:
        return None
    low = ys > ys.max() - 30
    lx = xs[low]
    sx = np.sort(lx)
    gaps = np.diff(sx)
    if len(gaps) == 0:
        return None
    k = int(np.argmax(gaps))
    if gaps[k] < 20:
        return None
    split = (sx[k] + sx[k + 1]) / 2.0
    return x0, y0, ys, xs, split


def tips_A(img, center, band=8):
    """정의 A: 손가락마다 가장 아래 band행의 평균 위치."""
    fb = finger_blobs(img, center)
    if fb is None:
        return None
    x0, y0, ys, xs, split = fb
    out = []
    for side in (xs < split, xs >= split):
        sy, sx = ys[side], xs[side]
        low = sy > sy.max() - 30
        sy, sx = sy[low], sx[low]
        b = sy > sy.max() - band
        out.append((sx[b].mean() + x0, sy[b].mean() + y0))
    return out


def tips_B(img, center):
    """정의 B: 손가락 아래 30행의 주축(PCA) 중심선이 두 손가락의 공통 높이(두 손가락 아래 끝의 평균 y)를 지나는 점."""
    fb = finger_blobs(img, center)
    if fb is None:
        return None
    x0, y0, ys, xs, split = fb
    parts = []
    for side in (xs < split, xs >= split):
        sy, sx = ys[side], xs[side]
        low = sy > sy.max() - 30
        parts.append((sx[low].astype(float), sy[low].astype(float)))
    ytarget = np.mean([p[1].max() for p in parts]) - 4
    out = []
    for px, py in parts:
        m, c = np.polyfit(py, px, 1)  # x = m y + c (세로 손가락)
        out.append((m * ytarget + c + x0, ytarget + y0))
    return out


def tips_C(img, center):
    """정의 C: 손가락 아래 25행 안쪽 모서리(서로 마주보는 쪽 가장자리)의 평균 x와 y."""
    fb = finger_blobs(img, center)
    if fb is None:
        return None
    x0, y0, ys, xs, split = fb
    out = []
    for side, inner in ((xs < split, "max"), (xs >= split, "min")):
        sy, sx = ys[side], xs[side]
        yb = sy.max()
        sel = sy > yb - 25
        sy, sx = sy[sel], sx[sel]
        edges = []
        for y in np.unique(sy):
            row = sx[sy == y]
            edges.append((row.max() if inner == "max" else row.min(), y))
        e = np.array(edges, float)
        out.append((e[:, 0].mean() + x0, e[:, 1].mean() + y0))
    return out


def tips_D(img, center):
    """정의 D: 손가락 원통(폭 8~24 화소, 세로로 25행 이상 이어지는 어두운 구간)을 행마다 추적해 아래 끝점을 잡는다. 창은 고정(손이 어느 자세에서도 들어오는 영역)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = 600, 130, 830, 270
    m = (g[y0:y1, x0:x1] < 80).astype(np.uint8)
    runs = {}
    for y in range(m.shape[0]):
        row = m[y]
        xs = np.nonzero(row)[0]
        if len(xs) == 0:
            continue
        splits = np.nonzero(np.diff(xs) > 1)[0]
        starts = np.r_[xs[0], xs[splits + 1]]
        ends = np.r_[xs[splits], xs[-1]]
        runs[y] = [((a + b) / 2.0, b - a + 1) for a, b in zip(starts, ends) if 8 <= b - a + 1 <= 24]
    # 아래에서 위로: 손가락 후보 = 서로 이어지는 폭 8~24 구간의 사슬
    chains = []
    for y in sorted(runs, reverse=True):
        for cx, w in runs[y]:
            placed = False
            for ch in chains:
                if abs(ch[-1][1] - cx) < 5 and ch[-1][0] - y <= 2:
                    ch.append((y, cx)); placed = True; break
            if not placed:
                chains.append([(y, cx)])
    cand = [c for c in chains if len(c) >= 25]
    if len(cand) < 2:
        return None
    cand = sorted(cand, key=lambda c: -len(c))[:2]
    cand = sorted(cand, key=lambda c: np.mean([x for _, x in c]))
    out = []
    for c in cand:
        low = c[:6]  # 아래 끝 6행
        out.append((float(np.mean([x for _, x in low]) + x0), float(np.mean([y for y, _ in low]) + y0)))
    return out


def main() -> None:
    folder = Path(sys.argv[1])
    rows_mm = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    l1 = L1Map()
    recs = [json.loads(l) for l in (folder / "sweep.jsonl").read_text().splitlines()]
    defs = {"A 아래 띠": tips_A, "B 주축": tips_B, "C 안쪽 모서리": tips_C, "D 원통 추적": tips_D}
    table = {k: [] for k in defs}
    for r in recs:
        for k, fn in defs.items():
            vals = []
            for fr in r["frames"]:
                img = cv2.imread(str(folder / "frames" / f"{fr}.color.png"))
                h = marker_h(img)
                if h is None:
                    continue
                H, board = h
                j1, j5 = r["commanded"][0], r["commanded"][4]
                c = np.array(l1.forward(j1, rows_mm[5], j5)) + board  # 블록이 그 자리에 있을 때의 예측 중심
                t = fn(img, c)
                if t is None:
                    continue
                a, b = tf(H, t)
                vals.append((math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])), float(np.hypot(*(b - a)))))
            table[k].append((r["pose"], r["read"], vals))
    for k, rowsk in table.items():
        print(f"== 정의 {k}")
        det = sum(1 for _, _, v in rowsk if len(v) == 2)
        print(f"  검출(자세당 2장 모두) {det}/{len(rowsk)}")
        ref = [np.mean([a for a, _ in v]) for p, _, v in rowsk if p.startswith("ref") and v]
        if len(ref) >= 2:
            print(f"  기준 재방문 3번 보정 각도: {np.round(ref, 2).tolist()} 표준편차 {np.std(ref):.2f}°")
        within = [abs(v[0][0] - v[1][0]) for _, _, v in rowsk if len(v) == 2]
        if within:
            print(f"  자세 내 두 장 차이(평균 절대) {np.mean(within):.2f}°")
        for key, lo_hi in (("j5", None), ("j1", None)):
            pts = []
            for p, rd, v in rowsk:
                if not v or rd is None:
                    continue
                if key == "j5" and (p.startswith("j5-") or p.startswith("ref")):
                    pts.append((rd[4], np.mean([a for a, _ in v])))
                if key == "j1" and (p.startswith("j1") or p.startswith("ref")):
                    pts.append((rd[0], np.mean([a for a, _ in v])))
            if len(pts) >= 4:
                x = np.array([q[0] for q in pts]); y = np.array([q[1] for q in pts])
                kk, aa = np.polyfit(x, y, 1)
                res = y - (aa + kk * x)
                print(f"  {key.upper()} 민감도 {kk:+.3f} 보정°/읽은{key.upper()}° (n {len(pts)}), 선형 잔차 표준편차 {res.std():.2f}°, 최대 {abs(res).max():.2f}°")
        gaps = [np.mean([g for _, g in v]) for _, _, v in rowsk if v]
        print(f"  손끝 간격(보정 단위) 평균 {np.mean(gaps):.2f} 표준편차 {np.std(gaps):.2f}")


if __name__ == "__main__":
    main()
