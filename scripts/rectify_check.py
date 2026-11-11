#!/usr/bin/env python3
"""마커 호모그래피로 영상을 보정(평면 직교화)했을 때 J5 민감도와 J1당 반경선 방향 변화가 1에 가까운지 본다. 장치 없음, 쓰기 0.
보드 마커 id 1의 네 모서리로 호모그래피를 만들어 블록 윗면 점과 반경선 두 점의 각도를 보정 좌표에서 잰다(평행 평면은 같은 원형점을 공유한다).
사용: uv run python scripts/rectify_check.py runs/G0b/block-map.json
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, signed_diff
from learned_grasp.pose_table import generate


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


def ang(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def main() -> None:
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    l1 = L1Map()
    recs = [r for r in json.loads(Path(sys.argv[1]).read_text()) if "du" in r]
    raw, rect, j5s, used = [], [], [], 0
    for r in recs:
        img = cv2.imread(r["file"])
        if img is None:
            continue
        h = marker_h(img)
        if h is None:
            continue
        H, board = h
        hsv = cv2.cvtColor(img[100:470, 480:960], cv2.COLOR_BGR2HSV)
        m = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 200) & (hsv[:, :, 2] > 200)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        ys, xs = np.nonzero(lab == i)
        pts = np.c_[xs + 480, ys + 100].astype(np.int32)
        hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)
        (_, _), (w, hh), a = cv2.minAreaRect(hull)
        long_raw = (a if w >= hh else a + 90) % 180.0
        (_, _), (w, hh), a = cv2.minAreaRect(tf(H, hull))
        long_rect = (a if w >= hh else a + 90) % 180.0
        rr = rows[r["r_index"]]
        pa = np.array(l1.forward(r["j1"], rr + 8, 90)) + board
        pb = np.array(l1.forward(r["j1"], rr - 8, 90)) + board
        rad_raw = ang(pb, pa) % 180.0
        rad_rect = ang(*tf(H, [pb, pa])) % 180.0
        raw.append(signed_diff(long_raw, rad_raw))
        rect.append(signed_diff(long_rect, rad_rect))
        j5s.append(r["j5"])
        used += 1
    j5 = np.array(j5s)
    print(f"사진 {used}장(마커 id 1 검출)")
    for name, d in (("영상 각도", np.array(raw)), ("보정 각도(마커 호모그래피)", np.array(rect))):
        k, a = np.polyfit(j5 - 90, d, 1)
        res = d - (a + k * (j5 - 90))
        print(f"  {name}: d = {a:+.1f}° {k:+.3f}·(J5−90), 잔차 표준편차 {res.std():.1f}°")
    # J1당 반경선 방향 변화(마커 호모그래피 하나, 첫 사진 기준)
    first = next(h for h in (marker_h(cv2.imread(r["file"])) for r in recs if cv2.imread(r["file"]) is not None) if h)
    H, board = first
    print("J1당 반경선 방향 변화(영상 각도 대 보정 각도, J1±3°):")
    for j1, ri in ((80, 3), (90, 5), (100, 4)):
        rr = rows[ri]
        out = []
        for jj in (j1 - 3, j1 + 3):
            pa = np.array(l1.forward(jj, rr + 8, 90)) + board
            pb = np.array(l1.forward(jj, rr - 8, 90)) + board
            out.append((ang(pb, pa) % 180.0, ang(*tf(H, [pb, pa])) % 180.0))
        print(f"  J1 {j1} 행 {ri}: 영상 {out[0][0] - out[1][0]:+.1f}° / 6° J1 = {(out[0][0] - out[1][0]) / 6:+.2f}°/°, 보정 {out[0][1] - out[1][1]:+.1f}° / 6° = {(out[0][1] - out[1][1]) / 6:+.2f}°/°")


if __name__ == "__main__":
    main()
