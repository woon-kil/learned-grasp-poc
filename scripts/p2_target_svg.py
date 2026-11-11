#!/usr/bin/env python3
"""P2 목표 자리 안내 SVG. 장치 없음. 사용: uv run python scripts/p2_target_svg.py p2-01 data/p2/blank.png data/p2/p2-01.svg

분홍 사각형(블록 윗면 60.33×29.89 mm)을 목표 방향 그대로 그린다. 사람은 블록을 이 사각형 쪽으로 **대충** 놓는다(방향 ±10° 안, 정렬 안 해도 된다).
"""
import base64
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

SCALE_TOP = 1.14


def main() -> None:
    tid, photo, out = sys.argv[1:4]
    t = {x["id"]: x for x in json.loads(Path("config/p2_targets.json").read_text())["targets"]}[tid]
    img = cv2.imread(photo)
    params = aruco.DetectorParameters()
    params.errorCorrectionRate = 1.0
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), params)
    corners, ids, _ = det.detectMarkers(img[:, :, 2])
    board = next((c.reshape(4, 2).mean(0) for c, i in zip(corners, [] if ids is None else ids.ravel()) if int(i) == 1), np.array([691.25, 291.6]))
    cx, cy = board[0] + t["du"], board[1] + t["dv"]
    ang = math.radians(t["block_angle_deg"])
    ux, uy = math.cos(ang), math.sin(ang)
    L, W = 60.33 * SCALE_TOP / 2, 29.89 * SCALE_TOP / 2
    pts = [(cx + sl * L * ux - sw * W * uy, cy + sl * L * uy + sw * W * ux) for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    x0, y0, x1, y1 = 440, 80, 960, 440
    ok, buf = cv2.imencode(".jpg", img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode()
    s = 2.0
    P = lambda x, y: ((x - x0) * s, (y - y0) * s)
    poly = " ".join("%.1f,%.1f" % P(*p) for p in pts)
    c = P(cx, cy)
    W_, H_ = int((x1 - x0) * s), int((y1 - y0) * s)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_ + 190}" font-family="Apple SD Gothic Neo, Noto Sans KR, sans-serif">',
           f'<rect width="100%" height="100%" fill="#fafafa"/><image href="data:image/jpeg;base64,{b64}" x="0" y="0" width="{W_}" height="{H_}"/>',
           f'<polygon points="{poly}" fill="#ffeb3b" fill-opacity="0.35" stroke="#d81b60" stroke-width="4"/>',
           f'<line x1="{c[0]-14:.1f}" y1="{c[1]:.1f}" x2="{c[0]+14:.1f}" y2="{c[1]:.1f}" stroke="#d81b60" stroke-width="3"/><line x1="{c[0]:.1f}" y1="{c[1]-14:.1f}" x2="{c[0]:.1f}" y2="{c[1]+14:.1f}" stroke="#d81b60" stroke-width="3"/>']
    lean = 90.0 - t["block_angle_deg"]  # 수직 기준 기울기: 양수면 위 끝이 왼쪽으로 기운다(영상 각도 < 90)
    lean_txt = f"수직에서 {abs(lean):.0f}° 기울임: 블록 위(베이스 쪽) 끝이 " + ("왼쪽" if lean > 0 else "오른쪽") + "으로"
    base = H_ + 34
    for i, (txt, col, sz) in enumerate((
            (f"{tid}: 블록을 분홍 사각형 쪽에 눕혀 놓는다 (위 = 베이스, 아래 = 사용자 쪽)", "#000", 24),
            (("자리 ±10 mm, 방향 ±5°(범위 끝 목표: 정확히). 정렬 왕복은 하지 않는다." if (t["target_j5"] <= 72 or t["target_j5"] >= 108) else "자리 ±10 mm, 방향 ±10°면 충분하다. 정렬하지 않는다.") + " 보드 마커 id 1·바깥 4X4 마커를 가리지 않게.", "#000", 20),
            (f"방향 안내: {lean_txt}.", "#000", 22),
            ("놓았으면 「놓았다」. 팔이 움직이는 동안 손을 넣지 않는다. 시도 뒤에는 블록을 집어 다음 자리에 바로 놓는다.", "#a00", 20))):
        svg.append(f'<text x="14" y="{base + 40 * i + 24}" font-size="{sz}" fill="{col}">{txt}</text>')
    svg.append("</svg>")
    Path(out).write_text("\n".join(svg))
    print(f"{tid}: J1 {t['j1']} r {t['r_mm']} 목표 J5 {t['target_j5']} d {t['target_d_deg']:+.0f}° 중심 ({cx:.1f},{cy:.1f}) -> {out}")


if __name__ == "__main__":
    main()
