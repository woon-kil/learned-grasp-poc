#!/usr/bin/env python3
"""P1 목표 자리 안내 SVG를 최근 빈 보드 사진 위에 그린다(개인 사진이 들어가므로 data/p1/에 둔다). 장치 없음.

목표 블록 외곽(윗면 60.33×29.89 mm, 긴 변을 그 자리의 반경 방향으로), 중심 십자, 반경선, 안내 문구.
사용: uv run python scripts/p1_target_svg.py p1-01 data/p1/blank.png data/p1/p1-01.svg
"""
import base64
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.l1 import L1Map

SCALE_TOP = 1.14  # 화소/mm, 블록 윗면 높이(docs 스케일 정정)


def main() -> None:
    tid, photo, out = sys.argv[1:4]
    targets = {t["id"]: t for t in json.loads(Path("config/p1_targets.json").read_text())["targets"]}
    t = targets[tid]
    img = cv2.imread(photo)
    from learned_grasp.l1 import block_from_image  # noqa: F401 (보드 마커 중심은 아래서 직접 검출)
    from cv2 import aruco
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img[:, :, 2])
    board = next(c.reshape(4, 2).mean(0) for c, i in zip(corners, ids.ravel()) if int(i) == 1)
    cx, cy = board[0] + t["du"], board[1] + t["dv"]
    ang = math.radians(t["radial_angle_deg"])
    ux, uy = math.cos(ang), math.sin(ang)
    L, W = 60.33 * SCALE_TOP / 2, 29.89 * SCALE_TOP / 2
    pts = []
    for sl, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        pts.append((cx + sl * L * ux - sw * W * uy, cy + sl * L * uy + sw * W * ux))
    x0, y0, x1, y1 = 440, 80, 960, 440
    crop = img[y0:y1, x0:x1]
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode()
    s = 2.0
    def P(x, y): return ((x - x0) * s, (y - y0) * s)
    poly = " ".join("%.1f,%.1f" % P(*p) for p in pts)
    c = P(cx, cy)
    a0, a1 = P(cx - 1.4 * L * ux, cy - 1.4 * L * uy), P(cx + 1.4 * L * ux, cy + 1.4 * L * uy)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {int((x1-x0)*s)} {int((y1-y0)*s)+330}" font-family="Apple SD Gothic Neo, Noto Sans KR, sans-serif">',
           f'<rect width="100%" height="100%" fill="#fafafa"/><image href="data:image/jpeg;base64,{b64}" x="0" y="0" width="{int((x1-x0)*s)}" height="{int((y1-y0)*s)}"/>',
           f'<line x1="{a0[0]:.1f}" y1="{a0[1]:.1f}" x2="{a1[0]:.1f}" y2="{a1[1]:.1f}" stroke="#1565c0" stroke-width="3" stroke-dasharray="10 6"/>',
           f'<polygon points="{poly}" fill="#ffeb3b" fill-opacity="0.35" stroke="#d81b60" stroke-width="4"/>',
           f'<line x1="{c[0]-14:.1f}" y1="{c[1]:.1f}" x2="{c[0]+14:.1f}" y2="{c[1]:.1f}" stroke="#d81b60" stroke-width="3"/><line x1="{c[0]:.1f}" y1="{c[1]-14:.1f}" x2="{c[0]:.1f}" y2="{c[1]+14:.1f}" stroke="#d81b60" stroke-width="3"/>']
    base = int((y1 - y0) * s) + 36
    lines = [(f"P1 {tid}: 블록을 분홍 사각형 자리에 놓는다 (사진 방향: 위 = 베이스, 아래 = 사용자 쪽)", "#000", 24, True),
             ("블록: 노란 블록을 눕혀 놓는다(마커 면은 위가 아님, 보드 마커·바깥 4X4 마커를 가리지 않게). 긴 변을 파란 점선(반경선)에 맞춘다.", "#000", 20, False),
             ("자리 정확도는 ±5 mm면 충분하다(시스템이 놓인 자리를 직접 잰다). 방향은 파란 점선과 5° 안이어야 한다(블록 긴 변의 양끝이 점선에서 5 mm 이내).", "#000", 20, False),
             ("놓은 뒤 「놓았다」라고 알린다. Claude가 Gemini 사진으로 자리·방향을 확인한다. 방향이 5°를 넘으면 다시 놓게 한다(최대 2회).", "#a00", 20, False),
             ("팔이 hover에서 멈추면 「하강 전 확인」을 묻는다. 손가락이 블록 위로 내려앉을 것 같으면 「멈춰」, 이상이면 전원 차단(손·자를 빼고).", "#a00", 20, False),
             ("시도가 끝나면 팔이 park로 돌아온 뒤에만 블록을 치운다. 팔이 움직이는 동안과 hover 대기 중에는 손을 넣지 않는다.", "#a00", 20, False)]
    for i, (txt, col, sz, bold) in enumerate(lines):
        svg.append(f'<text x="14" y="{base + 44*i + 24}" font-size="{sz}" fill="{col}"{" font-weight=\"bold\"" if bold else ""}>{txt}</text>')
    svg.append("</svg>")
    Path(out).write_text("\n".join(svg))
    print(f"{tid}: J1 {t['j1']} r {t['r_mm']} 중심 ({cx:.1f},{cy:.1f}) 반경각 {t['radial_angle_deg']}° -> {out}")


if __name__ == "__main__":
    main()
