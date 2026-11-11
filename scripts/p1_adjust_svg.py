#!/usr/bin/env python3
"""P1 방향 조정 안내 SVG: 놓은 블록 사진 위에 지금 방향(빨강)과 맞출 반경선(파랑)을 그린다. 장치 없음.
사용: uv run python scripts/p1_adjust_svg.py <놓은 사진> <측정 du> <측정 dv> <블록각> <반경각> <조정 n/2> <출력.svg>
"""
import base64
import math
import sys
from pathlib import Path

import cv2
from cv2 import aruco

photo, du, dv, ang, rad, n, out = sys.argv[1:8]
du, dv, ang, rad = float(du), float(dv), float(ang), float(rad)
img = cv2.imread(photo)
blank = cv2.imread("data/p1/blank.png")
params = aruco.DetectorParameters()
params.errorCorrectionRate = 1.0
det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), params)
c, i, _ = det.detectMarkers(blank[:, :, 2])
board = [cc.reshape(4, 2).mean(0) for cc, ii in zip(c, i.flatten()) if int(ii) == 1][0]
x0, y0, x1, y1, s = 440, 80, 960, 440, 2.0
ok, buf = cv2.imencode(".jpg", img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 88])
b64 = base64.b64encode(buf).decode()
cx, cy = board[0] + du, board[1] + dv
P = lambda x, y: ((x - x0) * s, (y - y0) * s)
def line(angle, color, dash, L=75):
    a = math.radians(angle)
    p0, p1 = P(cx - L * math.cos(a), cy - L * math.sin(a)), P(cx + L * math.cos(a), cy + L * math.sin(a))
    return f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{color}" stroke-width="4" {dash}/>'
W, H = int((x1 - x0) * s), int((y1 - y0) * s)
T = lambda y, t, sz=20, fill="#000", bold=False: f'<text x="14" y="{y}" font-size="{sz}" fill="{fill}"{" font-weight=\"bold\"" if bold else ""}>{t}</text>'
diff = ang - rad
move = "위(베이스 쪽) 끝을 왼쪽으로, 아래 끝을 오른쪽으로" if diff > 0 else "위(베이스 쪽) 끝을 오른쪽으로, 아래 끝을 왼쪽으로"
mm = abs(diff) / 2 * math.pi / 180 * 30.2
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H + 230}" font-family="Apple SD Gothic Neo, Noto Sans KR, sans-serif"><rect width="100%" height="100%" fill="#fafafa"/>',
       f'<image href="data:image/jpeg;base64,{b64}" x="0" y="0" width="{W}" height="{H}"/>', line(ang, "#d00", ""), line(rad, "#1565c0", 'stroke-dasharray="10 6"'),
       T(H + 34, f"방향 조정 {n}: 블록이 반경선보다 {abs(diff):.1f}° 틀어졌다 (사진 방향: 위 = 베이스, 아래 = 사용자 쪽)", 22, "#000", True),
       T(H + 72, f"빨간 실선 = 지금 블록 긴 변(측정 {ang:.1f}°), 파란 점선 = 맞출 반경선({rad:.1f}°). 허용 5°.", 20),
       T(H + 108, f"중심은 그대로 두고 {move} 각 약 {mm:.0f} mm 돌린다.", 20, "#a00"),
       T(H + 144, "팔이 park인 상태에서만 만진다. 다시 놓으면 「놓았다」.", 20), "</svg>"]
Path(out).write_text("\n".join(svg))
print(out)
