#!/usr/bin/env python3
"""G0d 윗면 임시 마커 인쇄물을 만든다. A4, 300 dpi, 100% 크기로 인쇄한다.

사용: uv run --with opencv-contrib-python-headless --with pillow python scripts/make_top_marker.py
"""
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

DPI = 300
ID = 7
SIDE_MM = 25.0
OUT = Path("results/G0d/top-marker-id7-25mm.png")


def mm(value: float) -> int:
    return round(value / 25.4 * DPI)


def main() -> None:
    sheet = Image.new("L", (mm(210), mm(297)), 255)
    draw = ImageDraw.Draw(sheet)
    tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), ID, mm(SIDE_MM))
    tag_image = Image.fromarray(tag)
    font = ImageFont.load_default(size=36)
    for index in range(4):
        x, y = mm(25 + index * 42), mm(25)
        sheet.paste(tag_image, (x, y))
        draw.text((x, y + mm(SIDE_MM) + 12), f"id {ID}", fill=0, font=font)
    x0, y0 = mm(25), mm(70)
    draw.line((x0, y0, x0 + mm(50), y0), fill=0, width=4)
    for tick in (0, 50):
        draw.line((x0 + mm(tick), y0 - 18, x0 + mm(tick), y0 + 18), fill=0, width=4)
    draw.text((x0, y0 + 26), "50 mm: check print scale with a ruler", fill=0, font=font)
    cx, cy, length = mm(105), mm(180), mm(70)
    for angle, label in ((0, "0"), (10, "+10"), (-10, "-10")):
        rad = math.radians(angle - 90)
        tip = (cx + length * math.cos(rad), cy + length * math.sin(rad))
        draw.line((cx, cy, *tip), fill=0, width=3)
        draw.text((tip[0] - 20, tip[1] - 50), label, fill=0, font=font)
    draw.text((mm(25), mm(195)), "angle guide", fill=0, font=font)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, dpi=(DPI, DPI))
    print(OUT, sheet.size)


if __name__ == "__main__":
    main()
