#!/usr/bin/env python3
"""H2-1 손 마커(DaBai 브래킷) 인쇄물을 만든다. A4, 300 dpi, 100% 크기로 인쇄한다.

AprilTag 36h11 id 9. 검은 정사각형 한 변 15 mm(주)와 20 mm(예비). 흰 여백은 한 변의 1/6.
사용: uv run --with pillow python scripts/make_hand_marker.py
"""
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

DPI = 300
ID = 9
OUT = Path("results/H2/hand-marker-id9.png")


def mm(value: float) -> int:
    return round(value / 25.4 * DPI)


def main() -> None:
    sheet = Image.new("L", (mm(210), mm(297)), 255)
    draw = ImageDraw.Draw(sheet)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    y = 20.0
    for side, copies in ((15.0, 3), (20.0, 2)):
        pad = side / 6.0
        total = side + 2 * pad
        for i in range(copies):
            x = 20.0 + i * (total + 8.0)
            tag = Image.fromarray(cv2.aruco.generateImageMarker(dictionary, ID, mm(side)))
            sheet.paste(tag, (mm(x + pad), mm(y + pad)))
            draw.rectangle([mm(x), mm(y), mm(x + total), mm(y + total)], outline=170, width=2)  # 자르는 선
        draw.text((mm(20), mm(y + total + 3)), f"id {ID}  검은 정사각형 {side:g} mm  스티커 {total:.1f} mm", fill=0)
        y += total + 14.0
    # 인쇄 배율 확인용 50 mm 눈금
    draw.line([mm(20), mm(y + 5), mm(70), mm(y + 5)], fill=0, width=4)
    draw.text((mm(20), mm(y + 9)), "이 선이 자로 50 mm여야 한다(100% 인쇄)", fill=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, dpi=(DPI, DPI))
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()
