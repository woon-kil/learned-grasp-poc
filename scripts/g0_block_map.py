#!/usr/bin/env python3
"""G0b·G0c·G0b-9의 「집기 전」 Gemini 사진(팔이 셀의 명령 자세로 놓은 블록)에서 블록의 영상 위치를 구한다. 장치 없음.

블록 위치 = 노란 영역(HSV H20~35, S>200, V>200, 탐색 영역 x 480~960, y 100~470. S·V를 높여 탁자 나뭇결을 제외한다)의 가장 큰 성분의 볼록 껍질 무게중심.
기준 = 보드 마커 id 1(36h11, 빨강 채널) 중심. 출력 위치는 기준 대비 (du, dv) 화소.
사용: uv run python scripts/g0_block_map.py <출력 json>
"""
import glob
import json
import re
import sys

import cv2
import numpy as np
from cv2 import aruco

CELLS = {0: (0, 90, 90), 1: (4, 110, 120), 2: (9, 110, 90), 3: (1, 70, 60), 4: (4, 110, 60), 5: (8, 70, 120), 6: (2, 110, 60), 7: (5, 110, 60), 8: (6, 70, 120), 9: (2, 70, 120)}  # runs/G0b·G0c replay 기록
ROI = (480, 100, 960, 470)


def board_center(img):
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img[:, :, 2])
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == 1:
            return c.reshape(4, 2).mean(0)
    return None


def block_center(img):
    x0, y0, x1, y1 = ROI
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] > 200) & (hsv[:, :, 2] > 200)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None, 0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == i)
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    m = cv2.moments(hull)
    return np.array([m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0]), int(stats[i, cv2.CC_STAT_AREA])


def main() -> None:
    rows = []
    items = []
    for src in ("G0b", "G0c"):
        for path in sorted(glob.glob(f"data/g0frames/{src}-before/c??-t?.color.png")):
            m = re.search(r"c(\d\d)-t(\d)\.color", path)
            cell = int(m.group(1))
            r, j1, j5 = CELLS[cell]
            items.append((path, src, f"c{cell:02d}", r, j1, j5, int(m.group(2))))
    for path in sorted(glob.glob("runs/G0b-9/frames/cal5-??-r*-j*-p*.color.png")):
        if "after" in path:
            continue
        m = re.search(r"cal5-(\d\d)-r(\d+)-j(\d+)-p(\d+)", path)
        items.append((path, "G0b-9", f"cal{m.group(1)}", int(m.group(2)), int(m.group(3)), int(m.group(4)), 0))
    boards = {}
    detected = []
    for path, src, cell, r, j1, j5, trial in items:
        img = cv2.imread(path)
        board, block = board_center(img), block_center(img)
        detected.append((path, src, cell, r, j1, j5, trial, board, block))
        if board is not None:
            boards.setdefault(src, []).append(board)
    median = {s: np.median(np.array(b), axis=0) for s, b in boards.items()}  # 세션별 보드 마커 중앙값(마커가 블록에 가려진 사진의 보충)
    for path, src, cell, r, j1, j5, trial, board, block in detected:
        entry = {"src": src, "cell": cell, "r_index": r, "j1": j1, "j5": j5, "trial": trial, "file": path, "board_fallback": board is None}
        ref = board if board is not None else median.get(src)
        if ref is not None and block[0] is not None:
            entry |= {"board": np.asarray(ref).round(2).tolist(), "block": block[0].round(2).tolist(), "area": block[1], "du": round(float(block[0][0] - ref[0]), 2), "dv": round(float(block[0][1] - ref[1]), 2)}
        rows.append(entry)
    print("세션별 보드 마커 중앙값", {s: m.round(2).tolist() for s, m in median.items()}, "표준편차", {s: np.array(b).std(0).round(2).tolist() for s, b in boards.items()})
    json.dump(rows, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
    print(f"{len(rows)}장, 검출 {sum('du' in r for r in rows)}장")


if __name__ == "__main__":
    main()
