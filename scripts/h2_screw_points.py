#!/usr/bin/env python3
"""H2-1: DaBai 브래킷의 나사 머리 두 점(밝은 점)을 Gemini 컬러 영상에서 찾아 위치와 잡음을 낸다. 장치 없음.

절차(실행 전 고정): 회색조 → DoG(σ1 − σ4) → 임계(최대의 0.6) 연결 성분 → 면적 ≥ 3 → 면적이 큰 두 성분 중
행 차이 ≤ 4 화소, 간격 10~25 화소인 쌍 → 각 점은 DoG 양수 가중 무게중심(하위 화소).
각 장에서 보드 마커 id 1(36h11, 빨강 채널, 기본 파라미터)의 중심도 기록해 카메라가 움직이지 않았는지 본다.
영상 가장자리까지의 여백(화소)을 함께 낸다.
자세와 무관한 넓은 탐색(pose=wide, H3용): 탐색 영역 x 560~1000(왼쪽 위 흰 전원 어댑터를 피함), y 0~430, DoG 임계 = 최대의 0.45,
성분 면적 6~70화소, 성분 중심 5×5의 최대 밝기 ≥ 190 이고 평균 채도 ≤ 45(흰 나사 머리), 두 성분의 면적비 ≤ 2.5, 간격 11~26화소, 행 차 ≤ 0.25×간격+1.5, 그리고 「어두운 판 위」 조건:
각 성분 둘레 반지름 8~14화소 고리의 평균 밝기 < 120 이고 두 점 아래쪽(간격×2) 영역 평균 < 110. 후보가 여러 쌍이면 판 조건 점수가 가장 좋은 쌍.
사용: uv run python scripts/h2_screw_points.py <hover|grasp|wide> <png...>
"""
import json
import sys

import cv2
import numpy as np
from cv2 import aruco

ROI = {"hover": (700, 0, 780, 70), "grasp": (690, 70, 770, 140), "wide": (560, 0, 1000, 430)}  # x0, y0, x1, y1 (H2-0 사진 기준 대략 위치)


def board_marker_center(path: str):
    img = cv2.imread(path)
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(img[:, :, 2])
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == 1:
            return c.reshape(4, 2).mean(0).tolist()
    return None


def _ring_mean(g, cx, cy, r0=8, r1=14):
    h, w = g.shape
    yy, xx = np.ogrid[:h, :w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = (d2 >= r0 * r0) & (d2 <= r1 * r1)
    return float(g[m].mean()) if m.any() else 255.0


def find_pair_wide(path: str):
    x0, y0, x1, y1 = ROI["wide"]
    bgr = cv2.imread(path)
    full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g = full[y0:y1, x0:x1].astype(np.float32)
    d = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 4.0)
    mask = (d > 0.45 * d.max()).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    comps = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not 6 <= area <= 70:
            continue
        ys, xs = np.nonzero(lab == i)
        w = np.clip(d[ys, xs], 0, None)
        cx, cy = float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())
        px, py = int(round(cx + x0)), int(round(cy + y0))
        patch_g = full[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3]
        patch_s = hsv[max(py - 2, 0):py + 3, max(px - 2, 0):px + 3, 1]
        if float(patch_g.max()) < 190 or float(patch_s.mean()) > 45:  # 나사 머리는 흰색(밝고 채도 낮음). 분홍 LED·유리 반사와 구분
            continue
        comps.append((area, cx, cy, _ring_mean(g, cx, cy)))
    best, best_score = None, None
    for a in range(len(comps)):
        for b in range(a + 1, len(comps)):
            A, B = comps[a], comps[b]
            gap = abs(A[1] - B[1])
            if not 11 <= gap <= 26 or abs(A[2] - B[2]) > 0.25 * gap + 1.5 or max(A[0], B[0]) / min(A[0], B[0]) > 2.5:
                continue
            if max(A[3], B[3]) >= 120:
                continue
            mx, my, below = (A[1] + B[1]) / 2, (A[2] + B[2]) / 2, None
            ya, yb = int(my + 4), int(min(my + 2 * gap, g.shape[0]))
            xa, xb = int(max(mx - gap, 0)), int(min(mx + gap, g.shape[1]))
            below = float(g[ya:yb, xa:xb].mean()) if yb > ya else 255.0
            if below >= 110:
                continue
            score = max(A[3], B[3]) + below
            if best_score is None or score < best_score:
                best_score, best = score, ((A[0], A[1] + x0, A[2] + y0), (B[0], B[1] + x0, B[2] + y0))
    if best is None:
        return None
    return best if best[0][1] < best[1][1] else (best[1], best[0])


def find_pair(path: str, pose: str):
    if pose == "wide":
        return find_pair_wide(path)
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
    rows, miss, boards = [], 0, []
    for path in paths:
        board = board_marker_center(path)
        if board is not None:
            boards.append(board)
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
        h, w = cv2.imread(paths[0]).shape[:2]
        margin = float(min(a[:, 0].min(), a[:, 1].min(), w - a[:, 2].max(), h - a[:, 3].max()))
        out |= {"edge_margin_px": round(margin, 1)}
        if boards:
            b = np.array(boards)
            out |= {"board_id1_frames": len(boards), "board_id1_mean": b.mean(0).round(2).tolist(), "board_id1_std": b.std(0, ddof=1).round(3).tolist() if len(b) > 1 else None}
        out |= {"mid_mean": mid.mean(0).round(2).tolist(), "mid_std": mid.std(0, ddof=1).round(3).tolist() if len(rows) > 1 else None,
                "spacing_mean": float((a[:, 2] - a[:, 0]).mean().round(2)), "left": a[:, :2].mean(0).round(2).tolist(), "right": a[:, 2:].mean(0).round(2).tolist()}
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
