"""열린 손 hover 사진에서 손끝 두 점의 축 각도를 잰다(Gemini 고정 카메라 영상, BGR).

손가락 끝 두 점(검은 손가락의 가장 아래 띠)을 이어 닫힘 축의 영상 각도를 낸다. 블록 긴 변은 닫힘 축에 수직이어야 하므로
어긋남 ψ_pick = 블록 긴 변 각도 − (손끝 축 각도 + 90°)(−90~90°로 감김)이다. 상수 편향(카메라 기하)은 Δ 사이 차이에서 빠지므로 차이 측정에 쓴다.
H2-1의 같은 자세 hover 30장에서 30/30 검출, 각도 표준편차 0.02°(정지 영상 재현성, 팔 재방문 변동은 별개).
"""

from __future__ import annotations

import math

import numpy as np

ROI = (630, 150, 790, 215)  # x0, y0, x1, y1: hover 자세의 손가락 끝이 들어오는 영역(영상 좌표)
DARK = 90
BAND = 8
SEARCH = 30  # 손가락을 가르는 데 쓰는 아래쪽 띠 높이(화소)
MIN_GAP = 20


GAP_PX = (60.0, 100.0)  # 두 손끝 간격(H2-1 hover 78~80 화소, X1 chain-003 hover 78)
HEIGHT_DIFF_PX = 30.0


def roi_for_block(center_uv: tuple[float, float]) -> tuple[int, int, int, int]:
    """park 사진의 블록 중심 아래쪽이 아니라 위쪽(베이스 쪽)에 손끝이 온다. hover의 손끝 중점은 블록 중심에서 영상 위쪽으로 약 36 화소(X1 chain-003 사진)."""
    cx, cy = center_uv
    return (int(cx - 95), int(cy - 62), int(cx + 95), int(cy - 8))


def tip_axis_deg(image_bgr: np.ndarray, roi: tuple[int, int, int, int] = ROI) -> dict | None:
    """손끝 두 점과 축 각도(도), 간격(화소). 못 찾거나 간격·높이가 손끝답지 않으면 None."""
    import cv2

    x0, y0, x1, y1 = roi
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    ys, xs = np.nonzero(gray < DARK)
    if len(ys) == 0:
        return None
    low = ys > ys.max() - SEARCH  # 손 몸통을 빼고 손가락 아래 부분만으로 두 손가락을 가른다
    lx, ly = xs[low], ys[low]
    order = np.argsort(lx)
    sx = lx[order]
    gaps = np.diff(sx)
    if len(gaps) == 0:
        return None
    k = int(np.argmax(gaps))
    if gaps[k] < MIN_GAP:
        return None
    split = (sx[k] + sx[k + 1]) / 2.0
    tips = []
    for side in (lx < split, lx >= split):  # 손가락마다 자기 아래 끝 띠를 따로 잡는다(한쪽이 더 낮아도 각도가 줄지 않게)
        sy, sxx = ly[side], lx[side]
        band = sy > sy.max() - BAND
        tips.append((float(sxx[band].mean() + x0), float(sy[band].mean() + y0)))
    left, right = tips
    gap = right[0] - left[0]
    if not (GAP_PX[0] <= gap <= GAP_PX[1]) or abs(right[1] - left[1]) > HEIGHT_DIFF_PX:
        return None
    return {"left": left, "right": right, "axis_deg": math.degrees(math.atan2(right[1] - left[1], right[0] - left[0])), "gap_px": right[0] - left[0]}


def misalignment_deg(block_angle_deg: float, axis_deg: float) -> float:
    """블록 긴 변 각도 − (손끝 축 + 90°), (−90, 90]로 감는다."""
    return ((block_angle_deg - (axis_deg + 90.0) + 90.0) % 180.0) - 90.0
