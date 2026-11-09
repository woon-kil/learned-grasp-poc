"""상시 카메라 서버의 순수 계산. ROS 없이 시험한다."""

import numpy as np


def change_fraction(before, after, roi, threshold=30):
    """ROI 안에서 밝기 차이가 threshold를 넘는 화소 비율. roi는 (u0, u1, v0, v1)이다."""
    u0, u1, v0, v1 = roi
    first = before[v0:v1, u0:u1].astype(np.int16).mean(axis=2)
    second = after[v0:v1, u0:u1].astype(np.int16).mean(axis=2)
    return float((np.abs(first - second) > threshold).mean())


def yellow_pixels(image, roi, min_sat=0.62):
    """ROI 안의 노란 블록 색 화소 수. image는 RGB이고 roi는 (u0, u1, v0, v1)이다."""
    u0, u1, v0, v1 = roi
    r, g, b = np.moveaxis(image[v0:v1, u0:u1].astype(np.float32), -1, 0)
    return int(((r > 140) & (g > 120) & (np.abs(r - g) < 60) & ((r - b) / np.maximum(r, 1) > min_sat)).sum())
