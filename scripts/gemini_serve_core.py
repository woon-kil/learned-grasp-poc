"""상시 카메라 서버의 순수 계산. ROS 없이 시험한다."""

import numpy as np


def change_fraction(before, after, roi, threshold=30):
    """ROI 안에서 밝기 차이가 threshold를 넘는 화소 비율. roi는 (u0, u1, v0, v1)이다."""
    u0, u1, v0, v1 = roi
    first = before[v0:v1, u0:u1].astype(np.int16).mean(axis=2)
    second = after[v0:v1, u0:u1].astype(np.int16).mean(axis=2)
    return float((np.abs(first - second) > threshold).mean())
