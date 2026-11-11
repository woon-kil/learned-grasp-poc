from __future__ import annotations

import cv2
import numpy as np

from learned_grasp.hand_tips import misalignment_deg, tip_axis_deg


def _image(left_y=205, right_y=205):
    img = np.full((720, 1280, 3), 200, np.uint8)
    cv2.rectangle(img, (665, 170), (680, left_y), (20, 20, 20), -1)  # 왼 손가락
    cv2.rectangle(img, (745, 170), (760, right_y), (20, 20, 20), -1)  # 오른 손가락
    return img


def test_tip_axis_horizontal_and_tilted():
    flat = tip_axis_deg(_image())
    assert abs(flat["axis_deg"]) < 0.5 and abs(flat["gap_px"] - 80.0) < 2.0
    tilted = tip_axis_deg(_image(left_y=205, right_y=209))  # 오른쪽 끝이 4화소 아래 → 약 +2.9°
    assert 2.0 < tilted["axis_deg"] < 4.0


def test_tip_axis_returns_none_without_fingers():
    assert tip_axis_deg(np.full((720, 1280, 3), 200, np.uint8)) is None


def test_misalignment_wraps_and_is_zero_when_perpendicular():
    assert abs(misalignment_deg(90.0, 0.0)) < 1e-9  # 닫힘 축 0°면 블록 긴 변은 90°
    assert abs(misalignment_deg(100.0, 0.0) - 10.0) < 1e-9
    assert abs(misalignment_deg(-80.0, 0.0) - 10.0) < 1e-9  # 180° 감김: −80° = 100°
