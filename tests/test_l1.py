from __future__ import annotations

import numpy as np
import pytest

from learned_grasp.config import load_config
from learned_grasp.l1 import PSI_MAX_DEG, L1Map, block_from_image, plan
from learned_grasp.pose_table import generate, table_for


@pytest.fixture(scope="module")
def table():
    return table_for(load_config("config/workspace.yaml"))


@pytest.fixture(scope="module")
def l1():
    return L1Map()


def test_inverse_recovers_forward_points(l1, table):
    for j1, r in ((90, 140.0), (75, 125.0), (105, 160.0), (82, 150.0)):
        du, dv = l1.forward(j1, r)
        j1c, rc, res = l1.inverse(du, dv, (table.entries[0].r_mm, table.entries[-1].r_mm))
        assert res < 0.01 and abs(j1c - j1) < 0.05 and abs(rc - r) < 0.2


def test_plan_rounds_j1_and_picks_nearest_row(l1, table):
    du, dv = l1.forward(88.4, 143.0)
    radial = l1.radial_angle_deg(88.4, 143.0) % 180.0
    p = plan({"du": du, "dv": dv, "block_angle_deg": radial}, table, l1)
    assert p.accept and p.j1 == 88 and abs(table.entries[p.r_index].r_mm - 143.0) <= 3.0
    assert p.psi_deg < 0.5


def test_plan_rejects_misaligned_block(l1, table):
    du, dv = l1.forward(90, 140.0)
    radial = l1.radial_angle_deg(90, 140.0) % 180.0
    p = plan({"du": du, "dv": dv, "block_angle_deg": (radial + PSI_MAX_DEG + 3.0) % 180.0}, table, l1)
    assert not p.accept and "어긋남" in p.reason and p.psi_deg > PSI_MAX_DEG


def test_plan_rejects_outside_map(l1, table):
    p = plan({"du": 400.0, "dv": 300.0, "block_angle_deg": 90.0}, table, l1)
    assert not p.accept
    assert not plan(None, table, l1).accept


def test_psi_wraps_around_180(l1, table):
    du, dv = l1.forward(90, 140.0)
    radial = l1.radial_angle_deg(90, 140.0) % 180.0
    p = plan({"du": du, "dv": dv, "block_angle_deg": (radial + 178.0) % 180.0}, table, l1)  # 178°는 -2°와 같다
    assert p.accept and p.psi_deg == pytest.approx(2.0, abs=0.3)


def test_block_from_image_on_real_photo(l1):
    import cv2

    img = cv2.imread("data/g0frames/G0b-before/c00-t0.color.png")
    if img is None:
        pytest.skip("사진 자료 없음(gitignore)")
    m = block_from_image(img)
    assert m is not None and 3 < m["du"] < 12 and -90 < m["dv"] < -75
