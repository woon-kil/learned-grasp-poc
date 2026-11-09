from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from learned_grasp.config import CHAIN_PATH, ROOT
from learned_grasp.kinematics import fk, fk_batch
from learned_grasp.pose_table import PoseTable, generate


@pytest.mark.parametrize(
    "sdk, expected, tilt",
    [
        # 기존 dofbot_fk로 계산해 플랜 검증 근거에 기록한 값
        ([89, 85, 25, 0, 108], (1.9, 144.6, 76.1), 20.0),
        ([90, 88, 25, 0, 108], (-0.6, 146.8, 83.7), 23.0),
        ([82, 85, 25, 0, 90], (19.5, 143.3, 76.1), 20.0),
    ],
)
def test_fk_matches_recorded_values(sdk, expected, tilt):
    position, angle = fk(sdk)
    assert position == pytest.approx(expected, abs=0.06)
    assert angle == pytest.approx(tilt, abs=0.06)


def test_fk_batch_matches_single():
    q = np.array([[90, 88, 19, 6, 90], [70, 100, 40, 10, 120]])
    positions, tilts = fk_batch(q)
    for index in range(2):
        position, tilt = fk(q[index])
        assert positions[index] == pytest.approx(position)
        assert tilts[index] == pytest.approx(tilt)


def test_table_covers_planned_radius(table, draft_config):
    assert table.entries[0].r_mm <= draft_config.table.r_min_mm + 1
    assert table.entries[-1].r_mm >= draft_config.table.r_max_mm - 1
    radii = [entry.r_mm for entry in table.entries]
    assert radii == sorted(set(radii))


def test_every_grasp_meets_spec(table, draft_config):
    spec = draft_config.table
    grasps = np.array([[90, *entry.grasp, 90] for entry in table.entries])
    positions, tilts = fk_batch(grasps)
    assert np.all(np.abs(positions[:, 2] - spec.z_grasp_mm) <= spec.z_tol_mm)
    assert np.all(np.abs(tilts - spec.tilt_deg) <= spec.tilt_tol_deg)


def test_hover_is_above_grasp(table):
    grasps = fk_batch(np.array([[90, *entry.grasp, 90] for entry in table.entries]))[0]
    hovers = fk_batch(np.array([[90, *entry.hover, 90] for entry in table.entries]))[0]
    assert np.all(hovers[:, 2] > grasps[:, 2] + 30)
    assert np.all(np.abs(hovers[:, 1] - grasps[:, 1]) <= 2.0 + 1e-6)


def test_label_restores_every_command(table):
    """라벨 r에서 표를 거쳐 같은 정수 명령이 복원된다. G0c의 전제다."""
    for entry in table.entries:
        assert table.recover(entry.grasp) == entry.grasp


def test_table_round_trips_through_file(table, tmp_path):
    path = tmp_path / "table.json"
    table.save(path)
    loaded = PoseTable.load(path)
    assert loaded.entries == table.entries
    assert loaded.sha256() == table.sha256()


def test_generation_is_deterministic(table, draft_config):
    assert generate(draft_config.table).sha256() == table.sha256()


def test_committed_table_matches_config(table):
    """실행 표는 구성에서 다시 만든다. 커밋한 표와 어긋나면 실패한다."""
    assert PoseTable.load(Path(ROOT) / "config" / "pose_table.json").sha256() == table.sha256()


def test_chain_records_source_model():
    chain = json.loads(CHAIN_PATH.read_text(encoding="utf-8"))
    assert chain["source_urdf_sha256"] == "b626a845cd67998cd5ce47f2b3d302d9cb3e5b9da764d9a1e7e1404a277ccc28"
    assert [joint["name"] for joint in chain["joints"]][-1] == "Gripping_point_Joint"


def test_hover_keeps_j3_j4_away_from_read_boundary(table):
    assert len(table.entries) == 10
    assert all(entry.hover[1] >= 5 and entry.hover[2] >= 5 for entry in table.entries)
