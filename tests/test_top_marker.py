from __future__ import annotations

import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from learned_grasp import top_marker
from learned_grasp.top_marker import MeasureError, Placement, TopMarker, require_opencv

ROOT = Path(__file__).resolve().parents[1]


def _paste(image, dictionary, marker_id, center, size):
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(dictionary), marker_id, size)
    pad = size // 4
    tile = np.full((size + 2 * pad, size + 2 * pad), 255, np.uint8)
    tile[pad:pad + size, pad:pad + size] = marker
    x, y = int(center[0] - tile.shape[1] / 2), int(center[1] - tile.shape[0] / 2)
    image[y:y + tile.shape[0], x:x + tile.shape[1]] = tile[..., None]


def scene(tmp_path, name="s", top=(690, 230), refs=((915, 600), (851, 211)), drop=()):
    image = np.full((720, 1280, 3), 200, np.uint8)
    for marker_id, center in enumerate(refs):
        if marker_id not in drop:
            _paste(image, cv2.aruco.DICT_4X4_50, marker_id, center, 80)
    if top is not None:
        _paste(image, cv2.aruco.DICT_APRILTAG_36h11, 7, top, 40)
    path = tmp_path / f"{name}.color.png"
    cv2.imwrite(str(path), image)
    return path


def test_measures_position_from_reference_mean(tmp_path):
    m = TopMarker().measure(scene(tmp_path))
    assert m.top and set(m.outer) == {0, 1}
    mean = np.mean([(915, 600), (851, 211)], axis=0)
    assert m.position == pytest.approx(tuple(np.array((690, 230)) - mean), abs=1.0)
    assert m.angle == pytest.approx(0.0, abs=1.0)
    assert len(m.sha256) == 64 and m.record()["outer"].keys() == {"0", "1"}


@pytest.mark.parametrize(("kwargs", "reason"), [({"top": None}, "top_missing"), ({"drop": (1,)}, "reference_missing")])
def test_placement_stops_without_markers(tmp_path, kwargs, reason):
    marker = TopMarker()
    placement = Placement(marker.measure(scene(tmp_path, "ref")).outer)
    with pytest.raises(MeasureError) as error:
        placement.check(marker.measure(scene(tmp_path, "x", **kwargs)))
    assert error.value.reason == reason


def test_placement_stops_when_reference_moves(tmp_path):
    marker = TopMarker()
    placement = Placement(marker.measure(scene(tmp_path, "ref")).outer)
    placement.check(marker.measure(scene(tmp_path, "same")))
    with pytest.raises(MeasureError) as error:
        placement.check(marker.measure(scene(tmp_path, "moved", refs=((918, 600), (851, 211)))))
    assert error.value.reason == "reference_moved"


def test_reference_needs_both_markers(tmp_path):
    with pytest.raises(MeasureError) as error:
        Placement(TopMarker().measure(scene(tmp_path, drop=(0,))).outer)
    assert error.value.reason == "reference_missing"


def test_missing_or_broken_photo_is_fetch_failure(tmp_path):
    marker = TopMarker()
    with pytest.raises(MeasureError) as error:
        marker.measure(tmp_path / "none.color.png")
    assert error.value.reason == "fetch_failed"
    broken = tmp_path / "broken.color.png"
    broken.write_bytes(b"not a png")
    with pytest.raises(MeasureError) as error:
        marker.measure(broken)
    assert error.value.reason == "fetch_failed"


def test_other_opencv_version_refuses(monkeypatch):
    monkeypatch.setattr(top_marker.importlib.metadata, "version", lambda name: "4.5.4.60")
    with pytest.raises(MeasureError) as error:
        require_opencv()
    assert error.value.reason == "opencv_version"


CONTRACT3 = ROOT / "runs" / "marker-top-3"


@pytest.mark.skipif(not (CONTRACT3 / "analysis.json").exists(), reason="계약 3 사진은 커밋하지 않는다(runs/)")
def test_matches_contract3_measurements():
    """측정 합격을 받은 계약 3 도구와 프레임별 값이 같아야 한다. 다르면 측정을 다시 봐야 한다(계약 3 64행)."""
    expected = json.loads((CONTRACT3 / "analysis.json").read_text())["frames"]
    marker = TopMarker()
    for name, record in expected.items():
        got = marker.measure(CONTRACT3 / "frames" / f"{name}.color.png").record()
        assert got["sha256"] == record["sha256"], name
        assert got["outer"] == record["outer"], name
        if "position" in record:
            assert got["position"] == record["position"] and got["angle"] == record["angle"], name
        if "board_tag" in record:
            assert (got["board_tag"], got["top"]) == (record["board_tag"], record["top"]), name
