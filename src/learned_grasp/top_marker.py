"""블록 윗면 마커 측정(G0d). 검출 설정과 식은 계약 3 도구(scripts/marker_precheck3.py)와 같다.

위치는 바깥쪽 4X4 마커 id 0, 1 중심의 평균에서 윗면 마커 id 7 중심까지의 화소 벡터다. 각도는 윗면 마커 첫 변의
영상 각도(360° 주기)이고 J5 명령과 따로 기록한다. mm 환산은 하지 않는다. 분석은 Mac의 고정 OpenCV로만 한다.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
from dataclasses import dataclass, field
from pathlib import Path

OPENCV_VERSION = "5.0.0.93"
TOP_ID = 7
BOARD_ID = 1
REFERENCE_IDS = (0, 1)
ROI = (480, 120, 920, 460)
SCALE = 3
CAMERA_LIMIT_PX = 2.0


class MeasureError(RuntimeError):
    """측정이 안 되면 쓰기 없이 멈춘다. reason은 기록용 짧은 코드다."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def require_opencv() -> None:
    try:
        installed = importlib.metadata.version("opencv-contrib-python-headless")
    except importlib.metadata.PackageNotFoundError as error:
        raise MeasureError("opencv_missing", "opencv-contrib-python-headless가 없다") from error
    if installed != OPENCV_VERSION:
        raise MeasureError("opencv_version", f"설치 {installed}, 고정 {OPENCV_VERSION}")


@dataclass(frozen=True)
class Measurement:
    name: str
    sha256: str
    outer: dict[int, tuple[float, float]]
    board_tag: bool
    top: bool
    position: tuple[float, float] | None = None
    angle: float | None = None
    extra: dict = field(default_factory=dict)

    def record(self) -> dict:
        return {
            "name": self.name, "sha256": self.sha256, "board_tag": self.board_tag, "top": self.top,
            "outer": {str(i): [round(v, 3) for v in c] for i, c in self.outer.items()},
            "position": None if self.position is None else [round(v, 3) for v in self.position],
            "angle": None if self.angle is None else round(self.angle, 3),
        }


def _detectors():
    import cv2

    aruco = cv2.aruco
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 53
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.05
    return (
        aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), params),
        aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_4X4_50), aruco.DetectorParameters()),
    )


def _found(detector, image) -> dict:
    corners, ids, _ = detector.detectMarkers(image)
    return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


class TopMarker:
    def __init__(self):
        require_opencv()
        self.tags, self.outer = _detectors()

    def measure(self, path: Path) -> Measurement:
        import cv2
        import numpy as np

        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise MeasureError("fetch_failed", str(error)) from error
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise MeasureError("fetch_failed", f"사진을 읽지 못했다: {path.name}")
        x0, y0, x1, y1 = ROI
        crop = cv2.resize(image[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
        top = _found(self.tags, crop)
        refs = _found(self.outer, image)
        outer = {i: tuple(float(v) for v in refs[i].mean(0)) for i in REFERENCE_IDS if i in refs}
        base = dict(name=path.name.removesuffix(".color.png"), sha256=hashlib.sha256(data).hexdigest(), outer=outer, board_tag=BOARD_ID in top, top=TOP_ID in top)
        if TOP_ID not in top or len(outer) < len(REFERENCE_IDS):
            return Measurement(**base)
        corners = top[TOP_ID] / SCALE + np.array([x0, y0])
        edge = corners[1] - corners[0]
        center = corners.mean(0) - np.mean([outer[i] for i in REFERENCE_IDS], axis=0)
        return Measurement(**base, position=(float(center[0]), float(center[1])), angle=float(math.degrees(math.atan2(edge[1], edge[0]))))


def wrap_deg(delta: float) -> float:
    return (delta + 180.0) % 360.0 - 180.0


class Placement:
    """시작 때 고정한 바깥쪽 마커 기준으로 사진마다 측정을 확인한다. 어긋나면 MeasureError로 멈춘다."""

    def __init__(self, reference: dict[int, tuple[float, float]], limit_px: float = CAMERA_LIMIT_PX):
        if set(reference) != set(REFERENCE_IDS):
            raise MeasureError("reference_missing", f"기준 마커 {sorted(REFERENCE_IDS)} 중 {sorted(reference)}만 있다")
        self.reference = dict(reference)
        self.limit_px = limit_px

    def check(self, measurement: Measurement) -> Measurement:
        self.check_reference(measurement)
        if not measurement.top:
            raise MeasureError("top_missing", f"{measurement.name}: 윗면 마커 id {TOP_ID} 없음")
        return measurement

    def check_reference(self, measurement: Measurement) -> Measurement:
        """바깥쪽 기준 마커만 본다. 블록이 없는 판정 사진에 쓴다."""
        missing = [i for i in REFERENCE_IDS if i not in measurement.outer]
        if missing:
            raise MeasureError("reference_missing", f"{measurement.name}: 바깥쪽 마커 {missing} 없음")
        for i in REFERENCE_IDS:
            moved = math.dist(measurement.outer[i], self.reference[i])
            if moved > self.limit_px:
                raise MeasureError("reference_moved", f"{measurement.name}: 마커 {i}가 기준에서 {moved:.2f}화소")
        return measurement
