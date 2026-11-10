#!/usr/bin/env python3
"""G0d 윗면 마커 쓰기 없는 재시험(계약 3). 팔에 쓰기를 보내지 않는다. 다시 찍기 기능은 없다.

촬영: uv run --with opencv-contrib-python-headless==5.0.0.93 python scripts/marker_precheck3.py shoot
가져오기: uv run python scripts/marker_precheck3.py fetch
분석: uv run --with opencv-contrib-python-headless==5.0.0.93 python scripts/marker_precheck3.py analyze --dir runs/marker-top-3/frames --p3-photo-covered yes
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path

PLACES = {
    "P1": "베이스와 보드 마커 사이에, 보드 마커가 보이게",
    "P3": "보드 마커 위에, 보드 마커가 위에서 전혀 보이지 않게 덮어서",
}
PREVIEW_WANT = {"P1": True, "P3": False}
STATIC_SHOTS = 10
ANGLE_SHOTS = 5
MIN_STATIC = 9
MIN_ANGLE_SHOTS = 4
MIN_ANGLE_DEG = 5.0
ANGLE_SIGMA = 3.0
MIN_POS_PX = 2.0
POS_SIGMA = 3.0
CAMERA_LIMIT_PX = 2.0
OPENCV_VERSION = "5.0.0.93"
JUDGE_PLACES = ("P1", "P3")
EDGES = {
    "E1": "분홍 표시 반대쪽 가장자리와 YAHBOOM 글자 쪽 가장자리가 만나는 종이 모서리에, 전부 종이 위에 있게 붙여",
    "E2": "분홍 표시 쪽 종이 가장자리에, 전부 종이 위에 있게 붙여",
    "E3": "YAHBOOM 글자 쪽 종이 가장자리에, 전부 종이 위에 있게 붙여",
}
CHANGE_MIN = 0.02
YELLOW_MAX = 100
PICK_ROI = (540, 860, 190, 420)
BOARD_ROI = (585, 825, 190, 395)
HOST = "p620"
TOP_ID = 7
BOARD_ID = 1
REFERENCE_IDS = (0, 1)
ROI = (480, 120, 920, 460)
SCALE = 3


def steps() -> list[tuple[str, str, int]]:
    items = []
    for place, where in PLACES.items():
        items += [
            (f"{place}-base", f"[{place}] 블록 긴 변을 베이스에서 YAHBOOM 글자 쪽으로 향하게 하고, 블록을 {where} 놓는다", STATIC_SHOTS),
            (f"{place}-rotp", f"[{place}] 베이스 쪽 끝은 그대로 두고 글자 쪽 끝만 옆으로 약 1 cm 민다(눈대중)", ANGLE_SHOTS),
            (f"{place}-rotm", f"[{place}] 같은 글자 쪽 끝을 반대쪽으로 약 2 cm 민다(처음보다 반대로 약 1 cm, 눈대중)", ANGLE_SHOTS),
        ]
        if place in JUDGE_PLACES:
            items += [
                (f"{place}-slide", f"[{place}] 처음 방향으로 되돌린 뒤 돌리지 말고 옆으로 약간 민다", 1),
                (f"{place}-removed", f"[{place}] 블록을 보드 밖으로 치운다", 1),
            ]
    for name, where in EDGES.items():
        items.append((f"{name}-00", f"[{name}] 블록 긴 변을 베이스에서 YAHBOOM 글자 쪽으로 향하게 하고, 블록을 {where} 놓는다", 1))
    return items


def spread_rms(points) -> float:
    """평균에서의 2차원 RMS 거리. 양쪽으로 같게 흔들려도 0이 되지 않는다."""
    points = list(points)
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in points) / len(points))


def expected_frames() -> list[str]:
    frames = []
    for name, _, shots in steps():
        frames += [name] if name.startswith("E") else [f"{name}-{index:02d}" for index in range(shots)]
    return frames


def fetch(out: Path) -> int:
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include=*.color.png", "--exclude=*", f"{HOST}:projects/learned-grasp-poc/{frames}/", f"{frames}/"], check=True)
    missing = [name for name in expected_frames() if not (frames / f"{name}.color.png").exists()]
    print(f"사진 {len(expected_frames()) - len(missing)}/{len(expected_frames())}, 빠짐 {missing}")
    return 1 if missing else 0


class PreviewError(RuntimeError):
    pass


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


def _ids(detector, image) -> set[int]:
    _, ids, _ = detector.detectMarkers(image)
    return set() if ids is None else {int(i) for i in ids.flatten()}


def board_visible(image, tags) -> bool:
    """미리보기 판단. 보드 마커 id 1만 본다. 윗면 마커와 바깥쪽 마커 값은 쓰지 않는다."""
    import cv2

    if image is None:
        raise PreviewError("미리보기 사진을 읽지 못했다")
    x0, y0, x1, y1 = ROI
    crop = cv2.resize(image[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    return BOARD_ID in _ids(tags, crop)


def _take_preview(camera, out: Path, name: str):
    import cv2

    camera.snapshot(name)
    frames = out / "previews"
    frames.mkdir(exist_ok=True)
    source = f"{HOST}:projects/learned-grasp-poc/{out}/frames/{name}.color.png"
    if subprocess.run(["scp", "-q", source, str(frames / f"{name}.color.png")]).returncode != 0:
        raise PreviewError(f"미리보기 사진을 가져오지 못했다: {name}")
    image = cv2.imread(str(frames / f"{name}.color.png"))
    if image is None:
        raise PreviewError(f"미리보기 사진을 읽지 못했다: {name}")
    return image


def shoot(out: Path) -> None:
    installed = importlib.metadata.version("opencv-contrib-python-headless")
    if installed != OPENCV_VERSION:
        sys.exit(f"촬영 중단: OpenCV {installed}, 계약은 {OPENCV_VERSION}")
    tags, outer = _detectors()
    remote = f"projects/learned-grasp-poc/{out}"
    if subprocess.run(["ssh", HOST, f"test ! -e {remote}"]).returncode != 0:
        sys.exit(f"p620에 이미 있다: {remote}")
    from learned_grasp.camera import P620Camera
    from learned_grasp.session import _load_env

    _load_env()
    camera = P620Camera(str(out / "frames"), host=HOST)
    try:
        plan = steps()
        log = {"session_id": uuid.uuid4().hex, "host": HOST, "complete": False, "p3_covered": None, "edge_confirmed": {}, "preview_attempts": {}, "steps": []}
        scene = _ids(outer, _take_preview(camera, out, "preview-scene"))
        if not set(REFERENCE_IDS) <= scene:
            sys.exit(f"시작 전 중단: 바깥쪽 마커 {sorted(REFERENCE_IDS)} 중 일부가 없다. 잡힌 마커 {sorted(scene)}. 시험 사진은 찍지 않았다.")
        for number, (name, text, shots) in enumerate(plan, 1):
            if name.endswith("-base"):
                place = name.split("-")[0]
                attempt = 0
                while True:
                    if place == "P3":
                        while input(f"({number}/{len(plan)}) {text}. 손을 빼고, 보드 마커가 위에서 안 보이면 y 입력 후 Enter: ").strip().lower() != "y":
                            print("  아직 찍지 않았다. 다시 놓는다.")
                        log["p3_covered"] = "y"
                    else:
                        input(f"({number}/{len(plan)}) {text}. 손을 빼고 Enter: ")
                    attempt += 1
                    log["preview_attempts"][place] = attempt
                    try:
                        visible = board_visible(_take_preview(camera, out, f"preview-{place}-{attempt:02d}"), tags)
                    except PreviewError as error:
                        print(f"  {error}. 시험 사진은 찍지 않았다. Enter로 다시 확인한다.")
                        continue
                    if visible == PREVIEW_WANT[place]:
                        print("  자리 확인 통과")
                        break
                    print("  자리가 아니다: 카메라에 보드 마커가 " + ("안 보인다. 블록을 베이스 쪽으로 옮긴다." if PREVIEW_WANT[place] else "보인다. 더 덮어 놓는다."))
            elif name.startswith("E"):
                attempt = 0
                while True:
                    while input(f"({number}/{len(plan)}) {text}. 손을 빼고, 블록 전체가 종이 위이고 지정 가장자리에 닿았으면 y 입력 후 Enter: ").strip().lower() != "y":
                        print("  아직 찍지 않았다. 다시 놓는다.")
                    attempt += 1
                    log["preview_attempts"][name] = attempt
                    try:
                        seen = _ids(outer, _take_preview(camera, out, f"preview-{name}-{attempt:02d}"))
                    except PreviewError as error:
                        print(f"  {error}. 시험 사진은 찍지 않았다. y로 다시 확인한다.")
                        continue
                    if set(REFERENCE_IDS) <= seen:
                        print("  자리 확인 통과")
                        break
                    print("  자리가 아니다: 오른쪽 큰 마커가 가려졌다. 블록을 큰 마커에서 떨어뜨려 놓는다.")
                log["edge_confirmed"][name] = "y"
            else:
                input(f"({number}/{len(plan)}) {text}. 손을 빼고 Enter: ")
            for index in range(shots):
                camera.snapshot(name if name.startswith("E") else f"{name}-{index:02d}")
                time.sleep(0.5)
            print(f"  찍음 {name} {shots}장", flush=True)
            log["steps"].append({"name": name, "shots": shots, "time": time.time()})
            log["complete"] = number == len(plan)
            (out / "shoot_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    finally:
        camera.close()


def analyze(folder: Path, p3_photo_covered: bool) -> dict:
    installed = importlib.metadata.version("opencv-contrib-python-headless")
    if installed != OPENCV_VERSION:
        sys.exit(f"판정 중단: OpenCV {installed}, 계약은 {OPENCV_VERSION}")
    missing = [name for name in expected_frames() if not (folder / f"{name}.color.png").exists()]
    if missing:
        sys.exit(f"판정 중단: 사진 없음 {missing}")
    log = json.loads((folder.parent / "shoot_log.json").read_text())
    if not log.get("complete") or [step["name"] for step in log["steps"]] != [name for name, _, _ in steps()]:
        sys.exit("판정 중단: 촬영 기록이 완결되지 않았다")
    import cv2
    import numpy as np

    tags, outer = _detectors()
    frames: dict = {}
    reference_points: dict = {i: [] for i in REFERENCE_IDS}

    def found(detector, image):
        corners, ids, _ = detector.detectMarkers(image)
        return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}

    def read(name):
        path = folder / f"{name}.color.png"
        return cv2.imread(str(path)), hashlib.sha256(path.read_bytes()).hexdigest()

    def measure(name, shots):
        rows, board = [], 0
        for index in range(shots):
            frame = f"{name}-{index:02d}"
            image, digest = read(frame)
            x0, y0, x1, y1 = ROI
            crop = cv2.resize(image[y0:y1, x0:x1], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
            top = found(tags, crop)
            refs = found(outer, image)
            board += BOARD_ID in top
            centers = {i: refs[i].mean(0) for i in REFERENCE_IDS if i in refs}
            for i, center in centers.items():
                reference_points[i].append(center)
            record = {"sha256": digest, "board_tag": BOARD_ID in top, "top": TOP_ID in top, "outer": {str(i): [round(float(v), 3) for v in c] for i, c in centers.items()}}
            if TOP_ID in top and len(centers) == len(REFERENCE_IDS):
                corners = top[TOP_ID] / SCALE + np.array([x0, y0])
                edge = corners[1] - corners[0]
                position = corners.mean(0) - np.mean(list(centers.values()), axis=0)
                angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
                rows.append((*position, angle))
                record |= {"position": [round(float(v), 3) for v in position], "angle": round(angle, 3)}
            frames[frame] = record
        return np.array(rows).reshape(-1, 3), board

    def circular_mean(degrees):
        radians = np.radians(degrees)
        return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))

    def wrap(delta):
        return (delta + 180) % 360 - 180

    def judge(before, after):
        def gray(image):
            u0, u1, v0, v1 = PICK_ROI
            return image[v0:v1, u0:u1].astype(np.int16).mean(axis=2)

        first, second = read(before)[0], read(after)[0]
        fraction = float((np.abs(gray(first) - gray(second)) > 30).mean())
        u0, u1, v0, v1 = BOARD_ROI
        b, g, r = np.moveaxis(second[v0:v1, u0:u1].astype(np.float32), -1, 0)
        yellow = int(((r > 140) & (g > 120) & (np.abs(r - g) < 60) & ((r - b) / np.maximum(r, 1) > 0.62)).sum())
        return {"change": round(fraction, 4), "yellow": yellow, "success": fraction >= CHANGE_MIN and yellow <= YELLOW_MAX}

    def record_only(name):
        image, digest = read(name)
        refs = found(outer, image)
        centers = {i: refs[i].mean(0) for i in REFERENCE_IDS if i in refs}
        for i, center in centers.items():
            reference_points[i].append(center)
        frames[name] = {"sha256": digest, "outer": {str(i): [round(float(v), 3) for v in c] for i, c in centers.items()}}

    report: dict = {"opencv": installed, "cv2": cv2.__version__, "p3_covered_answer": log.get("p3_covered"), "session_id": log["session_id"], "p3_photo_covered": p3_photo_covered, "preview_attempts": log.get("preview_attempts"), "edge_confirmed": log.get("edge_confirmed"), "places": {}, "judge": [], "valid": True, "measure_pass": None, "judge_pass": None, "pass": None}
    measured = {place: measure(f"{place}-base", STATIC_SHOTS) for place in PLACES}
    report["board_tag"] = {place: f"{board}/{STATIC_SHOTS}" for place, (_, board) in measured.items()}
    report["valid"] = measured["P1"][1] >= MIN_STATIC and measured["P3"][1] <= STATIC_SHOTS - MIN_STATIC and log.get("p3_covered") == "y" and p3_photo_covered
    if not report["valid"]:
        report["frames"] = frames
        return report

    turned_sets = {place: {label: measure(f"{place}-{label}", ANGLE_SHOTS)[0] for label in ("rotp", "rotm")} for place in PLACES}
    for place in JUDGE_PLACES:
        for label in ("slide", "removed"):
            record_only(f"{place}-{label}-00")
    for name in EDGES:
        record_only(f"{name}-00")

    measured_ok = True
    for place in PLACES:
        static = measured[place][0]
        entry = {"detected": f"{len(static)}/{STATIC_SHOTS}", "detect_ok": len(static) >= MIN_STATIC}
        ok = entry["detect_ok"]
        for label, turned in turned_sets[place].items():
            entry[f"{label}_detected"] = f"{len(turned)}/{ANGLE_SHOTS}"
        if len(static) >= 2:
            center = circular_mean(static[:, 2])
            home = static[:, :2].mean(0)
            sigma = float(wrap(static[:, 2] - center).std())
            spread = spread_rms(static[:, :2].tolist())
            angle_limit = max(MIN_ANGLE_DEG, ANGLE_SIGMA * sigma)
            pos_limit = max(MIN_POS_PX, POS_SIGMA * spread)
            entry |= {"pos_rms_px": round(spread, 3), "angle_std_deg": round(sigma, 3), "angle_limit_deg": round(angle_limit, 2), "pos_limit_px": round(pos_limit, 2)}
            signs = []
            for label, turned in turned_sets[place].items():
                good = len(turned) >= MIN_ANGLE_SHOTS
                if len(turned):
                    delta = float(wrap(circular_mean(turned[:, 2]) - center))
                    moved = float(np.linalg.norm(turned[:, :2].mean(0) - home))
                    entry[f"{label}_delta_deg"] = round(delta, 2)
                    entry[f"{label}_moved_px"] = round(moved, 2)
                    signs.append(delta)
                    good = good and abs(delta) >= angle_limit and moved >= pos_limit
                entry[f"{label}_ok"] = bool(good)
                ok = ok and good
            entry["opposite_ok"] = len(signs) == 2 and signs[0] * signs[1] < 0
            ok = ok and entry["opposite_ok"]
        else:
            ok = False
        report["places"][place] = entry
        measured_ok = measured_ok and bool(ok)
    expected_outer = len(expected_frames())
    report["outer_detected"] = {str(i): f"{len(points)}/{expected_outer}" for i, points in reference_points.items()}
    measured_ok = measured_ok and all(len(points) == expected_outer for points in reference_points.values())
    camera = {str(i): float(np.linalg.norm(np.array(points) - np.median(points, axis=0), axis=1).max()) if points else None for i, points in reference_points.items()}
    report["outer_max_px"] = {i: None if value is None else round(value, 3) for i, value in camera.items()}
    measured_ok = measured_ok and all(value is not None and value <= CAMERA_LIMIT_PX for value in camera.values())
    report["measure_pass"] = bool(measured_ok)

    judged_ok = True
    def check(before, after, want):
        result = {"after": after, "want_success": want, **judge(before, after)}
        result["ok"] = result["success"] == want
        report["judge"].append(result)
        return result["ok"]

    for place in JUDGE_PLACES:
        base = f"{place}-base-00"
        names = [f"{place}-base-{index:02d}" for index in range(1, STATIC_SHOTS)]
        names += [f"{place}-{label}-{index:02d}" for label in ("rotp", "rotm") for index in range(ANGLE_SHOTS)]
        names.append(f"{place}-slide-00")
        for after in names:
            judged_ok = check(base, after, False) and judged_ok
        judged_ok = check(base, f"{place}-removed-00", True) and judged_ok
    if all((log.get("edge_confirmed") or {}).get(f"{name}-00") == "y" for name in EDGES):
        for name in EDGES:
            judged_ok = check("P1-base-00", f"{name}-00", False) and judged_ok
        report["judge_pass"] = bool(judged_ok)
    report["pass"] = bool(report["measure_pass"] and report["judge_pass"])
    report["frames"] = frames
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["shoot", "fetch", "analyze"])
    parser.add_argument("--p3-photo-covered", choices=["yes", "no"])
    parser.add_argument("--out", type=Path, default=Path("runs/marker-top-3"))
    parser.add_argument("--dir", type=Path)
    args = parser.parse_args()
    if args.command == "shoot":
        sys.exit("계약 3은 끝났다. 촬영은 scripts/marker_precheck4.py shoot 만 쓴다.")
    if args.command == "fetch":
        return fetch(args.out)
    if args.p3_photo_covered is None:
        sys.exit("--p3-photo-covered 필요: 분석 전에 기록·커밋한 P3 사진 확인 결과")
    report = analyze(args.dir, args.p3_photo_covered == "yes")
    (args.dir.parent / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "frames"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
