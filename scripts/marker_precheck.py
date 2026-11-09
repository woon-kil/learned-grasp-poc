#!/usr/bin/env python3
"""G0d 윗면 마커 쓰기 없는 사전 시험. 팔에 쓰기를 보내지 않는다. 판정 계약은 docs의 사전 시험 계약 문서다.

촬영: uv run python scripts/marker_precheck.py shoot --out runs/marker-top
분석: uv run --with opencv-contrib-python-headless python scripts/marker_precheck.py analyze --dir runs/marker-top/frames
"""
import argparse
import json
import sys
import time
from pathlib import Path

SPOTS = {"A": "보드 마커보다 베이스 쪽", "B": "보드 마커 왼쪽 옆", "C": "보드 마커보다 앞쪽(YAHBOOM 글자 쪽)"}
STATIC_SHOTS = 10
ANGLE_SHOTS = 5
MARKER_ID = 7
BOARD_ID = 1
CHANGE_MIN = 0.02
YELLOW_MAX = 100
PICK_ROI = (540, 860, 190, 420)
BOARD_ROI = (585, 825, 190, 395)
MIN_DETECT = 9
MIN_ANGLE_DEG = 5.0
ANGLE_SIGMA = 3.0


def steps() -> list[tuple[str, str, int]]:
    items = []
    for spot, where in SPOTS.items():
        items += [
            (f"{spot}-base", f"[{spot}] 윗면 마커가 보이게 블록을 {where}에 놓는다", STATIC_SHOTS),
            (f"{spot}-rotp", f"[{spot}] 블록 한쪽 끝은 그대로 두고 다른 쪽 끝만 옆으로 약 1 cm 민다(눈대중)", ANGLE_SHOTS),
            (f"{spot}-rotm", f"[{spot}] 같은 끝을 반대쪽으로 약 2 cm 민다(처음보다 반대로 약 1 cm, 눈대중)", ANGLE_SHOTS),
            (f"{spot}-slide", f"[{spot}] 처음 방향으로 되돌린 뒤 돌리지 말고 옆으로 약간 민다", 1),
            (f"{spot}-removed", f"[{spot}] 블록을 보드 밖으로 치운다", 1),
        ]
    items.append(("edge", "블록을 보드 오른쪽 가장자리에 걸치게 놓는다", 1))
    return items


def shoot(out: Path) -> None:
    from learned_grasp.camera import P620Camera
    from learned_grasp.session import _load_env

    _load_env()
    camera = P620Camera(str(out / "frames"))
    try:
        plan = steps()
        for number, (name, text, shots) in enumerate(plan, 1):
            input(f"({number}/{len(plan)}) {text}. 손을 빼고 Enter: ")
            for index in range(shots):
                camera.snapshot(f"{name}-{index:02d}")
                time.sleep(0.5)
            print(f"  찍음 {name} {shots}장", flush=True)
    finally:
        camera.close()


def analyze(folder: Path) -> dict:
    import cv2
    import numpy as np

    def circular_mean(degrees):
        radians = np.radians(degrees)
        return float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())))

    aruco = cv2.aruco
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 53
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.05
    detector = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), params)

    def read(name):
        return cv2.imread(str(folder / f"{name}.color.png"))

    def detect(image, scale=3, roi=(480, 120, 920, 460)):
        x0, y0, x1, y1 = roi
        crop = cv2.resize(image[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        corners, ids, _ = detector.detectMarkers(crop)
        return {} if ids is None else {int(i): c.reshape(4, 2) / scale + np.array([x0, y0]) for i, c in zip(ids.flatten(), corners)}

    def angle(corners):
        edge = corners[1] - corners[0]
        return float(np.degrees(np.arctan2(edge[1], edge[0])))

    def measure(name, shots):
        rows = []
        for index in range(shots):
            found = detect(read(f"{name}-{index:02d}"))
            if MARKER_ID in found and BOARD_ID in found:
                rows.append((*(found[MARKER_ID].mean(0) - found[BOARD_ID].mean(0)), angle(found[MARKER_ID])))
        return np.array(rows).reshape(-1, 3)

    def judge(before, after):
        def gray(image, roi):
            u0, u1, v0, v1 = roi
            return image[v0:v1, u0:u1].astype(np.int16).mean(axis=2)

        first, second = read(before), read(after)
        fraction = float((np.abs(gray(first, PICK_ROI) - gray(second, PICK_ROI)) > 30).mean())
        u0, u1, v0, v1 = BOARD_ROI
        b, g, r = np.moveaxis(second[v0:v1, u0:u1].astype(np.float32), -1, 0)
        yellow = int(((r > 140) & (g > 120) & (np.abs(r - g) < 60) & ((r - b) / np.maximum(r, 1) > 0.62)).sum())
        return {"change": round(fraction, 4), "yellow": yellow, "success": fraction >= CHANGE_MIN and yellow <= YELLOW_MAX}

    report: dict = {"spots": {}, "judge": [], "pass": True}
    for spot in SPOTS:
        static = measure(f"{spot}-base", STATIC_SHOTS)
        detected = len(static)
        entry = {"detected": f"{detected}/{STATIC_SHOTS}", "detect_ok": detected >= MIN_DETECT}
        if detected >= 2:
            sigma = float(((static[:, 2] - static[0, 2] + 180) % 360 - 180).std())
            entry |= {"pos_std_px": [round(float(v), 3) for v in static[:, :2].std(0)], "angle_std_deg": round(sigma, 3)}
            for label in ("rotp", "rotm"):
                turned = measure(f"{spot}-{label}", ANGLE_SHOTS)
                if len(turned):
                    delta = abs(float(((circular_mean(turned[:, 2]) - circular_mean(static[:, 2]) + 180) % 360) - 180))
                    entry[f"{label}_delta_deg"] = round(delta, 2)
                    entry[f"{label}_ok"] = delta >= max(MIN_ANGLE_DEG, ANGLE_SIGMA * sigma)
                else:
                    entry[f"{label}_ok"] = False
        report["spots"][spot] = entry
        report["pass"] &= bool(entry["detect_ok"] and entry.get("rotp_ok") and entry.get("rotm_ok"))
        base = f"{spot}-base-00"
        cases = [("same", f"{spot}-base-{STATIC_SHOTS - 1:02d}", False), ("rot", f"{spot}-rotp-00", False), ("slide", f"{spot}-slide-00", False), ("removed", f"{spot}-removed-00", True)]
        for case, after, want in cases:
            result = {"spot": spot, "case": case, "want_success": want, **judge(base, after)}
            result["ok"] = result["success"] == want
            report["judge"].append(result)
            report["pass"] &= result["ok"]
    edge = {"spot": "A", "case": "edge", "want_success": False, **judge("A-base-00", "edge-00")}
    edge["ok"] = not edge["success"]
    report["judge"].append(edge)
    report["pass"] &= edge["ok"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["shoot", "analyze"])
    parser.add_argument("--out", type=Path, default=Path("runs/marker-top"))
    parser.add_argument("--dir", type=Path)
    args = parser.parse_args()
    if args.command == "shoot":
        if args.out.exists():
            sys.exit(f"이미 있다: {args.out}")
        args.out.mkdir(parents=True)
        shoot(args.out)
        return 0
    report = analyze(args.dir)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
