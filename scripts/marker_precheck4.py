#!/usr/bin/env python3
"""G0d 판정 재시험(계약 4). 팔에 쓰기를 보내지 않는다. 다시 찍기 기능은 없다.

판정은 운영 코드 make_pick_observer를 저장 사진 어댑터로 그대로 돌린다(docs/261110-1008-G0d-판정값-재계획-플랜.md).

촬영: uv run --with opencv-contrib-python-headless==5.0.0.93 python scripts/marker_precheck4.py shoot
가져오기: uv run python scripts/marker_precheck4.py fetch
분석: uv run --with opencv-contrib-python-headless==5.0.0.93 python scripts/marker_precheck4.py analyze
"""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

from gemini_serve_core import change_fraction, yellow_pixels
from marker_precheck3 import HOST, OPENCV_VERSION, REFERENCE_IDS, CAMERA_LIMIT_PX, PreviewError, _detectors, _ids, _take_preview, board_visible

PLACES = {
    "N": "베이스 쪽 끝이 베이스 판 가장자리에 거의 닿게(닿지는 않게), 보드 마커가 보이게",
    "P1": "베이스와 보드 마커 사이 가운데에, 보드 마커가 보이게",
    "M": "글자 쪽 끝이 보드 마커 바로 앞에 오게, 보드 마커가 보이게",
    "P3": "보드 마커 위에, 보드 마커가 위에서 전혀 보이지 않게 덮어서",
}
PREVIEW_WANT = {"N": True, "P1": True, "M": True, "P3": False}
STATIC_SHOTS = 3
EDGES = {
    "E1": "분홍 표시 반대쪽 가장자리와 YAHBOOM 글자 쪽 가장자리가 만나는 종이 모서리에, 전부 종이 위에 있게 붙여",
    "E2": "분홍 표시 쪽 종이 가장자리에, 전부 종이 위에 있게 붙여",
    "E3": "YAHBOOM 글자 쪽 종이 가장자리에, 전부 종이 위에 있게 붙여",
    "E4": "분홍 표시 반대쪽 종이 가장자리의 가운데쯤에, 전부 종이 위에 있게 붙여",
}
OUT = Path("runs/judge-4")


def steps() -> list[tuple[str, str, int]]:
    items = []
    for place, where in PLACES.items():
        items += [
            (f"{place}-base", f"[{place}] 블록 긴 변을 베이스에서 YAHBOOM 글자 쪽으로 향하게 하고, 블록을 {where} 놓는다", STATIC_SHOTS),
            (f"{place}-rotp", f"[{place}] 베이스 쪽 끝은 그대로 두고 글자 쪽 끝만 옆으로 약 1 cm 민다(눈대중)", 1),
            (f"{place}-rotm", f"[{place}] 같은 글자 쪽 끝을 반대쪽으로 약 2 cm 민다(처음보다 반대로 약 1 cm, 눈대중)", 1),
            (f"{place}-slide", f"[{place}] 처음 방향으로 되돌린 뒤 돌리지 말고 옆으로 약간 민다", 1),
            (f"{place}-removed", f"[{place}] 블록을 보드 밖으로 치운다", 1),
        ]
    for name, where in EDGES.items():
        items.append((f"{name}-00", f"[{name}] 블록 긴 변을 베이스에서 YAHBOOM 글자 쪽으로 향하게 하고, 블록을 {where} 놓는다", 1))
    return items


def expected_frames() -> list[str]:
    frames = []
    for name, _, shots in steps():
        frames += [name] if name.startswith("E") else [f"{name}-{index:02d}" for index in range(shots)]
    return frames


def judge_cases() -> list[tuple[str, str, bool]]:
    """(기준, 판정 사진, 성공이어야 하는가). 28건이다."""
    cases = []
    for place in PLACES:
        base = f"{place}-base-00"
        cases += [(base, f"{place}-base-{index:02d}", False) for index in range(1, STATIC_SHOTS)]
        cases += [(base, f"{place}-{label}-00", False) for label in ("rotp", "rotm", "slide")]
        cases.append((base, f"{place}-removed-00", True))
    cases += [("P1-base-00", f"{name}-00", False) for name in EDGES]
    return cases


class StoredCamera:
    """운영 판정 함수가 부르는 카메라 자리에 저장 사진을 넣는다. 계산식은 p620 서버와 같은 gemini_serve_core다."""

    def __init__(self, folder: Path, before: str, after: str):
        self.folder, self.names = folder, {"case": before, "case-after": after}

    def _read(self, name):
        import cv2

        image = cv2.imread(str(self.folder / f"{self.names[name]}.color.png"))
        if image is None:
            raise FileNotFoundError(self.names[name])
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def snapshot(self, name):
        assert name in self.names

    def change(self, before, after, roi):
        return change_fraction(self._read(before), self._read(after), roi)

    def yellow_pixels(self, name, roi):
        return yellow_pixels(self._read(name), roi)


def run_case(folder: Path, before: str, after: str) -> dict:
    from learned_grasp.session import make_pick_observer

    lines = []
    outcome = make_pick_observer(StoredCamera(folder, before, after), log=lambda text, **_: lines.append(text))({"trial_id": "case"})
    values = json.loads(lines[-1])
    return {"change": values["change"], "yellow": values["yellow"], "success": outcome.success}


def fetch(out: Path) -> int:
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--include=*.color.png", "--exclude=*", f"{HOST}:projects/learned-grasp-poc/{frames}/", f"{frames}/"], check=True)
    missing = [name for name in expected_frames() if not (frames / f"{name}.color.png").exists()]
    print(f"사진 {len(expected_frames()) - len(missing)}/{len(expected_frames())}, 빠짐 {missing}")
    return 1 if missing else 0


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
                        print(f"  {error}. 시험 사진은 찍지 않았다. 다시 확인한다.")
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


def analyze(folder: Path) -> dict:
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

    _, outer = _detectors()
    points: dict = {i: [] for i in REFERENCE_IDS}
    hashes = {}
    for name in expected_frames():
        path = folder / f"{name}.color.png"
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        corners, ids, _ = outer.detectMarkers(cv2.imread(str(path)))
        found = {} if ids is None else {int(i): c.reshape(4, 2).mean(0) for i, c in zip(ids.flatten(), corners)}
        for i in REFERENCE_IDS:
            if i in found:
                points[i].append(found[i])
    total = len(expected_frames())
    camera = {str(i): float(np.linalg.norm(np.array(p) - np.median(p, axis=0), axis=1).max()) if p else None for i, p in points.items()}
    edges_ok = all((log.get("edge_confirmed") or {}).get(f"{name}-00") == "y" for name in EDGES)
    report: dict = {
        "opencv": installed,
        "session_id": log["session_id"],
        "preview_attempts": log.get("preview_attempts"),
        "p3_covered_answer": log.get("p3_covered"),
        "edge_confirmed": log.get("edge_confirmed"),
        "outer_detected": {str(i): f"{len(p)}/{total}" for i, p in points.items()},
        "outer_max_px": {i: None if v is None else round(v, 3) for i, v in camera.items()},
    }
    report["valid"] = bool(
        edges_ok
        and log.get("p3_covered") == "y"
        and all(len(p) == total for p in points.values())
        and all(v is not None and v <= CAMERA_LIMIT_PX for v in camera.values())
    )
    report["judge"] = []
    if report["valid"]:
        for before, after, want in judge_cases():
            result = {"before": before, "after": after, "want_success": want, **run_case(folder, before, after)}
            result["ok"] = result["success"] == want
            report["judge"].append(result)
    report["judge_pass"] = bool(report["valid"] and len(report["judge"]) == 28 and all(r["ok"] for r in report["judge"]))
    report["sha256"] = hashes
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["shoot", "fetch", "analyze"])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--dir", type=Path)
    args = parser.parse_args()
    if args.command == "shoot":
        if args.out.exists():
            sys.exit(f"이미 있다: {args.out}")
        args.out.mkdir(parents=True)
        shoot(args.out)
        return 0
    if args.command == "fetch":
        return fetch(args.out)
    folder = args.dir or args.out / "frames"
    report = analyze(folder)
    (folder.parent / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "sha256"}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
