#!/usr/bin/env python3
"""H3 수집: 표 grasp 자세 12곳에서 열린 손 정지 사진 5장씩을 찍고 나사점 위치를 저장한다.

자세마다 기존 `dabai-view --r-index R --j1 J`(park→hover→grasp→hover→park, 쓰기 5회)를 따로 실행하고,
hover는 사진 없이 바로 신호를 보내며 grasp 도착 줄이 나오면 상시 카메라로 5장을 찍고 신호를 보낸다.
검출 실패는 기록하고 다음 자세로 간다. 프로세스 실패·대기 상한 초과·이동 이상이면 멈춘다.
사용: uv run python scripts/h3_collect.py runs/H3 [--dry-run] [--start N] [--count M]   (N번째 자세부터 M개, 기본은 전부. 결과는 poses.json에 이어 붙인다)
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2_screw_points as screws  # noqa: E402

from learned_grasp.camera import P620Camera  # noqa: E402
from learned_grasp.config import load_config  # noqa: E402
from learned_grasp.pose_table import table_for  # noqa: E402

# (r_index, J1): 순서 1은 도구 시험(H2-1과 같은 자세), 12는 1의 재방문 반복. r0은 영상 밖 위험이 커서 뒤로 보낸다.
POSES = [(4, 90), (2, 90), (6, 90), (4, 70), (4, 110), (9, 90), (9, 70), (9, 110), (0, 90), (0, 70), (0, 110), (4, 90)]
FRAMES = 5
WRITES_PER_POSE = 5
ARRIVE_TIMEOUT_S = 90.0
CAMERA_INFO = "data/h2/gemini_camera_info.json"


def wait_for(path: Path, count: int, timeout: float, proc: subprocess.Popen) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if path.exists() and path.read_text().count("arrived") >= count:
            return
        if proc.poll() is not None:
            raise RuntimeError(f"세션 프로세스가 끝났다(코드 {proc.returncode}): {path.read_text()[-300:]}")
        time.sleep(0.5)
    raise TimeoutError(f"{timeout:g}초 안에 도착 줄 {count}개가 나오지 않았다")


def main() -> None:
    folder = Path(sys.argv[1])
    dry = "--dry-run" in sys.argv

    def option(name, default):
        return int(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    start, count = option("--start", 1), option("--count", len(POSES))
    cfg = load_config("config/workspace.yaml")
    table = table_for(cfg)
    if dry:
        print(f"자세 {len(POSES)}곳, 쓰기 {len(POSES) * WRITES_PER_POSE}회 예정")
        for i, (r, j1) in enumerate(POSES):
            print(i + 1, f"r_index {r} ({table.entries[r].r_mm:.1f} mm) J1 {j1} grasp {table.entries[r].grasp}")
        return
    camera = P620Camera(str(folder / "frames"))
    previous = folder / "poses.json"
    results = json.loads(previous.read_text())["poses"] if previous.exists() else []
    try:
        for order, (r, j1) in enumerate(POSES, start=1):
            if not start <= order < start + count:
                continue
            log = folder / f"pose-{order:02d}.log"
            signal = folder / "dabai_next"
            signal.unlink(missing_ok=True)
            proc = subprocess.Popen(
                ["uv", "run", "python", "-m", "learned_grasp.session", "dabai-view", "--session", str(folder / "approval.json"),
                 "--r-index", str(r), "--j1", str(j1), "--start-confirmed"], stdout=log.open("w"), stderr=subprocess.STDOUT)
            wait_for(log, 1, ARRIVE_TIMEOUT_S, proc)
            signal.write_text("1")  # hover에서는 찍지 않는다
            wait_for(log, 2, ARRIVE_TIMEOUT_S, proc)
            paths = []
            for k in range(FRAMES):
                name = f"{order:02d}-r{r}-j{j1}-{k}"
                camera.snapshot(name)
                paths.append(str(camera.fetch(name)))
            pairs = [screws.find_pair(p, "wide") for p in paths]
            found = [pr for pr in pairs if pr is not None]
            entry = {"order": order, "r_index": r, "j1": j1, "unique": f"r{r}_j{j1}", "q": [j1, *table.entries[r].grasp, 90],
                     "frames": FRAMES, "detected": len(found)}
            if found:
                mids = np.array([[(a[1] + b[1]) / 2, (a[2] + b[2]) / 2] for a, b in found])
                entry |= {"u": float(mids[:, 0].mean()), "v": float(mids[:, 1].mean()), "std": mids.std(0, ddof=1).round(3).tolist() if len(found) > 1 else None,
                          "spacing": float(np.mean([b[1] - a[1] for a, b in found]))}
                img = cv2.imread(paths[0])
                for _, x, y in found[0]:
                    cv2.circle(img, (int(x), int(y)), 16, (0, 255, 0), 2)
                cv2.imwrite(str(folder / f"overlay-{order:02d}-r{r}-j{j1}.png"), img[0:450, 450:1100])
            signal.write_text("1")
            code = proc.wait(timeout=120)
            if code != 0:
                raise RuntimeError(f"자세 {order} 세션이 비정상 종료({code}): {log.read_text()[-300:]}")
            results.append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
    finally:
        camera.close()
        (folder / "poses.json").write_text(json.dumps({"camera_info": CAMERA_INFO, "poses": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
