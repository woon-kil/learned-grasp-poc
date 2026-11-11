#!/usr/bin/env python3
"""P2 시도 준비: 사용자가 블록을 놓은 뒤 사진을 찍고 plan_aligned로 (r_index, J1, J5)를 낸다. 쓰기 0(카메라 촬영만).
사용: uv run python scripts/p2_plan.py <시도 id> <세션 폴더> [--board-ref 691.0,291.0]
출력: 한 줄 JSON과 data/p2/plan-<id>.json. 수락이면 이어서 p1-pick --r-index R --j1 J --j5 J5 --trial <id>를 실행한다.
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2

from learned_grasp.config import load_config
from learned_grasp.l1 import YawModel, block_from_image, plan_aligned
from learned_grasp.pose_table import generate


def main() -> None:
    tid, folder = sys.argv[1], sys.argv[2]
    ref = (691.0, 291.0)
    if "--board-ref" in sys.argv:
        ref = tuple(float(v) for v in sys.argv[sys.argv.index("--board-ref") + 1].split(","))
    shot = f"{folder}/plan-{tid}"
    subprocess.run(["ssh", "p620", f'cd ~/projects/learned-grasp-poc && bash scripts/p620_camera.sh capture "$PWD/{shot}"'], check=True, capture_output=True)
    local = Path("data/p2") / f"{tid}-placed.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scp", "-q", f"p620:projects/learned-grasp-poc/{shot}/color.png", str(local)], check=True)
    table = generate(load_config("config/workspace.yaml").table)
    m = block_from_image(cv2.imread(str(local)), board_ref=ref)
    p = plan_aligned(m, table, yaw=YawModel())
    rec = {"trial": tid, **p.asdict()}
    Path("data/p2").joinpath(f"plan-{tid}.json").write_text(json.dumps(rec, ensure_ascii=False))
    print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
