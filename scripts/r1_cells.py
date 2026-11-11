#!/usr/bin/env python3
"""R1(반경 허용 측정) 일정을 시드로 만들어 실행 전에 고정한다. 장치 없음.

놓을 자리는 행 3~6, J1 72~108만(집는 행 = L1 행 + 오프셋이 0~9 안에 있게 한다). 오프셋은 {−3, 0, +3}행을 각 N/3번 섞은 순서다.
사용: uv run python scripts/r1_cells.py config/r1_schedule.json [시도 수(3의 배수, 기본 30)]
"""
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

SEED = 20261113
sys.path.insert(0, str(Path(__file__).parent))
from chain_cells import inside  # noqa: E402


def main() -> None:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    assert n % 3 == 0
    rng = np.random.default_rng(SEED)
    l1 = L1Map()
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    cells: list[dict] = []
    while len(cells) < n:
        c = {"r_index": int(rng.integers(3, 7)), "j1": int(rng.integers(72, 109))}
        if not inside(l1, c["j1"], rows[c["r_index"]]):
            continue
        if cells and abs(c["j1"] - cells[-1]["j1"]) < 2 and c["r_index"] == cells[-1]["r_index"]:
            continue
        cells.append(c)
    offsets = [-3, 0, 3] * (n // 3)
    rng.shuffle(offsets)
    Path(sys.argv[1]).write_text(json.dumps({"seed": SEED, "n": n, "cells": cells, "offsets": [int(o) for o in offsets]}, ensure_ascii=False, indent=1))
    print(f"{n}개 -> {sys.argv[1]}, 행 분포 {np.bincount([c['r_index'] for c in cells], minlength=10).tolist()}, 오프셋 {offsets[:12]}...")


if __name__ == "__main__":
    main()
