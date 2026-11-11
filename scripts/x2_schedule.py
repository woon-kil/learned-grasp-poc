#!/usr/bin/env python3
"""X2(횡방향 허용 측정) 일정을 시드로 만들어 실행 전에 고정한다. 장치 없음.

놓을 자리: 행 3~6, J1 76~104(±3° 오프셋 여유, 관찰기 영역 점검). 오프셋 j1_offsets(J1°): A {−1, 0, +1} 각 8번(24), B {−2, +2} 각 6번(12), sanity [0, +1, −1].
B는 confirm_abs_ge 2: 모든 |Δ| ≥ 2 하강을 사용자가 확인한다.
사용: uv run python scripts/x2_schedule.py config
"""
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

SEED = 20261116
sys.path.insert(0, str(Path(__file__).parent))
from chain_cells import inside  # noqa: E402


def cells_for(rng, n, l1, rows):
    cells: list[dict] = []
    while len(cells) < n:
        c = {"r_index": int(rng.integers(3, 7)), "j1": int(rng.integers(76, 105))}
        if not inside(l1, c["j1"], rows[c["r_index"]]):
            continue
        if cells and abs(c["j1"] - cells[-1]["j1"]) < 2 and c["r_index"] == cells[-1]["r_index"]:
            continue
        cells.append(c)
    return cells


def main() -> None:
    out = Path(sys.argv[1])
    rng = np.random.default_rng(SEED)
    l1 = L1Map()
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    a = [-1, 0, 1] * 8
    rng.shuffle(a)
    b = [-2, 2] * 6
    rng.shuffle(b)
    files = {
        "x2_a.json": {"seed": SEED, "stage": "A", "n": 24, "cells": cells_for(rng, 24, l1, rows), "j1_offsets": [int(x) for x in a], "confirm_abs_ge": None},
        "x2_b.json": {"seed": SEED, "stage": "B", "n": 12, "cells": cells_for(rng, 12, l1, rows), "j1_offsets": [int(x) for x in b], "confirm_abs_ge": 2},
        "x2_sanity.json": {"seed": SEED, "stage": "sanity", "n": 3, "cells": cells_for(rng, 3, l1, rows), "j1_offsets": [0, 1, -1], "confirm_abs_ge": None},
    }
    for name, data in files.items():
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(name, data["n"], "오프셋", data["j1_offsets"][:12], "...")


if __name__ == "__main__":
    main()
