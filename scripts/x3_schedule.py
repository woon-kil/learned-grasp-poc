#!/usr/bin/env python3
"""X3(J1 보정 hover 사진 진단) 일정. 장치 없음. 자리·시드 방식은 x2_schedule.py와 같다.
config/x3.json: Δ {−1, 0, +1} 각 8번(24시도), config/x3_sanity.json: [0, +1, −1](3시도).
사용: uv run python scripts/x3_schedule.py config
"""
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

sys.path.insert(0, str(Path(__file__).parent))
from x2_schedule import cells_for  # noqa: E402

SEED = 20261117


def main() -> None:
    out = Path(sys.argv[1])
    rng = np.random.default_rng(SEED)
    l1 = L1Map()
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    a = [-1, 0, 1] * 8
    rng.shuffle(a)
    files = {
        "x3.json": {"seed": SEED, "stage": "X3", "n": 24, "cells": cells_for(rng, 24, l1, rows), "j1_offsets": [int(x) for x in a], "confirm_abs_ge": None},
        "x3_sanity.json": {"seed": SEED, "stage": "X3-sanity", "n": 3, "cells": cells_for(rng, 3, l1, rows), "j1_offsets": [0, 1, -1], "confirm_abs_ge": None},
    }
    for name, data in files.items():
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=1))
        print(name, data["n"], data["j1_offsets"][:12])


if __name__ == "__main__":
    main()
