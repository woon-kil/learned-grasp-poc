#!/usr/bin/env python3
"""X1(J5 극단 연쇄 검증) 일정을 시드로 만들어 실행 전에 고정한다. 장치 없음.

시도 k(1부터): 홀수는 극단 J5로 놓고(place_j5 = {64,66,68,112,114,116} 각 N/12번, 섞음), 짝수는 정렬 J5로 놓는다(null).
극단으로 돌아 놓인 블록은 다음(짝수) 시도가 L1이 정한 극단 J5로 집는다. 놓을 자리는 행 3~6, J1 72~108(관찰기 영역 점검 포함).
사용: uv run python scripts/x1_schedule.py config/x1_schedule.json [시도 수(12의 배수, 기본 24)]
"""
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

SEED = 20261115
sys.path.insert(0, str(Path(__file__).parent))
from chain_cells import inside  # noqa: E402


def main() -> None:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    assert n % 12 == 0
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
    extremes = [64, 66, 68, 112, 114, 116] * (n // 12)
    rng.shuffle(extremes)
    place = [int(extremes[i // 2]) if i % 2 == 0 else None for i in range(n)]
    Path(sys.argv[1]).write_text(json.dumps({"seed": SEED, "n": n, "cells": cells, "place_j5": place}, ensure_ascii=False, indent=1))
    print(f"{n}개 -> {sys.argv[1]}, 극단 놓기 J5 {[p for p in place if p is not None]}")


if __name__ == "__main__":
    main()
