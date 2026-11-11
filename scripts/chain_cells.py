#!/usr/bin/env python3
"""연쇄 시험의 내려놓을 자리 목록을 시드로 뽑아 실행 전에 고정한다. 장치 없음.

표의 격자 자리(r_index, 정수 J1)다. 가장자리 행(0, 9)과 J1 70·110은 놓기 흩어짐(약 3 mm)이 구역 밖으로 나가기 쉬워 피한다.
앞 자리와 J1이 2° 미만·같은 행이면 다시 뽑아 연속 자리가 겹치지 않게 한다.
사용: uv run python scripts/chain_cells.py config/chain_cells.json [개수]
"""
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

SEED = 20261112
# 관찰기 영역(변화 PICK_ROI, 노란 화소 PICK_BOARD_ROI 첫 영역, 픽셀 x0 x1 y0 y1)과 보드 마커 기준. 블록 중심이 가장자리에서 MARGIN 안쪽이어야 한다.
PICK_ROI = (540, 860, 190, 420)
BOARD_ROI = (585, 825, 150, 395)
BOARD_REF = (691.25, 291.5)
MARGIN = 16
R_INDEX = range(1, 9)
J1 = range(72, 109)


def inside(l1: L1Map, j1: int, r_mm: float) -> bool:
    du, dv = l1.forward(j1, r_mm)
    x, y = BOARD_REF[0] + du, BOARD_REF[1] + dv
    return all(a + MARGIN <= x <= b - MARGIN and c + MARGIN <= y <= d - MARGIN for a, b, c, d in (PICK_ROI, BOARD_ROI))


def main() -> None:
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rng = np.random.default_rng(SEED)
    l1 = L1Map()
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    cells: list[dict] = []
    while len(cells) < n:
        c = {"r_index": int(rng.choice(list(R_INDEX))), "j1": int(rng.choice(list(J1)))}
        if not inside(l1, c["j1"], rows[c["r_index"]]):
            continue  # 관찰기 영역 밖이면 거짓 실패가 난다
        if cells and abs(c["j1"] - cells[-1]["j1"]) < 2 and c["r_index"] == cells[-1]["r_index"]:
            continue
        cells.append(c)
    Path(sys.argv[1]).write_text(json.dumps({"seed": SEED, "n": n, "cells": cells}, ensure_ascii=False, indent=1))
    print(f"{n}개 -> {sys.argv[1]}, r_index 분포", np.bincount([c["r_index"] for c in cells], minlength=10).tolist())


if __name__ == "__main__":
    main()
