#!/usr/bin/env python3
"""P2 목표 20자리를 시드로 뽑아 실행 전에 고정한다. 장치 없음.

위치는 J1 72~108, r 128~160 mm의 라틴 하이퍼큐브. **목표 J5를 62~118에서 층화**(62~70 4개, 110~118 4개, 70~110 12개)해 뽑고,
자리마다 목표 방향 d = hand_d(J5, J1, r_index)를 역산한다(방향으로 뽑으면 셀마다 d0이 달라 J5가 범위 밖이 되는 일이 많다).
블록 영상 위치는 J5를 넣은 순방향 맵으로 낸다. 사용: uv run python scripts/p2_targets.py config/p2_targets.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import MAP_PATH, YAW_PATH, L1Map, YawModel
from learned_grasp.pose_table import generate

SEED = 20261114
N = 20


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    cells = (np.arange(N) + rng.random(N)) / N
    j1 = 72.0 + rng.permutation(cells) * 36.0
    r = 128.0 + rng.permutation(cells) * 32.0
    j5 = np.concatenate([rng.uniform(66, 70, 4), rng.uniform(110, 114, 4), np.linspace(71, 109, 12) + rng.uniform(-1.5, 1.5, 12)])
    j5 = rng.permutation(j5)
    l1, yaw = L1Map(), YawModel()
    targets = []
    for i, (a, b, c) in enumerate(zip(j1, r, j5), start=1):
        r_idx = int(np.argmin([abs(x - b) for x in rows]))
        du, dv = l1.forward(a, b, c)
        rad = l1.radial_angle_deg(a, b) % 180.0
        d = yaw.hand_d(c, round(a), r_idx)
        targets.append({"id": f"p2-{i:02d}", "j1": round(float(a), 2), "r_mm": round(float(b), 2), "r_index": r_idx, "target_j5": round(float(c), 1),
                        "target_d_deg": round(float(d), 1), "radial_angle_deg": round(float(rad), 1), "block_angle_deg": round(float((rad + d) % 180.0), 1),
                        "du": round(float(du), 1), "dv": round(float(dv), 1)})
    out = {"seed": SEED, "n": N, "map_sha256": hashlib.sha256(Path(MAP_PATH).read_bytes()).hexdigest(), "yaw_sha256": hashlib.sha256(Path(YAW_PATH).read_bytes()).hexdigest(), "targets": targets}
    Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{N}개 -> {sys.argv[1]}, 목표 J5 극단(≤70 또는 ≥110) {sum(1 for t in targets if t['target_j5'] <= 70 or t['target_j5'] >= 110)}개")
    for t in targets:
        print(t["id"], "J1 %.0f r %.0f J5 %.0f d %+.0f°" % (t["j1"], t["r_mm"], t["target_j5"], t["target_d_deg"]))


if __name__ == "__main__":
    main()
