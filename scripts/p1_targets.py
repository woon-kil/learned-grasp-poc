#!/usr/bin/env python3
"""P1 목표 자리 20개를 시드로 뽑아 실행 전에 고정한다. 장치 없음.

맵 부채꼴 안에서 연속값 (J1, r)를 라틴 하이퍼큐브로 뽑는다(J1 72~108, r 125~160 mm: 표 가장자리 행을 피함).
순방향 맵으로 보드 마커 대비 영상 위치(du, dv)와 반경 방향 각도(블록을 놓을 방향)를 함께 낸다.
사용: uv run python scripts/p1_targets.py config/p1_targets.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.l1 import L1Map, MAP_PATH

SEED = 20261111
N = 20
J1_RANGE = (72.0, 108.0)
R_RANGE = (125.0, 160.0)


def main() -> None:
    rng = np.random.default_rng(SEED)
    cells = (np.arange(N) + rng.random(N)) / N  # 라틴 하이퍼큐브: 칸마다 한 점
    j1 = J1_RANGE[0] + rng.permutation(cells) * (J1_RANGE[1] - J1_RANGE[0])
    r = R_RANGE[0] + rng.permutation(cells) * (R_RANGE[1] - R_RANGE[0])
    l1 = L1Map()
    targets = []
    for i, (a, b) in enumerate(zip(j1, r), start=1):
        du, dv = l1.forward(a, b)
        targets.append({"id": f"p1-{i:02d}", "j1": round(float(a), 2), "r_mm": round(float(b), 2), "du": round(float(du), 1), "dv": round(float(dv), 1),
                        "radial_angle_deg": round(l1.radial_angle_deg(a, b) % 180.0, 1)})
    out = {"seed": SEED, "n": N, "j1_range": J1_RANGE, "r_range": R_RANGE, "map_sha256": hashlib.sha256(Path(MAP_PATH).read_bytes()).hexdigest(), "targets": targets}
    Path(sys.argv[1]).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{N}개 -> {sys.argv[1]}, J1 정수 아닌 값 {sum(abs(t['j1'] - round(t['j1'])) > 0.25 for t in targets)}개")
    for t in targets:
        print(t["id"], "J1 %.1f r %.1f du %.0f dv %.0f 반경각 %.0f°" % (t["j1"], t["r_mm"], t["du"], t["dv"], t["radial_angle_deg"]))


if __name__ == "__main__":
    main()
