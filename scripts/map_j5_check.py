#!/usr/bin/env python3
"""L1 맵의 J5 항 선형성 점검. 장치 없음, 쓰기 0.

연쇄 시험에서 팔이 (손 행, 공칭 J1, 놓기 J5)로 놓은 블록의 영상 위치(다음 시도의 plan du, dv)를 맵의 순방향 예측 forward(J1, r, J5)와 비교한다.
맵은 G0b·G0c 20점(J5 60·90·120)으로 맞췄으므로 J5 중간값의 표본 밖 점검이다. 반경 오프셋(R1)·J1 오프셋(X2) 시도는 블록 위치가 손과 다르므로 빼고 CH-3·X1 장부만 쓴다.
사용: uv run python scripts/map_j5_check.py runs/CH-3/chain.jsonl runs/X1-0/chain.jsonl runs/X1-1/chain.jsonl
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map
from learned_grasp.pose_table import generate

PX_PER_MM = 1.14


def main() -> None:
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    l1 = L1Map()
    data = []
    for path in sys.argv[1:]:
        recs = [json.loads(line) for line in Path(path).read_text().splitlines()]
        st = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
        fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
        ids = sorted(st)
        for i, t in enumerate(ids[:-1]):
            s = st[t]
            if t not in fin or not fin[t]["success"] or s.get("offset", 0) or s.get("j1_offset", 0) or st[ids[i + 1]].get("human_placed"):
                continue
            n = s["place_at"]
            nxt = st[ids[i + 1]]["plan"]
            pred = l1.forward(n["j1"], rows[s.get("place_row", n["r_index"])], s["place_j5"])
            rad = math.radians(l1.radial_angle_deg(n["j1"], rows[s.get("place_row", n["r_index"])]))
            e = np.array([nxt["du"], nxt["dv"]]) - pred
            lat = float(e @ np.array([-math.sin(rad), math.cos(rad)])) / PX_PER_MM
            rd = float(e @ np.array([math.cos(rad), math.sin(rad)])) / PX_PER_MM
            data.append((s["place_j5"], lat, rd, path.split("/")[1]))
    print(f"놓인 블록 {len(data)}개(오프셋 없음, 성공, 사람 재배치 제외)")
    for lo, hi in ((55, 72), (72, 86), (86, 96), (96, 111), (111, 125)):
        g = [d for d in data if lo <= d[0] < hi]
        if g:
            lat = np.array([d[1] for d in g]); rd = np.array([d[2] for d in g])
            print(f"  J5 [{lo},{hi}): n {len(g)}, 횡방향 잔차 평균 {lat.mean():+.1f} mm (표준편차 {lat.std():.1f}), 반경 잔차 평균 {rd.mean():+.1f} mm (표준편차 {rd.std():.1f})")
    j5 = np.array([d[0] for d in data]); lat = np.array([d[1] for d in data]); rd = np.array([d[2] for d in data])
    kl, al = np.polyfit(j5 - 90, lat, 1); kr, ar = np.polyfit(j5 - 90, rd, 1)
    print(f"횡방향 잔차 ≈ {al:+.2f} mm {kl:+.3f} mm/J5°, 반경 잔차 ≈ {ar:+.2f} mm {kr:+.3f} mm/J5° (J5 편차 ±30°에서 횡방향 ±{abs(kl) * 30:.1f} mm, 반경 ±{abs(kr) * 30:.1f} mm)")
    k2 = np.polyfit(j5 - 90, lat, 2)
    print(f"횡방향 2차 항 {k2[0]:+.4f} mm/J5°² (J5 ±30°에서 {abs(k2[0]) * 900:.1f} mm)")


if __name__ == "__main__":
    main()
