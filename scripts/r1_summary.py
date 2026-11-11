#!/usr/bin/env python3
"""R1(반경 허용) 기록 요약. 장치 없음. 사용: uv run python scripts/r1_summary.py runs/R1-1/chain.jsonl runs/R1-1/frames/chain-031.color.png"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, YawModel, block_from_image, plan_aligned
from learned_grasp.pose_table import generate


def main() -> None:
    recs = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    st = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
    fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
    ids = sorted(st)
    table = generate(load_config("config/workspace.yaml").table)
    rows = [e.r_mm for e in table.entries]
    yaw = YawModel()
    print(f"시도 {len(ids)}: " + ", ".join(f"{k} {sum(1 for t in fin.values() if t['outcome'] == k)}" for k in ("success", "failed", "interrupted")))
    print("거절:", [(r["trial_id"], r["reason"][:40]) for r in recs if r.get("kind") == "rejected"], " 정지 요청:", [r["trial_id"] for r in recs if r.get("kind") == "stop_requested"])
    print("오프셋(행)별 성공/시도, 실제 손 어긋남(mm, 집는 행 r − L1 반경 추정), 서보 6 판독:")
    for d in (-3, 0, 3):
        g = [t for t in ids if st[t]["offset"] == d]
        mm = np.array([rows[st[t]["r_pick"]] - st[t]["plan"]["r_cont"] for t in g])
        grips = [fin[t]["grip_read"] for t in g if fin[t].get("grip_read") is not None]
        print(f"  Δ{d:+d}행: {sum(1 for t in g if fin[t]['success'])}/{len(g)}, 손 어긋남 평균 {mm.mean():+.1f} mm (범위 {mm.min():+.1f}~{mm.max():+.1f}), 서보 판독 {min(grips)}~{max(grips)}")
    # 놓인 블록의 L1 반경 추정 − 공칭 행 r : 다음 시도의 plan.r_cont, 마지막은 마지막 사진
    dev = []
    for i, t in enumerate(ids):
        n = st[t]["place_at"]
        if i + 1 < len(ids):
            rc = st[ids[i + 1]]["plan"]["r_cont"]
        else:
            m = block_from_image(cv2.imread(sys.argv[2]), board_ref=(691.0, 291.0))
            p = plan_aligned(m, table, yaw=yaw)
            rc = p.r_cont
        dev.append((st[t]["offset"], rc - rows[n["r_index"]]))
    for d in (-3, 0, 3):
        v = np.array([x[1] for x in dev if x[0] == d])
        print(f"  놓인 블록 반경 − 공칭 행 r (Δ{d:+d}행으로 쥔 블록): 평균 {v.mean():+.1f}, 표준편차 {v.std():.1f}, 범위 {v.min():+.1f}~{v.max():+.1f} mm (n {len(v)})")
    pj = [st[t]["pick_j5"] for t in ids]
    print(f"집기 J5 범위 {min(pj)}~{max(pj)}, 쓰기 {sum(1 for _ in ids) * 10}회 안팎")


if __name__ == "__main__":
    main()
