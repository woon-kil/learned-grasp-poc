#!/usr/bin/env python3
"""연쇄 시험 기록 요약. 장치 없음. 사전 고정 판정(docs/*연쇄-시험-구현과-L2-반올림-계산-결과.md 2.2)과 놓인 뒤 방향 분포를 낸다.
사용: uv run python scripts/chain_summary.py runs/CH-3/chain.jsonl runs/CH-3/frames/chain-041.color.png
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, YawModel, block_from_image, signed_diff
from learned_grasp.pose_table import generate


def main() -> None:
    recs = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    starts = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
    fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
    ids = sorted(starts)
    print(f"시도 {len(ids)}: " + ", ".join(f"{k} {sum(1 for t in fin.values() if t['outcome'] == k)}" for k in ("success", "failed", "interrupted")))
    print("거절 기록(L1):", [(r["trial_id"], r["reason"][:40]) for r in recs if r.get("kind") == "rejected"])
    print("정지 요청:", [r["trial_id"] for r in recs if r.get("kind") == "stop_requested"], " 확인(ack):", [r["trial_id"] for r in recs if r.get("kind") == "acknowledged"])
    ok = [t for t in ids if fin[t]["outcome"] in ("success", "failed")]
    succ = [t for t in ok if fin[t]["outcome"] == "success"]
    print(f"집기 성공 {len(succ)}/{len(ok)}(중단 제외). 보수적으로 중단을 실패로 세면 {len(succ)}/{len(ids)}")
    for name, flag in (("집는 자리 ≠ 놓은 자리", False), ("같은 자리", True)):
        grp = [t for t in ok if starts[t]["same_cell"] is flag]
        print(f"  {name}: 시도 {len(grp)}, 성공 {sum(1 for t in grp if fin[t]['outcome'] == 'success')}")
    unk = [t for t in ok if starts[t].get("placed_at") is None]
    print(f"  (직전 놓은 자리를 모르는 시도 {len(unk)}개는 same_cell 거짓으로 기록됨)")
    pj = [starts[t]["pick_j5"] for t in ids]; qj = [starts[t]["place_j5"] for t in ids]
    print(f"집기 J5 범위 {min(pj)}~{max(pj)}(≠90 {sum(1 for j in pj if j != 90)}개), 놓기 J5 범위 {min(qj)}~{max(qj)}(≠90 {sum(1 for j in qj if j != 90)}개)")
    d_pick = np.array([starts[t]["plan"]["d_deg"] for t in ids])
    print(f"집기 전 반경선 대비 d: 평균 {d_pick.mean():+.1f}°, 표준편차 {d_pick.std():.1f}°, |d|>5° {int((abs(d_pick) > 5).sum())}개")
    # 놓인 뒤 d = 다음 시도의 집기 전 d(같은 블록). 마지막은 마지막 사진.
    table = generate(load_config("config/workspace.yaml").table); rows = [e.r_mm for e in table.entries]
    l1, yaw = L1Map(), YawModel()
    placed = []
    for i, t in enumerate(ids):
        pl = starts[t]["place_at"]
        if t not in succ:
            continue
        if i + 1 < len(ids) and starts[ids[i + 1]]["placed_at"] is not None:
            d = starts[ids[i + 1]]["plan"]["d_deg"]
        elif i + 1 == len(ids):
            m = block_from_image(cv2.imread(sys.argv[2]), board_ref=(691.0, 291.0))
            d = signed_diff(m["block_angle_deg"], l1.radial_angle_deg(pl["j1"], rows[pl["r_index"]]) % 180.0)
        else:
            continue
        pred = yaw.hand_d(starts[t]["place_j5"], pl["j1"], pl["r_index"])
        placed.append((t, d, pred))
    d = np.array([x[1] for x in placed]); pr = np.array([x[2] for x in placed])
    print(f"놓인 뒤 반경선 대비 d(n={len(d)}): 평균 {d.mean():+.1f}°, 표준편차 {d.std():.1f}°, |d|>5° {int((abs(d) > 5).sum())}개, |d|>10° {int((abs(d) > 10).sum())}개, 범위 {d.min():+.1f}~{d.max():+.1f}°")
    print(f"  모델 예측(놓기 정렬) 평균 {pr.mean():+.1f}°, 잔차(실측−예측) 평균 {np.mean(d - pr):+.1f}°, 표준편차 {np.std(d - pr):.1f}°")
    t0 = [r["t"] for r in recs if r.get("event") == "start"]; t1 = [r["t"] for r in recs if r.get("event") == "finish"]
    dur = np.array([f - s for s, f in zip(t0, t1)])
    print(f"시도 시간: 중앙값 {np.median(dur):.0f}초, 최대 {dur.max():.0f}초(쉼·확인 대기 포함)")


if __name__ == "__main__":
    main()
