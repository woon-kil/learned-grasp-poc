#!/usr/bin/env python3
"""X1 기록 요약. 장치 없음. 사용: uv run python scripts/x1_summary.py runs/X1-1/chain.jsonl runs/X1-1/frames/chain-025.color.png"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, YawModel, block_from_image, plan_aligned, signed_diff
from learned_grasp.pose_table import generate


def main() -> None:
    recs = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    st = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
    fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
    ids = sorted(st)
    table = generate(load_config("config/workspace.yaml").table)
    yaw = YawModel()
    n_succ = sum(1 for t in ids if t in fin and fin[t]["success"])
    print(f"시도 {len(ids)}: 성공 {n_succ}, 실패 {sum(1 for t in fin.values() if t['outcome'] == 'failed')}, 중단 {sum(1 for t in fin.values() if t['outcome'] == 'interrupted')}")
    print("거절:", [(r["trial_id"], r["reason"][:40]) for r in recs if r.get("kind") == "rejected"], " 정지 요청:", [r["trial_id"] for r in recs if r.get("kind") == "stop_requested"])
    ext = [t for i, t in enumerate(ids) if i % 2 == 1 and not st[t].get("human_placed")]  # 짝수 번호 시도(극단 집기). 사람이 다시 놓은 블록은 제외
    print(f"극단 집기(짝수 시도) 성공 {sum(1 for t in ext if fin.get(t, {}).get('success'))}/{len(ext)}")
    devs = np.array([abs(st[t]["pick_j5"] - 90) for t in ext]) if ext else np.array([])
    if len(devs):
        print(f"  극단 집기의 실제 |J5−90|: 평균 {devs.mean():.1f}, 범위 {devs.min()}~{devs.max()}, ≥20 {int((devs >= 20).sum())}개, ≥26 {int((devs >= 26).sum())}개")
    print("클램프:", [(t, st[t]["pick_j5"]) for t in ids if st[t].get("clamped")])
    l1 = L1Map()
    rows = [e.r_mm for e in table.entries]
    clean, after_ext, human = [], [], []
    for i, t in enumerate(ids):
        if t not in fin or not fin[t]["success"]:
            continue
        n = st[t]["place_at"]
        if i + 1 < len(ids):
            d = st[ids[i + 1]]["plan"]["d_deg"]
            nxt_human = st[ids[i + 1]].get("human_placed")
        else:
            m = block_from_image(cv2.imread(sys.argv[2]), board_ref=(691.0, 291.0))
            d = signed_diff(m["block_angle_deg"], l1.radial_angle_deg(n["j1"], rows[st[t]["place_row"]]) % 180.0)
            nxt_human = False
        resid = d - yaw.hand_d(st[t]["place_j5"], n["j1"], st[t]["place_row"])
        if nxt_human:
            human.append(resid)
        elif i % 2 == 0 and i > 0:
            clean.append((st[t]["place_j5"], d, resid))  # 홀수 번호 시도: near-90 집기 뒤 극단 놓기
        elif i % 2 == 1:
            after_ext.append(resid)
    if clean:
        r = np.array([x[2] for x in clean])
        print(f"near-90 집기 뒤 극단 놓기(n {len(r)}): 놓인 뒤 d 실측−예측 평균 {r.mean():+.1f}°, 표준편차 {r.std():.1f}°, |최대| {abs(r).max():.1f}° (판정 ≤ 5.0°, 기준선 CH-3 정렬 놓기 3.4°. 3.4°를 넘는 만큼을 극단 모델 오차로 읽는다)")
        print("  (놓기 J5, 측정 d, 잔차):", [(a, round(b, 1), round(c, 1)) for a, b, c in clean])
    if after_ext:
        r = np.array(after_ext)
        print(f"극단 집기 뒤 정렬 놓기(n {len(r)}): 잔차 평균 {r.mean():+.1f}°, 표준편차 {r.std():.1f}° (이월 효과, 따로 보고)")
    if human:
        print(f"사람이 다시 놓은 블록 뒤 값 {len(human)}개는 지표에서 제외")


if __name__ == "__main__":
    main()
