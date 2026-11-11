#!/usr/bin/env python3
"""X2 기록 요약. 장치 없음. 사용: uv run python scripts/x2_summary.py runs/X2-1/chain.jsonl runs/X2-1/frames/chain-025.color.png"""
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    recs = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    st = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
    fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
    ids = sorted(st)
    print(f"시도 {len(ids)}: " + ", ".join(f"{k} {sum(1 for t in fin.values() if t['outcome'] == k)}" for k in ("success", "failed", "interrupted")))
    print("거절:", [(r["trial_id"], r["reason"][:40]) for r in recs if r.get("kind") == "rejected"], " 정지 요청:", [r["trial_id"] for r in recs if r.get("kind") == "stop_requested"])
    print("J1 오프셋별 성공/시도, 실제 횡방향 어긋남(mm), 서보 6 판독, 집기 J5 범위:")
    for d in sorted({st[t]["j1_offset"] for t in ids}):
        g = [t for t in ids if st[t]["j1_offset"] == d and t in fin]
        if not g:
            continue
        mm = [st[t]["lateral_mm"] for t in g]
        grips = [fin[t]["grip_read"] for t in g if fin[t].get("grip_read") is not None]
        j5 = [st[t]["pick_j5"] for t in g]
        print(f"  Δ{d:+d}°: {sum(1 for t in g if fin[t]['success'])}/{len(g)}, 횡방향 {np.mean(mm):+.1f} mm, 서보 {min(grips) if grips else '-'}~{max(grips) if grips else '-'}, 집기 J5 {min(j5)}~{max(j5)}")
    print("실제 횡방향 어긋남(= (집는 J1 − L1 연속 J1) × r, 반올림 포함) 구간별 성공/시도:")
    edges = [(-9, -4.5), (-4.5, -1.5), (-1.5, 1.5), (1.5, 4.5), (4.5, 9)]
    for lo, hi in edges:
        g = [t for t in ids if t in fin and lo <= st[t]["lateral_true_mm"] < hi]
        if g:
            print(f"  [{lo:+.1f}, {hi:+.1f}) mm: {sum(1 for t in g if fin[t]['success'])}/{len(g)}")
    print("실패·중단:", [(t, fin[t]["outcome"], fin[t]["reason"][:50]) for t in ids if t in fin and not fin[t]["success"]])
    # 놓인 블록의 L1 J1 − 손 J1(공칭): 0이면 오프셋 쥠이 가운데로 옮겨졌다는 뜻이 아니라 놓는 손이 공칭이므로 블록이 손 J1 근처에 놓였는지 본다
    dj = []
    for i, t in enumerate(ids):
        if t not in fin or not fin[t]["success"]:
            continue
        n = st[t]["place_at"]
        if i + 1 < len(ids):
            j1c = st[ids[i + 1]]["plan"]["j1_cont"]
        else:
            j1c = None  # 마지막 시도의 놓인 블록은 다음 시도가 없어 이 지표에서 뺀다(사진은 따로 분석)
        if j1c is not None:
            dj.append((st[t]["j1_offset"], j1c - n["j1"]))
    res = residuals(st, fin, ids)
    for d in sorted(res):
        v = np.array(res[d])
        print(f"  놓인 뒤 방향 잔차(실측−예측), Δ{d:+d}° 집기 뒤: 평균 {v.mean():+.1f}° (n {len(v)})")
    if 1 in res and -1 in res:
        print(f"  J1 오프셋 보정 진단: +1 뒤 평균 − −1 뒤 평균 = {np.mean(res[1]) - np.mean(res[-1]):+.1f}° (보정이 맞으면 약 0, 없으면 약 2.6°, 부호 반대면 약 5°)")
    for d in sorted({x[0] for x in dj}):
        v = np.array([x[1] for x in dj if x[0] == d])
        print(f"  놓인 블록 J1 − 놓은 손 J1(공칭), Δ{d:+d}° 집기 뒤: 평균 {v.mean():+.2f}° 표준편차 {v.std():.2f}° (n {len(v)}). 0이면 블록이 손 가운데로 옮겨져 놓임, −Δ면 오프셋 보존")


def residuals(st, fin, ids):
    """놓인 뒤 d(다음 시도 plan.d_deg) − 모델 예측 hand_d(놓기 J5), 앞 집기의 Δ 부호별(J1 오프셋 보정 진단)."""
    from learned_grasp.l1 import YawModel

    yaw = YawModel()
    out = {}
    for i, t in enumerate(ids[:-1]):
        if t not in fin or not fin[t]["success"] or st[ids[i + 1]].get("human_placed"):
            continue
        n = st[t]["place_at"]
        resid = st[ids[i + 1]]["plan"]["d_deg"] - yaw.hand_d(st[t]["place_j5"], n["j1"], st[t]["place_row"])
        out.setdefault(st[t]["j1_offset"], []).append(resid)
    return out


if __name__ == "__main__":
    main()
