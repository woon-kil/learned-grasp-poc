#!/usr/bin/env python3
"""X3 hover 사진 분석. 장치 없음(사진은 p620에서 scp). 닫히기 전 손과 블록의 방향 어긋남 ψ_pick과 좌우 간격 비대칭을 Δ별로 낸다.

ψ_pick = 블록 긴 변 각도(park 사진) − (손끝 축 각도(hover 사진) + 90°). 카메라 기하 상수 편향은 Δ 사이 차이에서 빠진다.
좌우 비대칭 = (블록 중심 − 손끝 중점)을 손끝 축 방향으로 투영한 화소(블록 중심은 park 사진 노란 영역 중심). 상수 편향(시차)은 Δ 차이에서 빠진다.
사용: uv run python scripts/x3_summary.py runs/X3-1 [runs/X3-0 ...]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.hand_tips import misalignment_deg, tip_axis_deg
from learned_grasp.l1 import block_from_image


def fetch(folder: str, name: str) -> Path:
    local = Path(folder) / "frames" / f"{name}.color.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        subprocess.run(["scp", "-q", f"p620:projects/learned-grasp-poc/{folder}/frames/{name}.color.png", str(local)], check=False)
    return local


def main() -> None:
    rows = []
    for folder in sys.argv[1:]:
        recs = [json.loads(line) for line in (Path(folder) / "chain.jsonl").read_text().splitlines()]
        st = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
        fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
        for t in sorted(st):
            r = st[t]
            hov, park = cv2.imread(str(fetch(folder, f"{t}-hover"))), cv2.imread(str(fetch(folder, r["before_frame"])))
            if hov is None or park is None:
                continue
            tips = tip_axis_deg(hov)
            blk = block_from_image(park, board_ref=(691.0, 291.0))
            if tips is None or blk is None:
                continue
            mid = ((tips["left"][0] + tips["right"][0]) / 2.0, (tips["left"][1] + tips["right"][1]) / 2.0)
            cx, cy = blk["board"][0] + blk["du"], blk["board"][1] + blk["dv"]
            ax = math.radians(tips["axis_deg"])
            lateral_px = (cx - mid[0]) * math.cos(ax) + (cy - mid[1]) * math.sin(ax)
            rows.append({"folder": folder, "t": t, "d1": r["j1_offset"], "psi": misalignment_deg(blk["block_angle_deg"], tips["axis_deg"]), "lat_px": lateral_px,
                         "cmd_mm": r["lateral_true_mm"], "ok": fin.get(t, {}).get("success"), "gap": tips["gap_px"], "axis": tips["axis_deg"]})
    print(f"hover 사진이 있는 시도 {len(rows)}개")
    by = {}
    for r in rows:
        by.setdefault(r["d1"], []).append(r)
    for d in sorted(by):
        psi = np.array([r["psi"] for r in by[d]])
        lat = np.array([r["lat_px"] for r in by[d]])
        print(f"  Δ{d:+d}°: n {len(psi)}, ψ_pick 평균 {psi.mean():+.2f}° (표준편차 {psi.std():.2f}), 좌우 비대칭 평균 {lat.mean():+.1f} px (표준편차 {lat.std():.1f}), 손끝 간격 {np.mean([r['gap'] for r in by[d]]):.1f}px, 성공 {sum(1 for r in by[d] if r['ok'])}/{len(by[d])}")
    if 1 in by and -1 in by:
        a, b = np.array([r["psi"] for r in by[1]]), np.array([r["psi"] for r in by[-1]])
        d = a.mean() - b.mean()
        se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        print(f"D_pick = ψ_pick(+1) − ψ_pick(−1) = {d:+.2f}° ± {se:.2f} (95% 구간 {d - 1.96 * se:+.2f} ~ {d + 1.96 * se:+.2f}). 보정이 맞으면 약 0, 보정이 없으면 약 +2.6°(부호는 기하 규약에 따름), 과보정이면 반대 부호")
        la, lb = np.array([r["lat_px"] for r in by[1]]), np.array([r["lat_px"] for r in by[-1]])
        print(f"좌우 비대칭 차 = {la.mean() - lb.mean():+.1f} px (명령한 횡방향 차 약 {np.mean([r['cmd_mm'] for r in by[1]]) - np.mean([r['cmd_mm'] for r in by[-1]]):+.1f} mm, 1.1 px/mm 환산 {1.1 * (np.mean([r['cmd_mm'] for r in by[1]]) - np.mean([r['cmd_mm'] for r in by[-1]])):+.1f} px)")
    if rows:
        lat = np.array([r["lat_px"] for r in rows]); cmd = np.array([r["cmd_mm"] for r in rows])
        slope = np.polyfit(cmd, lat, 1)[0]
        print(f"좌우 비대칭(px) 대 명령 횡방향(mm) 기울기 {slope:+.2f} px/mm (기대 약 ±1.1: 부호는 영상 방향 규약), 잔차 표준편차 {np.std(lat - np.polyval(np.polyfit(cmd, lat, 1), cmd)):.1f} px")


if __name__ == "__main__":
    main()
