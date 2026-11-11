#!/usr/bin/env python3
"""P2 기록 요약(결과·J5 구간별 성공·집기 전후 블록 각도 비틀림). 장치 없음, 사진은 p620에서 scp로 가져온다.
사용: uv run python scripts/p2_summary.py runs/P2-1
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from learned_grasp.l1 import block_from_image


def fetch(folder: str, name: str) -> Path:
    local = Path(folder) / "frames" / f"{name}.color.png"
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        subprocess.run(["scp", "-q", f"p620:projects/learned-grasp-poc/{folder}/frames/{name}.color.png", str(local)], check=False)
    return local


def angle(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    m = block_from_image(img, board_ref=(691.0, 291.0))
    return None if m is None else m["block_angle_deg"]


def wrap(x: float) -> float:
    return ((x + 90.0) % 180.0) - 90.0


def main() -> None:
    folder = sys.argv[1]
    recs = [json.loads(line) for line in (Path(folder) / "p1.jsonl").read_text().splitlines()]
    st = {r["trial_id"]: r for r in recs if r.get("event") == "start"}
    fin = {r["trial_id"]: r for r in recs if r.get("event") == "finish"}
    rows = []
    for t in sorted(st):
        j5 = st[t]["j5"]
        a0, a1 = angle(fetch(folder, t)), angle(fetch(folder, f"{t}-putback"))
        twist = None if a0 is None or a1 is None else wrap(a1 - a0)
        rows.append((t, j5, abs(j5 - 90), fin.get(t, {}).get("outcome"), twist))
        print(f"{t}: J5 {j5} (|J5−90| {abs(j5 - 90)}), 결과 {fin.get(t, {}).get('outcome')}, 서보 {fin.get(t, {}).get('grip_read')}, 집기 전→도로 놓은 뒤 블록 각도 변화(비틀림) " + ("없음" if twist is None else f"{twist:+.1f}°"))
    done = [r for r in rows if r[3] in ("success", "failed")]
    print(f"시도 {len(done)}: 성공 {sum(1 for r in done if r[3] == 'success')}")
    for lo, hi, name in ((0, 10, "≤10°"), (10, 20, "10~20°"), (20, 40, "20~35°")):
        g = [r for r in done if lo <= r[2] < hi] if hi < 40 else [r for r in done if r[2] >= lo]
        tw = np.array([r[4] for r in g if r[4] is not None])
        print(f"  |J5−90| {name}: 시도 {len(g)}, 성공 {sum(1 for r in g if r[3] == 'success')}" + (f", 비틀림 평균 {tw.mean():+.1f}°, 표준편차 {tw.std():.1f}°, |비틀림| 최대 {abs(tw).max():.1f}°" if len(tw) else ""))


if __name__ == "__main__":
    main()
