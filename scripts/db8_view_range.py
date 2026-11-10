"""DB-8 후속: r_index별 hover에서 블록 윗면 마커(변 25 mm, 중심 z 30 mm)가 영상 안(여백 포함)에 있는 앞뒤 y 범위를 모델로 계산한다. 장치 없음.

모델 절대 위치의 알려진 어긋남(열 +59.9, 행 −7.1 화소, DB-7 측정 대비)을 r_index 0에서 구해 모두에 더한다(가정).
사용법: uv run python scripts/db8_view_range.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import db8_model_projection as m  # noqa: E402
from learned_grasp.config import load_config  # noqa: E402
from learned_grasp.pose_table import table_for  # noqa: E402

k = json.loads((ROOT / "data/dabai/view/db7-hover/probe.json").read_text())["info"]["color"]["k"]
intr = (k[0], k[4], k[2], k[5])
W, H = 640, 480
OFF = (59.9, -7.1)  # (열, 행) 모델에 더하는 값. DB-7 측정(378.0, 411.8) − 모델(318.1, 418.9)
tab = table_for(load_config(ROOT / "config" / "workspace.yaml"))


def corners(sdk, x, y, half):
    out = []
    for dx, dz in ((-half, 0), (half, 0)):
        pass
    pts = [(x - half, y), (x + half, y)]  # 좌우 변
    rows = []
    cols = []
    for sx in (-half, half):
        for sy in (-half, half):
            c, r, _ = m.project(sdk, (x + sx, y + sy, 30.0), intr)
            cols.append(c + OFF[0]); rows.append(r + OFF[1])
    return min(cols), max(cols), min(rows), max(rows)


def window(sdk, half, margin):
    ok = []
    for y in np.arange(60, 220, 1.0):
        c0, c1, r0, r1 = corners(sdk, 0.0, y, half)
        if c0 >= margin and c1 <= W - margin and r0 >= margin and r1 <= H - margin:
            ok.append(y)
    return (min(ok), max(ok)) if ok else None


for idx, e in enumerate(tab.entries[:6]):
    sdk = np.array((90,) + tuple(e.hover) + (90,), float)
    row = f"r_index {idx} hover {tuple(e.hover)}:"
    for size in (20, 25, 30):
        for margin in (5, 20):
            w = window(sdk, size / 2, margin)
            row += f" [{size}mm 여백{margin}: " + (f"y {w[0]:.0f}~{w[1]:.0f}" if w else "없음") + "]"
    print(row)
