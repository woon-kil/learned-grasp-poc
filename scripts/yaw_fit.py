#!/usr/bin/env python3
"""손 yaw 모델을 맞춘다: 부호 있는 d(블록 − 반경선) = a + k·(J5−90) + b·(J1−90) + c·(r_index−4.5). 장치 없음.

자료: scripts/g0_j5_yaw.py가 만든 runs/G0b/j5-yaw.json(G0b·G0c·G0b-9 70장 = 20 포즈). 포즈 단위 LOO 표준편차를 함께 남긴다.
사용: uv run python scripts/yaw_fit.py runs/G0b/j5-yaw.json config/l1_yaw.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = json.loads(src.read_text())
    X = np.array([[1, r["j5"] - 90, r["j1"] - 90, r["r_index"] - 4.5] for r in rows], float)
    y = np.array([r["d"] for r in rows])
    g = np.array([r["cell"] for r in rows])
    W, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = np.zeros(len(y))
    for p in sorted(set(g)):
        tr, te = g != p, g == p
        Wp, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        res[te] = X[te] @ Wp - y[te]
    out.write_text(json.dumps({
        "model": "d = a + k*(J5-90) + b*(J1-90) + c*(r_index-4.5), d = 블록 긴 변 영상 각도 - 반경 방향 영상 각도(부호 있음, 도)",
        "a": round(float(W[0]), 4), "k": round(float(W[1]), 4), "b": round(float(W[2]), 4), "c": round(float(W[3]), 4),
        "photos": len(rows), "poses": len(set(g)), "j5_90_poses": len({r["cell"] for r in rows if r["j5"] == 90}),
        "lopo_std_deg": round(float(res.std()), 2), "lopo_p95_deg": round(float(np.percentile(abs(res), 95)), 2),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
    }, ensure_ascii=False, indent=1))
    print(out.read_text())


if __name__ == "__main__":
    main()
