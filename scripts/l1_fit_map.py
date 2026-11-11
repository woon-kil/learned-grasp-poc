#!/usr/bin/env python3
"""L1 맵을 맞춘다: 블록의 영상 위치 (du, dv) = 2차 다항식(J1, r) + J5 선형항. 장치 없음.

자료: scripts/g0_block_map.py가 만든 block-map.json(G0b·G0c 셀 평균 10 + G0b-9 교정 사진 10). 위치는 보드 마커 id 1 대비 화소.
출력: config/l1_map.json (계수, 훈련 점, 교차 검증, 입력 해시).
사용: uv run python scripts/l1_fit_map.py runs/G0b/block-map.json config/l1_map.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from learned_grasp.config import load_config
from learned_grasp.pose_table import table_for


def features(a, b, j5):
    return np.array([1.0, a, b, a * a, a * b, b * b, (j5 - 90.0)])


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [r for r in json.loads(src.read_text()) if "du" in r]
    table = table_for(load_config("config/workspace.yaml"))
    cells = {}
    for r in rows:
        if r["src"] in ("G0b", "G0c"):
            cells.setdefault(r["cell"], []).append(r)
    points = []
    for c in sorted(cells):
        v = cells[c]
        points.append({"id": c, "r_index": v[0]["r_index"], "j1": v[0]["j1"], "j5": v[0]["j5"], "du": float(np.mean([x["du"] for x in v])), "dv": float(np.mean([x["dv"] for x in v])), "n": len(v)})
    for r in rows:
        if r["src"] == "G0b-9":
            points.append({"id": r["cell"], "r_index": r["r_index"], "j1": r["j1"], "j5": r["j5"], "du": r["du"], "dv": r["dv"], "n": 1})
    for p in points:
        p["r_mm"] = table.entries[p["r_index"]].r_mm
    X = np.array([features(p["j1"] - 90.0, p["r_mm"] - 140.0, p["j5"]) for p in points])
    U = np.array([[p["du"], p["dv"]] for p in points])
    W, *_ = np.linalg.lstsq(X, U, rcond=None)
    loo = []
    for i in range(len(points)):
        keep = [k for k in range(len(points)) if k != i]
        Wi, *_ = np.linalg.lstsq(X[keep], U[keep], rcond=None)
        loo.append(X[i] @ Wi - U[i])
    loo = np.array(loo)
    fit = np.linalg.norm(X @ W - U, axis=1)
    result = {
        "model": "du,dv = c0 + c1*a + c2*b + c3*a^2 + c4*a*b + c5*b^2 + c6*(J5-90), a=J1-90, b=r_mm-140",
        "coef_du": W[:, 0].round(6).tolist(), "coef_dv": W[:, 1].round(6).tolist(),
        "source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "points": points,
        "fit_rms_px": round(float(np.sqrt((fit ** 2).mean())), 3),
        "loo_rms_px": {"du": round(float(np.sqrt((loo[:, 0] ** 2).mean())), 3), "dv": round(float(np.sqrt((loo[:, 1] ** 2).mean())), 3),
                       "2d": round(float(np.sqrt((np.linalg.norm(loo, axis=1) ** 2).mean())), 3)},
        "domain": {"j1": [70, 110], "r_mm": [round(table.entries[0].r_mm, 1), round(table.entries[-1].r_mm, 1)], "board_marker_id1": [693.0, 294.5],
                   "note": "du, dv는 보드 마커 id 1 중심 대비 화소. 훈련 때 세션별 마커 중앙값은 (693.0, 294.5), (693.0, 294.0), (693.0, 294.9)"},
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps({k: result[k] for k in ("fit_rms_px", "loo_rms_px")}, ensure_ascii=False), f"{len(points)}점 -> {out}")


if __name__ == "__main__":
    main()
