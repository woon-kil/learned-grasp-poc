#!/usr/bin/env python3
"""P2 남은 목표(p2-11 이후)의 블록 윗면 사각형이 로봇 받침대(영상 y 약 190)와 겹치지 않게 r를 바깥으로 옮긴다. 장치 없음.
사용자 지적(2026-11-11): p2-11 SVG가 받침대에 걸쳐 있었다. 사용: uv run python scripts/p2_targets_clear.py config/p2_targets.json p2-11
"""
import json
import math
import sys
from pathlib import Path

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, YawModel
from learned_grasp.pose_table import generate

PEDESTAL_Y = 195.0  # 받침대 아래 가장자리(약 190)와 흡착판(약 197) 사이, 윗면 사각형의 위 끝이 이보다 아래여야 한다
BOARD = (691.25, 291.5)
J1_MAX_TARGET = 102.0  # 사람의 위치 오차 ±10 mm(J1 약 ±4°)가 허용 한계 110을 넘지 않게
L, W = 60.33 * 1.14 / 2, 29.89 * 1.14 / 2


def top_y(cy: float, angle_deg: float) -> float:
    a = math.radians(angle_deg)
    return cy - (L * abs(math.sin(a)) + W * abs(math.cos(a)))


def main() -> None:
    path, first = Path(sys.argv[1]), sys.argv[2]
    data = json.loads(path.read_text())
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    l1, yaw = L1Map(), YawModel()
    changed = []
    for t in data["targets"]:
        if t["id"] < first:
            continue
        if t["j1"] > J1_MAX_TARGET:  # 오른쪽 끝(J1 110 허용 한계, 종이띠 가장자리) 바로 앞이면 사람의 오차가 범위 밖 거절로 이어진다
            t["j1"] = J1_MAX_TARGET
        r = t["r_mm"]
        while True:
            du, dv = l1.forward(t["j1"], r, t["target_j5"])
            rad = l1.radial_angle_deg(t["j1"], r) % 180.0
            r_idx = min(range(len(rows)), key=lambda i: abs(rows[i] - r))
            d = yaw.hand_d(t["target_j5"], round(t["j1"]), r_idx)
            ang = (rad + d) % 180.0
            if top_y(BOARD[1] + dv, ang) >= PEDESTAL_Y or r >= 160.0:
                break
            r += 0.5
        if abs(r - t["r_mm"]) > 1e-6 or True:
            changed.append((t["id"], t["r_mm"], round(r, 1)))
            t.update({"r_mm": round(r, 2), "r_index": r_idx, "target_d_deg": round(d, 1), "radial_angle_deg": round(rad, 1), "block_angle_deg": round(ang, 1), "du": round(float(du), 1), "dv": round(float(dv), 1)})
    data["note"] = f"{first} 이후 목표는 받침대 겹침을 피하도록 r를 바깥으로 옮김(scripts/p2_targets_clear.py): {changed}"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print("옮긴 목표(id, r 전 → 후):", changed)


if __name__ == "__main__":
    main()
