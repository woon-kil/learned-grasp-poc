#!/usr/bin/env python3
"""연쇄 4개 전이에서 「집을 때 어긋남」이 「놓인 뒤 방향 잔차」로 이어지는지 본다. 장치 없음, 쓰기 0.

δ_pick = 집기 전 블록 d − 손 yaw 모델(J5 90, 집는 자리)  (집을 때 손과 블록의 어긋남 추정)
ρ_place = 놓인 뒤 블록 d − 손 yaw 모델(J5 90, 놓은 자리)  (놓은 뒤 잔차)
턱이 블록을 손 방향으로 완전히 돌려 쥐면 ρ_place는 δ_pick과 무관하고(놓는 순간의 흔들림), 쥔 각도를 보존하면 ρ_place ≈ δ_pick이다.
사용: uv run python scripts/j5_carryover.py
"""
import cv2
import numpy as np

from learned_grasp.config import load_config
from learned_grasp.l1 import L1Map, YawModel, block_from_image, signed_diff
from learned_grasp.pose_table import generate


def main() -> None:
    rows = [e.r_mm for e in generate(load_config("config/workspace.yaml").table).entries]
    l1, yaw = L1Map(), YawModel()

    def d_of(path, j1, r_index):
        m = block_from_image(cv2.imread(path), board_ref=(691.0, 291.0))
        return signed_diff(m["block_angle_deg"], l1.radial_angle_deg(j1, rows[r_index]) % 180.0)

    # (이름, 집기 전 사진, 집은 자리(r_index, J1), 놓인 뒤 사진, 놓은 자리)
    transitions = [
        ("A CH-0 1→2", "runs/CH-0/frames/chain-001.color.png", (5, 78), "runs/CH-0/frames/chain-002.color.png", (8, 104)),
        ("B CH-0 2→CH-1 1", "runs/CH-0/frames/chain-002.color.png", (8, 103), "runs/CH-1/frames/chain-001.color.png", (4, 75)),
        ("C CH-1 1→2", "runs/CH-1/frames/chain-001.color.png", (5, 75), "runs/CH-1/frames/chain-002.color.png", (8, 104)),
        ("D CH-1 2→3", "runs/CH-1/frames/chain-002-r1.color.png", (7, 103), "runs/CH-1/frames/chain-003.color.png", (4, 75)),
    ]
    xs, ys = [], []
    for name, pre, (rp, jp), post, (rq, jq) in transitions:
        d_pre, d_post = d_of(pre, jp, rp), d_of(post, jq, rq)
        delta = d_pre - yaw.hand_d(90, jp, rp)
        rho = d_post - yaw.hand_d(90, jq, rq)
        xs.append(delta)
        ys.append(rho)
        print(f"{name}: 집기 전 d {d_pre:+.1f}°(자리 {rp},{jp}), δ_pick {delta:+.1f}° → 놓인 뒤 d {d_post:+.1f}°(자리 {rq},{jq}), ρ_place {rho:+.1f}°")
    xs, ys = np.array(xs), np.array(ys)
    print(f"δ_pick 표준편차 {xs.std():.1f}°, ρ_place 표준편차 {ys.std():.1f}°, 상관 {np.corrcoef(xs, ys)[0, 1]:+.2f}(표본 {len(xs)}개)")


if __name__ == "__main__":
    main()
