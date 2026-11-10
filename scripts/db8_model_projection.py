"""DB-8 사후 비교용: URDF FK로 DaBai 컬러 카메라 투영을 계산한다. 모델 값이다. 장치 없음.

사용: python3 scripts/db8_model_projection.py  (프로젝트 루트에서)
카메라 = Arm4_Link에 붙은 DaBai_DCW2_Joint. 광축 = 링크 +Y, 영상 아래 = 링크 −Z(DB-6 관측에 맞춘 부호), 영상 오른쪽 = 링크 +X.
블록 윗면 마커 중심을 세계 좌표(mm)로 받아 화소(열, 행)와 한 변 화소를 낸다.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learned_grasp import kinematics as K  # noqa: E402

root = ET.parse(ROOT / "assets" / "urdf" / "DOFBOT_Pro-V24.urdf").getroot()


def vec(s, d=(0, 0, 0)):
    return np.array([float(x) for x in s.split()]) if s else np.array(d, float)


joints = {}
for j in root.findall("joint"):
    o, ax = j.find("origin"), j.find("axis")
    joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                 xyz=vec(o.get("xyz")) if o is not None else np.zeros(3),
                                 rpy=vec(o.get("rpy")) if o is not None else np.zeros(3),
                                 axis=vec(ax.get("xyz"), (1, 0, 0)) if ax is not None else np.array([1.0, 0, 0]))
parent_joint = {j["child"]: (n, j) for n, j in joints.items()}


def link_pose(link, sdk):
    q = {"Arm1_Joint": sdk[0], "Arm2_Joint": sdk[1], "Arm3_Joint": sdk[2], "Arm4_Joint": sdk[3], "Arm5_Joint": sdk[4]}
    if link not in parent_joint:
        return np.eye(4)
    n, j = parent_joint[link]
    P = link_pose(j["parent"], sdk)
    M = np.eye(4); M[:3, :3] = K._rpy(j["rpy"]); M[:3, 3] = j["xyz"]
    Q = np.eye(4)
    if j["type"] != "fixed":
        Q[:3, :3] = K._axis_rotations(j["axis"], np.array([np.radians(q.get(n, 90.0) - 90.0)]))[0]
    return P @ M @ Q


def project(sdk, world_mm, intr):
    """world_mm: 세계 좌표(mm). 반환 (열, 행, 깊이 mm)."""
    T = link_pose("DaBai_DCW2_Link", sdk)
    Tw = T.copy(); Tw[:3, 3] *= 1000.0
    p = np.linalg.inv(Tw) @ np.r_[np.asarray(world_mm, float), 1.0]
    xl, yl, zl = p[:3]
    xc, yc, zc = xl, -zl, yl  # 광축 +Y, 영상 오른쪽 +X, 영상 아래 −Z
    fx, fy, cx, cy = intr
    return fx * xc / zc + cx, fy * yc / zc + cy, zc


if __name__ == "__main__":
    probe = json.loads((ROOT / "data/dabai/view/db7-hover/probe.json").read_text())
    k = probe["info"]["color"]["k"]
    intr = (k[0], k[4], k[2], k[5])
    sdk = np.array([90, 94, 6, 5, 90], float)  # r_index 0 hover
    for y in (130.0, 140.0):
        c0, r0, d0 = project(sdk, (0.0, y, 30.0), intr)
        print(f"y={y:.0f} 기준 중심 열 {c0:.1f} 행 {r0:.1f} 깊이 {d0:.1f}")
        for dx in (-10.0, 10.0):
            c, r, _ = project(sdk, (dx, y, 30.0), intr)
            print(f"  x{dx:+.0f} mm: Δ열 {c - c0:+.1f} Δ행 {r - r0:+.1f} -> {(c - c0) / dx:+.2f} 화소/mm")
        side = 25.0 * intr[0] / d0
        print(f"  마커 한 변(25 mm) {side:.1f} 화소, {side / 25.0:.2f} 화소/mm")
