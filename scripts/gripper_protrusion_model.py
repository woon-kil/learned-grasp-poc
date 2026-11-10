"""URDF(grip_joint, 평행 링크 mimic)로 손끝 돌출과 간격을 grip_joint 각도 전체 범위에서 계산한다. 모델 값이다. 장치 없음.

사용법: uv run python scripts/gripper_protrusion_model.py <메시 디렉터리>
손끝 돌출 = Arm5_Link 좌표에서 접근축(+Y) 방향 최대 정점 y의 변화. 간격 = 두 손가락 안쪽 면의 x 차(손끝 부근).
SDK 각도와 grip_joint의 대응은 이 계산에 없다(미확인). 간격으로 대응을 본다.
"""
import re
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MESH = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "src"))
from learned_grasp import kinematics as K  # noqa: E402

root = ET.parse(ROOT / "assets/urdf/DOFBOT_Pro-V24.urdf").getroot()


def vec(s, d=(0, 0, 0)):
    return np.array([float(x) for x in s.split()]) if s else np.array(d, float)


def stl(path):
    b = Path(path).read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:300]:
        return np.array([[float(x) for x in m.groups()] for m in re.finditer(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", b)])
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


links = {}
for l in root.findall("link"):
    v = l.find("visual")
    if v is None:
        continue
    m, o = v.find("geometry/mesh"), v.find("origin")
    if m is not None:
        links[l.get("name")] = dict(file=Path(m.get("filename")).name, scale=vec(m.get("scale"), (1, 1, 1)),
                                    xyz=vec(o.get("xyz")) if o is not None else np.zeros(3),
                                    rpy=vec(o.get("rpy")) if o is not None else np.zeros(3))
joints = {}
for j in root.findall("joint"):
    o, a, mi = j.find("origin"), j.find("axis"), j.find("mimic")
    joints[j.find("child").get("link")] = dict(name=j.get("name"), parent=j.find("parent").get("link"), type=j.get("type"),
                                  xyz=vec(o.get("xyz")), rpy=vec(o.get("rpy")), axis=vec(a.get("xyz"), (0, 0, 1)) if a is not None else np.zeros(3),
                                  mimic=float(mi.get("multiplier")) if mi is not None else None, q=None)


def rz(rpy):
    return K._rpy(rpy)


def pose_in_arm5(link, q):
    """Arm5_Link 좌표에서 link의 변환(4×4)."""
    if link == "Arm5_Link":
        return np.eye(4)
    j = joints[link]
    P = pose_in_arm5(j["parent"], q)
    M = np.eye(4); M[:3, :3] = rz(j["rpy"]); M[:3, 3] = j["xyz"]
    R = np.eye(3)
    if j["type"] != "fixed":
        ang = q if j["name"] == "grip_joint" else (j["mimic"] * q if j["mimic"] is not None else 0.0)
        R = K._axis_rotations(j["axis"], np.array([ang]))[0]
    Q = np.eye(4); Q[:3, :3] = R
    return P @ M @ Q


def verts(link, q):
    d = links[link]
    V = stl(MESH / d["file"]) * d["scale"]
    Mv = np.eye(4); Mv[:3, :3] = rz(d["rpy"]); Mv[:3, 3] = d["xyz"]
    T = pose_in_arm5(link, q) @ Mv
    return (T @ np.c_[V, np.ones(len(V))].T).T[:, :3] * 1000.0  # mm


print("grip_joint(rad)  손끝 최대 y(mm)  돌출(열림 대비)  손끝 부근 간격(mm)")
base_tip = None
rows = []
for q in np.linspace(0.0, -1.6, 17):
    R = verts("rlink3", q)
    Lf = verts("llink3", q)
    tip = max(R[:, 1].max(), Lf[:, 1].max())
    near = lambda V: V[V[:, 1] >= V[:, 1].max() - 8.0]
    gap = near(R)[:, 0].min() - near(Lf)[:, 0].max()
    rows.append((q, tip, gap))
tip0 = rows[0][1]
for q, tip, gap in rows:
    print(f"{q:+.2f}          {tip:8.1f}        {tip - tip0:+7.1f}        {gap:7.1f}")
