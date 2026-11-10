"""H5: 정수 관절 이웃 탐색(L2)이 모델상 위치 오차를 줄이는가. 오프라인, 장치 없음, 모델 FK 값이다.

기준안 = 목표를 연속 IK(J1 = atan2)로 구해 J1 반올림, r은 표에서 가장 가까운 행.
탐색안 = 허용 J1(구성 범위) × 파지 후보 정수 (J2,J3,J4)(표 생성과 같은 높이·기울기 조건) 전부에서
         접선(횡방향) 오차² + w_r × 반경 오차²가 최소인 조합. 반경 방향은 손가락이 지나도 블록이 남아 여유가 크다(HW7)고 보고 w_r < 1.
사용법: uv run python scripts/h5_integer_neighbor.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from learned_grasp import pose_table as P  # noqa: E402
from learned_grasp.config import load_config  # noqa: E402
from learned_grasp.kinematics import fk_batch  # noqa: E402

cfg = load_config(ROOT / "config" / "workspace.yaml")
spec = cfg.table
tab = P.table_for(cfg)
grid, pos0, tilt0 = P._grid()
mask = (np.abs(pos0[:, 2] - spec.z_grasp_mm) <= spec.z_tol_mm) & (np.abs(tilt0 - spec.tilt_deg) <= spec.tilt_tol_deg) \
    & (pos0[:, 1] >= spec.r_min_mm - 5) & (pos0[:, 1] <= spec.r_max_mm + 5)
cand = grid[mask]
print(f"파지 후보 (J2,J3,J4) {len(cand)}개, r {pos0[mask,1].min():.1f}~{pos0[mask,1].max():.1f} mm")
j1_lo, j1_hi = cfg.j1_range
j1s = np.arange(j1_lo, j1_hi + 1)

# 후보마다, J1마다 TCP (x, y)
Q = np.array([[j1, *c, 90] for j1 in j1s for c in cand], float)
pos, _ = fk_batch(Q)
X = pos[:, 0].reshape(len(j1s), len(cand))
Y = pos[:, 1].reshape(len(j1s), len(cand))
Jarr = j1s[:, None] * np.ones((1, len(cand)))

rng = np.random.default_rng(0)
N = 3000
rs = rng.uniform(spec.r_min_mm, spec.r_max_mm, N)
ang = np.radians(rng.uniform(-25, 25, N))  # J1 90 기준 ±25° 안(구성 J1 범위 ±20°를 넘지 않게 아래서 거른다)
tx_all = rs * np.sin(ang)
ty_all = rs * np.cos(ang)
# 목표 TCP는 J1=90 기준 좌표(x 옆, y 앞)로 둔다. J1 부호는 fk_batch가 정한 대로 쓴다.
keep = np.abs(np.degrees(ang)) <= 18
tx, ty = tx_all[keep], ty_all[keep]


def errors(sel_x, sel_y, tgx, tgy):
    ex, ey = sel_x - tgx, sel_y - tgy
    nr = np.hypot(tgx, tgy)
    ux, uy = tgx / nr, tgy / nr            # 목표 방향(반경)
    rad = ex * ux + ey * uy
    tan = -ex * uy + ey * ux               # 접선(횡방향)
    return tan, rad


# 기준안: 연속 IK → J1 반올림, 표에서 r이 가장 가까운 행
rows = tab.entries
row_r = np.array([e.r_mm for e in rows])
row_grasp = np.array([e.grasp for e in rows])
base_x, base_y = [], []
for gx, gy in zip(tx, ty):
    r = np.hypot(gx, gy)
    i = int(np.argmin(np.abs(row_r - r)))
    # J1 후보 중 목표 방위각에 가장 가까운 정수 도(연속 해 반올림과 같다고 보고 방위각을 계산)
    p, _ = fk_batch(np.array([[90 + d, *row_grasp[i], 90] for d in (-1.0,)]))  # 부호 확인용
    base_x.append(None); base_y.append(None)
# 부호는 위 호출 대신 단위 점검으로 정한다
pp, _ = fk_batch(np.array([[91, *row_grasp[len(rows) // 2], 90], [90, *row_grasp[len(rows) // 2], 90]]))
sign = np.sign(pp[0, 0] - pp[1, 0]) or 1.0  # J1 +1°가 x를 +로 움직이면 +1
print(f"J1 +1° 이동의 x 방향 부호 {sign:+.0f}")

bx, by = [], []
for gx, gy in zip(tx, ty):
    r = np.hypot(gx, gy)
    i = int(np.argmin(np.abs(row_r - r)))
    az = np.degrees(np.arctan2(gx, gy)) * sign          # 필요한 J1 변화(도)
    j1 = 90 + int(np.round(az))
    j1 = min(max(j1, j1_lo), j1_hi)
    p, _ = fk_batch(np.array([[j1, *row_grasp[i], 90]]))
    bx.append(p[0, 0]); by.append(p[0, 1])
bx, by = np.array(bx), np.array(by)
bt, br = errors(bx, by, tx, ty)

W_R = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
nx, ny = [], []
for gx, gy in zip(tx, ty):
    ex, ey = X - gx, Y - gy
    nr = np.hypot(gx, gy); ux, uy = gx / nr, gy / nr
    rad = ex * ux + ey * uy; tan = -ex * uy + ey * ux
    cost = tan ** 2 + W_R * rad ** 2
    k = np.unravel_index(int(np.argmin(cost)), cost.shape)
    nx.append(X[k]); ny.append(Y[k])
nt, nrad = errors(np.array(nx), np.array(ny), tx, ty)


def stat(name, a):
    a = np.abs(a)
    print(f"  {name}: 중앙 {np.median(a):.2f}  p95 {np.percentile(a, 95):.2f}  최대 {a.max():.2f} mm | >1.5 mm {np.mean(a>1.5)*100:.1f}%  >3.5 mm {np.mean(a>3.5)*100:.1f}%")


print(f"목표 {len(tx)}점 (r {spec.r_min_mm}~{spec.r_max_mm} mm, 방위 ±18°), 반경 가중 w_r={W_R}")
print("기준안(J1 반올림 + 표 행):"); stat("접선(횡방향)", bt); stat("반경", br)
print("탐색안(J1 × J2·J3·J4 전수):"); stat("접선(횡방향)", nt); stat("반경", nrad)
