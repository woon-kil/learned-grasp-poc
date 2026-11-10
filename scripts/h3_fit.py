#!/usr/bin/env python3
"""H3: 고정 Gemini 영상의 나사점 위치로 강체 손-눈(모델 A)을 맞추고 교차 검증 잔차를 낸다. 장치 없음.

모델 A (모수 9): Gemini 카메라 자세(회전 rvec 3, 이동 t 3: 베이스 좌표 → 카메라 좌표) + 나사점의 Arm4_Link 좌표 s(3, mm).
투영: Arm4_Link FK(URDF, 모델 값, 명령 관절값)로 베이스 좌표를 얻고 카메라 좌표로 바꾼 뒤 cv2.projectPoints(내부 파라미터와 왜곡 포함).
모델 B (탐색용, 모수 13): A + 관절 J1~J4 상수 오프셋(도). 판정에는 A만 쓴다.
최적화: Levenberg–Marquardt(numpy, 수치 야코비안), 여러 초기값. 교차 검증은 고유 자세 하나를 빼고 맞춘 뒤 예측.

입력 JSON: {"camera_info": "경로", "poses": [{"id": "...", "unique": "고유키", "q": [J1,J2,J3,J4,J5], "u": 열, "v": 행}, ...]}
사용: uv run python scripts/h3_fit.py <입력.json>  |  uv run python scripts/h3_fit.py --selftest
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db8_model_projection as fk  # noqa: E402


def load_camera(path: str):
    info = json.loads(Path(path).read_text())["color"]
    return np.array(info["k"], float).reshape(3, 3), np.array(info["d"], float)


def arm4_points(q, s):
    """베이스 좌표(mm)의 나사점. s는 Arm4_Link 좌표(mm)."""
    T = fk.link_pose("Arm4_Link", np.asarray(q, float))
    return T[:3, :3] @ s + T[:3, 3] * 1000.0


def project(params, qs, K, D, joint_offsets=None):
    rvec, t, s = params[0:3], params[3:6], params[6:9]
    R = cv2.Rodrigues(rvec)[0]
    pts = []
    for q in qs:
        qq = np.asarray(q, float).copy()
        if joint_offsets is not None:
            qq[:4] += joint_offsets
        pts.append(R @ arm4_points(qq, s) + t)
    uv, _ = cv2.projectPoints(np.array(pts).reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(-1, 2)


def residuals(x, qs, uv, K, D, model):
    off = x[9:13] if model == "B" else None
    return (project(x[:9], qs, K, D, off) - uv).ravel()


def lm(fun, x0, iters=80):
    x, lam = x0.copy(), 1e-2
    r = fun(x)
    cost = float(r @ r)
    for it in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = 1e-5 * max(1.0, abs(x[j]))
            xp = x.copy()
            xp[j] += h
            J[:, j] = (fun(xp) - r) / h
        A, g = J.T @ J, J.T @ r
        improved = False
        for _ in range(12):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            rn = fun(x + step)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost, lam, improved = x + step, rn, cn, max(lam / 3, 1e-9), True
                break
            lam *= 4
        if not improved or np.linalg.norm(step) < 1e-9:
            break
    return x, cost, J


def initial_guesses(qs, uv, K, D, rng, n=8):
    s0 = np.array([0.0, 68.9, 70.0])  # DaBai 브래킷 근처 추정(URDF DaBai_DCW2_Joint xyz에서 위로)
    base = np.array([arm4_points(q, s0) for q in qs])
    starts = []
    ok, rv, tv = cv2.solvePnP(base.reshape(-1, 1, 3), uv.reshape(-1, 1, 2), K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if ok:
        starts.append(np.r_[rv.ravel(), tv.ravel(), s0])
    for _ in range(n):
        s = s0 + rng.normal(0, 20, 3)
        b = np.array([arm4_points(q, s) for q in qs])
        ok, rv, tv = cv2.solvePnP(b.reshape(-1, 1, 3), uv.reshape(-1, 1, 2), K, D, flags=cv2.SOLVEPNP_ITERATIVE,
                                  rvec=rng.normal(0, 0.3, (3, 1)), tvec=rng.normal(0, 200, (3, 1)), useExtrinsicGuess=True)
        if ok:
            starts.append(np.r_[rv.ravel(), tv.ravel(), s])
    return starts


def fit(qs, uv, K, D, model="A", seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for x0 in initial_guesses(qs, uv, K, D, rng):
        if model == "B":
            x0 = np.r_[x0, np.zeros(4)]
        x, cost, J = lm(lambda x: residuals(x, qs, uv, K, D, model), x0)
        if best is None or cost < best[1]:
            best = (x, cost, J)
    return best


def loo(qs, uv, uniq, K, D, model="A"):
    out = []
    for key in sorted(set(uniq)):
        keep = [i for i, u in enumerate(uniq) if u != key]
        test = [i for i, u in enumerate(uniq) if u == key]
        x, _, _ = fit([qs[i] for i in keep], uv[keep], K, D, model)
        off = x[9:13] if model == "B" else None
        pred = project(x[:9], [qs[i] for i in test], K, D, off)
        out.append((key, float(np.linalg.norm(pred - uv[test], axis=1).mean())))
    return out


def report(data, selftest=False):
    K, D = load_camera(data["camera_info"])
    poses = data["poses"]
    qs = [p["q"] for p in poses]
    uv = np.array([[p["u"], p["v"]] for p in poses], float)
    uniq = [p["unique"] for p in poses]
    out = {"points": len(poses), "unique_poses": len(set(uniq))}
    for model in ("A", "B"):
        x, cost, J = fit(qs, uv, K, D, model)
        fit_res = np.linalg.norm(residuals(x, qs, uv, K, D, model).reshape(-1, 2), axis=1)
        sv = np.linalg.svd(J, compute_uv=False)
        cv = loo(qs, uv, uniq, K, D, model)
        errs = np.array([e for _, e in cv])
        out[model] = {"fit_rms_px": round(float(np.sqrt((fit_res ** 2).mean())), 3),
                      "loo_rms_px": round(float(np.sqrt((errs ** 2).mean())), 3), "loo_p95_px": round(float(np.percentile(errs, 95)), 3),
                      "loo_per_pose": {k: round(e, 3) for k, e in cv}, "params": x.round(3).tolist(),
                      "jacobian_singular_values": sv.round(4).tolist(), "condition": float(sv[0] / max(sv[-1], 1e-12))}
    return out


def selftest():
    rng = np.random.default_rng(1)
    K = np.array([[610.5, 0, 641.8], [0, 610.3, 364.4], [0, 0, 1.0]])
    D = np.array([-0.0317, 0.0342, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    truth = np.r_[0.3, 2.6, -0.4, 40.0, -180.0, 520.0, 3.0, 69.0, 72.0]
    q_list = []
    for r in (0, 4, 9):
        for j1 in (70, 90, 110):
            from learned_grasp.config import load_config
            from learned_grasp.pose_table import table_for
            e = table_for(load_config("config/workspace.yaml")).entries[r]
            q_list.append([j1, *e.grasp, 90])
    uv = project(truth, q_list, K, D) + rng.normal(0, 0.05, (len(q_list), 2))
    data = {"camera_info": None}
    poses = [{"q": q, "u": float(a), "v": float(b), "unique": str(i)} for i, (q, (a, b)) in enumerate(zip(q_list, uv))]
    x, cost, _ = fit(q_list, uv, K, D, "A")
    err = np.linalg.norm(project(x[:9], q_list, K, D) - uv, axis=1)
    cv = loo(q_list, uv, [p["unique"] for p in poses], K, D, "A")
    print(json.dumps({"selftest": True, "fit_rms_px": round(float(np.sqrt((err ** 2).mean())), 4),
                      "loo_rms_px": round(float(np.sqrt(np.mean([e ** 2 for _, e in cv]))), 4)}))


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        print(json.dumps(report(json.loads(Path(sys.argv[1]).read_text())), ensure_ascii=False, indent=1))
