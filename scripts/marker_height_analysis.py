#!/usr/bin/env python3
"""보드 마커(id 1, 20 mm)와 블록 윗면 마커(id 7, 25 mm)의 PnP 높이 분석. 저장 영상만 쓴다. 장치·쓰기 없음.

고정 절차(2026-11-10 DB-4 사후 탐색에서 정함, 다음 시험에는 사전 고정으로 적용):
  1. 빨강 채널, DICT_APRILTAG_36h11, errorCorrectionRate=1.0(나머지 기본값)
  2. 내부 파라미터 K와 왜곡 d(8계수)는 같은 폴더 probe.json 값 그대로
  3. cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)로 마커마다 해 2개
  4. 두 마커의 법선 사이 각도가 가장 작은 해 쌍을 고른다(둘 다 같은 작업대 평면이라 법선이 같아야 한다)
  5. 그 각도가 MAX_NORMAL_DEG(10°) 이하일 때만 높이를 낸다. 넘으면 「높이 미확인」
  6. 높이 = |평균 법선 · (t7 − t1)| (mm)
사용: python scripts/marker_height_analysis.py data/dabai/view/db4-hover data/dabai/view/db4-grasp ...
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

SIZE_MM = {1: 20.0, 7: 25.0}
MAX_NORMAL_DEG = 10.0


def _obj(s):
    return np.array([[-s / 2, s / 2, 0], [s / 2, s / 2, 0], [s / 2, -s / 2, 0], [-s / 2, -s / 2, 0]], np.float32)


def analyze(folder: Path) -> dict:
    probe = json.loads((folder / "probe.json").read_text())
    K = np.array(probe["info"]["color"]["k"]).reshape(3, 3)
    D = np.array(probe["info"]["color"]["d"], float)
    img = cv2.imread(str(folder / "color.png"))
    params = aruco.DetectorParameters()
    params.errorCorrectionRate = 1.0
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11), params)
    corners, ids, _ = det.detectMarkers(img[:, :, 2])
    out = {"folder": folder.name, "markers": {}}
    sols = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if i not in SIZE_MM:
                continue
            pts = c.reshape(4, 2).astype(np.float32)
            side = float(np.mean([np.linalg.norm(pts[k] - pts[(k + 1) % 4]) for k in range(4)]))
            _, rvs, tvs, errs = cv2.solvePnPGeneric(_obj(SIZE_MM[i]), pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            errs = np.array(errs).ravel()
            sols[i] = [(cv2.Rodrigues(rv)[0][:, 2], tv.ravel(), float(errs[k])) for k, (rv, tv) in enumerate(zip(rvs, tvs))]
            out["markers"][i] = {"side_px": round(side, 1), "reproj_px_per_solution": [round(s[2], 2) for s in sols[i]]}
    if 1 in sols and 7 in sols:
        best = None
        for a, (n1, t1, _) in enumerate(sols[1]):
            for b, (n7, t7, _) in enumerate(sols[7]):
                ang = float(np.degrees(np.arccos(np.clip(abs(np.dot(n1, n7)), 0, 1))))
                if best is None or ang < best[0]:
                    best = (ang, a, b)
        ang, a, b = best
        n1, t1, _ = sols[1][a]
        n7, t7, _ = sols[7][b]
        avg = n1 * np.sign(-n1[2]) + n7 * np.sign(-n7[2])
        avg /= np.linalg.norm(avg)
        out["normal_angle_deg"] = round(ang, 1)
        out["Z_board_mm"] = round(float(t1[2]), 1)
        out["Z_block_mm"] = round(float(t7[2]), 1)
        out["height_mm"] = round(float(abs(np.dot(avg, t7 - t1))), 1) if ang <= MAX_NORMAL_DEG else None
        out["height_status"] = "ok" if ang <= MAX_NORMAL_DEG else "높이 미확인(법선 각도 초과)"
    else:
        out["height_status"] = "높이 미확인(마커 검출 부족)"
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(json.dumps(analyze(Path(arg)), ensure_ascii=False))
