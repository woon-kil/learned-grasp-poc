#!/usr/bin/env python3
"""H3 적합 입력을 만든다: runs/H3/frames의 5장씩에서 나사점 중점을 구한다. 장치 없음.

자세별 출처: 자동(h2_screw_points wide) 또는 「눈 지정 + 자동 하위 화소 정밀화」(MANUAL).
MANUAL은 자동 검출이 엉뚱한 쌍을 잡은 자세(확인 이미지로 눈으로 확인)에만 쓴다. 지정 위치는 적합 잔차가 아니라 확대 이미지에서 읽었다.
사용: uv run python scripts/h3_build_input.py runs/H3
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2_screw_points as H  # noqa: E402

# 자세 순서 → (왼쪽 나사, 오른쪽 나사) 눈으로 읽은 영상 위치. 4: r4 J1 70, 7: r9 J1 70.
MANUAL = {4: ((643, 80), (659, 84)), 7: ((634, 104), (650, 107))}


def main() -> None:
    folder = Path(sys.argv[1])
    data = json.loads((folder / "poses.json").read_text())
    out = []
    for e in data["poses"]:
        paths = sorted(glob.glob(str(folder / f"frames/{e['order']:02d}-r{e['r_index']}-j{e['j1']}-*.color.png")))
        if e["order"] in MANUAL:
            pairs = [H.refine_pair(p, *MANUAL[e["order"]]) for p in paths]
            source = "눈 지정 + 정밀화"
        else:
            pairs = [pr for pr in (H.find_pair_wide(p) for p in paths) if pr]
            source = "자동"
        mids = np.array([[(a[1] + b[1]) / 2, (a[2] + b[2]) / 2] for a, b in pairs]) if pairs else None
        entry = {"order": e["order"], "unique": e["unique"], "r_index": e["r_index"], "j1": e["j1"], "q": e["q"], "source": source, "detected": len(pairs)}
        if mids is not None:
            entry |= {"u": float(mids[:, 0].mean()), "v": float(mids[:, 1].mean()), "std": mids.std(0, ddof=1).round(3).tolist(),
                      "spacing": float(np.mean([b[1] - a[1] for a, b in pairs]))}
        out.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    (folder / "fit-input.json").write_text(json.dumps({"camera_info": data["camera_info"], "poses": [x for x in out if "u" in x]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
