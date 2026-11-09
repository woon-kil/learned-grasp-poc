"""구성에서 허용 자세 표를 만들어 저장한다. 장치를 쓰지 않는다.

사용: uv run python scripts/generate_pose_table.py [--config config/workspace.yaml] [--out config/pose_table.json]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from learned_grasp.config import load_config
from learned_grasp.pose_table import table_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/workspace.yaml"))
    parser.add_argument("--out", type=Path, default=Path("config/pose_table.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    table = table_for(config)
    table.save(args.out)
    print(f"구성 {config.config_version}, 항목 {len(table.entries)}개, r {table.entries[0].r_mm}~{table.entries[-1].r_mm} mm, sha256 {table.sha256()}")


if __name__ == "__main__":
    main()
