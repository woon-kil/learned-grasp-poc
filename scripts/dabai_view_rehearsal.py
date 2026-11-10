#!/usr/bin/env python3
"""dabai-view 리허설. FakeArm이라 장치 쓰기가 없다. 실제 sleep·clock으로 자세당 소요 시간을 잰다.

사용: uv run python scripts/dabai_view_rehearsal.py runs/rehearsal-1/dabai_next
도착 줄이 나오면 그 사이에 프로브를 돌리고 신호 파일을 만든다.
"""
import sys
import time
from pathlib import Path

from learned_grasp.arm import FakeArm, SafeArm
from learned_grasp.config import ROOT, load_config
from learned_grasp.pose_table import table_for
from learned_grasp.safety import SafetyGate
from learned_grasp.session import run_dabai_view

signal = Path(sys.argv[1])
signal.parent.mkdir(parents=True, exist_ok=True)
config = load_config(Path(ROOT) / "config" / "workspace.yaml")
arm = SafeArm(FakeArm(), SafetyGate(config, table_for(config)))
start = time.monotonic()
run_dabai_view(arm, signal, log=lambda line, **k: print(f"{time.monotonic() - start:6.1f}s {line}", flush=True))
print(f"{time.monotonic() - start:6.1f}s 완료", flush=True)
