#!/usr/bin/env python3
"""G0d-0과 G0d 1차의 쓰기 수를 Fake로 센다. 승인 max_writes를 정할 때 쓴다. 장치를 쓰지 않는다.

uv run python scripts/g0d_write_budget.py
"""
import tempfile
from pathlib import Path

from learned_grasp.arm import CloseWatch, FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import FakeCamera
from learned_grasp.config import ROOT, load_config
from learned_grasp.g0_runner import Outcome, tolerance_sheet
from learned_grasp.pose_table import table_for
from learned_grasp.safety import SafetyGate
from learned_grasp.tolerance import G0dContext, run_natural_spread, run_tolerance, spread_sheet
from learned_grasp.top_marker import Measurement, Placement

REF = {0: (915.0, 654.0), 1: (851.0, 211.0)}


class Camera(FakeCamera):
    def fetch(self, name):
        return Path(name)


class Marker:
    def measure(self, path):
        return Measurement(str(path), "x" * 64, dict(REF), True, True, (0.0, 0.0), 0.0)


def count(run, sheet, outcome, answers, folder):
    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    driver = FakeArm()
    arm = SafeArm(driver, SafetyGate(config, table), sleep=lambda _: None, close_watch=CloseWatch(3600, on_expire=lambda _: None))
    ctx = G0dContext(
        arm=arm, camera=Camera(), ledger=AttemptLedger(folder / "a.jsonl"), table=table, observer=outcome, marker=Marker(),
        placement=Placement(REF), ask=lambda kind, prompt, deadline: answers.get(kind, "y"), load_block=lambda trial: None, release_s=1.0, log=lambda _: None,
    )
    run(ctx, sheet(config, table))
    closes = sum(1 for (_, grip, _), (_, prev, _) in zip(driver.writes[1:], driver.writes) if grip != prev and grip == config.close_sdk)
    return len(driver.writes), closes


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "s").mkdir(), (tmp / "t").mkdir(), (tmp / "w").mkdir()
    print("G0d-0 전부 성공: 쓰기 %d, 닫기 %d" % count(run_natural_spread, spread_sheet, lambda t: Outcome(True), {}, tmp / "s"))
    print("G0d 1차 전부 성공(72회): 쓰기 %d, 닫기 %d" % count(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), lambda t: Outcome(True), {}, tmp / "t"))

    # 상한: 변형마다 5회. 4번째 시도만 실패(미파지 답)해 재적재까지 거친다.
    def worst(trial):
        return Outcome(not trial["trial_id"].endswith("-t3"))

    print("G0d 1차 상한(90회, 각 변형 4번째 실패): 쓰기 %d, 닫기 %d" % count(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), worst, {"failure": "miss"}, tmp / "w"))
