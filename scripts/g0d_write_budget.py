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
    """쉼 묶음마다 새 SafeArm(새 세션)으로 끝까지 돈다. 전체 쓰기·닫기와 묶음당 최대 쓰기를 돌려준다."""
    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    driver = FakeArm()
    ledger = AttemptLedger(folder / "a.jsonl")
    per_batch, closes, batches = [], 0, 0
    while True:
        arm = SafeArm(driver, SafetyGate(config, table), sleep=lambda _: None, close_watch=CloseWatch(3600, on_expire=lambda _: None))
        ctx = G0dContext(
            arm=arm, camera=Camera(), ledger=ledger, table=table, observer=outcome, marker=Marker(),
            placement=Placement(REF), ask=lambda kind, prompt, deadline: answers.get(kind, "y"), load_block=lambda trial: None, release_s=1.0, log=lambda _: None,
        )
        before = len(driver.writes)
        result = run(ctx, sheet(config, table))
        per_batch.append(len(driver.writes) - before)
        closes += arm.close_count
        batches += 1
        if not ctx.counters["batch_end"] or result is True:
            return len(driver.writes), closes, batches, max(per_batch)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "s").mkdir(), (tmp / "t").mkdir(), (tmp / "w").mkdir()
    print("G0d-0 전부 성공: 쓰기 %d, 닫기 %d, 묶음 %d, 묶음당 최대 쓰기 %d" % count(run_natural_spread, spread_sheet, lambda t: Outcome(True), {}, tmp / "s"))
    print("G0d 1차 전부 성공(72회): 쓰기 %d, 닫기 %d, 묶음 %d, 묶음당 최대 쓰기 %d" % count(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), lambda t: Outcome(True), {}, tmp / "t"))

    # 상한: 변형마다 5회. 4번째 시도만 실패(미파지 답)해 재적재까지 거친다.
    def worst(trial):
        return Outcome(not trial["trial_id"].endswith("-t3"))

    print("G0d 1차 상한(90회, 각 변형 4번째 실패): 쓰기 %d, 닫기 %d, 묶음 %d, 묶음당 최대 쓰기 %d" % count(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), worst, {"failure": "miss"}, tmp / "w"))


# ---- 닫힘 시간 (advisor 재검토 4번) ----
# SafeArm sleep(명령 이동시간+settle)과 CloseWatch가 같은 가짜 시계를 쓴다. 카메라 지연은 아래 값을 더한다.
# 촬영 0.2초·가져오기 0.17~0.32초·검출 0.04초는 실측(PROJECT-STATUS, G0d 플랜 57행), diff·yellow 호출 0.05초는 가정이다.
LATENCY = {"snapshot": 0.2, "fetch": 0.32, "change": 0.05, "yellow": 0.05, "measure": 0.04}


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TimedCamera(Camera):
    def __init__(self, clock):
        super().__init__()
        self.clock_ = clock

    def snapshot(self, name):
        self.clock_.sleep(LATENCY["snapshot"])
        return super().snapshot(name)

    def fetch(self, name):
        self.clock_.sleep(LATENCY["fetch"])
        return Path(name)

    def change(self, before, after, roi):
        self.clock_.sleep(LATENCY["change"])
        return 1.0

    def yellow_pixels(self, name, roi):
        self.clock_.sleep(LATENCY["yellow"])
        return 0


class TimedMarker(Marker):
    def __init__(self, clock):
        self.clock_ = clock

    def measure(self, path):
        self.clock_.sleep(LATENCY["measure"])
        return super().measure(path)


class Recording(CloseWatch):
    def __init__(self, clock):
        super().__init__(10**6, on_expire=lambda _: None, clock=clock)
        self.spans = []

    def stop(self):
        if self.started_at is not None:
            self.spans.append(self.clock() - self.started_at)
        super().stop()


def closed_spans(run, sheet, outcome, folder):
    from learned_grasp.session import make_pick_observer

    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    ledger = AttemptLedger(folder / "a.jsonl")
    spans = []
    while True:
        clock = Clock()
        watch = Recording(clock)
        camera = TimedCamera(clock)
        arm = SafeArm(FakeArm(), SafetyGate(config, table), sleep=clock.sleep, close_watch=watch)
        observer = make_pick_observer(camera, log=lambda *a, **k: None) if outcome is None else outcome
        ctx = G0dContext(
            arm=arm, camera=camera, ledger=ledger, table=table, observer=observer, marker=TimedMarker(clock),
            placement=Placement(REF), ask=lambda kind, prompt, deadline: "y", load_block=lambda trial: None, release_s=1.0, log=lambda _: None,
        )
        result = run(ctx, sheet(config, table))
        spans += watch.spans
        if not ctx.counters["batch_end"] or result is True:
            return max(spans), len(spans)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "s").mkdir(), (tmp / "t").mkdir()
    print("닫힘 최대(가짜 시계, 사람 답 대기 제외): G0d-0 %.1f초(%d구간), G0d 1차 %.1f초(%d구간)" % (
        *closed_spans(run_natural_spread, spread_sheet, None, tmp / "s"),
        *closed_spans(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), None, tmp / "t"),
    ))
