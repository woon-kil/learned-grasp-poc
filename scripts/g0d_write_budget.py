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


# ---- 실측 보정 (advisor 3차 3번) ----
# G0b 실제 시도 시간(runs/G0b, 중앙값 35.1초)과 같은 시계 모형의 계산을 비교해 팔 호출당 숨은 지연(SSH 왕복, 계수 파일 fsync)을 구한다.
# 이 값은 모형 차이를 호출 수로 나눈 추정이다. 호출마다 같다고 가정한다. release_s 후보도 같은 방식이다.
import json as _json
import statistics as _statistics

from learned_grasp.g0_runner import run_replay
from learned_grasp.session import make_pick_observer


class CountingFake(FakeArm):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def write6(self, q5, grip, time_ms):
        self.calls += 1
        super().write6(q5, grip, time_ms)

    def write_grip(self, grip, time_ms):
        self.calls += 1
        super().write_grip(grip, time_ms)

    def read(self):
        self.calls += 1
        return super().read()


def model_g0b_trial_seconds():
    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    sheet = _json.loads((Path(ROOT) / "results" / "G0b" / "sheet-config6-seed1.json").read_text())
    clock = Clock()
    driver = CountingFake()
    arm = SafeArm(driver, SafetyGate(config, table), sleep=clock.sleep)
    camera = TimedCamera(clock)
    with tempfile.TemporaryDirectory() as tmp:
        ledger = AttemptLedger(Path(tmp) / "a.jsonl", clock=clock)
        run_replay(arm, camera, ledger, sheet, make_pick_observer(camera, log=lambda *a, **k: None), table, "stored", lambda trial: None)
        finished, _ = ledger.state()
    durations = [r["t"] - r["started_at"] for r in finished.values()]
    # 첫 시도 시작 전 park·개방과 적재 집기는 시도 시간에 들지 않는다. 시도당 호출 수는 시작 이후만 센다.
    per_trial_calls = (driver.calls - 1) / len(sheet)  # 시작 park 1회 제외 근사
    return _statistics.median(durations), per_trial_calls


ACTUAL_G0B_TRIAL_S = 35.08  # runs/G0b/replay-stored.jsonl 30건 중앙값
model_s, calls_per_trial = model_g0b_trial_seconds()
hidden = max(0.0, (ACTUAL_G0B_TRIAL_S - model_s) / calls_per_trial)
print("G0b 시도 시간: 실제 %.2f초, 모형 %.2f초, 시도당 팔 호출 %.1f회 → 호출당 숨은 지연 %.3f초" % (ACTUAL_G0B_TRIAL_S, model_s, calls_per_trial, hidden))


class CallRecording(Recording):
    """닫힘 구간마다 (모형 시간, 그 사이 팔 호출 수)를 기록한다."""

    def __init__(self, clock, calls):
        super().__init__(clock)
        self.calls_fn, self.pairs, self.mark = calls, [], 0

    def start(self):
        self.mark = self.calls_fn()
        super().start()

    def stop(self):
        if self.started_at is not None:
            self.pairs.append((self.clock() - self.started_at, self.calls_fn() - self.mark))
        super().stop()


def corrected_span(run, sheet, folder, hidden_per_call):
    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    ledger = AttemptLedger(folder / "a.jsonl")
    worst = (0.0, 0.0, 0)
    while True:
        clock = Clock()
        driver = CountingFake()
        watch = CallRecording(clock, lambda: driver.calls)
        camera = TimedCamera(clock)
        arm = SafeArm(driver, SafetyGate(config, table), sleep=clock.sleep, close_watch=watch)
        from learned_grasp.session import make_pick_observer as _observer

        ctx = G0dContext(
            arm=arm, camera=camera, ledger=ledger, table=table, observer=_observer(camera, log=lambda *a, **k: None), marker=TimedMarker(clock),
            placement=Placement(REF), ask=lambda kind, prompt, deadline: "y", load_block=lambda trial: None, release_s=1.0, log=lambda _: None,
        )
        result = run(ctx, sheet(config, table))
        for seconds, calls in watch.pairs:
            total = seconds + hidden_per_call * calls
            if total > worst[0]:
                worst = (total, seconds, calls)
        if not ctx.counters["batch_end"] or result is True:
            return worst


def release_candidate(hidden_per_call):
    """판정 실패 뒤 hover 이동과 개방. 후보값이고 실측이 아니다(플랜 125행)."""
    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    from learned_grasp.safety import HOVER, PARK_POSE, Pose

    worst = 0.0
    for r_index in range(len(table.entries)):
        clock = Clock()
        driver = CountingFake()
        arm = SafeArm(driver, SafetyGate(config, table), sleep=clock.sleep)
        arm.goto(PARK_POSE)
        arm.set_grip("open")
        arm.set_grip("close") if False else None
        before, calls0 = clock.now, driver.calls
        arm.goto(Pose(HOVER, r_index, 90, 90))
        arm.set_grip("open")
        worst = max(worst, (clock.now - before) + hidden_per_call * (driver.calls - calls0))
    return worst


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "s").mkdir(), (tmp / "t").mkdir()
    s_total, s_model, s_calls = corrected_span(run_natural_spread, spread_sheet, tmp / "s", hidden)
    t_total, t_model, t_calls = corrected_span(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), tmp / "t", hidden)
    print("닫힘 최대(보정): G0d-0 %.1f초(모형 %.1f초+호출 %d회), G0d 1차 %.1f초(모형 %.1f초+호출 %d회)" % (s_total, s_model, s_calls, t_total, t_model, t_calls))
print("release_s 후보(park에서 hover 이동+개방, 최대 r): %.1f초 — 후보이며 실측 전" % release_candidate(hidden))


def elapsed_at_failure_question(run, sheet, folder):
    """모든 집기가 실패 판정이 되는 경우, 닫은 뒤 사람에게 답을 묻는 순간까지의 모형 경과 시간(최대)."""
    config = load_config(Path(ROOT) / "config" / "workspace.yaml")
    table = table_for(config)
    clock = Clock()
    driver = CountingFake()
    watch = CallRecording(clock, lambda: driver.calls)
    camera = TimedCamera(clock)
    arm = SafeArm(driver, SafetyGate(config, table), sleep=clock.sleep, close_watch=watch)
    seen = []

    def ask(kind, prompt, deadline):
        if kind == "failure":
            seen.append(watch.clock() - watch.started_at)
            return "miss"
        return "y"

    ctx = G0dContext(
        arm=arm, camera=camera, ledger=AttemptLedger(folder / "a.jsonl"), table=table, observer=lambda t: Outcome(t["trial_id"].endswith("-w"), "각본"), marker=TimedMarker(clock),
        placement=Placement(REF), ask=ask, load_block=lambda trial: None, release_s=1.0, log=lambda _: None,
    )
    run(ctx, sheet(config, table))
    return max(seen)


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "f").mkdir()
    q = elapsed_at_failure_question(run_tolerance, lambda c, t: tolerance_sheet(c, t, 1), tmp / "f")
    print("판정 실패 시 사람에게 묻는 순간의 닫힘 경과(모형 최대): %.1f초. close_limit_s ≥ 이 값 + release_s + 사람 답 대기 시간 + 여유" % (q + hidden * 11))
