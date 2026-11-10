from __future__ import annotations

from pathlib import Path

import pytest

from learned_grasp.arm import CloseWatch, FakeArm, SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import FakeCamera
from learned_grasp.g0_runner import Outcome, compute_tolerance, tolerance_sheet, tolerance_variants
from learned_grasp.safety import GRASP, HOVER, PARK_POSE, Pose, SafetyGate
from learned_grasp.tolerance import G0dContext, PlacementFilter, StopSession, run_tolerance, variant_states
from learned_grasp.top_marker import Measurement, MeasureError, Placement

REF = {0: (915.0, 654.0), 1: (851.0, 211.0)}


class Camera(FakeCamera):
    def fetch(self, name):
        return Path(name)


class Marker:
    def __init__(self):
        self.special: dict[str, object] = {}

    def measure(self, path):
        value = self.special.get(str(path))
        if isinstance(value, Exception):
            raise value
        if value is not None:
            return value
        return Measurement(str(path), "x" * 64, dict(REF), True, True, (0.0, 0.0), 0.0)


class World:
    """가짜 장치 묶음. ask와 판정 결과를 각본으로 준다. 사람 답 시점의 쓰기 수를 남긴다."""

    def __init__(self, ready_config, table, tmp_path, limit_s=60.0):
        self.driver = FakeArm()
        self.watch = CloseWatch(limit_s, on_expire=lambda text: None)
        self.arm = SafeArm(self.driver, SafetyGate(ready_config, table), sleep=lambda _: None, close_watch=self.watch)
        ticks = iter(range(10**6))
        self.ledger = AttemptLedger(tmp_path / "attempts.jsonl", clock=lambda: float(next(ticks)))
        self.camera = Camera()
        self.marker = Marker()
        self.outcomes: dict[str, bool] = {}
        self.answers: dict[tuple[str, str], str | None] = {}
        self.asked: list[tuple[str, str, int, float | None]] = []
        self.loads = 0
        self.ctx = G0dContext(
            arm=self.arm, camera=self.camera, ledger=self.ledger, table=table, observer=self.observe, marker=self.marker,
            placement=Placement(REF), ask=self.ask, load_block=self.load, release_s=5.0, log=lambda text: None,
        )

    def observe(self, trial):
        return Outcome(self.outcomes.get(trial["trial_id"], True), "각본")

    def ask(self, kind, prompt, deadline):
        trial_id = prompt.split(" ")[0].rstrip(":")
        self.asked.append((kind, trial_id, len(self.driver.writes), deadline))
        return self.answers.get((kind, trial_id), "y" if kind in ("descent", "reload") else "miss")

    def load(self, trial):
        self.loads += 1

    def run(self, sheet):
        return run_tolerance(self.ctx, sheet)


@pytest.fixture
def sheet(ready_config, table):
    return tolerance_sheet(ready_config, table, 1)


def _q(arm, kind, trial, prefix=""):
    return arm.gate.joints(Pose(kind, trial[f"{prefix}r_index"], trial[f"{prefix}j1"], trial[f"{prefix}j5"]))


def test_all_success_passes_every_variant_with_four_trials(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    states = world.run(sheet)
    assert len(states) == 18 and all(s["status"] == "pass" and s["outcomes"] == [True] * 4 for s in states.values())
    finished, open_ids = world.ledger.state()
    assert len(finished) == 72 and not open_ids
    assert not any(t["trial_id"].endswith("-t4") for t in finished.values())
    result = compute_tolerance(tolerance_variants(ready_config, table), states, "verified")
    assert all(axis["usable"] for axis in result.values())
    assert world.loads == 1  # 복귀 성공이면 블록을 쥔 채 다음 시도로 간다


def test_return_cycle_command_order(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.run(sheet[:1])
    trial, arm = sheet[0], world.arm
    q = [w[0] for w in world.driver.writes]
    park = arm.gate.joints(PARK_POSE)
    expected = [
        park,  # 시작 park(이미 열려 있어 개방 쓰기 없음)
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # 적재 집기
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # P 놓기
        _q(arm, HOVER, trial, "pick_"), _q(arm, GRASP, trial, "pick_"), _q(arm, GRASP, trial, "pick_"), _q(arm, HOVER, trial, "pick_"), park,  # P+δ 집기
        _q(arm, HOVER, trial, "pick_"), _q(arm, GRASP, trial, "pick_"), _q(arm, GRASP, trial, "pick_"), _q(arm, HOVER, trial, "pick_"), park,  # P+δ 복귀 놓기
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # P 재집기
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # 끝 내려놓기
    ]
    assert q == expected
    kinds = [(k, tid) for k, tid, _, _ in world.asked]
    assert kinds == [("descent", trial["trial_id"]), ("descent", trial["trial_id"])]


def test_descent_stop_sends_no_grasp_before_answer(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.answers[("descent", trial["trial_id"])] = "n"
    world.run(sheet[:1])
    kind, _, writes_at_ask, _ = world.asked[0]
    assert kind == "descent"
    assert world.driver.writes[writes_at_ask - 1][0] == _q(world.arm, HOVER, trial, "pick_")
    after = [w[0] for w in world.driver.writes[writes_at_ask:]]
    assert _q(world.arm, GRASP, trial, "pick_") not in after
    finished, _ = world.ledger.state()
    assert finished[trial["trial_id"]]["outcome"] == "safety_unmeasured"
    assert variant_states(sheet, world.ledger.records())[trial["variant_id"]]["status"] == "incomplete"


def test_failure_branch_waits_with_zero_writes_then_opens_on_miss(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.outcomes[trial["trial_id"]] = False
    world.run(sheet[:1])
    failure = next(a for a in world.asked if a[0] == "failure")
    _, _, writes_at_ask, deadline = failure
    assert world.driver.writes[writes_at_ask - 1][0] == world.arm.gate.joints(PARK_POSE)  # 판정 뒤 쓰기 없음
    assert 0 < deadline <= 60.0 - 5.0
    tail = [w[0] for w in world.driver.writes[writes_at_ask:writes_at_ask + 3]]
    assert tail == [_q(world.arm, HOVER, trial, "pick_"), _q(world.arm, HOVER, trial, "pick_"), world.arm.gate.joints(PARK_POSE)]
    assert world.driver.writes[writes_at_ask + 1][1] == ready_config.open_sdk
    finished, _ = world.ledger.state()
    assert finished[trial["trial_id"]]["outcome"] == "failure"


@pytest.mark.parametrize("answer", ["held", "unknown", None])
def test_failure_branch_stops_without_writes_unless_miss(ready_config, table, tmp_path, sheet, answer):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.outcomes[trial["trial_id"]] = False
    world.answers[("failure", trial["trial_id"])] = answer
    with pytest.raises(StopSession):
        world.run(sheet)
    _, _, writes_at_ask, _ = next(a for a in world.asked if a[0] == "failure")
    assert len(world.driver.writes) == writes_at_ask
    finished, _ = world.ledger.state()
    assert finished[trial["trial_id"]]["outcome"] == "stopped"


def test_expired_answer_window_is_refused(ready_config, table, tmp_path, sheet):
    """정상 해제 시간을 빼면 남는 시간이 없으면 묻지 않고 멈춘다(늦은 답 거부와 같다)."""
    world = World(ready_config, table, tmp_path, limit_s=4.0)
    world.outcomes[sheet[0]["trial_id"]] = False
    with pytest.raises(StopSession):
        world.run(sheet)
    assert not [a for a in world.asked if a[0] == "failure"]


def test_two_failures_end_variant_and_skip_its_slots(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    target = sheet[0]["variant_id"]
    for slot in sheet:
        if slot["variant_id"] == target and slot["round"] in (0, 1):
            world.outcomes[slot["trial_id"]] = False
    states = world.run(sheet)
    assert states[target] == {"status": "fail", "outcomes": [False, False]}
    finished, _ = world.ledger.state()
    assert sorted(t for t in finished if t.startswith(target + "-")) == [f"{target}-t0", f"{target}-t1"]


def test_placement_failure_waits_for_reload_without_consuming_id(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.marker.special[f"{trial['trial_id']}-p1"] = MeasureError("top_missing")
    world.run(sheet[:1])
    reload = next(a for a in world.asked if a[0] == "reload")
    assert world.driver.writes[reload[2] - 1][0] == world.arm.gate.joints(PARK_POSE)
    descent = next(a for a in world.asked if a[0] == "descent")
    assert descent[2] == reload[2] + 1  # 재적재 답 뒤 첫 쓰기가 P+δ hover다
    finished, _ = world.ledger.state()
    assert finished[trial["trial_id"]]["outcome"] == "success"


def test_two_placement_failures_make_variant_incomplete_without_id(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    for k in (1, 2):
        world.marker.special[f"{trial['trial_id']}-p{k}"] = MeasureError("top_missing")
    world.run(sheet[:2])
    finished, _ = world.ledger.state()
    assert trial["trial_id"] not in finished
    assert variant_states(sheet, world.ledger.records())[trial["variant_id"]] == {"status": "incomplete", "reason": "배치 확인 2회 연속 실패"}
    assert not [a for a in world.asked if a[0] == "descent" and a[1] == trial["trial_id"]]


def test_placement_filter_rejects_far_block(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.ctx.placement_filter = PlacementFilter({0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0)}, {0: 3, 1: 3, 2: 3}, {0: 2, 1: 2, 2: 2})
    trial = sheet[0]
    world.marker.special[f"{trial['trial_id']}-p1"] = Measurement("p1", "x" * 64, dict(REF), True, True, (5.0, 0.0), 0.0)
    world.run(sheet[:1])
    assert [a[0] for a in world.asked][:2] == ["reload", "descent"]


def test_reference_move_stops_session_without_writes(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    moved = {0: (918.0, 654.0), 1: (851.0, 211.0)}
    world.marker.special[f"{trial['trial_id']}-p1"] = Measurement("p1", "x" * 64, moved, True, True, (0.0, 0.0), 0.0)
    with pytest.raises(StopSession, match="카메라 기준"):
        world.run(sheet)
    assert world.driver.writes[-1][0] == world.arm.gate.joints(PARK_POSE)
    assert not world.asked


def test_return_pick_failure_is_not_counted(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.outcomes[f"{trial['trial_id']}-ret"] = False
    world.run(sheet[:1])
    finished, _ = world.ledger.state()
    assert finished[trial["trial_id"]]["outcome"] == "success"
    assert world.ctx.counters["return_failures"] == 1
    assert [a[0] for a in world.asked] == ["descent", "descent", "failure"]


def test_return_repick_descent_refusal_stops_before_grasp(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    answers = iter(["y", "n"])
    world.ctx.ask = lambda kind, prompt, deadline: (world.asked.append((kind, "", len(world.driver.writes), deadline)), next(answers))[1]
    with pytest.raises(StopSession, match="복귀 재집기"):
        world.run(sheet)
    assert world.driver.writes[-1][0] == _q(world.arm, HOVER, trial)


def test_resume_skips_used_ids_and_marks_interrupted_incomplete(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.run(sheet[:2])
    world.ledger.start(sheet[2])  # 이전 세션이 이 시도 중에 끊겼다
    again = World(ready_config, table, tmp_path)
    states = again.run(sheet[:4])
    finished, _ = again.ledger.state()
    starts = [r["trial_id"] for r in again.ledger.records() if r["event"] == "start"]
    assert len(starts) == len(set(starts))
    assert finished[sheet[2]["trial_id"]]["outcome"] == "interrupted"
    assert states[sheet[2]["variant_id"]]["status"] == "incomplete"
    assert sheet[3]["trial_id"] in finished


def test_requires_close_watch(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.arm.close_watch = None
    with pytest.raises(ValueError):
        world.run(sheet)
