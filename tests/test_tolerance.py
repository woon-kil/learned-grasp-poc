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


class LimitedArm(FakeArm):
    """승인 max_writes처럼 상한에 닿으면 전송 전에 SafetyViolation을 낸다."""

    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def _check(self):
        if self.limit is not None and len(self.writes) >= self.limit:
            from learned_grasp.safety import SafetyViolation

            raise SafetyViolation("세션 쓰기 상한")

    def write6(self, q5, grip, time_ms):
        self._check()
        super().write6(q5, grip, time_ms)

    def write_grip(self, grip, time_ms):
        self._check()
        super().write_grip(grip, time_ms)


class World:
    """가짜 장치 묶음. ask와 판정 결과를 각본으로 준다. 사람 답 시점의 쓰기 수를 남긴다."""

    def __init__(self, ready_config, table, tmp_path, limit_s=60.0, write_limit=None):
        self.driver = LimitedArm(write_limit)
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
        self.batches = 0
        self.judged: list[tuple[str, int]] = []
        self.ctx = G0dContext(
            arm=self.arm, camera=self.camera, ledger=self.ledger, table=table, observer=self.observe, marker=self.marker,
            placement=Placement(REF), ask=self.ask, load_block=self.load, release_s=5.0, log=lambda text: None,
            writes_left=None if write_limit is None else (lambda: write_limit - len(self.driver.writes)),
        )

    def observe(self, trial):
        self.judged.append((trial["trial_id"], len(self.driver.writes)))
        return Outcome(self.outcomes.get(trial["trial_id"], True), "각본")

    def ask(self, kind, prompt, deadline):
        trial_id = prompt.split(" ")[0].rstrip(":")
        self.asked.append((kind, trial_id, len(self.driver.writes), deadline))
        return self.answers.get((kind, trial_id), "y" if kind in ("descent", "reload") else "miss")

    def load(self, trial):
        self.loads += 1

    def _new_batch(self):
        """쉼 묶음 뒤 새 세션과 같다. 닫기 계수는 새 SafeArm처럼 0에서 시작한다."""
        self.batches += 1
        self.arm.close_count = 0
        self.arm.pose = None
        self.ctx.counters["batch_end"] = False

    def run(self, sheet):
        for _ in range(300):  # 진행 없이 묶음만 되풀이되면 멈추지 않고 실패시킨다
            self._new_batch()
            states = run_tolerance(self.ctx, sheet)
            if not self.ctx.counters["batch_end"]:
                return states
        raise AssertionError("묶음이 300번 되풀이돼도 끝나지 않는다")

    def spread(self, sheet):
        from learned_grasp.tolerance import run_natural_spread

        for _ in range(300):
            self._new_batch()
            if run_natural_spread(self.ctx, sheet) or not self.ctx.counters["batch_end"]:
                return
        raise AssertionError("묶음이 300번 되풀이돼도 끝나지 않는다")


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
    assert world.loads == world.batches > 1  # 복귀 성공이면 블록을 쥔 채 다음 시도로 가고, 묶음마다 다시 적재한다


def test_return_cycle_command_order(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.run(sheet[:1])
    trial, arm = sheet[0], world.arm
    q = [w[0] for w in world.driver.writes]
    park = arm.gate.joints(PARK_POSE)
    expected = [
        park,  # 시작 park(이미 열려 있어 개방 쓰기 없음)
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # 적재 집기
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # 준비 순환 놓기
        _q(arm, HOVER, trial), _q(arm, GRASP, trial), _q(arm, GRASP, trial), _q(arm, HOVER, trial), park,  # 준비 순환 집기
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
    world.ctx.close_batch = 7
    world._new_batch()
    run_tolerance(world.ctx, sheet)
    done = [r["trial_id"] for r in world.ledger.records() if r["event"] == "finish"]
    nxt = next(s for s in sheet if s["trial_id"] not in done)
    world.ledger.start(nxt)  # 이전 세션이 이 시도 중에 끊겼다
    again = World(ready_config, table, tmp_path)
    again.ctx.close_batch = 7
    again._new_batch()
    states = run_tolerance(again.ctx, sheet)
    finished, _ = again.ledger.state()
    starts = [r["trial_id"] for r in again.ledger.records() if r["event"] == "start"]
    assert len(starts) == len(set(starts))
    assert finished[nxt["trial_id"]]["outcome"] == "interrupted"
    assert states[nxt["variant_id"]]["status"] == "incomplete"
    assert len(finished) > len(done) + 1


def test_requires_close_watch(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.arm.close_watch = None
    with pytest.raises(ValueError):
        world.run(sheet)


def test_spread_sheet_shape(ready_config, table):
    from learned_grasp.tolerance import spread_sheet

    sheet = spread_sheet(ready_config, table)
    assert len(sheet) == 42 and len({t["trial_id"] for t in sheet}) == 42
    assert [t["cell"] for t in sheet[:6]] == [0, 1, 2, 0, 1, 2]
    assert sum(t["path"] == "plain" for t in sheet) == 24 + 9 and sum(t["path"] == "return" for t in sheet) == 9
    assert {t["r_index"] for t in sheet} == {1, 4, 7}
    for t in sheet:
        same = (t["pick_r_index"], t["pick_j1"], t["pick_j5"]) == (t["r_index"], t["j1"], t["j5"])
        assert same is (t["path"] == "plain")


def _spread_world(ready_config, table, tmp_path):
    world = World(ready_config, table, tmp_path)
    world.camera.yellow_value = 0
    return world


def test_natural_spread_runs_and_summarizes(ready_config, table, tmp_path):
    from learned_grasp.tolerance import filter_from_spread, run_natural_spread, spread_sheet, summarize_spread

    world = _spread_world(ready_config, table, tmp_path)
    world.ctx.observer = lambda trial: Outcome(True, "변화 0.0500, 노란 화소 30")
    sheet = spread_sheet(ready_config, table)
    shift = iter(range(1000))
    def measure(path):
        k = next(shift)
        return Measurement(str(path), "x" * 64, dict(REF), True, True, (float(k % 3), 0.0), float(k % 2))
    world.marker.measure = measure
    world.spread(sheet)
    descents = [a for a in world.asked if a[0] == "descent"]
    assert len(descents) == 9 * 2  # 복귀 경로만 하강 전 확인(오프셋 집기와 복귀 재집기)
    assert all(c["base_n"] == 8 for c in summarize_spread(world.ledger.records())["cells"].values())
    summary = summarize_spread(world.ledger.records())
    assert summary["trials"] == 42 and summary["failures"] == 0
    assert summary["yellow_max_success"] == 30 and summary["yellow_ok"]
    assert set(summary["cells"]) == {0, 1, 2}
    assert all(c["return_n"] == 3 for c in summary["cells"].values())
    flt = filter_from_spread(summary)
    assert set(flt.center) == {0, 1, 2}


def test_natural_spread_stops_on_first_failure(ready_config, table, tmp_path):
    from learned_grasp.tolerance import run_natural_spread, spread_sheet

    world = _spread_world(ready_config, table, tmp_path)
    sheet = spread_sheet(ready_config, table)
    world.outcomes[sheet[3]["trial_id"]] = False
    with pytest.raises(StopSession, match="G0d-0 집기 실패"):
        world.spread(sheet)
    finished, _ = world.ledger.state()
    assert len(finished) == 4 and not finished[sheet[3]["trial_id"]]["success"]


def test_spread_yellow_over_limit_blocks(tmp_path):
    from learned_grasp.tolerance import summarize_spread

    records = [{"event": "finish", "success": True, "reason": "변화 0.0500, 노란 화소 53"}]
    assert summarize_spread(records)["yellow_ok"] is False


def test_spread_return_outside_makes_transfer_unverified():
    from learned_grasp.tolerance import summarize_spread

    records = []
    def place(tid, cell, after, x, angle=0.0):
        records.append({"event": "note", "kind": "placement", "ok": True, "trial_id": tid, "frame": f"{tid}-p1", "measurement": {"position": [x, 0.0], "angle": angle}})
        records.append({"event": "note", "kind": "spread_place", "trial_id": tid, "cell": cell, "phase": "spread" if after == "plain" else "compare", "after_path": after, "frame": f"{tid}-p1"})
    for k, x in enumerate((0.0, 1.0, 2.0)):
        place(f"a{k}", 0, "plain", x)
    for k, x in enumerate((1.0, 1.5, 5.0)):
        place(f"b{k}", 0, "return", x)
    summary = summarize_spread(records)
    assert summary["cells"][0]["position_max_px"] == 1.0 and summary["cells"][0]["return_outside"] == 1
    assert summary["transfer"] == "unverified"


def test_failure_branch_sends_nothing_between_judge_and_answer(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.outcomes[sheet[0]["trial_id"]] = False
    world.run(sheet[:1])
    judged = dict(world.judged)[sheet[0]["trial_id"]]
    asked = next(a for a in world.asked if a[0] == "failure")[2]
    assert judged == asked


def test_judge_photo_reference_move_stops_before_any_write(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.marker.special[f"{trial['trial_id']}-after"] = Measurement("a", "x" * 64, {0: (919.0, 654.0), 1: (851.0, 211.0)}, False, False)
    with pytest.raises(StopSession, match="판정 사진 기준"):
        world.run(sheet)
    assert len(world.driver.writes) == dict(world.judged)[trial["trial_id"]]
    finished, _ = world.ledger.state()
    assert finished[trial["trial_id"]]["outcome"] == "stopped"


def test_close_batch_stops_with_block_down_and_resumes(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world._new_batch()
    run_tolerance(world.ctx, sheet)
    assert world.ctx.counters["batch_end"]
    assert 0 < world.arm.close_count <= 10
    assert not world.arm.holding and world.driver.writes[-1][0] == world.arm.gate.joints(PARK_POSE)
    first = len(world.ledger.state()[0])
    world.run(sheet)
    assert len(world.ledger.state()[0]) == 72 > first


def test_return_stop_marks_variant_incomplete_for_resume(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.outcomes[f"{trial['trial_id']}-ret"] = False
    world.answers[("failure", f"{trial['trial_id']}-ret")] = "unknown"
    with pytest.raises(StopSession):
        world.run(sheet)
    states = variant_states(sheet, world.ledger.records())
    assert states[trial["variant_id"]]["status"] == "incomplete"
    again = World(ready_config, table, tmp_path)
    again.run(sheet)
    starts = [r["trial_id"] for r in again.ledger.records() if r["event"] == "start"]
    assert [s for s in starts if s.startswith(trial["variant_id"] + "-")] == [trial["trial_id"]]


def test_answer_without_release_time_left_is_refused(ready_config, table, tmp_path, sheet):
    """답이 왔어도 그 사이 시간이 흘러 해제 시간이 모자라면 움직이지 않는다."""
    world = World(ready_config, table, tmp_path, limit_s=60.0)
    trial = sheet[0]
    world.outcomes[trial["trial_id"]] = False
    clock = {"now": 0.0}
    world.watch.clock = lambda: clock["now"]

    def slow(kind, prompt, deadline):
        world.asked.append((kind, prompt.split(" ")[0].rstrip(":"), len(world.driver.writes), deadline))
        if kind == "failure":
            clock["now"] += 58.0
            return "miss"
        return "y"

    world.ctx.ask = slow
    with pytest.raises(StopSession, match="해제 시간"):
        world.run(sheet)
    assert len(world.driver.writes) == next(a for a in world.asked if a[0] == "failure")[2]


def test_ledger_bound_to_binding_and_sheet(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world.ctx.binding = {"spread_sha256": "a"}
    world.run(sheet[:1])
    again = World(ready_config, table, tmp_path)
    again.ctx.binding = {"spread_sha256": "b"}
    with pytest.raises(RuntimeError, match="이어 쓸 수 없다"):
        again.run(sheet[:1])
    other = World(ready_config, table, tmp_path)
    other.ctx.binding = {"spread_sha256": "a"}
    with pytest.raises(RuntimeError, match="이어 쓸 수 없다"):
        other.run(tolerance_sheet(ready_config, table, 2))


def test_spread_yellow_counts_return_picks():
    from learned_grasp.tolerance import summarize_spread

    records = [
        {"event": "finish", "success": True, "reason": "변화 0.0500, 노란 화소 30"},
        {"event": "note", "kind": "return_pick", "success": True, "reason": "변화 0.0500, 노란 화소 80"},
    ]
    assert summarize_spread(records)["yellow_max_success"] == 80 and not summarize_spread(records)["yellow_ok"]


def test_spread_angle_center_wraps():
    from learned_grasp.tolerance import summarize_spread

    records = []
    for k, angle in enumerate((179.0, -179.0, 178.0)):
        records.append({"event": "note", "kind": "placement", "ok": True, "trial_id": f"a{k}", "frame": f"a{k}", "measurement": {"position": [0.0, 0.0], "angle": angle}})
        records.append({"event": "note", "kind": "spread_place", "trial_id": f"a{k}", "cell": 0, "phase": "spread", "after_path": "plain", "frame": f"a{k}"})
    records.append({"event": "note", "kind": "placement", "ok": True, "trial_id": "r", "frame": "r", "measurement": {"position": [0.0, 0.0], "angle": 0.0}})
    records.append({"event": "note", "kind": "spread_place", "trial_id": "r", "cell": 0, "phase": "compare", "after_path": "return", "frame": "r"})
    cell = summarize_spread(records)["cells"][0]
    assert abs(abs(cell["center"][2]) - 179.33) < 0.1 and cell["angle_max_deg"] < 2.0
    assert cell["return_outside"] == 1


def test_spread_baseline_uses_spread_phase_only():
    from learned_grasp.tolerance import summarize_spread

    records = []
    for k, (phase, x) in enumerate((("spread", 0.0), ("spread", 1.0), ("compare", 9.0))):
        records.append({"event": "note", "kind": "placement", "ok": True, "trial_id": f"a{k}", "frame": f"a{k}", "measurement": {"position": [x, 0.0], "angle": 0.0}})
        records.append({"event": "note", "kind": "spread_place", "trial_id": f"a{k}", "cell": 0, "phase": phase, "after_path": "plain", "frame": f"a{k}"})
    cell = summarize_spread(records)["cells"][0]
    assert cell["base_n"] == 2 and cell["position_max_px"] == 0.5


def test_close_count_counts_real_close_commands(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    world._new_batch()
    world.ctx.close_batch = 10**6
    run_tolerance(world.ctx, sheet[:1])
    closes = sum(1 for (_, grip, _), (_, prev, _) in zip(world.driver.writes[1:], world.driver.writes) if grip != prev and grip == ready_config.close_sdk)
    assert world.arm.close_count == closes == 4  # 적재, 준비 순환, 오프셋 집기, 복귀 재집기


@pytest.mark.parametrize("batch", [7, 10])
def test_spread_batches_never_split_return_pairs(ready_config, table, tmp_path, batch):
    from learned_grasp.tolerance import spread_sheet, summarize_spread

    world = _spread_world(ready_config, table, tmp_path)
    world.ctx.close_batch = batch
    world.ctx.observer = lambda trial: Outcome(True, "변화 0.0500, 노란 화소 30")
    world.spread(spread_sheet(ready_config, table))
    summary = summarize_spread(world.ledger.records())
    assert summary["trials"] == 42
    assert all(c["return_n"] == 3 for c in summary["cells"].values())
    assert summary["transfer"] == "verified"
    assert all(c["base_n"] == 8 for c in summary["cells"].values())
    records = world.ledger.records()
    assert sum(1 for r in records if r["event"] == "note" and r.get("kind") == "warmup_pick") == world.batches
    warm_frames = [r for r in records if r["event"] == "note" and r.get("kind") == "placement" and r["trial_id"].endswith("-w")]
    assert len(warm_frames) == world.batches


def test_warmup_failure_blocks_resume_and_counts_as_failure(ready_config, table, tmp_path):
    from learned_grasp.tolerance import run_natural_spread, spread_sheet, summarize_spread

    world = _spread_world(ready_config, table, tmp_path)
    sheet = spread_sheet(ready_config, table)
    world.outcomes[f"{sheet[0]['trial_id']}-w"] = False
    with pytest.raises(StopSession, match="준비 순환"):
        world.spread(sheet)
    summary = summarize_spread(world.ledger.records())
    assert summary["failures"] >= 1 and summary["incomplete"] >= 1
    again = _spread_world(ready_config, table, tmp_path)
    again._new_batch()
    with pytest.raises(StopSession, match="실패나 미완료"):
        run_natural_spread(again.ctx, sheet)
    assert not again.driver.writes


@pytest.mark.parametrize("spread_run", [False, True])
def test_held_answer_blocks_any_resume(ready_config, table, tmp_path, sheet, spread_run):
    from learned_grasp.tolerance import run_natural_spread, spread_sheet

    world = _spread_world(ready_config, table, tmp_path) if spread_run else World(ready_config, table, tmp_path)
    target = spread_sheet(ready_config, table) if spread_run else sheet
    world.outcomes[target[0]["trial_id"]] = False
    world.answers[("failure", target[0]["trial_id"])] = "held"
    with pytest.raises(StopSession):
        world.spread(target) if spread_run else world.run(target)
    again = World(ready_config, table, tmp_path)
    again._new_batch()
    with pytest.raises(StopSession, match="오판정"):
        run_natural_spread(again.ctx, target) if spread_run else run_tolerance(again.ctx, target)
    assert not again.driver.writes


def test_spread_return_uses_smallest_r_step_per_cell(ready_config, table):
    from learned_grasp.tolerance import spread_sheet

    picks = {t["cell"]: (t["pick_r_index"], t["model_offset"]) for t in spread_sheet(ready_config, table) if t["path"] == "return"}
    assert picks == {0: (0, -4.29), 1: (5, 4.398), 2: (8, 3.951)}


def test_tolerance_batch_first_placement_is_after_warmup(ready_config, table, tmp_path, sheet):
    """묶음 첫 배치도 필터를 만든 조건(일반 경로 다음)이다. 준비 순환 놓기는 필터를 쓰지 않는다."""
    world = World(ready_config, table, tmp_path)
    world.ctx.placement_filter = PlacementFilter({0: (0, 0, 0), 1: (0, 0, 0), 2: (0, 0, 0)}, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1})
    trial = sheet[0]
    world.marker.special[f"{trial['trial_id']}-w-p1"] = Measurement("w", "x" * 64, dict(REF), True, True, (50.0, 0.0), 0.0)  # 필터 밖
    world.run(sheet[:1])
    kinds = [a[0] for a in world.asked]
    assert "reload" not in kinds  # 준비 순환 놓기에는 필터가 없다
    notes = [r for r in world.ledger.records() if r["event"] == "note" and r.get("kind") == "placement"]
    assert [n["trial_id"] for n in notes][:2] == [f"{trial['trial_id']}-w", trial["trial_id"]]
    assert sum(1 for r in world.ledger.records() if r["event"] == "note" and r.get("kind") == "warmup_pick") == 1


def _scenarios(sheet_ids):
    return {
        "all_success": lambda tid: True,
        "return_fails": lambda tid: not tid.endswith("-ret"),
        "second_round_fails": lambda tid: not tid.endswith("-t1") and not tid.endswith("-s1"),
    }


@pytest.mark.parametrize("scenario", ["all_success", "return_fails", "second_round_fails"])
@pytest.mark.parametrize("which", ["tolerance", "spread"])
def test_write_limit_sweep_never_hits_cap_while_holding(ready_config, table, tmp_path, scenario, which):
    """남은 쓰기 예약이 맞으면 어떤 상한에서도 쓰기 상한 오류가 나지 않고 블록을 내려놓은 채 park에서 끝난다."""
    from learned_grasp.safety import SafetyViolation
    from learned_grasp.tolerance import run_natural_spread, spread_sheet

    sheets = {"tolerance": tolerance_sheet(ready_config, table, 1), "spread": spread_sheet(ready_config, table)}
    sheet = sheets[which]
    rule = _scenarios(None)[scenario]
    for limit in (60, 61, 75, 90, 111, 130, 157, 199, 240):
        world = World(ready_config, table, tmp_path / f"{which}-{scenario}-{limit}", write_limit=limit)
        world.ctx.close_batch = 10**6
        world.ctx.observer = lambda trial: Outcome(rule(trial["trial_id"]), "변화 0.0500, 노란 화소 10")
        world.arm.pose = None
        try:
            (run_tolerance if which == "tolerance" else run_natural_spread)(world.ctx, sheet)
        except SafetyViolation as error:
            raise AssertionError(f"상한 {limit}에서 쓰기 상한 오류: {error}") from error
        except StopSession:
            pass  # 판정 실패 답 등 의도된 정지는 이 시험의 대상이 아니다
        else:
            assert len(world.driver.writes) <= limit
            assert not world.arm.holding and world.driver.writes[-1][0] == world.arm.gate.joints(PARK_POSE), (which, scenario, limit)
            assert world.ctx.counters["end"] in ("batch", "complete")


def test_budget_too_small_stops_before_any_write(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path, write_limit=40)
    with pytest.raises(StopSession, match="남은 쓰기"):
        run_tolerance(world.ctx, sheet)
    assert not world.driver.writes


@pytest.mark.parametrize("which", ["tolerance", "spread"])
def test_batch_smaller_than_one_slot_stops_instead_of_looping(ready_config, table, tmp_path, sheet, which):
    from learned_grasp.tolerance import run_natural_spread, spread_sheet

    world = World(ready_config, table, tmp_path)
    world.ctx.close_batch = 2
    world._new_batch()
    with pytest.raises(StopSession, match="한 자리 진행에 모자라다"):
        (run_tolerance if which == "tolerance" else run_natural_spread)(world.ctx, sheet if which == "tolerance" else spread_sheet(ready_config, table))


def test_spread_batch_four_cannot_progress_return_slot(ready_config, table, tmp_path):
    from learned_grasp.tolerance import spread_sheet

    world = _spread_world(ready_config, table, tmp_path)
    world.ctx.close_batch = 4
    world.ctx.observer = lambda trial: Outcome(True, "변화 0.0500, 노란 화소 30")
    with pytest.raises(StopSession, match="한 자리 진행에 모자라다"):
        world.spread(spread_sheet(ready_config, table))


def test_resume_never_reuses_placement_frame_names(ready_config, table, tmp_path, sheet):
    world = World(ready_config, table, tmp_path)
    trial = sheet[0]
    world.marker.special[f"{trial['trial_id']}-w-p1"] = MeasureError("reference_moved", "x")
    with pytest.raises(StopSession):
        world.run(sheet[:1])
    world.marker.special.clear()
    world.run(sheet[:1])
    frames = [r["frame"] for r in world.ledger.records() if r["event"] == "note" and r.get("kind") == "placement"]
    assert len(frames) == len(set(frames))
    assert f"{trial['trial_id']}-w-p1-r2" in frames
    photo_names = [n for n in world.camera.names if "-w-p1" in n]
    assert len(photo_names) == len(set(photo_names)) == 2


def test_spread_reports_warmup_placements_inside_filter():
    from learned_grasp.tolerance import summarize_spread

    records = []
    def place(tid, cell, phase, after, x):
        records.append({"event": "note", "kind": "placement", "ok": True, "trial_id": tid, "frame": f"{tid}-p1", "measurement": {"position": [x, 0.0], "angle": 0.0}})
        if not tid.endswith("-w"):
            records.append({"event": "note", "kind": "spread_place", "trial_id": tid, "cell": cell, "phase": phase, "after_path": after, "frame": f"{tid}-p1"})
    for k, x in enumerate((0.0, 1.0)):
        place(f"n-c0-s{k}", 0, "spread", "plain", x)
    place("n-c0-s0-w", 0, "spread", "plain", 0.5)
    place("n-c0-s2-w", 0, "spread", "plain", 9.0)
    assert summarize_spread(records)["cells"][0]["warmup_in_filter"] == [1, 2]
