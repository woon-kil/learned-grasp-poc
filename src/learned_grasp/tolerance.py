"""G0d 실행기. 한 시도 흐름은 `_attempt` 하나이고 G0d 1차(`run_tolerance`)가 쓴다.

순서(docs/261109-2129-G0d-재설계-플랜.md v9 「한 시도」):
1. 블록을 쥔 채 P에 놓고 park에서 사진. 2. 배치 확인(실패면 쓰기 없이 사람 재적재, 2회 연속이면 변형 미완료).
3. P+δ의 hover에서 멈춤. 4. 하강 전 확인(여기서 시도 ID 소비, 거부면 안전상 미측정).
5. 집기, park, 사진 판정. 6. 실패면 쓰기 없이 사람 답(마감 안). 7. 성공이면 복귀: P+δ에 놓고 P에서 다시 집는다.
자동 복구 이동은 없다. 판정 실패 뒤 답 전 쓰기는 0건이다.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

from learned_grasp.arm import SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import Camera
from learned_grasp.g0_runner import Observer, _bind, _pick, _place, _pose, variant_verdict
from learned_grasp.pose_table import PoseTable
from learned_grasp.safety import GRASP, HOVER, PARK_POSE
from learned_grasp.top_marker import Measurement, MeasureError, Placement, wrap_deg

# 사람 답. ask(kind, prompt, deadline_s)는 마감 안의 답을 돌려주고, 마감을 넘기거나 늦은 답이면 None이다.
Ask = Callable[[str, str, "float | None"], "str | None"]
FAILURE_ANSWERS = ("miss", "held", "unknown")


class StopSession(RuntimeError):
    """추가 쓰기 없이 세션을 끝낸다. 팔은 그 자리에 있다."""


@dataclass(frozen=True)
class PlacementFilter:
    """G0d-0에서 실행 전에 고정한 셀별 기준과 자연 흩어짐 최대. 값은 계약에서 받는다."""

    center: dict[int, tuple[float, float, float]]  # cell -> (x, y, angle)
    position_max_px: dict[int, float]
    angle_max_deg: dict[int, float]

    def deviation(self, cell: int, m: Measurement) -> tuple[float, float]:
        x, y, angle = self.center[cell]
        return math.dist(m.position, (x, y)), abs(wrap_deg(m.angle - angle))

    def accepts(self, cell: int, m: Measurement) -> bool:
        position, angle = self.deviation(cell, m)
        return position <= self.position_max_px[cell] and angle <= self.angle_max_deg[cell]


@dataclass
class G0dContext:
    arm: SafeArm
    camera: Camera
    ledger: AttemptLedger
    table: PoseTable
    observer: Observer
    marker: object  # measure(path) -> Measurement
    placement: Placement
    ask: Ask
    load_block: Callable[[dict], None]
    release_s: float  # 미파지 답 뒤 hover 이동과 개방에 드는 시간. 실측으로 계약에 고정한다.
    placement_filter: PlacementFilter | None = None
    log: Callable[[str], None] = print
    counters: dict = field(default_factory=lambda: {"return_failures": 0, "reloads": 0})


def _measure(ctx: G0dContext, name: str) -> Measurement:
    ctx.camera.snapshot(name)
    try:
        path = ctx.camera.fetch(name)
    except (FileNotFoundError, OSError) as error:
        raise MeasureError("fetch_failed", str(error)) from error
    return ctx.placement.check(ctx.marker.measure(path))


def _answer_deadline(ctx: G0dContext) -> float:
    remaining = ctx.arm.close_watch.remaining()
    if remaining is None:
        raise StopSession("닫힘 감시가 돌고 있지 않다")
    return remaining - ctx.release_s


def _failure_path(ctx: G0dContext, trial: dict, prefix: str, counted: bool) -> None:
    """판정 실패. 쓰기 없이 사람 답을 받는다. 미파지만 hover에서 열고 park로 간다."""
    deadline = _answer_deadline(ctx)
    answer = ctx.ask("failure", f"{trial['trial_id']} 판정 실패. 손이 비었고 블록이 보드 위면 miss, 쥐었으면 held, 모르면 unknown", deadline) if deadline > 0 else None
    ctx.ledger.note("failure_answer", trial_id=trial["trial_id"], answer=answer, counted=counted)
    if answer != "miss":
        raise StopSession(f"{trial['trial_id']}: 판정 실패 뒤 답 {answer!r}. 추가 쓰기 없이 끝낸다")
    ctx.arm.goto(_pose(HOVER, trial, prefix))
    ctx.arm.set_grip("open")
    ctx.arm.goto(PARK_POSE)


def _confirm_descent(ctx: G0dContext, trial: dict, prefix: str, on_arrival: Callable[[], None] | None = None) -> bool:
    """hover에서 멈추고 사람 확인을 받는다. y가 아니면 내려가지 않는다."""
    ctx.arm.goto(_pose(HOVER, trial, prefix))
    if on_arrival is not None:
        on_arrival()
    answer = ctx.ask("descent", f"{trial['trial_id']}: 양쪽 손가락 하강 경로가 블록에 닿지 않으면 y", None)
    return answer == "y"


def _place_and_check(ctx: G0dContext, trial: dict) -> Measurement | None:
    """P에 놓고 배치를 확인한다. 실패하면 쓰기 없이 사람이 다시 놓는다. 2회 연속 실패면 None."""
    _place(ctx.arm, trial)
    for attempt in (1, 2):
        name = f"{trial['trial_id']}-p{attempt}"
        reason = None
        try:
            m = _measure(ctx, name)
            if ctx.placement_filter is not None and not ctx.placement_filter.accepts(trial["cell"], m):
                reason = f"배치 벗어남 {ctx.placement_filter.deviation(trial['cell'], m)}"
        except MeasureError as error:
            if error.reason == "reference_moved":
                raise StopSession(f"카메라 기준이 움직였다: {error}") from error
            m, reason = None, error.reason
        ctx.ledger.note("placement", trial_id=trial["trial_id"], frame=name, ok=reason is None, reason=reason, measurement=None if m is None else m.record(), reloaded=attempt > 1)
        if reason is None:
            trial["before_frame"] = name
            return m
        if attempt == 1:
            ctx.counters["reloads"] += 1
            if ctx.ask("reload", f"{trial['trial_id']}: 배치 확인 실패({reason}). 블록을 P에 손으로 다시 놓으면 y", None) != "y":
                raise StopSession("재적재 답이 없다")
    return None


def _return_cycle(ctx: G0dContext, trial: dict) -> bool:
    """P+δ에 도로 놓고 P에서 다시 집는다. 시도 결과에 세지 않는다. 성공이면 블록을 쥔 상태다."""
    _place(ctx.arm, trial, "pick_")
    name = f"{trial['trial_id']}-ret"
    try:
        m = _measure(ctx, name)
    except MeasureError as error:
        raise StopSession(f"복귀 배치 확인 실패: {error}") from error
    ctx.ledger.note("return_place", trial_id=trial["trial_id"], frame=name, measurement=m.record())
    if not _confirm_descent(ctx, trial, ""):
        raise StopSession(f"{trial['trial_id']}: 복귀 재집기 하강 거부")
    _pick(ctx.arm, trial)
    ctx.arm.goto(PARK_POSE)
    outcome = ctx.observer({**trial, "trial_id": f"{trial['trial_id']}-ret", "before_frame": name})
    ctx.ledger.note("return_pick", trial_id=trial["trial_id"], success=outcome.success, reason=outcome.reason)
    if outcome.success:
        return True
    ctx.counters["return_failures"] += 1
    _failure_path(ctx, {**trial, "trial_id": f"{trial['trial_id']}-ret"}, "", counted=False)
    return False


def variant_states(sheet: list[dict], records: list[dict]) -> dict[str, dict]:
    """기록에서 변형 상태를 만든다. 미완료(안전상 미측정, 배치 2회 실패, 중단)가 하나라도 있으면 그 변형은 미완료다."""
    by_id = {slot["trial_id"]: slot["variant_id"] for slot in sheet}
    outcomes: dict[str, list[bool]] = {}
    states: dict[str, dict] = {}
    for record in records:
        variant = by_id.get(record.get("trial_id"))
        if variant is None:
            continue
        if record["event"] == "note" and record.get("kind") == "incomplete":
            states[variant] = {"status": "incomplete", "reason": record["reason"]}
        elif record["event"] == "finish":
            if record.get("outcome") in ("success", "failure"):
                outcomes.setdefault(variant, []).append(bool(record["success"]))
            else:
                states[variant] = {"status": "incomplete", "reason": record.get("reason", record.get("outcome"))}
    for variant, results in outcomes.items():
        if variant in states:
            continue
        verdict = variant_verdict(results)
        if verdict is not None:
            states[variant] = {"status": verdict, "outcomes": results}
    return states


def run_tolerance(ctx: G0dContext, sheet: list[dict]) -> dict[str, dict]:
    """G0d 1차. 종결된 변형의 남은 자리와 이미 쓴 시도 ID는 건너뛴다. StopSession이면 그 자리에서 끝난다."""
    arm, ledger = ctx.arm, ctx.ledger
    if arm.close_watch is None:
        raise ValueError("G0d는 닫힘 한도 감시가 필요하다")
    _bind(arm, ledger, ctx.table)
    ledger.close_interrupted()
    arm.goto(PARK_POSE)
    arm.set_grip("open")
    has_block = False
    for slot in sheet:
        started = {r["trial_id"] for r in ledger.records() if r["event"] in ("start", "finish")}
        if slot["variant_id"] in variant_states(sheet, ledger.records()) or slot["trial_id"] in started:
            continue
        trial = dict(slot)
        if not has_block:
            ctx.load_block(trial)
            _pick(arm, trial)
            arm.goto(PARK_POSE)
        measured = _place_and_check(ctx, trial)
        has_block = False
        if measured is None:
            ledger.note("incomplete", trial_id=trial["trial_id"], reason="배치 확인 2회 연속 실패")
            ctx.log(json.dumps({"incomplete": trial["variant_id"], "reason": "placement"}, ensure_ascii=False))
            continue
        if not _confirm_descent_offset(ctx, trial):
            continue
        _pick(arm, trial, "pick_")
        arm.goto(PARK_POSE)
        outcome = ctx.observer(trial)
        if not outcome.success:
            try:
                _failure_path(ctx, trial, "pick_", counted=True)
            except StopSession:
                ledger.finish(trial["trial_id"], outcome="stopped", success=False, reason=outcome.reason, human_intervention=True)
                raise
            ledger.finish(trial["trial_id"], outcome="failure", success=False, reason=outcome.reason, human_intervention=True, read_back=arm.read())
            continue
        ledger.finish(trial["trial_id"], outcome="success", success=True, reason=outcome.reason, read_back=arm.read())
        has_block = _return_cycle(ctx, trial)
    if has_block:
        _place(arm, trial)
    return variant_states(sheet, ledger.records())


def _confirm_descent_offset(ctx: G0dContext, trial: dict) -> bool:
    """P+δ hover에서 확인을 받는다. 확인을 묻는 순간부터 시도 ID를 쓴다. 거부면 안전상 미측정으로 park에 간다."""
    joints = {"place_q": ctx.arm.gate.joints(_pose(GRASP, trial)), "pick_q": ctx.arm.gate.joints(_pose(GRASP, trial, "pick_"))}
    if _confirm_descent(ctx, trial, "pick_", on_arrival=lambda: ctx.ledger.start(trial, **joints)):
        return True
    ctx.ledger.finish(trial["trial_id"], outcome="safety_unmeasured", success=False, reason="하강 전 확인 거부")
    ctx.arm.goto(PARK_POSE)
    return False
