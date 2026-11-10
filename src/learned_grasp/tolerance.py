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
CLOSE_BATCH = 10  # 쉼 묶음. 실제로 보낸 닫기 명령 수로 센다(플랜 v9 172행). 묶음 끝에서 블록을 내려놓고 멈춘다.
# 쓰기 예약. 한 자리의 최악 쓰기(적재 5, 준비 순환 13, 놓기 5, 하강 확인 1, 집기 4, 실패 경로 3, 복귀 13 = 44에 여유)와
# 묶음 끝 내려놓기 쓰기. 남은 쓰기가 이보다 적으면 닫기 예약처럼 블록을 내려놓고 묶음을 끝낸다. tests/test_tolerance.py의
# 상한 훑기 시험이 이 값에서 상한 오류가 나지 않음을 확인한다.
SLOT_WRITES = 48
END_WRITES = 6
START_WRITES = 2


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
    binding: dict = field(default_factory=dict)  # 기록에 묶을 기준 마커·spread 해시. 다르면 이어 쓰지 않는다
    close_batch: int = CLOSE_BATCH
    writes_left: Callable[[], int] | None = None  # 승인 max_writes까지 남은 쓰기 수. None이면 상한 예약을 하지 않는다(Fake 시험용)
    counters: dict = field(default_factory=lambda: {"return_failures": 0, "reloads": 0, "batch_end": False, "end": None})


def _measure(ctx: G0dContext, name: str) -> Measurement:
    ctx.camera.snapshot(name)
    try:
        path = ctx.camera.fetch(name)
    except (FileNotFoundError, OSError) as error:
        raise MeasureError("fetch_failed", str(error)) from error
    return ctx.placement.check(ctx.marker.measure(path))


def _judge(ctx: G0dContext, trial: dict):
    """집기 판정. 판정 사진의 바깥 기준 마커도 확인한다. 어긋나면 어떤 쓰기도 하지 않고 멈춘다."""
    outcome = ctx.observer(trial)
    name = f"{trial['trial_id']}-after"
    try:
        ctx.placement.check_reference(ctx.marker.measure(ctx.camera.fetch(name)))
    except (MeasureError, FileNotFoundError, OSError) as error:
        raise StopSession(f"{name}: 판정 사진 기준 확인 실패({error}). 추가 쓰기 없이 끝낸다") from error
    return outcome


def _batch_full(ctx: G0dContext, needed: int) -> bool:
    """다음 자리에 필요한 닫기 수를 더하면 묶음을 넘는가, 또는 남은 쓰기가 한 자리와 끝 내려놓기에 모자라는가."""
    if ctx.arm.close_count + needed > ctx.close_batch:
        return True
    left = None if ctx.writes_left is None else ctx.writes_left()
    return left is not None and left < SLOT_WRITES + END_WRITES


def _require_budget(ctx: G0dContext) -> None:
    """첫 쓰기 전에 한 자리와 끝 내려놓기가 들어갈 쓰기가 남았는지 본다. 모자라면 아무것도 보내지 않는다."""
    left = None if ctx.writes_left is None else ctx.writes_left()
    if left is not None and left < START_WRITES + SLOT_WRITES + END_WRITES:
        raise StopSession(f"남은 쓰기 {left}회로는 한 자리와 끝 내려놓기를 보장하지 못한다. 쓰기 전에 멈춘다")


def _no_progress(ctx: G0dContext, done: int) -> None:
    """묶음이 한 자리도 못 하고 끝나면 같은 묶음이 영원히 되풀이된다. 이 경우 오류로 멈춘다."""
    if ctx.counters["batch_end"] and done == 0:
        raise StopSession("묶음 한도(닫기 또는 남은 쓰기)가 한 자리 진행에 모자라다. 한도를 다시 정한다")


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
    remaining = ctx.arm.close_watch.remaining()
    if remaining is None or remaining < ctx.release_s:
        raise StopSession(f"{trial['trial_id']}: 답은 왔지만 정상 해제 시간({ctx.release_s:g}초)이 남지 않았다")
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


def _unique_frame(ctx: G0dContext, base: str) -> str:
    """이어하기에서 같은 이름으로 다시 찍어 p620·Mac 사진과 기록 키를 덮어쓰지 않게 한다."""
    used = {r.get("frame") for r in ctx.ledger.records() if r["event"] == "note" and r.get("kind") == "placement"}
    name, run = base, 1
    while name in used:
        run += 1
        name = f"{base}-r{run}"
    return name


def _place_and_check(ctx: G0dContext, trial: dict, filtered: bool = True) -> Measurement | None:
    """P에 놓고 배치를 확인한다. 실패하면 쓰기 없이 사람이 다시 놓는다. 2회 연속 실패면 None.

    filtered=False는 적재 직후 준비 순환의 놓기다. 필터가 아니라 기준 마커와 윗면 마커만 확인한다.
    """
    _place(ctx.arm, trial)
    for attempt in (1, 2):
        name = _unique_frame(ctx, f"{trial['trial_id']}-p{attempt}")
        reason = None
        try:
            m = _measure(ctx, name)
            if filtered and ctx.placement_filter is not None and not ctx.placement_filter.accepts(trial["cell"], m):
                reason = f"배치 벗어남 {ctx.placement_filter.deviation(trial['cell'], m)}"
        except MeasureError as error:
            m, reason = None, error.reason
            moved = str(error) if error.reason == "reference_moved" else None
        else:
            moved = None
        # 정지하는 경우에도 사진 이름을 기록에 남겨 이어하기가 같은 이름으로 덮어쓰지 않게 한다.
        ctx.ledger.note("placement", trial_id=trial["trial_id"], frame=name, ok=reason is None, reason=reason, measurement=None if m is None else m.record(), reloaded=attempt > 1)
        if moved is not None:
            raise StopSession(f"카메라 기준이 움직였다: {moved}")
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
        # 세션을 끝낸다. 그리퍼는 열려 있고 블록은 P+δ에 놓여 있어 추가 이동 없이 hover에서 멈춘다(플랜 v10).
        raise StopSession(f"{trial['trial_id']}: 복귀 재집기 하강 거부")
    _pick(ctx.arm, trial)
    ctx.arm.goto(PARK_POSE)
    outcome = _judge(ctx, {**trial, "trial_id": f"{trial['trial_id']}-ret", "before_frame": name})
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


def sheet_sha256(sheet: list[dict]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(sheet, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _refuse_after_misjudge(ledger: AttemptLedger) -> None:
    """판정 실패에 held(실제로 쥠)가 한 번이라도 있으면 같은 판정값으로 잇지 않는다(플랜 121행: 판정값 재검토)."""
    held = [r["trial_id"] for r in ledger.records() if r["event"] == "note" and r.get("kind") == "failure_answer" and r.get("answer") == "held"]
    if held:
        raise StopSession(f"오판정 기록 {held}. 판정값을 재검토하기 전에는 이어서 하지 않는다")


def run_tolerance(ctx: G0dContext, sheet: list[dict]) -> dict[str, dict]:
    """G0d 1차. 종결된 변형의 남은 자리와 이미 쓴 시도 ID는 건너뛴다. StopSession이면 그 자리에서 끝난다.

    쉼 묶음(실제 닫기 CLOSE_BATCH회)에 닿기 전에 블록을 내려놓고 멈춘다. 같은 기록으로 다시 부르면 이어서 한다.
    """
    arm, ledger = ctx.arm, ctx.ledger
    if arm.close_watch is None:
        raise ValueError("G0d는 닫힘 한도 감시가 필요하다")
    _bind(arm, ledger, ctx.table, sheet_sha256=sheet_sha256(sheet), **ctx.binding)
    _refuse_after_misjudge(ledger)
    _require_budget(ctx)
    ledger.close_interrupted()
    arm.goto(PARK_POSE)
    arm.set_grip("open")
    has_block = False
    trial = None
    done = 0
    for slot in sheet:
        started = {r["trial_id"] for r in ledger.records() if r["event"] in ("start", "finish")}
        if slot["variant_id"] in variant_states(sheet, ledger.records()) or slot["trial_id"] in started:
            continue
        if _batch_full(ctx, (0 if has_block else 2) + 2):  # 적재 집기+준비 순환 집기, 오프셋 집기, 복귀 재집기
            ctx.counters["batch_end"] = True
            break
        done += 1
        trial = dict(slot)
        if not has_block:
            ctx.load_block(trial)
            _pick(arm, trial)
            arm.goto(PARK_POSE)
            # 적재 직후 배치는 필터를 만든 조건(일반 경로 다음)과 다르다. G0d-0과 같은 준비 순환을 거친다.
            _warmup(ctx, trial)
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
        try:
            outcome = _judge(ctx, trial)
        except StopSession:
            ledger.finish(trial["trial_id"], outcome="stopped", success=False, reason="판정 사진 기준 확인 실패", human_intervention=True)
            raise
        if not outcome.success:
            try:
                _failure_path(ctx, trial, "pick_", counted=True)
            except StopSession:
                ledger.finish(trial["trial_id"], outcome="stopped", success=False, reason=outcome.reason, human_intervention=True)
                raise
            ledger.finish(trial["trial_id"], outcome="failure", success=False, reason=outcome.reason, human_intervention=True, read_back=arm.read())
            continue
        ledger.finish(trial["trial_id"], outcome="success", success=True, reason=outcome.reason, read_back=arm.read())
        try:
            has_block = _return_cycle(ctx, trial)
        except StopSession as error:
            # 복귀 중 오판정·정지면 그 변형 결과를 믿지 않는다(플랜 70행). 원래 시도 기록은 그대로 둔다.
            ledger.note("incomplete", trial_id=trial["trial_id"], reason=f"복귀 중 정지: {error}")
            raise
    if has_block:
        _place(arm, trial)
    _no_progress(ctx, done)
    ctx.counters["end"] = "batch" if ctx.counters["batch_end"] else "complete"
    return variant_states(sheet, ledger.records())


def _confirm_descent_offset(ctx: G0dContext, trial: dict) -> bool:
    """P+δ hover에서 확인을 받는다. 확인을 묻는 순간부터 시도 ID를 쓴다. 거부면 안전상 미측정으로 park에 간다."""
    joints = {"place_q": ctx.arm.gate.joints(_pose(GRASP, trial)), "pick_q": ctx.arm.gate.joints(_pose(GRASP, trial, "pick_"))}
    if _confirm_descent(ctx, trial, "pick_", on_arrival=lambda: ctx.ledger.start(trial, **joints)):
        return True
    ctx.ledger.finish(trial["trial_id"], outcome="safety_unmeasured", success=False, reason="하강 전 확인 거부")
    ctx.arm.goto(PARK_POSE)
    return False


# ---- G0d-0 자연 흩어짐과 경로 비교 (플랜 v9 72~83행) ----
SPREAD_CELL_REPEATS = 8
COMPARE_REPEATS = 3
YELLOW_HOLD_MAX = 52  # 판정값 재계획 플랜 v3: 집기 성공 사진 노란 화소가 이를 넘으면 G0d 1차에 들어가지 않는다


def spread_sheet(config, table) -> list[dict]:
    """일반 경로 24회(셀 고정 순서 교차) 뒤 경로 비교 18회(셀마다 복귀·일반 교차 3번). 복귀는 셀마다 |δ| 최소 r 단계다."""
    from learned_grasp.g0_runner import tolerance_variants

    variants = {v["variant_id"]: v for v in tolerance_variants(config, table)}
    plain = []
    for cell in range(3):
        base = variants[f"c{cell}-r+"]
        plain.append({key: base[key] for key in ("cell", "r_index", "j1", "j5")} | {"pick_r_index": base["r_index"], "pick_j1": base["j1"], "pick_j5": base["j5"]})
    sheet = []
    for rep in range(SPREAD_CELL_REPEATS):
        for cell in range(3):
            sheet.append({**plain[cell], "trial_id": f"n-c{cell}-s{rep}", "path": "plain", "phase": "spread"})
    # 복귀 경로의 효과는 같은 셀에 다시 놓은 사진으로 본다. 셀마다 복귀 다음에 일반을 두어 복귀 뒤 배치가 같은 셀이 되게 한다.
    # P+δ는 플랜 76행대로 그 셀의 r 단계 중 |δ|가 가장 작은 것이다(셀 0은 r−, r_index 0 자리).
    for cell in range(3):
        ret = min((variants[f"c{cell}-r-"], variants[f"c{cell}-r+"]), key=lambda v: abs(v["model_offset"]))
        for rep in range(COMPARE_REPEATS):
            sheet.append({**{k: ret[k] for k in ("cell", "r_index", "j1", "j5", "pick_r_index", "pick_j1", "pick_j5", "model_offset")}, "trial_id": f"n-c{cell}-r{rep}", "path": "return", "phase": "compare"})
            sheet.append({**plain[cell], "trial_id": f"n-c{cell}-p{rep}", "path": "plain", "phase": "compare"})
    return sheet


def _warm_id(ctx: G0dContext, slot_id: str) -> str:
    """준비 순환 시도 ID. 이어하기에서 같은 ID로 판정 사진(`{id}-after`)을 덮어쓰지 않게 번호를 붙인다."""
    used = {r.get("trial_id") for r in ctx.ledger.records() if r["event"] == "note" and r.get("kind") == "placement"}
    name, run = f"{slot_id}-w", 1
    while name in used:
        run += 1
        name = f"{slot_id}-w{run}"
    return name


def _warmup(ctx: G0dContext, trial: dict) -> None:
    """적재 직후 같은 셀에서 놓기→P 집기를 한 번 더 한다. 다음 배치가 사람 적재가 아니라 일반 경로 다음이 되게 한다.
    시도로 세지 않지만 배치 확인, 판정, 기준 확인, 실패 경로는 같다. 실패면 G0d-0을 멈춘다.
    """
    warm = {**trial, "trial_id": _warm_id(ctx, trial["trial_id"]), "pick_r_index": trial["r_index"], "pick_j1": trial["j1"], "pick_j5": trial["j5"]}
    try:
        if _place_and_check(ctx, warm, filtered=False) is None:
            raise StopSession(f"{warm['trial_id']}: 준비 순환 배치 확인 2회 연속 실패")
        _pick(ctx.arm, warm)
        ctx.arm.goto(PARK_POSE)
        outcome = _judge(ctx, warm)
        ctx.ledger.note("warmup_pick", trial_id=trial["trial_id"], success=outcome.success, reason=outcome.reason)
        if not outcome.success:
            _failure_path(ctx, warm, "", counted=False)
            raise StopSession(f"{warm['trial_id']}: 준비 순환 집기 실패. 마커 영향을 재검토한다")
    except StopSession as error:
        # 시도는 아니지만 G0d-0의 「실패 1건이면 1차 금지」에 들어간다. 이어하기를 막도록 남긴다.
        ctx.ledger.note("incomplete", trial_id=warm["trial_id"], reason=f"준비 순환 정지: {error}")
        raise


def run_natural_spread(ctx: G0dContext, sheet: list[dict]) -> bool:
    """G0d-0. 집기 실패 1건이면 실패 경로대로 놓아 주고 멈춘다. 배치 필터는 쓰지 않는다(여기서 정한다).

    쉼 묶음마다 블록을 내려놓고 멈추며, 같은 기록으로 다시 부르면 이어서 한다. 중단된 시도가 있으면 잇지 않는다.
    시험표를 다 마치면 True다.
    """
    arm, ledger = ctx.arm, ctx.ledger
    if arm.close_watch is None:
        raise ValueError("G0d는 닫힘 한도 감시가 필요하다")
    if ctx.placement_filter is not None:
        raise ValueError("G0d-0은 배치 필터 없이 돈다")
    _bind(arm, ledger, ctx.table, sheet_sha256=sheet_sha256(sheet), **ctx.binding)
    _refuse_after_misjudge(ledger)
    if ledger.close_interrupted():
        raise StopSession("G0d-0은 중단된 시도 뒤로 잇지 않는다. 새 기록으로 다시 시작한다")
    finished, _ = ledger.state()
    if any(not record["success"] for record in finished.values()) or any(r["event"] == "note" and r.get("kind") == "incomplete" for r in ledger.records()):
        raise StopSession("G0d-0 기록에 실패나 미완료가 있다. 이어서 하지 않는다")
    _require_budget(ctx)
    arm.goto(PARK_POSE)
    arm.set_grip("open")
    has_block, previous, trial = False, "load", None
    done = 0
    for slot in sheet:
        if slot["trial_id"] in finished:
            continue
        # 복귀 경로 다음의 같은 셀 일반 시도가 같은 묶음에 들어가도록 그 집기 1회를 미리 남긴다. 경로 비교 쌍이 묶음 경계로 끊기지 않는다.
        needed = (0 if has_block else 2) + 1 + (2 if slot["path"] == "return" else 0)  # 적재 집기+준비 순환, 시도 집기, 복귀 재집기+다음 일반 집기
        if _batch_full(ctx, needed):
            ctx.counters["batch_end"] = True
            break
        done += 1
        trial = dict(slot)
        if not has_block:
            ctx.load_block(trial)
            _pick(arm, trial)
            arm.goto(PARK_POSE)
            _warmup(ctx, trial)
            previous = "plain"
        if _place_and_check(ctx, trial) is None:
            ledger.note("incomplete", trial_id=trial["trial_id"], reason="배치 확인 2회 연속 실패")
            raise StopSession(f"{trial['trial_id']}: 배치 확인 2회 연속 실패")
        has_block = False
        ledger.note("spread_place", trial_id=trial["trial_id"], cell=trial["cell"], phase=trial["phase"], after_path=previous, frame=trial["before_frame"])
        if trial["path"] == "return":
            if not _confirm_descent_offset(ctx, trial):
                raise StopSession(f"{trial['trial_id']}: 하강 전 확인 거부")
        else:
            ledger.start(trial)
        _pick(arm, trial, "pick_")
        arm.goto(PARK_POSE)
        try:
            outcome = _judge(ctx, trial)
        except StopSession:
            ledger.finish(trial["trial_id"], outcome="stopped", success=False, reason="판정 사진 기준 확인 실패", human_intervention=True)
            raise
        if not outcome.success:
            try:
                _failure_path(ctx, trial, "pick_", counted=True)
            finally:
                ledger.finish(trial["trial_id"], outcome="failure", success=False, reason=outcome.reason, human_intervention=True)
            raise StopSession(f"{trial['trial_id']}: G0d-0 집기 실패. 마커 영향을 재검토한다")
        ledger.finish(trial["trial_id"], outcome="success", success=True, reason=outcome.reason, read_back=arm.read())
        if trial["path"] == "return":
            try:
                if not _return_cycle(ctx, trial):
                    raise StopSession(f"{trial['trial_id']}: 복귀 재집기 실패")
            except StopSession as error:
                ledger.note("incomplete", trial_id=trial["trial_id"], reason=f"복귀 중 정지: {error}")
                raise
        has_block = True
        previous = trial["path"]
    complete = all(slot["trial_id"] in ledger.state()[0] for slot in sheet)
    if complete and has_block:
        last = dict(sheet[-1])
        last["trial_id"] = "n-end"
        if _place_and_check(ctx, last) is None:
            raise StopSession("끝 배치 확인 2회 연속 실패")
        ledger.note("spread_place", trial_id="n-end", cell=last["cell"], phase="end", after_path=previous, frame=last["before_frame"])
    elif has_block:
        _place(arm, trial)
    _no_progress(ctx, done)
    ctx.counters["end"] = "complete" if complete else "batch"
    return complete


def ledger_counters(records: list[dict]) -> dict:
    """묶음마다 새로 만드는 ctx 값이 아니라 장부에서 다시 센 합계. 최종 JSON에는 이 값을 쓴다(플랜 83·112행)."""
    notes = [r for r in records if r["event"] == "note"]
    placements = [r for r in notes if r.get("kind") == "placement"]
    warm = [r for r in notes if r.get("kind") == "warmup_pick"]
    ret = [r for r in notes if r.get("kind") == "return_pick"]
    return {
        "placements": len(placements),
        "placements_ok": sum(1 for r in placements if r.get("ok")),
        "reloads": sum(1 for r in placements if r.get("reloaded")),
        "warmup_picks": len(warm),
        "warmup_failures": sum(1 for r in warm if not r.get("success")),
        "return_picks": len(ret),
        "return_failures": sum(1 for r in ret if not r.get("success")),
    }


def _circular_mean_deg(values: list[float]) -> float:
    return math.degrees(math.atan2(sum(math.sin(math.radians(v)) for v in values), sum(math.cos(math.radians(v)) for v in values)))


def summarize_spread(records: list[dict]) -> dict:
    """기준 = spread 단계(일반 경로 24회)에서 일반 경로 다음 배치의 셀별 위치 중앙값과 각도 원형 평균.
    흩어짐 = 그 기준에서의 편차 최대. 적재 직후에는 준비 순환을 거치므로 묶음 첫 배치도 일반 경로 다음이다.
    경로 비교 단계 배치는 기준에 넣지 않는다.
    경로 비교에서 복귀 경로 다음(같은 셀) 배치가 그 셀 흩어짐을 넘으면 transfer=unverified다.
    노란 화소는 집기 성공 사진(일반·오프셋·복귀 재집기) 전부의 최대다.
    """
    import re
    import statistics

    frames = {r.get("frame"): r for r in records if r["event"] == "note" and r.get("kind") == "placement" and r.get("ok")}
    rows = []
    for note in (r for r in records if r["event"] == "note" and r.get("kind") == "spread_place"):
        m = frames[note["frame"]]["measurement"]
        rows.append({"cell": note["cell"], "phase": note.get("phase"), "after": note["after_path"], "x": m["position"][0], "y": m["position"][1], "angle": m["angle"]})
    cells = {}
    for cell in sorted({row["cell"] for row in rows}):
        base = [row for row in rows if row["cell"] == cell and row["after"] == "plain" and row["phase"] == "spread"]
        if not base:
            continue
        center = (statistics.median(r["x"] for r in base), statistics.median(r["y"] for r in base), _circular_mean_deg([r["angle"] for r in base]))

        def dev(row, center=center):
            return math.dist((row["x"], row["y"]), center[:2]), abs(wrap_deg(row["angle"] - center[2]))

        pos_max = max(dev(r)[0] for r in base)
        ang_max = max(dev(r)[1] for r in base)
        returned = [dev(r) for r in rows if r["cell"] == cell and r["after"] == "return"]
        # 적재 직후 준비 순환 배치가 이 셀의 필터(기준±흩어짐 최대)를 얼마나 통과하는지. 1차는 이 배치에 필터를 쓰지 않는다.
        warm = []
        for r in records:
            if r["event"] == "note" and r.get("kind") == "placement" and r.get("ok") and re.match(rf"n-c{cell}-.*-w\d*$", r.get("trial_id", "")):
                m = r["measurement"]
                warm.append(dev({"x": m["position"][0], "y": m["position"][1], "angle": m["angle"]}))
        cells[cell] = {
            "warmup_in_filter": [sum(1 for p_, a_ in warm if p_ <= pos_max and a_ <= ang_max), len(warm)],
            "center": [round(v, 3) for v in center], "position_max_px": round(pos_max, 3), "angle_max_deg": round(ang_max, 3),
            "base_n": len(base), "return_n": len(returned),
            "return_dev": [[round(p, 3), round(a, 3)] for p, a in returned],
            "return_outside": sum(1 for p, a in returned if p > pos_max or a > ang_max),
        }
    successes = [r.get("reason", "") for r in records if (r["event"] == "finish" and r.get("success")) or (r["event"] == "note" and r.get("kind") in ("return_pick", "warmup_pick") and r.get("success"))]
    yellow = [int(m.group(1)) for reason in successes if (m := re.search(r"노란 화소 (\d+)", reason))]
    finishes = [r for r in records if r["event"] == "finish"]
    side_failures = sum(1 for r in records if r["event"] == "note" and r.get("kind") in ("return_pick", "warmup_pick") and not r.get("success"))
    incomplete = sum(1 for r in records if r["event"] == "note" and r.get("kind") == "incomplete")
    return {
        "cells": cells,
        "transfer": "verified" if len(cells) == 3 and all(c["return_outside"] == 0 and c["return_n"] == COMPARE_REPEATS for c in cells.values()) else "unverified",
        "trials": len(finishes),
        "failures": sum(1 for r in finishes if not r.get("success")) + side_failures,
        "incomplete": incomplete,
        "yellow_max_success": max(yellow) if yellow else None,
        "yellow_ok": bool(yellow) and max(yellow) <= YELLOW_HOLD_MAX,
    }


def filter_from_spread(summary: dict) -> PlacementFilter:
    cells = {int(k): v for k, v in summary["cells"].items()}
    return PlacementFilter(
        {c: tuple(v["center"]) for c, v in cells.items()},
        {c: v["position_max_px"] for c, v in cells.items()},
        {c: v["angle_max_deg"] for c, v in cells.items()},
    )
