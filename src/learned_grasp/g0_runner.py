"""G0 시험표 생성과 실행. 판정 수치는 플랜 §6에 고정돼 있다.

사용:
  python -m learned_grasp.g0_runner sheet --gate G0b --seed 1 --out runs/G0b/sheet.json
  python -m learned_grasp.g0_runner summary --sheet runs/G0b/sheet.json --ledger runs/G0b/attempts.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from learned_grasp.arm import SafeArm
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import Camera
from learned_grasp.config import Config, load_config
from learned_grasp.pose_table import PoseTable, table_for
from learned_grasp.safety import GRASP, HOVER, PARK_POSE, Pose
from learned_grasp.stats import percentile, wilson_interval

G0_REQUIRED_SUCCESSES = 24
G0_REQUIRED_TOTAL = 30
G0E_MAX_SECONDS_PER_SAMPLE = 90.0
TOL_MIN_SUCCESSES = 4
TOL_MAX_FAILURES = 2
TOL_TRIALS = 5
# G0d 1차(docs/261109-2129-G0d-재설계-플랜.md v9). 셀은 r_index, J1·J5는 범위 가운데. 축·부호마다 가장 작은 서로 다른 자세 1단계.
TOL_CELLS = (1, 4, 7)
TOL_AXES = ("r", "lateral", "j5")
TOL_J1_STEP_DEG = 1
TOL_J5_STEP_DEG = 10


@dataclass(frozen=True)
class Outcome:
    success: bool
    reason: str = ""
    human_intervention: bool = False


Observer = Callable[[dict], Outcome]


def _levels(low: int, high: int, count: int) -> list[int]:
    return [round(low + (high - low) * index / (count - 1)) for index in range(count)]


def replay_sheet(config: Config, table: PoseTable, seed: int, cells: int = 10, repeats: int = 3) -> list[dict]:
    """r×J1×J5 격자에서 r 구간을 고르게 덮도록 셀을 고른다. 실패 셀을 제외하지 않는다."""
    rng = random.Random(seed)
    count = len(table.entries)
    grid = [
        (band, r_index, j1, j5)
        for band in range(3)
        for r_index in range(count * band // 3, count * (band + 1) // 3)
        for j1 in _levels(*config.j1_range, 3)
        for j5 in _levels(*config.j5_range, 3)
    ]
    rng.shuffle(grid)
    chosen: list[tuple[int, int, int, int]] = []
    for band_cycle in range(cells):
        band = band_cycle % 3
        pick = next((item for item in grid if item[0] == band and item not in chosen), None)
        if pick is None:
            pick = next(item for item in grid if item not in chosen)
        chosen.append(pick)
    trials = [
        {"trial_id": f"c{cell:02d}-t{rep}", "cell": cell, "r_index": r_index, "j1": j1, "j5": j5}
        for cell, (_, r_index, j1, j5) in enumerate(chosen)
        for rep in range(repeats)
    ]
    rng.shuffle(trials)
    return trials


def dryrun_sheet(config: Config, table: PoseTable) -> list[dict]:
    """G0a. 표 항목 전체를 J1 양끝·중앙, J5 양끝 조합에서 빈 그리퍼로 확인한다."""
    return [
        {"trial_id": f"r{r_index:02d}-j{j1}-p{j5}", "r_index": r_index, "j1": j1, "j5": j5}
        for r_index in range(len(table.entries))
        for j1 in _levels(*config.j1_range, 3)
        for j5 in config.j5_range
    ]


def tolerance_variants(config: Config, table: PoseTable) -> list[dict]:
    """G0d 1차 18변형. 놓기는 기준 자세 P, 집기는 P+δ다. 오프셋은 표 모델 값이고 물리 허용 오차가 아니다."""
    j1 = _levels(*config.j1_range, 3)[1]
    j5 = _levels(*config.j5_range, 3)[1]
    variants = []
    for cell, r_index in enumerate(TOL_CELLS):
        base = table.entries[r_index].r_mm
        for axis in TOL_AXES:
            for sign in (-1, 1):
                pick = {"pick_r_index": r_index, "pick_j1": j1, "pick_j5": j5}
                if axis == "r":
                    pick["pick_r_index"] = r_index + sign
                    if not 0 <= pick["pick_r_index"] < len(table.entries):
                        raise ValueError(f"표 끝을 넘는 r 단계: r_index={r_index}")
                    value = table.entries[pick["pick_r_index"]].r_mm - base
                elif axis == "lateral":
                    pick["pick_j1"] = j1 + sign * TOL_J1_STEP_DEG
                    value = base * math.radians(sign * TOL_J1_STEP_DEG)
                else:
                    pick["pick_j5"] = j5 + sign * TOL_J5_STEP_DEG
                    value = float(sign * TOL_J5_STEP_DEG)
                if not (config.j1_range[0] <= pick["pick_j1"] <= config.j1_range[1] and config.j5_range[0] <= pick["pick_j5"] <= config.j5_range[1]):
                    raise ValueError(f"범위를 넘는 변형: {axis}{sign:+d} 셀 {cell}")
                variants.append(
                    {
                        "variant_id": f"c{cell}-{axis}{'+' if sign > 0 else '-'}", "cell": cell, "axis": axis, "sign": sign,
                        "model_offset": round(value, 3), "r_index": r_index, "j1": j1, "j5": j5, **pick,
                    }
                )
    return variants


def tolerance_sheet(config: Config, table: PoseTable, seed: int) -> list[dict]:
    """변형마다 최대 5회 자리를 미리 만든다. 회차마다 변형 순서를 시드로 섞어 교차한다.

    결과를 본 뒤 순서를 바꾸지 않는다. 종결된 변형의 남은 자리는 실행기가 건너뛴다.
    """
    rng = random.Random(seed)
    variants = tolerance_variants(config, table)
    sheet = []
    for round_index in range(TOL_TRIALS):
        order = list(variants)
        rng.shuffle(order)
        sheet += [{**variant, "trial_id": f"{variant['variant_id']}-t{round_index}", "round": round_index} for variant in order]
    return sheet


def variant_verdict(outcomes: list[bool]) -> str | None:
    """4회 성공이면 통과, 2회 실패면 미달. 그 전에는 None이다. 5회 고정·4성공 기준과 같은 결론을 낸다."""
    if sum(outcomes) >= TOL_MIN_SUCCESSES:
        return "pass"
    if len(outcomes) - sum(outcomes) >= TOL_MAX_FAILURES:
        return "fail"
    if len(outcomes) >= TOL_TRIALS:
        raise ValueError(f"5회를 넘어 판정이 없다: {outcomes}")
    return None


def compute_tolerance(variants: list[dict], states: dict[str, dict], transfer: str) -> dict[str, dict]:
    """시험표 기준 축별 집계. states[variant_id] = {"status": pass|fail|incomplete, "reason": ...}.

    기록에 없는 변형도 미완료로 센다. 미완료가 있는 축은 confirmed=false다. 미달이 있으면 tol_cmd=0이고 G1을 통과할 수 없다.
    transfer가 verified가 아니면 usable=false다.
    """
    if transfer not in ("verified", "unverified"):
        raise ValueError(transfer)
    result = {}
    for axis in TOL_AXES:
        members = [v for v in variants if v["axis"] == axis]
        status = {v["variant_id"]: states.get(v["variant_id"], {"status": "incomplete", "reason": "기록 없음"}) for v in members}
        incomplete = {vid: s.get("reason", "") for vid, s in status.items() if s["status"] not in ("pass", "fail")}
        failed = sorted(vid for vid, s in status.items() if s["status"] == "fail")
        steps = sorted(abs(v["model_offset"]) for v in members)
        if failed:
            tol = 0.0
        elif incomplete:
            tol = None
        else:
            tol = steps[0]
        confirmed = not incomplete
        result[axis] = {
            "tol_cmd": tol,
            "unit": "deg" if axis == "j5" else "mm",
            "tested_steps": steps,
            "failed": failed,
            "incomplete": incomplete,
            "confirmed": confirmed,
            "transfer": transfer,
            "usable": bool(confirmed and not failed and transfer == "verified"),
        }
    return result


def _pose(kind: str, trial: dict, prefix: str = "") -> Pose:
    return Pose(kind, trial[f"{prefix}r_index"], trial[f"{prefix}j1"], trial[f"{prefix}j5"])


def _place(arm: SafeArm, trial: dict, prefix: str = "") -> None:
    arm.goto(_pose(HOVER, trial, prefix))
    arm.goto(_pose(GRASP, trial, prefix))
    arm.set_grip("open")
    arm.goto(_pose(HOVER, trial, prefix))
    arm.goto(PARK_POSE)


def _pick(arm: SafeArm, trial: dict, prefix: str = "") -> None:
    if arm.pose != _pose(HOVER, trial, prefix):
        arm.goto(_pose(HOVER, trial, prefix))
    arm.goto(_pose(GRASP, trial, prefix))
    arm.set_grip("close")
    arm.goto(_pose(HOVER, trial, prefix))


def _bind(arm: SafeArm, ledger: AttemptLedger, table: PoseTable) -> None:
    """기록을 구성 버전과 표에 묶는다. 다른 구성으로 같은 기록을 이어 쓰지 못한다."""
    ledger.bind(config_version=arm.gate.config.config_version, table_sha256=table.sha256())


def run_dryrun(
    arm: SafeArm,
    ledger: AttemptLedger,
    sheet: list[dict],
    observer: Observer,
    table: PoseTable,
    at_grasp: Callable[[], None] | None = None,
) -> None:
    """빈 그리퍼 드라이런. 관찰자가 이상을 보고하면 즉시 끝낸다."""
    _bind(arm, ledger, table)
    ledger.close_interrupted()
    finished, _ = ledger.state()
    stopped = sorted(trial_id for trial_id, record in finished.items() if not record["success"])
    if stopped:
        raise RuntimeError(f"이상 보고나 중단이 있는 드라이런은 재개하지 않는다. 팔을 확인하고 새 기록으로 시작한다: {stopped}")
    remaining = [trial for trial in sheet if trial["trial_id"] not in finished]
    if not remaining:
        return
    arm.goto(PARK_POSE)
    arm.set_grip("open")
    for trial in remaining:
        ledger.start(trial, joints_hover=arm.gate.joints(_pose(HOVER, trial)), joints_grasp=arm.gate.joints(_pose(GRASP, trial)))
        arm.goto(_pose(HOVER, trial))
        arm.goto(_pose(GRASP, trial))
        if at_grasp is not None:
            at_grasp()
        arm.goto(_pose(HOVER, trial))
        arm.goto(PARK_POSE)
        outcome = observer(trial)
        ledger.finish(trial["trial_id"], outcome="ok" if outcome.success else "stop", success=outcome.success, reason=outcome.reason)
        if not outcome.success:
            return


def run_replay(
    arm: SafeArm,
    camera: Camera,
    ledger: AttemptLedger,
    sheet: list[dict],
    observer: Observer,
    table: PoseTable,
    mode: str,
    load_block: Callable[[dict], None],
) -> None:
    """놓기→촬영→집기 반복.

    mode=stored는 저장한 r_index를 그대로 쓴다. mode=label은 FK로 라벨을 만든 뒤 표에서 복원한 항목으로 집는다.
    실패하면 공중에서 열고 사람이 다음 시도 위치에 물체를 놓는다. 재시도와 셀 제외는 없다.
    """
    if mode not in ("stored", "label"):
        raise ValueError(mode)
    _bind(arm, ledger, table)
    finished, _ = ledger.state()
    interrupted = ledger.close_interrupted()
    remaining = [trial for trial in sheet if trial["trial_id"] not in finished and trial["trial_id"] not in interrupted]
    if not remaining:
        return
    arm.goto(PARK_POSE)
    arm.set_grip("open")
    has_block = False
    for trial in remaining:
        trial = dict(trial)
        trial.setdefault("pick_r_index", trial["r_index"])
        trial.setdefault("pick_j1", trial["j1"])
        trial.setdefault("pick_j5", trial["j5"])
        if not has_block:
            # 사람 작업은 시도 기록 전에 끝낸다. 기다리다 끝나도 시도가 중단으로 남지 않는다.
            load_block(trial)
            _pick(arm, trial)
            arm.goto(PARK_POSE)
        ledger.start(
            trial,
            mode=mode,
            place_q=arm.gate.joints(_pose(GRASP, trial)),
            pick_q=arm.gate.joints(_pose(GRASP, trial, "pick_")),
        )
        _place(arm, trial)
        camera.snapshot(trial["trial_id"])
        if mode == "label":
            original = table.entries[trial["pick_r_index"]].grasp
            restored = table.recover(original)
            if restored != original:
                raise RuntimeError(f"라벨 복원 불일치 {trial['trial_id']}: {original} != {restored}")
        _pick(arm, trial, "pick_")
        arm.goto(PARK_POSE)
        outcome = observer(trial)
        if not outcome.success:
            # 물체가 남아 있을 수 있어 높은 park에서 열지 않는다. 낮은 대기 자세로 내려가 연다.
            arm.goto(_pose(HOVER, trial, "pick_"))
            arm.set_grip("open")
            arm.goto(PARK_POSE)
        has_block = outcome.success
        ledger.finish(
            trial["trial_id"],
            outcome="success" if outcome.success else "failure",
            success=outcome.success,
            reason=outcome.reason,
            human_intervention=outcome.human_intervention,
            read_back=arm.read(),
        )
    # 정상 종료에서만 내려놓는다. 예외와 중단에서는 자동 복구 이동을 하지 않는다. 시도 기록에는 넣지 않는다.
    if has_block:
        _place(arm, trial, "pick_")


def summarize(sheet: list[dict], ledger: AttemptLedger) -> dict:
    """사전 고정 기준으로 집계한다. 미완료면 합격 처리하지 않는다."""
    finished, open_ids = ledger.state()
    records = [finished[trial["trial_id"]] for trial in sheet if trial["trial_id"] in finished]
    successes = sum(1 for record in records if record["success"])
    total = len(sheet)
    durations = [record["t"] - record["started_at"] for record in records if record["success"] and record.get("started_at") is not None]
    complete = len(records) == total and not open_ids
    result = {
        "planned": total,
        "finished": len(records),
        "successes": successes,
        "interrupted": sum(1 for record in records if record.get("outcome") == "interrupted"),
        "human_interventions": sum(1 for record in records if record.get("human_intervention")),
        "complete": complete,
        "wilson95": wilson_interval(successes, total) if total else None,
        "seconds_per_success_mean": sum(durations) / len(durations) if durations else None,
        "seconds_per_success_p90": percentile(durations, 90) if durations else None,
    }
    # G0a·G0d에는 성공률 합격선이 없다. 30회 시험이 아니면 판정 자체를 내지 않는다.
    result["gate_pass"] = (complete and successes >= G0_REQUIRED_SUCCESSES) if total == G0_REQUIRED_TOTAL else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["sheet", "summary"])
    parser.add_argument("--config", default="config/workspace.yaml")
    parser.add_argument("--gate", choices=["G0a", "G0b", "G0d"], default="G0b")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if args.command == "summary":
        print(json.dumps(summarize(json.loads(args.sheet.read_text(encoding="utf-8")), AttemptLedger(args.ledger)), ensure_ascii=False, indent=1))
        return
    config = load_config(Path(args.config))
    table = table_for(config)
    if args.gate == "G0a":
        sheet = dryrun_sheet(config, table)
    elif args.gate == "G0d":
        sheet = tolerance_sheet(config, table, args.seed)
    else:
        sheet = replay_sheet(config, table, args.seed)
    text = json.dumps(sheet, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(f"{args.gate} 시도 {len(sheet)}건, 표 항목 {len(table.entries)}개, 구성 {config.config_version}")


if __name__ == "__main__":
    main()
