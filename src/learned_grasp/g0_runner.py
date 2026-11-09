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
TOL_TRIALS = 5
R_OFFSETS_MM = (3, 6, 10)
LATERAL_OFFSETS_MM = (3, 6, 10)
J5_OFFSETS_DEG = (10, 20)


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


def offset_sheet(config: Config, table: PoseTable, seed: int, cells: int = 3) -> list[dict]:
    """G0d. 놓은 자세에서 명령만 어긋나게 해 허용 오차를 잰다. 오프셋은 모델 좌표다."""
    rng = random.Random(seed)
    count = len(table.entries)
    j1_mid = _levels(*config.j1_range, 3)[1]
    j5_mid = _levels(*config.j5_range, 3)[1]
    centers = [count * band // cells + (count // cells) // 2 for band in range(cells)]
    trials = []
    for cell, r_index in enumerate(centers):
        base = table.entries[r_index].r_mm
        mm_per_j1_deg = math.radians(1.0) * base
        variants = [("r", sign * value, table.nearest_index(base + sign * value), j1_mid, j5_mid) for value in R_OFFSETS_MM for sign in (1, -1)]
        variants += [
            ("lateral", sign * value, r_index, j1_mid + sign * max(1, round(value / mm_per_j1_deg)), j5_mid)
            for value in LATERAL_OFFSETS_MM
            for sign in (1, -1)
        ]
        variants += [("j5", sign * value, r_index, j1_mid, j5_mid + sign * value) for value in J5_OFFSETS_DEG for sign in (1, -1)]
        for axis, offset, pick_index, pick_j1, pick_j5 in variants:
            actual_r = table.entries[pick_index].r_mm - base
            for rep in range(TOL_TRIALS):
                trials.append(
                    {
                        "trial_id": f"c{cell}-{axis}{offset:+d}-t{rep}", "cell": cell, "axis": axis, "offset": offset,
                        "actual_r_offset_mm": round(actual_r, 3),
                        "actual_lateral_offset_mm": round((pick_j1 - j1_mid) * mm_per_j1_deg, 3),
                        "r_index": r_index, "j1": j1_mid, "j5": j5_mid, "pick_r_index": pick_index, "pick_j1": pick_j1, "pick_j5": pick_j5,
                    }
                )
    rng.shuffle(trials)
    return trials


def compute_tolerance(records: list[dict], axis: str, offsets: tuple[int, ...]) -> int:
    """셀·부호별로 작은 오프셋부터 연속으로 기준을 넘은 최대값을 구하고 그중 최솟값을 쓴다.

    작은 오프셋에서 실패하면 큰 오프셋이 성공해도 인정하지 않는다. 모두 실패하면 0이고, 그러면 G1을 통과할 수 없다.
    """
    cells = sorted({record["cell"] for record in records if record.get("axis") == axis})
    if not cells:
        raise ValueError(f"{axis} 축 기록이 없다")
    tolerances = []
    for cell in cells:
        for sign in (1, -1):
            best = 0
            for value in sorted(offsets):
                matching = [r for r in records if r.get("axis") == axis and r["cell"] == cell and r["offset"] == sign * value]
                if sum(1 for r in matching if r["success"]) < TOL_MIN_SUCCESSES:
                    break
                best = value
            tolerances.append(best)
    return min(tolerances)


def _pose(kind: str, trial: dict, prefix: str = "") -> Pose:
    return Pose(kind, trial[f"{prefix}r_index"], trial[f"{prefix}j1"], trial[f"{prefix}j5"])


def _place(arm: SafeArm, trial: dict) -> None:
    arm.goto(_pose(HOVER, trial))
    arm.goto(_pose(GRASP, trial))
    arm.set_grip("open")
    arm.goto(_pose(HOVER, trial))
    arm.goto(PARK_POSE)


def _pick(arm: SafeArm, trial: dict, prefix: str = "") -> None:
    arm.goto(_pose(HOVER, trial, prefix))
    arm.goto(_pose(GRASP, trial, prefix))
    arm.set_grip("close")
    arm.goto(_pose(HOVER, trial, prefix))


def _bind(arm: SafeArm, ledger: AttemptLedger, table: PoseTable) -> None:
    """기록을 구성 버전과 표에 묶는다. 다른 구성으로 같은 기록을 이어 쓰지 못한다."""
    ledger.bind(config_version=arm.gate.config.config_version, table_sha256=table.sha256())


def run_dryrun(arm: SafeArm, ledger: AttemptLedger, sheet: list[dict], observer: Observer, table: PoseTable) -> None:
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
        ledger.start(
            trial,
            mode=mode,
            place_q=arm.gate.joints(_pose(GRASP, trial)),
            pick_q=arm.gate.joints(_pose(GRASP, trial, "pick_")),
        )
        if not has_block:
            load_block(trial)
            _pick(arm, trial)
            arm.goto(PARK_POSE)
        _place(arm, trial)
        camera.snapshot(trial["trial_id"])
        if mode == "label":
            original = table.entries[trial["pick_r_index"]].grasp
            restored = table.recover(original)
            if restored != original:
                raise RuntimeError(f"라벨 복원 불일치 {trial['trial_id']}: {original} != {restored}")
        _pick(arm, trial, "pick_")
        outcome = observer(trial)
        if not outcome.success:
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
        sheet = offset_sheet(config, table, args.seed)
    else:
        sheet = replay_sheet(config, table, args.seed)
    text = json.dumps(sheet, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(f"{args.gate} 시도 {len(sheet)}건, 표 항목 {len(table.entries)}개, 구성 {config.config_version}")


if __name__ == "__main__":
    main()
