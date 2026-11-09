"""실기 세션 실행. 사용자가 승인한 세션 파일 없이는 드라이버를 만들지 않는다.

사용:
  python -m learned_grasp.session read    --session runs/G0a-1/approval.json
  python -m learned_grasp.session ladder  --session runs/G0a-2/approval.json --z-offset 15 --start-confirmed
  python -m learned_grasp.session observe --session runs/G0a-2/approval.json --z-offset 15 --ok --note "간격 충분"
  python -m learned_grasp.session dryrun  --session runs/G0a-1/approval.json --start-confirmed
  python -m learned_grasp.session replay  --session runs/G0b-1/approval.json --sheet runs/G0b-1/sheet.json --mode stored --start-confirmed
  python -m learned_grasp.session ready   --session runs/G0b-1/approval.json

대화형 입력이 없다. 감독자는 전원 차단으로 멈춘다. --start-confirmed는 사용자가 빈 그리퍼, 시작 자세,
park까지의 경로를 눈으로 확인했다고 답한 뒤에만 붙인다. 높이 탐색은 앞 단계의 관찰 통과 기록이 있어야 다음 단계로 간다.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from dataclasses import replace
from pathlib import Path

from learned_grasp.arm import RemoteArm, SafeArm, SessionApproval, load_approval
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import P620Camera
from learned_grasp.config import Config, load_config
from learned_grasp.g0_runner import Outcome, dryrun_sheet, run_dryrun, run_replay
from learned_grasp.pose_table import PoseTable, generate
from learned_grasp.safety import HOVER, Pose, SafetyGate, SafetyViolation

# 구성 높이보다 위에서 시작해 내려온다. G0a-1은 중앙 한 곳만 확인했으므로 위치별 차이를 본다.
LADDER_OFFSETS_MM = (15, 5, 0)


class CountingDriver:
    """세션 전체의 쓰기 수를 파일에 누적해 승인 범위를 넘지 못하게 한다."""

    def __init__(self, driver, path: Path, limit: int):
        self.driver = driver
        self.path = Path(path)
        self.limit = limit

    def used(self) -> int:
        return int(self.path.read_text()) if self.path.exists() else 0

    def write6(self, q5, grip, time_ms):
        used = self.used()
        if used >= self.limit:
            raise SafetyViolation(f"세션 쓰기 상한 {self.limit}회에 도달했다")
        self._count(used + 1)
        self.driver.write6(q5, grip, time_ms)

    def write_grip(self, grip, time_ms):
        used = self.used()
        if used >= self.limit:
            raise SafetyViolation(f"세션 쓰기 상한 {self.limit}회에 도달했다")
        self._count(used + 1)
        self.driver.write_grip(grip, time_ms)

    def _count(self, value: int) -> None:
        """전송 전에 원자적으로 저장한다. 전송 실패도 쓴 것으로 센다."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("w") as handle:
            handle.write(str(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def read(self):
        return self.driver.read()

    def close(self):
        self.driver.close()


def ladder_config(config: Config, offset_mm: float) -> Config:
    """높이 탐색 표의 구성. 구성 버전에 오프셋을 넣어 기록이 섞이지 않게 한다."""
    table = replace(config.table, z_grasp_mm=config.table.z_grasp_mm + offset_mm)
    return replace(config, table=table, config_version=f"{config.config_version}+z{offset_mm:g}")


def ladder_sheet(config: Config, table: PoseTable) -> list[dict]:
    """표의 양끝·중앙 반경과 J1 양끝을 내려간다. 작업 영역 모서리의 높이 차이를 보기 위해서다."""
    last = len(table.entries) - 1
    return [
        {
            "trial_id": f"ladder-{config.config_version}-r{r_index}-j{j1}",
            "r_index": r_index,
            "j1": j1,
            "j5": sum(config.j5_range) // 2,
        }
        for r_index in (0, last // 2, last)
        for j1 in config.j1_range
    ]


def record_observation(folder: Path, offset: float, ok: bool, note: str) -> None:
    with (folder / "observations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"z_offset": offset, "ok": ok, "note": note}, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def require_previous_observation(folder: Path, offset: float) -> None:
    """명령 완료는 관찰 통과가 아니다. 앞 단계의 사람 관찰 통과 없이는 더 내려가지 않는다."""
    index = LADDER_OFFSETS_MM.index(offset)
    path = folder / "observations.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    if any(not record["ok"] for record in records):
        raise SafetyViolation("이상 관찰 기록이 있어 높이 탐색을 멈춘다. 새 세션이 필요하다")
    if index > 0 and not any(record["z_offset"] == LADDER_OFFSETS_MM[index - 1] and record["ok"] for record in records):
        raise SafetyViolation(f"앞 단계 z+{LADDER_OFFSETS_MM[index - 1]}의 관찰 통과 기록이 없다")


def read_start_joints(driver, folder: Path) -> tuple[int, ...]:
    """첫 쓰기 전 시작값을 읽어 남긴다. 거리 검사에만 쓰고 목표로 쓰지 않는다."""
    reading = driver.read()
    if reading.get("status") != ["sdk_return"] * 6:
        raise SafetyViolation(f"시작 조회 실패로 이동하지 않는다: {reading}")
    (folder / "start_read.json").write_text(json.dumps(reading, ensure_ascii=False) + "\n", encoding="utf-8")
    return tuple(reading["values"][:5])


def run_ladder(arm: SafeArm, ledger: AttemptLedger, config: Config, table: PoseTable, hold_s: float = 0.0) -> None:
    """hold_s 동안 파지 자세에 머물러 사진과 관찰을 받는다. 추가 쓰기는 없다."""
    at_grasp = (lambda: arm.sleep(hold_s)) if hold_s > 0 else None
    run_dryrun(arm, ledger, ladder_sheet(config, table), _unobserved, table, at_grasp)


# 작업 영역과 그 둘레. 카메라를 옮기면 다시 정한다. 화소 좌표이고 실측 2026-11-09 사진 기준이다.
PICK_ROI = (540, 860, 190, 420)
# 블록 윗면은 약 45×71 화소로 ROI의 약 4%다. 첫 시도에서 실측해 다시 정한다.
PICK_CHANGE_MIN = 0.015


class NeedHuman(RuntimeError):
    """사람 작업을 정해진 시간 안에 받지 못했다."""


def make_pick_observer(camera, threshold: float = PICK_CHANGE_MIN, roi=PICK_ROI, log=print):
    """놓은 뒤 사진과 집은 뒤 사진의 차이로 물체가 자리를 떠났는지 본다.

    공중에 들려 유지되는지는 이 값으로 알 수 없다. 낙하는 사진과 감독자 관찰로 따로 본다.
    """

    def observe(trial: dict) -> Outcome:
        after = f"{trial['trial_id']}-after"
        camera.snapshot(after)
        fraction = camera.change(trial["trial_id"], after, roi)
        log(json.dumps({"trial_id": trial["trial_id"], "change": round(fraction, 4)}), flush=True)
        if fraction >= threshold:
            return Outcome(True, f"변화 {fraction:.4f}")
        return Outcome(False, f"물체가 자리에 남은 것으로 판정, 변화 {fraction:.4f}", human_intervention=True)

    return observe


def make_block_loader(arm: SafeArm, ready_path: Path, wait_s: float = 300.0, sleep=None, clock=None, log=print):
    """팔을 그 자리 대기 자세로 보내고 사람이 물체를 놓을 때까지 기다린다. 위치 표시 역할이다."""
    import time as _time

    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic

    def load(trial: dict) -> None:
        arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))
        ready_path.unlink(missing_ok=True)
        log(json.dumps({"need_block": trial["trial_id"]}, ensure_ascii=False), flush=True)
        end = clock() + wait_s
        while not ready_path.exists():
            if clock() >= end:
                raise NeedHuman(f"{wait_s:g}초 안에 물체 적재 확인을 받지 못했다")
            sleep(1.0)
        ready_path.unlink(missing_ok=True)

    return load


def _unobserved(trial: dict) -> Outcome:
    """프로그램은 무간섭을 판정하지 못한다. 이 성공은 명령 완료이고 관찰은 세션 문서에 따로 남긴다."""
    return Outcome(True, "명령 완료, 관찰은 세션 문서")


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def _driver(approval: SessionApproval, used: int) -> RemoteArm:
    remaining = approval.max_writes - used
    limited = SessionApproval(approval.session_id, approval.gate, max(remaining, 1), approval.token)
    return RemoteArm(limited)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["read", "ladder", "observe", "dryrun", "replay", "ready"])
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--mode", choices=["stored", "label"], default="stored")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/workspace.yaml"))
    parser.add_argument("--z-offset", type=float)
    parser.add_argument("--start-confirmed", action="store_true", help="사용자가 빈 그리퍼, 시작 자세, park 경로를 확인했다")
    parser.add_argument("--hold-sec", type=float, default=0.0)
    parser.add_argument("--ok", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    _load_env()
    config = load_config(args.config)
    approval = load_approval(args.session, config)
    folder = args.session.parent
    lock = (folder / "session.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("이 세션을 쓰는 다른 프로세스가 있다")
    counter = folder / "writes.count"
    used = int(counter.read_text()) if counter.exists() else 0

    if args.command in ("ladder", "observe") and args.z_offset not in LADDER_OFFSETS_MM:
        raise SystemExit(f"--z-offset는 {LADDER_OFFSETS_MM} 중 하나다")
    if args.command == "observe":
        record_observation(folder, args.z_offset, args.ok, args.note)
        print(f"관찰 기록: z+{args.z_offset:g} ok={args.ok}")
        return
    if args.command == "ready":
        (folder / "block_ready").write_text("1")
        print("물체 적재 확인을 기록했다")
        return
    if args.command in ("ladder", "dryrun", "replay") and not args.start_confirmed:
        raise SystemExit("--start-confirmed 없이는 움직이지 않는다")
    if args.command == "ladder":
        require_previous_observation(folder, args.z_offset)
        config = ladder_config(config, args.z_offset)
    table = generate(config.table)
    driver = _driver(approval, used)
    try:
        if args.command == "read":
            print(json.dumps(driver.read(), ensure_ascii=False))
            return
        start = read_start_joints(driver, folder)
        arm = SafeArm(CountingDriver(driver, counter, approval.max_writes), SafetyGate(config, table), start_joints=start)
        if args.command == "replay":
            camera = P620Camera(f"runs/{approval.session_id}/frames")
            try:
                sheet = json.loads(args.sheet.read_text(encoding="utf-8"))
                run_replay(
                    arm, camera, AttemptLedger(folder / f"replay-{args.mode}.jsonl"), sheet,
                    make_pick_observer(camera), table, args.mode, make_block_loader(arm, folder / "block_ready"),
                )
            finally:
                camera.close()
        elif args.command == "ladder":
            run_ladder(arm, AttemptLedger(folder / f"ladder-z{args.z_offset:g}.jsonl"), config, table, min(args.hold_sec, 60.0))
        else:
            run_dryrun(arm, AttemptLedger(folder / "dryrun.jsonl"), dryrun_sheet(config, table), _unobserved, table)
        print(f"완료. 세션 쓰기 누적 {counter.read_text()}회. 무간섭은 관찰 기록으로만 판정한다.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
