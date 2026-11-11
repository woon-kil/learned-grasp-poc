"""실기 세션 실행. 사용자가 승인한 세션 파일 없이는 드라이버를 만들지 않는다.

사용:
  python -m learned_grasp.session read    --session runs/G0a-1/approval.json
  python -m learned_grasp.session ladder  --session runs/G0a-2/approval.json --z-offset 15 --start-confirmed
  python -m learned_grasp.session observe --session runs/G0a-2/approval.json --z-offset 15 --ok --note "간격 충분"
  python -m learned_grasp.session dryrun  --session runs/G0a-1/approval.json --start-confirmed
  python -m learned_grasp.session replay  --session runs/G0b-1/approval.json --sheet runs/G0b-1/sheet.json --mode stored --start-confirmed
  python -m learned_grasp.session ready   --session runs/G0b-1/approval.json
  python -m learned_grasp.session natural-spread --session runs/G0d-0/approval.json --start-confirmed
  python -m learned_grasp.session tolerance --session runs/G0d-1/approval.json --sheet results/G0d/sheet.json --spread runs/G0d-0/spread.json --start-confirmed
  python -m learned_grasp.session answer  --session runs/G0d-0/approval.json   (다른 터미널, 실행 중 질문에 답한다)
  python -m learned_grasp.session dabai-view --session runs/DB-1/approval.json --start-confirmed [--r-index N] [--j1 J] [--hover-only]   (도착 줄마다 촬영 뒤 `touch runs/DB-1/dabai_next`)
  python -m learned_grasp.session p1-pick --session runs/P1/approval.json --r-index R --j1 J --trial p1-NN --start-confirmed   (hover에서 멈춤, 하강 확인 뒤 `touch runs/P1/dabai_next`)
  python -m learned_grasp.session chain --session runs/CH/approval.json --cells config/chain_cells.json --count N --board-ref X,Y --start-confirmed   (매 하강 전 hover에서 멈춤, 확인 뒤 `touch runs/CH/dabai_next`)
  python -m learned_grasp.session grip-max --session runs/H2-C-r4/approval.json --z-offset 15 --r-index 4 --max-sdk 180 --start-confirmed   (단계마다 읽은 뒤 `touch runs/H2-C-r4/dabai_next`, 중지는 `touch runs/H2-C-r4/grip_stop`)

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

from learned_grasp.answers import FileAnswers, answer_once
from learned_grasp.arm import CloseWatch, RemoteArm, SafeArm, SessionApproval, WatchedDriver, load_approval
from learned_grasp.attempts import AttemptLedger
from learned_grasp.camera import P620Camera
from learned_grasp.config import Config, load_config
from learned_grasp.g0_runner import Outcome, compute_tolerance, dryrun_sheet, run_dryrun, run_replay, tolerance_variants
from learned_grasp.l1 import YawModel, block_from_image, plan, plan_aligned
from learned_grasp.pose_table import PoseTable, generate
from learned_grasp.safety import GRASP, GRIP_SWEEP_MAX_SDK, GRIP_SWEEP_STEP, HOVER, PARK_POSE, Pose, SafetyGate, SafetyViolation

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
# 판정값 재계획(docs/261110-1008-G0d-판정값-재계획-플랜.md). 변화는 보조 조건이다. 움직임 없는 장면 0.0005,
# 블록 아닌 변화 0.007과 윗면 마커 블록 제거 최소 0.0182 사이다. 블록이 남았는지는 노란색이 가른다.
PICK_CHANGE_MIN = 0.01
# 보드 안쪽 사각형들의 합. 바깥의 나무는 노란색으로 잡힌다. 위쪽 y 150까지와 종이 왼쪽 띠다.
# 빈 보드 0~4, 쥔 블록 운반 사진 최대 52, 계약 3 남은 블록 최소 415화소 사이의 고정값이다.
PICK_BOARD_ROI = ((585, 825, 150, 395), (565, 585, 250, 395))
PICK_YELLOW_MAX = 100


class NeedHuman(RuntimeError):
    """사람 작업을 정해진 시간 안에 받지 못했다."""


def make_pick_observer(
    camera,
    threshold: float = PICK_CHANGE_MIN,
    roi=PICK_ROI,
    log=print,
    board_roi=PICK_BOARD_ROI,
    yellow_max: int = PICK_YELLOW_MAX,
):
    """놓은 뒤 사진과의 차이가 있고 보드에 노란색이 없으면 물체가 보드를 떠났다고 본다.

    밀린 블록은 차이가 커도 보드에 노란색이 남아 실패가 된다. 보드 밖 낙하와 공중 유지는 이 값으로
    알 수 없어 감독자가 본다(런북 G0b 낙하 규칙).
    """

    def observe(trial: dict) -> Outcome:
        after = f"{trial['trial_id']}-after"
        camera.snapshot(after)
        fraction = camera.change(trial.get("before_frame", trial["trial_id"]), after, roi)
        yellow = sum(camera.yellow_pixels(after, roi) for roi in board_roi)
        log(json.dumps({"trial_id": trial["trial_id"], "change": round(fraction, 4), "yellow": yellow}), flush=True)
        values = f"변화 {fraction:.4f}, 노란 화소 {yellow}"
        if fraction >= threshold and yellow <= yellow_max:
            return Outcome(True, values)
        return Outcome(False, f"물체가 보드에 남은 것으로 판정, {values}", human_intervention=True)

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


# 닫으면 손끝이 약 16~20 mm 돌출하므로 높은 곳부터 닫아 본다. 모델 TCP 높이다.
# 17.5 mm에서 닫은 손끝은 작업대 위 약 2~6 mm로 추정된다. 그보다 낮게 닫지 않는다.
FIT_Z_MM = (27.5, 22.5, 17.5)


def fit_config(config: Config, z_mm: float) -> Config:
    if z_mm not in FIT_Z_MM:
        raise SafetyViolation(f"--z-mm는 {FIT_Z_MM} 중 하나다")
    return ladder_config(config, z_mm - config.table.z_grasp_mm)


def fit_trial(config: Config, table: PoseTable) -> dict:
    return {
        "trial_id": f"fit-{config.config_version}",
        "r_index": table.nearest_index(140.0),
        "j1": sum(config.j1_range) // 2,
        "j5": sum(config.j5_range) // 2,
    }


def run_fit(arm: SafeArm, config: Config, table: PoseTable, load_block, hold_s: float) -> None:
    """닫지 않는다. 대기 자세에서 물체 확인을 받고 내려가 머문 뒤 돌아온다."""
    trial = fit_trial(config, table)
    arm.goto(PARK_POSE)
    load_block(trial)
    arm.goto(Pose(GRASP, trial["r_index"], trial["j1"], trial["j5"]))
    arm.sleep(hold_s)
    arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))
    arm.goto(PARK_POSE)


def run_grip_test(arm: SafeArm, config: Config, table: PoseTable, load_block, hold_s: float) -> None:
    """집어 들어 머문 뒤 같은 자리에 내려놓고 연다. park로 들고 가지 않는다."""
    trial = fit_trial(config, table)
    hover = Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"])
    grasp = Pose(GRASP, trial["r_index"], trial["j1"], trial["j5"])
    arm.goto(PARK_POSE)
    load_block(trial)
    arm.goto(grasp)
    arm.set_grip("close")
    arm.goto(hover)
    arm.sleep(hold_s)
    arm.goto(grasp)
    arm.set_grip("open")
    arm.goto(hover)
    arm.goto(PARK_POSE)


CARRY_ATTEMPTS = 3


def run_carry_test(
    arm: SafeArm, ledger: AttemptLedger, config: Config, table: PoseTable, load_block, hold_s: float, attempt: int
) -> None:
    """구성 높이에서 집어 park까지 들고 갔다가 같은 자리에 내려놓는다. 판정은 관찰이고 재시도는 없다."""
    if not 1 <= attempt <= CARRY_ATTEMPTS:
        raise SafetyViolation(f"--attempt는 1~{CARRY_ATTEMPTS}이다")
    ledger.bind(config_version=config.config_version, table_sha256=table.sha256())
    ledger.close_interrupted()
    finished, _ = ledger.state()
    stopped = sorted(trial_id for trial_id, record in finished.items() if record.get("outcome") == "interrupted")
    if stopped:
        raise SafetyViolation(f"중단된 시도가 있어 더 진행하지 않는다: {stopped}")
    trial = {**fit_trial(config, table), "trial_id": f"carry-{config.config_version}-{attempt}"}
    if trial["trial_id"] in finished:
        raise SafetyViolation(f"이미 수행한 시도다: {trial['trial_id']}")
    hover = Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"])
    grasp = Pose(GRASP, trial["r_index"], trial["j1"], trial["j5"])
    arm.goto(PARK_POSE)
    load_block(trial)
    ledger.start(trial, joints_grasp=arm.gate.joints(grasp), joints_hover=arm.gate.joints(hover))
    arm.goto(grasp)
    arm.set_grip("close")
    arm.goto(hover)
    arm.goto(PARK_POSE)
    arm.sleep(hold_s)
    read_at_park = arm.read()
    arm.goto(hover)
    arm.goto(grasp)
    arm.set_grip("open")
    arm.goto(hover)
    arm.goto(PARK_POSE)
    ledger.finish(
        trial["trial_id"], outcome="ok", success=True, reason="명령 완료, 관찰은 세션 문서", read_at_park=read_at_park
    )


def require_site_folder(folder: Path, r_index: int) -> None:
    """관찰 기록에 위치가 없으므로 위치마다 폴더를 따로 두어 앞 단계 관찰이 섞이지 않게 한다."""
    if not folder.name.endswith(f"-r{r_index}"):
        raise SafetyViolation(f"--r-index {r_index}는 이름이 -r{r_index}로 끝나는 세션 폴더에서만 쓴다: {folder.name}")


def run_close_check(arm: SafeArm, config: Config, table: PoseTable, hold_s: float, r_index: int | None = None) -> None:
    """빈 손으로 닫아 머문 뒤 연다. 닫은 손끝과 작업대의 간격은 사람이 읽는다."""
    trial = fit_trial(config, table)
    if r_index is not None:
        if not 0 <= r_index < len(table.entries):
            raise SafetyViolation(f"--r-index {r_index}는 표 범위 밖이다")
        trial = {**trial, "r_index": r_index}
    hover = Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"])
    grasp = Pose(GRASP, trial["r_index"], trial["j1"], trial["j5"])
    arm.goto(PARK_POSE)
    arm.goto(hover)
    arm.goto(grasp)
    arm.set_grip("close")
    arm.sleep(hold_s)
    arm.set_grip("open")
    arm.goto(hover)
    arm.goto(PARK_POSE)


def descend_and_stop(arm: SafeArm, config: Config, table: PoseTable) -> None:
    """전원 차단 거동 확인용. 낮은 자세까지 간 뒤 복귀하지 않고 끝난다. 서보가 그 자세를 유지한다."""
    trial = ladder_sheet(config, table)[2]
    trial = {**trial, "j1": sum(config.j1_range) // 2}
    arm.goto(PARK_POSE)
    arm.goto(Pose(HOVER, trial["r_index"], trial["j1"], trial["j5"]))
    arm.goto(Pose(GRASP, trial["r_index"], trial["j1"], trial["j5"]))


DABAI_VIEW_R_INDEX = 4


def _wait_dabai_signal(signal: Path, name: str, wait_s: float, sleep, clock, log, stop: Path | None = None) -> bool:
    """도착 줄을 내고 신호 파일을 기다린다. stop 파일이 먼저 생기면 True(정지 요청), 신호면 False를 돌려준다."""
    signal.unlink(missing_ok=True)
    log(json.dumps({"arrived": name}, ensure_ascii=False), flush=True)
    end = clock() + wait_s
    while not signal.exists() and not (stop is not None and stop.exists()):
        if clock() >= end:
            raise NeedHuman(f"{wait_s:g}초 안에 촬영 끝 신호를 받지 못했다({name})")
        sleep(1.0)
    signal.unlink(missing_ok=True)
    return stop is not None and stop.exists()


def run_dabai_view(
    arm: SafeArm,
    signal: Path,
    wait_s: float = 300.0,
    sleep=None,
    clock=None,
    log=print,
    r_index: int = DABAI_VIEW_R_INDEX,
    hover_only: bool = False,
    j1: int = 90,
) -> None:
    """DaBai 작업 거리 정지 촬영용. park→hover→grasp→hover→park를 열린 손으로 한 번 돈다.

    hover_only면 hover에서 신호를 받은 뒤 grasp로 가지 않고 park로 돌아온다(쓰기 3회).

    hover와 grasp에 도착하면 도착 줄을 내고 signal 파일이 생길 때까지 쉰다. 촬영은 그 사이에 한다. 닫지 않는다.
    대기 상한을 넘기면(안전 위반이 아니다) 정상 전이로 park까지 돌아온다. 안전 위반은 자동 쓰기 없이 올라간다.
    """
    import time as _time

    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    hover = Pose(HOVER, r_index, j1, 90)  # J1 범위는 게이트가 검사한다(j1_range)
    grasp = Pose(GRASP, r_index, j1, 90)
    arm.goto(PARK_POSE)
    arm.goto(hover)
    if hover_only:
        try:
            _wait_dabai_signal(signal, "hover", wait_s, sleep, clock, log)
        except NeedHuman:
            arm.goto(PARK_POSE)
            raise
        arm.goto(PARK_POSE)
        return
    try:
        _wait_dabai_signal(signal, "hover", wait_s, sleep, clock, log)
        arm.goto(grasp)
        _wait_dabai_signal(signal, "grasp", wait_s, sleep, clock, log)
    except NeedHuman:
        if arm.pose.kind == "grasp":
            arm.goto(hover)
        arm.goto(PARK_POSE)
        raise
    arm.goto(hover)
    arm.goto(PARK_POSE)


P1_GRIP_READ_MAX = 151  # 블록을 쥐면 서보 6 판독 148~149(G0b-4), 빈 손 SDK 155에서 154(H2-C). 이보다 크면 빈 손으로 본다


def run_p1(
    arm: SafeArm,
    camera,
    observer,
    ledger: AttemptLedger,
    config: Config,
    table: PoseTable,
    trial: dict,
    signal: Path,
    hold_s: float = 5.0,
    wait_s: float = 300.0,
    sleep=None,
    clock=None,
    log=print,
) -> None:
    """사람이 놓은 블록을 L1이 정한 (r_index, J1, J5 90)에서 집어 park까지 들고 갔다가 같은 자리에 내려놓는다.

    시작 전 사진을 찍고 hover에서 멈춰 하강 전 확인 신호(signal)를 기다린다. 성공 = 집기 관찰기(변화·노란 화소)와
    서보 6 판독(P1_GRIP_READ_MAX 이하)이 모두 참이다. 재시도는 없다. 확인 신호가 없으면 park로 돌아가고 시도는 중단으로 끝난다.
    """
    import time as _time

    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    ledger.bind(config_version=config.config_version, table_sha256=table.sha256())
    ledger.close_interrupted()
    finished, _ = ledger.state()
    stopped = sorted(trial_id for trial_id, record in finished.items() if record.get("outcome") == "interrupted")
    if stopped:
        raise SafetyViolation(f"중단된 시도가 있어 더 진행하지 않는다: {stopped}")
    if trial["trial_id"] in finished:
        raise SafetyViolation(f"이미 수행한 시도다: {trial['trial_id']}")
    hover = Pose(HOVER, trial["r_index"], trial["j1"], 90)
    grasp = Pose(GRASP, trial["r_index"], trial["j1"], 90)
    camera.snapshot(trial["trial_id"])  # 블록이 보드에 있고 팔은 park일 때의 사진
    arm.goto(PARK_POSE)
    arm.goto(hover)
    ledger.start(trial, joints_grasp=arm.gate.joints(grasp), joints_hover=arm.gate.joints(hover))
    try:
        _wait_dabai_signal(signal, "descend", wait_s, sleep, clock, log)
    except NeedHuman:
        arm.goto(PARK_POSE)
        ledger.finish(trial["trial_id"], outcome="interrupted", success=False, reason="하강 전 확인을 받지 못했다")
        raise
    arm.goto(grasp)
    arm.set_grip("close")
    arm.goto(hover)
    arm.goto(PARK_POSE)
    arm.sleep(hold_s)
    read_at_park = arm.read()
    outcome = observer(trial)
    values = read_at_park.get("values")
    grip = values[5] if values and len(values) > 5 else None
    holds = grip is not None and grip <= P1_GRIP_READ_MAX
    success = bool(outcome.success and holds)
    reason = f"{outcome.reason}, 서보 6 판독 {grip}" + ("" if holds else f" > {P1_GRIP_READ_MAX}(빈 손으로 봄)")
    arm.goto(hover)
    arm.goto(grasp)
    arm.set_grip("open")
    arm.goto(hover)
    arm.goto(PARK_POSE)
    ledger.finish(trial["trial_id"], outcome="success" if success else "failed", success=success, reason=reason, read_at_park=read_at_park, grip_read=grip)


def run_chain(
    arm: SafeArm,
    camera,
    observer,
    ledger: AttemptLedger,
    config: Config,
    table: PoseTable,
    cells: list[dict],
    count: int,
    signal: Path,
    measure,
    hold_s: float = 5.0,
    wait_s: float = 300.0,
    sleep=None,
    clock=None,
    log=print,
    confirm_each: bool = False,
    rest_every: int = 10,
    rest_s: float = 300.0,
    stop: Path | None = None,
    align: bool = False,
    yaw: YawModel | None = None,
) -> None:
    """연쇄 종단 시험. park 사진 → L1이 고른 자리에서 집기 → park → 목록의 다음 자리에 내려놓기를 count번 이어 간다.

    사람은 시작 때 블록을 놓고, **이 실행의 첫 집기 하강 전에만** 확인 신호(signal)를 보낸다(사용자 결정 2026-11-11: 그 뒤는 전원 차단으로 감독). confirm_each면 집기·내려놓기마다 받는다. 닫기 rest_every번마다 rest_s초 쉰다(그리퍼 쉼, 쓰기 없음). stop 파일이 있으면 시도 사이(팔 park, 손 열림, 블록 보드 위)나 쉼 중에 쓰기 없이 멈춘다. 시도 중에는 전원 차단뿐이다. 실패·중단 기록은 사람이 확인(ledger note acknowledged)해야 이어 간다. 다음 시도의 입력 사진은 직전 내려놓기 뒤의 park 사진이다.
    align면 J5로 손을 블록 방향에 맞추고(plan_aligned) 놓는 자리에서도 J5로 손을 반경선에 맞춘다(YawModel). J5 ≠ 90 하강은 처음 한 번씩만 확인한다.
    성공 = P1과 같다(관찰기 그리고 서보 6 판독). L1이 거절(검출 실패, 구역·방향 밖)하거나 시도가 실패하거나 확인 신호가 없으면
    쓰기 없이(실패면 P1처럼 같은 자리에 도로 놓고) 멈춘다. 자동 복구 이동은 없다. cells[k-1]이 k번째 시도의 내려놓을 자리다.
    """
    import time as _time

    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    ledger.bind(config_version=config.config_version, table_sha256=table.sha256())
    ledger.close_interrupted()
    finished, _ = ledger.state()
    acknowledged = {r.get("trial_id") for r in ledger.records() if r.get("event") == "note" and r.get("kind") == "acknowledged"}
    stopped = sorted(trial_id for trial_id, record in finished.items() if record.get("outcome") in ("interrupted", "failed") and trial_id not in acknowledged)
    if stopped:
        raise SafetyViolation(f"중단·실패한 시도가 있어 사람 확인 전에는 진행하지 않는다(--ack): {stopped}")
    yaw = yaw or (YawModel() if align else None)
    confirmed_j5 = {"pick": False, "place": False}
    first = len(finished) + 1
    last_success = first > 1 and finished.get(f"chain-{first - 1:03d}", {}).get("outcome") == "success"
    if first - 1 + count > len(cells):
        raise SafetyViolation(f"자리 목록({len(cells)}개)이 모자란다: 이미 {first - 1}개, 요청 {count}개")
    for k in range(first, first + count):
        tid = f"chain-{k:03d}"
        nxt = cells[k - 1]
        if stop is not None and stop.exists():
            ledger.note("stop_requested", trial_id=tid)
            raise NeedHuman(f"{tid}: 정지 파일이 있어 시도 전에 멈춘다(팔 park, 쓰기 없음)")
        rejected_before = sum(1 for r in ledger.records() if r.get("event") == "note" and r.get("kind") == "rejected" and r.get("trial_id") == tid)
        frame = tid if rejected_before == 0 else f"{tid}-r{rejected_before}"  # 거절된 사진을 덮어쓰지 않는다
        camera.snapshot(frame)  # 블록이 보드에 있고 팔은 park일 때의 사진
        measured = measure(camera.fetch(frame))
        planned = plan_aligned(measured, table, yaw=yaw) if align else plan(measured, table)
        if not planned.accept:
            ledger.note("rejected", trial_id=tid, **planned.asdict())
            raise NeedHuman(f"{tid}: L1이 시도하지 않는다({planned.reason}). 사람이 블록을 확인한다")
        pick_j5 = planned.j5 if align else 90
        place_j5 = yaw.place_j5(nxt["j1"], nxt["r_index"]) if align else 90
        trial = {"trial_id": tid, "r_index": planned.r_index, "j1": planned.j1, "j5": pick_j5, "before_frame": frame}
        previous = cells[k - 2] if k > 1 and last_success else None  # 직전 시도가 이 블록을 내려놓은 자리(성공일 때만 알 수 있다)
        same_cell = previous is not None and (previous["r_index"], previous["j1"]) == (planned.r_index, planned.j1)
        hover = Pose(HOVER, planned.r_index, planned.j1, pick_j5)
        grasp = Pose(GRASP, planned.r_index, planned.j1, pick_j5)
        if arm.pose != PARK_POSE:
            arm.goto(PARK_POSE)
        arm.goto(hover)
        ledger.start(trial, joints_grasp=arm.gate.joints(grasp), joints_hover=arm.gate.joints(hover), plan=planned.asdict(), place_at={"r_index": nxt["r_index"], "j1": nxt["j1"]}, placed_at=previous, same_cell=same_cell, pick_j5=pick_j5, place_j5=place_j5)
        need_j5 = align and pick_j5 != 90 and not confirmed_j5["pick"]
        if confirm_each or k == first or need_j5:
            confirmed_j5["pick"] = confirmed_j5["pick"] or pick_j5 != 90
            try:
                _wait_dabai_signal(signal, "descend", wait_s, sleep, clock, log)
            except NeedHuman:
                arm.goto(PARK_POSE)
                ledger.finish(tid, outcome="interrupted", success=False, reason="하강 전 확인을 받지 못했다")
                raise
        arm.goto(grasp)
        arm.set_grip("close")
        arm.goto(hover)
        arm.goto(PARK_POSE)
        arm.sleep(hold_s)
        read_at_park = arm.read()
        outcome = observer(trial)
        values = read_at_park.get("values")
        grip = values[5] if values and len(values) > 5 else None
        holds = grip is not None and grip <= P1_GRIP_READ_MAX
        success = bool(outcome.success and holds)
        reason = f"{outcome.reason}, 서보 6 판독 {grip}" + ("" if holds else f" > {P1_GRIP_READ_MAX}(빈 손으로 봄)")
        target = (hover, grasp) if not success else (Pose(HOVER, nxt["r_index"], nxt["j1"], place_j5), Pose(GRASP, nxt["r_index"], nxt["j1"], place_j5))
        arm.goto(target[0])
        need_place = success and align and place_j5 != 90 and not confirmed_j5["place"]
        if success and (confirm_each or need_place):  # 내려놓는 하강 확인(선택·처음 J5≠90). 없으면 블록을 쥔 채 park로 돌아가 멈춘다
            confirmed_j5["place"] = confirmed_j5["place"] or place_j5 != 90
            try:
                _wait_dabai_signal(signal, "place", wait_s, sleep, clock, log)
            except NeedHuman as exc:
                arm.goto(PARK_POSE)
                ledger.finish(tid, outcome="interrupted", success=False, picked=True, reason=f"{reason}. 내려놓기 전 확인을 받지 못했다(블록을 쥐고 park)", read_at_park=read_at_park, grip_read=grip)
                raise NeedHuman(f"{tid}: 내려놓기 전 확인을 받지 못했다. 블록을 쥔 채 park에 있다. 사람이 확인한다") from exc
        arm.goto(target[1])
        arm.set_grip("open")
        arm.goto(target[0])
        arm.goto(PARK_POSE)
        ledger.finish(tid, outcome="success" if success else "failed", success=success, reason=reason, read_at_park=read_at_park, grip_read=grip)
        last_success = success
        if not success:
            raise NeedHuman(f"{tid} 실패({reason}). 같은 자리에 도로 놓고 멈춘다. 사람이 확인한다")
        if rest_every > 0 and k % rest_every == 0 and k < first + count - 1:
            log(json.dumps({"rest_s": rest_s, "after": tid}, ensure_ascii=False), flush=True)
            end = clock() + rest_s
            while clock() < end:
                if stop is not None and stop.exists():
                    ledger.note("stop_requested", trial_id=tid, during="rest")
                    raise NeedHuman(f"{tid} 뒤 쉬는 중 정지 파일이 있어 멈춘다(팔 park, 쓰기 없음)")
                sleep(1.0)
    camera.snapshot(f"chain-{first + count:03d}")  # 마지막으로 놓은 블록의 방향도 잰다(쓰기 없음, 다음 시도의 입력 사진 이름)


def release_here(arm: SafeArm, r_index: int, j1: int, j5: int) -> None:
    """멈춘 자리(grasp 자세, 블록을 쥔 채)에서 정상 흐름의 나머지(열기 → hover → park)를 한다.

    프로세스가 도착 확인 실패로 끝난 뒤 새 프로세스가 자세를 모를 때 쓴다. 조회가 이 자리의 관절값과 맞지 않으면 쓰기 전에 멈춘다.
    조회값은 확인에만 쓰고 목표로 쓰지 않는다.
    """
    here = Pose(GRASP, r_index, j1, j5)
    expected = arm.gate.joints(here)
    reading = arm.read()
    values = reading.get("values")
    tolerance = arm.gate.config.motion.arrival_tol_deg
    if not values or any(abs(v - e) > tolerance for v, e in zip(values[:5], expected)):
        raise SafetyViolation(f"조회가 grasp 자세와 맞지 않아 움직이지 않는다. 기대 {expected}, 조회 {values}")
    arm.pose = here
    arm.grip = arm.gate.config.close_sdk
    arm.holding = True
    arm.set_grip("open")
    arm.goto(Pose(HOVER, r_index, j1, j5))
    arm.goto(PARK_POSE)


GRIP_MAX_WAIT_S = 300.0  # H2-A 실측: 사람이 자를 대고 읽고 손을 빼 답하는 데 약 211초


def run_grip_max(
    arm: SafeArm,
    config: Config,
    signal: Path,
    stop: Path,
    folder: Path,
    r_index: int,
    max_sdk: int = GRIP_SWEEP_MAX_SDK,
    wait_s: float = GRIP_MAX_WAIT_S,
    sleep=None,
    clock=None,
    log=print,
) -> None:
    """빈 손을 grasp(z+15 구성)에서 열림, close_sdk, 그 위로 5씩 max_sdk까지 올리며 단계마다 정지해 손끝을 읽게 한다.

    단계마다 서보 6 판독을 기록한다(쓰기 아님). 판독이 명령보다 (close_sdk 때의 지연 + 도착 허용치) 넘게 뒤지면
    막힘(접촉)으로 보고 더 올리지 않는다. stop 파일, 대기 상한 초과, 막힘, max_sdk 도달 어느 쪽이든 손을 열고
    hover, park로 정상 전이로 돌아간다. 안전 위반은 자동 쓰기 없이 올라간다. CloseWatch는 쓰지 않는다.
    """
    import time as _time

    if not config.close_sdk < max_sdk <= GRIP_SWEEP_MAX_SDK:
        raise SafetyViolation(f"--max-sdk {max_sdk}는 {config.close_sdk}보다 크고 {GRIP_SWEEP_MAX_SDK} 이하여야 한다")
    sleep = sleep or _time.sleep
    clock = clock or _time.monotonic
    hover = Pose(HOVER, r_index, 90, 90)
    grasp = Pose(GRASP, r_index, 90, 90)
    ledger = folder / "grip-max.jsonl"
    stop.unlink(missing_ok=True)

    def servo6() -> int | None:
        values = arm.read().get("values")
        return values[5] if values and len(values) > 5 else None

    def record(**item) -> None:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    arm.goto(PARK_POSE)
    arm.goto(hover)
    arm.goto(grasp)
    reason, human = "max_sdk 도달", None
    try:
        record(step="open", command=arm.grip, servo6=servo6())
        if _wait_dabai_signal(signal, "open", wait_s, sleep, clock, log, stop):
            reason = "정지 파일"
        else:
            arm.set_grip("close")
            record(step="close", command=arm.grip, servo6=servo6())
            if _wait_dabai_signal(signal, f"close:{arm.grip}", wait_s, sleep, clock, log, stop):
                reason = "정지 파일"
            else:
                # 기준 지연은 서보가 자리 잡은 뒤(읽기 시간 뒤) 다시 읽어 얻는다. 닫은 직후 값은 아직 가는 중일 수 있다.
                settled = servo6()
                record(step="close_settled", command=arm.grip, servo6=settled)
                base_lag = None if settled is None else arm.grip - settled
                if base_lag is None:
                    reason = "서보 판독 없음"
                elif base_lag > config.motion.arrival_tol_deg:
                    reason = "close_sdk에서 이미 판독이 뒤짐(손가락 사이에 물체가 있을 수 있다)"
                while reason == "max_sdk 도달" and arm.grip + GRIP_SWEEP_STEP <= max_sdk:
                    arm.set_grip(f"level:{arm.grip + GRIP_SWEEP_STEP}")
                    reading = servo6()
                    lag = None if reading is None else arm.grip - reading
                    record(step="level", command=arm.grip, servo6=reading, lag=lag, base_lag=base_lag)
                    if reading is None:
                        reason = "서보 판독 없음"
                        break
                    if lag - base_lag > config.motion.arrival_tol_deg:
                        reason = "막힘(서보 판독이 명령에 못 미침)"
                        break
                    if _wait_dabai_signal(signal, f"level:{arm.grip}", wait_s, sleep, clock, log, stop):
                        reason = "정지 파일"
                        break
    except NeedHuman as error:
        human, reason = error, "대기 상한 초과"
    record(step="end", reason=reason, command=arm.grip)
    log(json.dumps({"end": reason}, ensure_ascii=False), flush=True)
    arm.set_grip("open")
    arm.goto(hover)
    arm.goto(PARK_POSE)
    if human is not None:
        raise human


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


G0D_COMMANDS = ("natural-spread", "tolerance")


def g0d_limits(session: Path) -> tuple[float, float]:
    """닫힘 한도와 정상 해제 시간. 실측으로 승인 파일에 고정한다. 기본값을 두지 않는다(G0d 플랜 124~126행)."""
    raw = json.loads(Path(session).read_text(encoding="utf-8"))
    values = []
    for key in ("close_limit_s", "release_s"):
        value = raw.get(key)
        if type(value) not in (int, float) or value <= 0:
            raise SystemExit(f"G0d 승인 파일에 양수 {key}가 없다")
        values.append(float(value))
    if values[1] >= values[0]:
        raise SystemExit("release_s가 close_limit_s보다 작아야 답할 시간이 남는다")
    return values[0], values[1]


def run_g0d(command: str, arm: SafeArm, config: Config, table: PoseTable, folder: Path, session_id: str, release_s: float, sheet_path: Path | None, spread_path: Path | None, writes_left=None) -> None:
    """G0d-0과 G0d 1차. 쉼 묶음마다 블록을 내려놓고 끝나며, 같은 명령을 다시 실행하면 이어서 한다."""
    import hashlib

    from learned_grasp.tolerance import G0dContext, StopSession, filter_from_spread, ledger_counters, run_natural_spread, run_tolerance, spread_sheet, summarize_spread
    from learned_grasp.top_marker import Placement, TopMarker

    marker = TopMarker()  # OpenCV 버전이 다르면 쓰기 전에 멈춘다
    spread = None
    binding: dict = {}
    if command == "tolerance":
        if sheet_path is None or spread_path is None:
            raise SystemExit("tolerance는 --sheet와 --spread가 필요하다")
        raw = spread_path.read_bytes()
        spread = json.loads(raw)
        if not (spread.get("complete") and spread.get("failures") == 0 and spread.get("incomplete") == 0 and spread.get("yellow_ok")):
            raise SystemExit("G0d-0이 완결·무실패·노란 화소 기준을 채우지 않아 G0d 1차로 가지 않는다")
        binding["spread_sha256"] = hashlib.sha256(raw).hexdigest()
    camera = P620Camera(f"runs/{session_id}/frames")
    try:
        if spread is not None:
            reference = {int(k): tuple(v) for k, v in spread["reference"].items()}
        else:
            # 기준은 G0d-0 첫 실행에서 한 번 고정하고 이어지는 묶음에서 그대로 쓴다(플랜 55행).
            saved = folder / "reference.json"
            if saved.exists():
                reference = {int(k): tuple(v) for k, v in json.loads(saved.read_text(encoding="utf-8")).items()}
            else:
                camera.snapshot("reference")
                reference = marker.measure(camera.fetch("reference")).outer
                Placement(reference)  # 기준 마커가 모자라면 쓰기 전에 멈춘다
                saved.write_text(json.dumps({str(k): list(v) for k, v in reference.items()}), encoding="utf-8")
        binding["reference"] = {str(k): [round(c, 3) for c in v] for k, v in sorted(reference.items())}
        ctx = G0dContext(
            arm=arm, camera=camera, ledger=AttemptLedger(folder / ("spread.jsonl" if spread is None else "tolerance.jsonl")), table=table,
            observer=make_pick_observer(camera), marker=marker, placement=Placement(reference), ask=FileAnswers(folder),
            load_block=make_block_loader(arm, folder / "block_ready"), release_s=release_s,
            placement_filter=None if spread is None else filter_from_spread(spread), binding=binding, writes_left=writes_left,
        )
        def end_kind() -> str:
            return ctx.counters["end"] or "stop"  # 실행기가 끝 종류를 못 정하고 나왔으면 정지다

        try:
            if spread is None:
                complete = False
                try:
                    complete = run_natural_spread(ctx, spread_sheet(config, table))
                finally:
                    summary = summarize_spread(ctx.ledger.records()) | {"complete": complete, "end": end_kind(), "reference": binding["reference"], "counters": ledger_counters(ctx.ledger.records())}
                    (folder / "spread.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
                    print(json.dumps(summary, ensure_ascii=False))
                    print(f"끝 종류: {summary['end']}")
            else:
                sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
                states = None
                try:
                    states = run_tolerance(ctx, sheet)
                finally:
                    from learned_grasp.tolerance import variant_states

                    states = states if states is not None else variant_states(sheet, ctx.ledger.records())
                    result = compute_tolerance(tolerance_variants(config, table), states, spread["transfer"]) | {"end": end_kind(), "counters": ledger_counters(ctx.ledger.records())}
                    (folder / "tolerance.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
                    print(json.dumps(result, ensure_ascii=False))
                    print(f"끝 종류: {result['end']}")
            if ctx.counters["batch_end"]:
                print("쉼 묶음 끝. 블록을 내려놓았다. 열을 확인하고 같은 명령으로 이어서 한다")
        except StopSession as error:
            raise SystemExit(f"추가 쓰기 없이 멈췄다: {error}. 끝 종류 stop: 블록을 쥐고 있을 수 있다. 받친 상태에서 손으로 회수하고 --start-confirmed로 다시 시작한다") from error
    finally:
        camera.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["read", "ladder", "observe", "dryrun", "replay", "ready", "descend", "park", "fit", "grip-test", "carry-test", "close-check", "dabai-view", "grip-max", "p1-pick", "chain", "release-here", *G0D_COMMANDS, "answer"])
    parser.add_argument("--spread", type=Path)
    parser.add_argument("--attempt", type=int)
    parser.add_argument("--r-index", type=int)
    parser.add_argument("--max-sdk", type=int, default=GRIP_SWEEP_MAX_SDK, help="grip-max: 올릴 최대 SDK(close_sdk 초과, 180 이하)")
    parser.add_argument("--j1", type=int, help="dabai-view·p1-pick: hover·grasp의 J1(dabai-view 기본 90, 범위는 구성 j1_range)")
    parser.add_argument("--trial", help="p1-pick: 시도 id(예: p1-03)")
    parser.add_argument("--cells", type=Path, help="chain: 내려놓을 자리 목록 JSON(scripts/chain_cells.py)")
    parser.add_argument("--count", type=int, help="chain: 이번 실행의 시도 수")
    parser.add_argument("--board-ref", help="chain: 보드 마커 id 1이 안 보일 때 쓸 중심 'x,y'(최근 빈 보드 사진)")
    parser.add_argument("--confirm-each", action="store_true", help="chain: 집기·내려놓기마다 하강 전 확인(기본은 이 실행의 첫 집기만)")
    parser.add_argument("--rest-every", type=int, default=10, help="chain: 닫기 N번마다 쉰다(0이면 쉬지 않음)")
    parser.add_argument("--rest-sec", type=float, default=300.0, help="chain: 쉬는 시간(초)")
    parser.add_argument("--ack", help="chain: 실패·중단한 시도 id를 사람이 확인했다고 기록하고 이어 간다(예: chain-012)")
    parser.add_argument("--align", action="store_true", help="chain: J5로 손을 블록 방향(집기)·반경선(놓기)에 맞춘다(config/l1_yaw.json)")
    parser.add_argument("--j5", type=int, help="release-here: 멈춘 grasp 자세의 J5")
    parser.add_argument("--hover-only", action="store_true", help="dabai-view: hover에서 신호 뒤 grasp 없이 park로 돌아온다")
    parser.add_argument("--sheet", type=Path)
    parser.add_argument("--mode", choices=["stored", "label"], default="stored")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/workspace.yaml"))
    parser.add_argument("--z-offset", type=float)
    parser.add_argument("--start-confirmed", action="store_true", help="사용자가 빈 그리퍼, 시작 자세, park 경로를 확인했다")
    parser.add_argument("--hold-sec", type=float, default=0.0)
    parser.add_argument("--z-mm", type=float)
    parser.add_argument("--ok", action="store_true")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    if args.hover_only and args.command != "dabai-view":
        raise SystemExit("--hover-only는 dabai-view에서만 쓴다")
    if args.j1 is not None and args.command not in ("dabai-view", "p1-pick", "release-here"):
        raise SystemExit("--j1은 dabai-view, p1-pick, release-here에서만 쓴다")
    if args.command == "answer":
        # 실행 중인 세션 옆 터미널에서 쓴다. 잠금과 드라이버를 잡지 않는다.
        print(answer_once(args.session.parent))
        return
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

    if args.command in ("ladder", "observe", "close-check", "grip-max") and args.z_offset not in LADDER_OFFSETS_MM:
        raise SystemExit(f"--z-offset는 {LADDER_OFFSETS_MM} 중 하나다")
    if args.command == "observe":
        record_observation(folder, args.z_offset, args.ok, args.note)
        print(f"관찰 기록: z+{args.z_offset:g} ok={args.ok}")
        return
    if args.command == "ready":
        (folder / "block_ready").write_text("1")
        print("물체 적재 확인을 기록했다")
        return
    if args.command == "p1-pick" and (args.r_index is None or args.j1 is None or args.trial is None):
        raise SystemExit("p1-pick은 --r-index, --j1, --trial이 필요하다")
    if args.command == "release-here" and (args.r_index is None or args.j1 is None or args.j5 is None):
        raise SystemExit("release-here는 --r-index, --j1, --j5가 필요하다")
    if args.command == "chain" and (args.cells is None or args.count is None or args.count < 1):
        raise SystemExit("chain은 --cells와 1 이상의 --count가 필요하다")
    if args.command == "grip-max":
        if args.z_offset != 15 or args.r_index is None:
            raise SystemExit("grip-max는 --z-offset 15와 --r-index가 필요하다")
    if args.r_index is not None:
        if args.command not in ("close-check", "dabai-view", "grip-max", "p1-pick", "release-here"):
            raise SystemExit("--r-index는 close-check, dabai-view, grip-max, p1-pick, release-here에서만 쓴다")
        if args.command in ("close-check", "grip-max"):
            require_site_folder(folder, args.r_index)
    if args.command == "carry-test" and args.attempt is None:
        raise SystemExit("--attempt가 필요하다")
    if args.command in ("ladder", "dryrun", "replay", "descend", "park", "fit", "grip-test", "carry-test", "close-check", "dabai-view", "grip-max", "p1-pick", "chain", "release-here", *G0D_COMMANDS) and not args.start_confirmed:
        raise SystemExit("--start-confirmed 없이는 움직이지 않는다")
    if args.command in ("ladder", "close-check", "grip-max"):
        require_previous_observation(folder, args.z_offset)
        config = ladder_config(config, args.z_offset)
    if args.command in ("fit", "grip-test"):
        config = fit_config(config, args.z_mm)
    limits = g0d_limits(args.session) if args.command in G0D_COMMANDS else None
    table = generate(config.table)
    driver = _driver(approval, used)
    try:
        if args.command == "read":
            print(json.dumps(driver.read(), ensure_ascii=False))
            return
        start = read_start_joints(driver, folder)
        watch = None if limits is None else CloseWatch(limits[0])
        sender = driver if watch is None else WatchedDriver(driver, watch)
        counting = CountingDriver(sender, counter, approval.max_writes)
        arm = SafeArm(counting, SafetyGate(config, table, grip_sweep=args.command == "grip-max"), start_joints=start, close_watch=watch)
        loader = make_block_loader(arm, folder / "block_ready")
        if args.command == "fit":
            run_fit(arm, config, table, loader, min(args.hold_sec, 60.0))
        elif args.command == "grip-test":
            run_grip_test(arm, config, table, loader, min(args.hold_sec, 60.0))
        elif args.command == "carry-test":
            run_carry_test(arm, AttemptLedger(folder / "carry.jsonl"), config, table, loader, min(args.hold_sec, 60.0), args.attempt)
        elif args.command == "close-check":
            run_close_check(arm, config, table, min(args.hold_sec, 60.0), args.r_index)
        elif args.command == "descend":
            descend_and_stop(arm, config, table)
        elif args.command == "dabai-view":
            run_dabai_view(arm, folder / "dabai_next", r_index=args.r_index if args.r_index is not None else DABAI_VIEW_R_INDEX, hover_only=args.hover_only, j1=args.j1 if args.j1 is not None else 90)
        elif args.command == "p1-pick":
            camera = P620Camera(f"runs/{approval.session_id}/frames")
            try:
                run_p1(arm, camera, make_pick_observer(camera), AttemptLedger(folder / "p1.jsonl"), config, table,
                       {"trial_id": args.trial, "r_index": args.r_index, "j1": args.j1, "j5": 90}, folder / "dabai_next", min(args.hold_sec or 5.0, 60.0))
            finally:
                camera.close()
        elif args.command == "release-here":
            release_here(arm, args.r_index, args.j1, args.j5)
        elif args.command == "chain":
            import cv2

            board_ref = tuple(float(v) for v in args.board_ref.split(",")) if args.board_ref else None

            def measure(path):
                image = cv2.imread(str(path))
                if image is None:
                    raise FileNotFoundError(f"사진을 읽지 못했다: {path}")
                return block_from_image(image, board_ref)

            cells = json.loads(args.cells.read_text(encoding="utf-8"))["cells"]
            chain_ledger = AttemptLedger(folder / "chain.jsonl")
            if args.ack:
                done, _ = chain_ledger.state()
                if done.get(args.ack, {}).get("outcome") not in ("failed", "interrupted"):
                    raise SystemExit(f"--ack는 실패·중단한 시도에만 쓴다: {args.ack}")
                chain_ledger.note("acknowledged", trial_id=args.ack)
            camera = P620Camera(f"runs/{approval.session_id}/frames")
            try:
                run_chain(arm, camera, make_pick_observer(camera), chain_ledger, config, table, cells, args.count,
                          folder / "dabai_next", measure, min(args.hold_sec or 5.0, 60.0),
                          confirm_each=args.confirm_each, rest_every=args.rest_every, rest_s=args.rest_sec, stop=folder / "chain_stop", align=args.align)
            finally:
                camera.close()
        elif args.command == "grip-max":
            run_grip_max(arm, config, folder / "dabai_next", folder / "grip_stop", folder, args.r_index, args.max_sdk)
        elif args.command == "park":
            arm.goto(PARK_POSE)
        elif args.command == "replay":
            camera = P620Camera(f"runs/{approval.session_id}/frames")
            try:
                sheet = json.loads(args.sheet.read_text(encoding="utf-8"))
                run_replay(
                    arm, camera, AttemptLedger(folder / f"replay-{args.mode}.jsonl"), sheet,
                    make_pick_observer(camera), table, args.mode, make_block_loader(arm, folder / "block_ready"),
                )
            finally:
                camera.close()
        elif args.command in G0D_COMMANDS:
            run_g0d(args.command, arm, config, table, folder, approval.session_id, limits[1], args.sheet, args.spread, writes_left=lambda: counting.limit - counting.used())
        elif args.command == "ladder":
            run_ladder(arm, AttemptLedger(folder / f"ladder-z{args.z_offset:g}.jsonl"), config, table, min(args.hold_sec, 60.0))
        else:
            run_dryrun(arm, AttemptLedger(folder / "dryrun.jsonl"), dryrun_sheet(config, table), _unobserved, table)
        print(f"완료. 세션 쓰기 누적 {counter.read_text()}회. 무간섭은 관찰 기록으로만 판정한다.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
