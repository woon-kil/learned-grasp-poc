# AGENTS.md

대화와 문서는 한글로 작성한다. 기술 용어는 예외다.

세션 시작 시 `PROJECT-STATUS.md`를 읽는다. 플랜 정본은 `docs/261109-1017-learned-grasp-poc-플랜.md`다.

## 절대 규칙

1. **실기 쓰기는 사용자가 승인한 세션 안에서만 보낸다.** 승인 요청에는 대상, 명령 범위·횟수, 관찰, 중단 조건을 한 번에 적는다. 이 문서는 승인이 아니다.
2. **무인 수집 금지.** 모든 실기 세션에 전원 차단 담당자가 상주한다.
3. **`arm_driver`를 실행하지 않는다.** 생성자가 서보를 움직인다. `Arm_Lib`는 `jetson_arm_server.py`에서만 쓴다.
4. **모든 쓰기는 `SafeArm`과 `SafetyGate`를 통과한다.** 허용 자세 표와 전이 밖의 명령을 우회하는 코드를 만들지 않는다.
5. **조회값을 목표로 쓰지 않는다.** 복귀는 원래 쓰기 목표를 재전송한다. 과거 프로젝트의 각도를 기본값으로 쓰지 않는다.
6. **판정 수치를 사후에 바꾸지 않는다.** 실패 셀 제외, 재시도, 시험 데이터 튜닝을 하지 않는다. 구성이 바뀌면 의존 게이트를 다시 한다.
7. **미측정 값을 쓰지 않는다.** 명령값, SDK 반환, 영상 관찰을 서로 대체하지 않는다. 모델 값은 모델 값이라고 적는다.
8. **`simple-vision-tech-poc`를 수정하지 않는다.** 그 증거를 학습·평가에 쓰지 않는다. p620의 다른 컨테이너를 멈추지 않는다.

## 명령

```bash
uv sync
uv run pytest -q
uv run python scripts/generate_pose_table.py
uv run python -m learned_grasp.g0_runner sheet --gate G0b --seed 1
```

## 코드 구조

- `kinematics.py` URDF FK. `pose_table.py` 정수 허용 자세 표와 라벨 복원.
- `safety.py` 표·전이·이동시간 검사. `arm.py` `SafeArm`, 세션 승인, Fake·Remote 드라이버.
- `jetson_arm_server.py` Jetson에 복사하는 단독 파일. 표준 라이브러리만 쓴다.
- `attempts.py` 증분 시도 기록. `g0_runner.py` 시험표·실행·집계. `stats.py` Wilson.
- `camera.py` 인터페이스. 실장치 스냅샷은 G0b 전에 구현한다.

학습 코드는 G0 통과 후 S1에서 만든다.

## Git

정본: `.cursor/rules/git-autonomy.mdc`. **public 저장소**이고 푸시 대상은 `origin/main`뿐이다. 푸시는 `bash scripts/safe_push.sh`로만 한다. 우회와 `--force` 금지. 내부 주소, 시크릿, 제조사 파일, 체크포인트, `data/`, `runs/`는 커밋하지 않는다. `results/`와 `docs/`는 커밋 대상이다. 시험표와 승인 내역은 실행 전에 커밋한다.

## 문서

`docs/YYMMDD-HHMM-제목.md`. 이슈는 현상·원인·수정·교훈. 시험 결과는 실패와 미완료를 포함한다.

사람이 블록·마커를 손으로 놓는 시험은 놓는 자리를 SVG 그림으로 `results/`에 만들어 촬영 안내와 함께 준다. 위치는 실제 카메라 사진을 보고 그린다(사용자 2026-11-10).
