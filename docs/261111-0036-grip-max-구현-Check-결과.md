# grip-max 구현 Check 결과 (PDCA Check)

2026-11-11 KST. 대상: 플랜 `docs/*그리퍼-최대-닫힘-돌출-측정-플랜.md` v2의 (a) 구현. 장치 쓰기 0건(Fake 시험만).

## 변경 (Do)

| 파일 | 변경 |
| --- | --- |
| `src/learned_grasp/safety.py` | `GRIP_SWEEP_STEP`(5), `GRIP_SWEEP_MAX_SDK`(180 = 서버 `SDK_MAX[6]`), `SafetyGate(grip_sweep=False)`, `check_grip`에 `level:<sdk>`(grasp, 플래그 켜짐, 닫힌 뒤, 현재값 초과 5 이하, 상한 이하) |
| `src/learned_grasp/arm.py` | `set_grip`이 현재 그립값을 게이트에 넘기고 level 뒤에도 닫힘 상태 유지. 기존 open/close 동작은 그대로 |
| `src/learned_grasp/session.py` | `run_grip_max`, CLI `grip-max`(`--max-sdk`, `--z-offset 15`와 `--r-index` 필수, 폴더 `-r4` 규칙, 이전 이상 관찰이면 거부), `_wait_dabai_signal`에 `stop` 파일 |
| `tests/test_safety.py`, `tests/test_session.py` | level 게이트 시험(단계 크기, 상한, 플래그, 자세, 닫힘 여부), grip-max 시험 7개 |

## 안전 점검 (자체 리뷰)

- 새 동작은 `grip-max` 명령만 플래그를 켠다. 기본 `False`라 다른 명령은 level을 쓸 수 없다(시험 `test_level_requires_sweep_flag_grasp_and_closed_hand`, `test_grip_max_requires_gate_flag`).
- 대기 상한 300초(H2-A 실측: 사람이 읽고 손을 빼 답하는 데 약 211초). 막힘은 서보 6 판독으로 자동 정지한다. 기준 지연은 닫은 뒤 대기가 끝나고 다시 읽은 값이며, 빈 손 `close_sdk`에서 이미 허용치(3도)를 넘게 뒤지면 올리지 않는다.
- 어느 종료(정지 파일, 상한 초과, 막힘, 최대 도달)든 열기 → hover → park의 기존 정상 전이만 쓴다. 안전 위반은 복구 쓰기 없이 올라간다(`test_grip_max_requires_gate_flag`: 닫기까지만 나가고 멈춤).
- `CloseWatch`는 연결하지 않았다.
- **한계(미확인 가정)**: 서보 6 판독이 빈 손에서 실제 각도를 따르는지는 이번에 처음 확인한다. 근거로 쓸 수 있는 기록은 `G0b-4`의 블록을 쥔 판독 148, 149, 149뿐이다(플랜 H6). 판독이 명령을 못 따라가면 자동 정지가 서지 않으므로 사용자 관찰(`grip_stop`)과 전원 차단이 남는 수단이다(손과 자를 뺀 뒤에만).

## 시험 결과

```text
$ uv run pytest -q
........................................................................ [ 96%]
...........                                                              [100%]
299 passed in 56.86s
```

기존 286개에 12(+1)개를 더했다. 실측 확인 없음(장치 미사용).
