"""Jetson 쪽 팔 서버. 표준 라이브러리만 쓰고 이 파일 하나만 복사해 실행한다.

표준입력으로 JSON 줄을 받고 표준출력으로 응답한다. 포트를 열지 않는다.
입력이 끊기면 후속 명령을 폐기하고 종료한다. 이미 전송된 서보 이동은 이 종료로 멈추지 않는다.
드라이버 노드를 쓰지 않는다. 그 생성자는 서보를 움직인다.

사용: python3 jetson_arm_server.py --token <세션토큰> --max-writes <n>
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys

SDK_MAX = {1: 180, 2: 180, 3: 180, 4: 180, 5: 270, 6: 180}
MIN_TIME_MS = 500


def classify(servo_id, value):
    """SDK 반환 분류. 값은 축 실측이 아니다."""
    if servo_id not in SDK_MAX or isinstance(value, bool) or not isinstance(value, (int, float)):
        return "fail"
    if not math.isfinite(value) or value < 0 or value > SDK_MAX[servo_id]:
        return "fail"
    return "sdk_return"


def _valid_write(q, time_ms):
    if not isinstance(q, list) or len(q) != 6:
        return "관절 6개가 필요하다"
    for servo_id, value in enumerate(q, start=1):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= SDK_MAX[servo_id]:
            return "범위 밖 id %d" % servo_id
    if isinstance(time_ms, bool) or not isinstance(time_ms, int) or time_ms < MIN_TIME_MS:
        return "이동시간 하한 %d ms 미달" % MIN_TIME_MS
    return None


def _sdk_call(function, *args):
    """SDK를 호출하고 출력을 잡는다.

    Arm_Lib은 I2C 오류와 범위 초과를 예외 없이 print만 하고 넘어간다.
    출력을 실패로 보지 않으면 실패한 쓰기가 성공으로 보고되고 JSON 통신이 깨진다.
    """
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        result = function(*args)
    return result, captured.getvalue().strip()


def serve(device_factory, lines, write_line, token, max_writes):
    """요청을 처리한다. 인증 전에는 장치를 만들지 않는다. 반환값은 전송한 쓰기 수다."""
    device = None
    writes = 0
    for line in lines:
        try:
            request = json.loads(line)
            operation = request["op"]
        except (ValueError, KeyError, TypeError):
            write_line(json.dumps({"ok": False, "error": "잘못된 요청"}))
            break
        if operation == "bye":
            write_line(json.dumps({"ok": True, "writes": writes}))
            break
        if operation == "hello":
            if request.get("token") != token:
                write_line(json.dumps({"ok": False, "error": "세션 토큰 불일치"}))
                break
            if device is not None:
                write_line(json.dumps({"ok": False, "error": "hello 중복"}))
                break
            device = device_factory()
            write_line(json.dumps({"ok": True}))
            continue
        if device is None:
            write_line(json.dumps({"ok": False, "error": "hello 전이다"}))
            break
        if operation == "write6":
            error = _valid_write(request.get("q"), request.get("time_ms"))
            if error is None and writes >= max_writes:
                error = "세션 쓰기 상한 초과"
            if error is not None:
                write_line(json.dumps({"ok": False, "error": error}))
                break
            q = request["q"]
            _, sdk_output = _sdk_call(device.Arm_serial_servo_write6, q[0], q[1], q[2], q[3], q[4], q[5], request["time_ms"])
            writes += 1
            if sdk_output:
                write_line(json.dumps({"ok": False, "error": "SDK 쓰기 실패: " + sdk_output, "writes": writes}))
                break
            write_line(json.dumps({"ok": True, "writes": writes}))
        elif operation == "grip":
            angle, time_ms = request.get("angle"), request.get("time_ms")
            error = _valid_write([90, 90, 90, 90, 90, angle], time_ms)
            if error is None and writes >= max_writes:
                error = "세션 쓰기 상한 초과"
            if error is not None:
                write_line(json.dumps({"ok": False, "error": error}))
                break
            _, sdk_output = _sdk_call(device.Arm_serial_servo_write, 6, angle, time_ms)
            writes += 1
            if sdk_output:
                write_line(json.dumps({"ok": False, "error": "SDK 쓰기 실패: " + sdk_output, "writes": writes}))
                break
            write_line(json.dumps({"ok": True, "writes": writes}))
        elif operation == "read":
            values = []
            status = []
            for servo_id in range(1, 7):
                value, sdk_output = _sdk_call(device.Arm_serial_servo_read, servo_id)
                values.append(value)
                status.append("fail" if sdk_output else classify(servo_id, value))
            write_line(json.dumps({"ok": True, "values": values, "status": status}))
        else:
            write_line(json.dumps({"ok": False, "error": "알 수 없는 op"}))
            break
    return writes


def _real_device():
    from Arm_Lib import Arm_Device

    device, sdk_output = _sdk_call(Arm_Device)
    if sdk_output:
        raise RuntimeError("SDK 생성 출력: " + sdk_output)
    return device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--max-writes", type=int, required=True)
    args = parser.parse_args()

    def write_line(text):
        sys.stdout.write(text + "\n")
        sys.stdout.flush()

    serve(_real_device, sys.stdin, write_line, args.token, args.max_writes)


if __name__ == "__main__":
    main()
