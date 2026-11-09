from __future__ import annotations

import ast
import json
from pathlib import Path

import subprocess
import sys

import pytest

from learned_grasp import jetson_arm_server as server
from learned_grasp.arm import ApprovalError, RemoteArm, SessionApproval
from learned_grasp.config import ROOT

TOKEN = "abc"


class FakeDevice:
    def __init__(self):
        self.writes = []

    def Arm_serial_servo_write6(self, *args):
        self.writes.append(args)

    def Arm_serial_servo_read(self, servo_id):
        return 90


def run(requests, max_writes=5):
    devices = []
    responses = []

    def factory():
        devices.append(FakeDevice())
        return devices[-1]

    writes = server.serve(factory, [json.dumps(r) for r in requests], responses.append, TOKEN, max_writes)
    return writes, devices, [json.loads(r) for r in responses]


HELLO = {"op": "hello", "token": TOKEN}
WRITE = {"op": "write6", "q": [90, 90, 90, 90, 90, 100], "time_ms": 1500}


def test_device_not_created_before_valid_hello():
    writes, devices, responses = run([{"op": "hello", "token": "wrong"}, WRITE])
    assert writes == 0 and devices == []
    assert responses == [{"ok": False, "error": "세션 토큰 불일치"}]


def test_duplicate_hello_ends_session():
    writes, devices, responses = run([HELLO, HELLO, WRITE])
    assert writes == 0 and len(devices) == 1 and responses[-1]["error"] == "hello 중복"


def test_write_before_hello_rejected():
    writes, devices, _ = run([WRITE])
    assert writes == 0 and devices == []


def test_valid_session():
    writes, devices, responses = run([HELLO, WRITE, {"op": "read"}, {"op": "bye"}])
    assert writes == 1
    assert devices[0].writes == [(90, 90, 90, 90, 90, 100, 1500)]
    assert responses[2]["status"] == ["sdk_return"] * 6


def test_grip_writes_only_servo_six():
    class Recording(FakeDevice):
        def __init__(self):
            super().__init__()
            self.single = []

        def Arm_serial_servo_write(self, *args):
            self.single.append(args)

    devices = []

    def factory():
        devices.append(Recording())
        return devices[-1]

    responses = []
    writes = server.serve(factory, [json.dumps(HELLO), json.dumps({"op": "grip", "angle": 120, "time_ms": 1500})], responses.append, TOKEN, 5)
    assert writes == 1 and devices[0].single == [(6, 120, 1500)] and devices[0].writes == []


def test_grip_range_and_cap_enforced():
    class Recording(FakeDevice):
        def Arm_serial_servo_write(self, *args):
            self.writes.append(args)

    devices = []

    def factory():
        devices.append(Recording())
        return devices[-1]

    bad = {"op": "grip", "angle": 200, "time_ms": 1500}
    assert server.serve(factory, [json.dumps(HELLO), json.dumps(bad)], lambda _: None, TOKEN, 5) == 0
    good = {"op": "grip", "angle": 120, "time_ms": 1500}
    assert server.serve(factory, [json.dumps(HELLO), json.dumps(good), json.dumps(good)], lambda _: None, TOKEN, 1) == 1


class PrintingDevice(FakeDevice):
    """실제 Arm_Lib처럼 I2C 오류를 예외 없이 print만 한다."""

    def Arm_serial_servo_write6(self, *args):
        print("Arm_serial_servo_write6 I2C error")

    def Arm_serial_servo_read(self, servo_id):
        print("Arm_serial_servo_read I2C error")


def test_silent_sdk_write_failure_ends_session(capsys):
    responses = []
    writes = server.serve(PrintingDevice, [json.dumps(HELLO), json.dumps(WRITE), json.dumps(WRITE)], responses.append, TOKEN, 5)
    assert writes == 1
    last = json.loads(responses[-1])
    assert last["ok"] is False and "I2C error" in last["error"]
    assert len(responses) == 2  # 뒤따르는 쓰기를 보내지 않는다
    assert capsys.readouterr().out == ""  # JSON 통신 채널을 오염시키지 않는다


def test_silent_sdk_read_failure_is_marked_fail(capsys):
    responses = []
    server.serve(PrintingDevice, [json.dumps(HELLO), json.dumps({"op": "read"})], responses.append, TOKEN, 5)
    assert json.loads(responses[-1])["status"] == ["fail"] * 6
    assert capsys.readouterr().out == ""


def test_input_end_stops_processing():
    """입력이 끊기면 그 뒤 명령이 없다. 서버는 쓰기를 더 보내지 않고 끝난다."""
    writes, devices, _ = run([HELLO, WRITE])
    assert writes == 1 and len(devices[0].writes) == 1


def test_invalid_write_ends_session_and_discards_followers():
    bad = {"op": "write6", "q": [90, 90, 90, 90, 300, 100], "time_ms": 1500}
    writes, devices, responses = run([HELLO, bad, WRITE])
    assert writes == 0 and devices[0].writes == []
    assert responses[-1]["ok"] is False and len(responses) == 2


def test_short_time_rejected():
    writes, devices, _ = run([HELLO, {**WRITE, "time_ms": 10}, WRITE])
    assert writes == 0 and devices[0].writes == []


def test_float_and_bool_angles_rejected():
    for q in ([90.5, 90, 90, 90, 90, 100], [True, 90, 90, 90, 90, 100]):
        writes, _, _ = run([HELLO, {**WRITE, "q": q}])
        assert writes == 0


def test_write_cap_enforced():
    writes, devices, responses = run([HELLO, WRITE, WRITE, WRITE], max_writes=2)
    assert writes == 2 and len(devices[0].writes) == 2
    assert responses[-1]["error"] == "세션 쓰기 상한 초과"


def test_malformed_json_ends_session():
    responses = []
    writes = server.serve(FakeDevice, [json.dumps(HELLO), "not json", json.dumps(WRITE)], responses.append, TOKEN, 5)
    assert writes == 0


def test_classify():
    assert server.classify(5, 270) == "sdk_return"
    assert server.classify(1, 181) == "fail"
    assert server.classify(1, None) == "fail"
    assert server.classify(1, True) == "fail"
    assert server.classify(9, 10) == "fail"


def test_no_module_uses_arm_driver():
    """드라이버 노드는 생성자가 서보를 움직인다. 어떤 모듈도 import하지 않는다."""
    for path in (Path(ROOT) / "src" / "learned_grasp").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any("arm_driver" in name or "dofbot_pro_driver" in name for name in names), path


def test_only_server_touches_arm_lib():
    for path in (Path(ROOT) / "src" / "learned_grasp").glob("*.py"):
        uses = any(
            isinstance(node, ast.ImportFrom) and node.module == "Arm_Lib"
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        )
        assert uses == (path.name == "jetson_arm_server.py"), path


def test_server_is_standalone():
    """Jetson에 이 파일만 복사하므로 패키지 내부를 import하지 않는다."""
    source = (Path(ROOT) / "src" / "learned_grasp" / "jetson_arm_server.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            assert node.module in ("__future__", "Arm_Lib")
        elif isinstance(node, ast.Import):
            assert all(alias.name in ("argparse", "contextlib", "io", "json", "math", "sys") for alias in node.names)


def test_server_parses_on_python38():
    """Jetson의 Python 버전을 아직 확인하지 않았다. 구버전 문법으로도 파싱돼야 한다."""
    source = (Path(ROOT) / "src" / "learned_grasp" / "jetson_arm_server.py").read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 8))


FAKE_ARM_LIB = """
import json, os
class Arm_Device:
    def __init__(self):
        self.log = os.environ["FAKE_LOG"]
    def Arm_serial_servo_write6(self, *args):
        open(self.log, "a").write(json.dumps(args) + "\\n")
    def Arm_serial_servo_read(self, servo_id):
        return 90
"""


@pytest.fixture
def process_command(tmp_path, monkeypatch):
    (tmp_path / "Arm_Lib.py").write_text(FAKE_ARM_LIB)
    log = tmp_path / "writes.log"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    script = Path(ROOT) / "src" / "learned_grasp" / "jetson_arm_server.py"
    return [sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(tmp_path)!r}); sys.argv=['s','--token','t','--max-writes','3']; exec(open({str(script)!r}, encoding='utf-8').read())"], log


APPROVAL = SessionApproval("s", "G0a", 3, "t")


def test_remote_arm_end_to_end_process(process_command):
    command, log = process_command
    arm = RemoteArm(APPROVAL, command=command)
    arm.write6((90, 90, 90, 90, 90), 100, 1500)
    assert arm.read()["status"] == ["sdk_return"] * 6
    arm.close()
    assert arm.process.returncode == 0
    assert json.loads(log.read_text()) == [90, 90, 90, 90, 90, 100, 1500]


def test_process_exits_when_connection_closes_and_sends_nothing_more(process_command):
    command, log = process_command
    arm = RemoteArm(APPROVAL, command=command)
    arm.write6((90, 90, 90, 90, 90), 100, 1500)
    arm.process.stdin.close()  # SSH 단절과 같은 입력 종료
    assert arm.process.wait(timeout=5) == 0
    assert len(log.read_text().splitlines()) == 1


def test_process_rejects_over_cap_and_client_raises(process_command):
    command, log = process_command
    arm = RemoteArm(APPROVAL, command=command)
    for _ in range(3):
        arm.write6((90, 90, 90, 90, 90), 100, 1500)
    with pytest.raises(RuntimeError, match="상한"):
        arm.write6((90, 90, 90, 90, 90), 100, 1500)
    assert arm.process.wait(timeout=5) == 0
    assert len(log.read_text().splitlines()) == 3


def test_client_times_out_on_stalled_server(tmp_path):
    """연결은 살아 있는데 서버가 멈춘 경우 무기한 기다리지 않는다."""
    arm = RemoteArm.__new__(RemoteArm)
    arm.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    with pytest.raises(ConnectionError, match="응답하지"):
        arm._call({"op": "read"}, timeout_s=0.2)
    arm.close()
    assert arm.process.poll() is not None


def test_client_detects_dead_server(process_command):
    command, _ = process_command
    arm = RemoteArm(APPROVAL, command=command)
    arm.process.kill()
    arm.process.wait(timeout=5)
    with pytest.raises((ConnectionError, BrokenPipeError)):
        arm.write6((90, 90, 90, 90, 90), 100, 1500)


def test_camera_script_only_removes_its_own_container():
    """p620은 공유 서버다. 다른 프로젝트 컨테이너를 지우거나 멈추지 않는다."""
    script = (Path(ROOT) / "scripts" / "p620_camera.sh").read_text(encoding="utf-8")
    assert 'NAME="learned-grasp-camera"' in script
    for line in script.splitlines():
        if line.lstrip().startswith("docker ") and any(word in line for word in (" rm ", " stop ", " kill ")):
            assert '"${NAME}"' in line, line


def test_remote_arm_requires_host_from_environment(monkeypatch):
    """공개 저장소에 내부 주소를 두지 않는다."""
    monkeypatch.delenv("LG_JETSON_HOST", raising=False)
    with pytest.raises(ApprovalError):
        RemoteArm(APPROVAL)
