"""run_g0d 실제 연결 시험. 카메라·마커·답·적재만 Fake로 바꾸고 SafeArm·CloseWatch·WatchedDriver·판정 함수는 운영 그대로다."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from learned_grasp import session
from learned_grasp.arm import CloseWatch, FakeArm, SafeArm, WatchedDriver
from learned_grasp.camera import FakeCamera
from learned_grasp.g0_runner import tolerance_sheet
from learned_grasp.safety import SafetyGate
from learned_grasp.top_marker import Measurement

REF = {0: (915.0, 654.0), 1: (851.0, 211.0)}


class Camera(FakeCamera):
    instances: list = []

    def __init__(self, output, host=None):
        super().__init__()
        self.output = output
        Camera.instances.append(self)

    def fetch(self, name):
        return Path(name)

    def close(self):
        pass


class Marker:
    def measure(self, path):
        return Measurement(str(path), "x" * 64, dict(REF), True, str(path) != "reference", (0.0, 0.0), 0.0)


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(session, "P620Camera", Camera)
    monkeypatch.setattr("learned_grasp.top_marker.TopMarker", Marker)
    monkeypatch.setattr(session, "FileAnswers", lambda folder: (lambda kind, prompt, deadline: "y"))
    monkeypatch.setattr(session, "make_block_loader", lambda arm, path: (lambda trial: None))


def _arm(config, table):
    watch = CloseWatch(3600, on_expire=lambda _: None)
    return SafeArm(WatchedDriver(FakeArm(), watch), SafetyGate(config, table), sleep=lambda _: None, close_watch=watch)


def test_natural_spread_batches_fix_reference_and_feed_tolerance(wired, ready_config, table, tmp_path, capsys):
    folder = tmp_path / "G0d-0"
    folder.mkdir()
    session.run_g0d("natural-spread", _arm(ready_config, table), ready_config, table, folder, "G0d-0", 5.0, None, None)
    assert "쉼 묶음 끝" in capsys.readouterr().out
    reference = json.loads((folder / "reference.json").read_text())
    first = json.loads((folder / "spread.json").read_text())
    assert first["complete"] is False and first["end"] == "batch"
    assert first["counters"]["warmup_picks"] >= 1
    snaps_before = sum(c.names.count("reference") for c in Camera.instances)
    for _ in range(20):
        session.run_g0d("natural-spread", _arm(ready_config, table), ready_config, table, folder, "G0d-0", 5.0, None, None)
        spread = json.loads((folder / "spread.json").read_text())
        if spread["complete"]:
            break
    assert spread["complete"] and spread["failures"] == 0 and spread["incomplete"] == 0 and spread["trials"] == 42 and spread["end"] == "complete"
    ledger_notes = [r for r in map(json.loads, (folder / "spread.jsonl").read_text().splitlines()) if r["event"] == "note"]
    warm_notes = [r for r in ledger_notes if r.get("kind") == "warmup_pick"]
    assert spread["counters"]["warmup_picks"] == len(warm_notes) > first["counters"]["warmup_picks"]  # 묶음 값이 아니라 장부 합계
    assert spread["counters"]["return_picks"] == 9 and spread["counters"]["return_failures"] == 0
    assert sum(c.names.count("reference") for c in Camera.instances) == snaps_before  # 기준을 다시 찍지 않는다
    assert json.loads((folder / "reference.json").read_text()) == reference
    meta = next(r for r in map(json.loads, (folder / "spread.jsonl").read_text().splitlines()) if r["event"] == "meta")
    assert meta["reference"] == spread["reference"] and "sheet_sha256" in meta

    sheet_path = tmp_path / "sheet.json"
    sheet_path.write_text(json.dumps(tolerance_sheet(ready_config, table, 1)))
    run = tmp_path / "G0d-1"
    run.mkdir()
    session.run_g0d("tolerance", _arm(ready_config, table), ready_config, table, run, "G0d-1", 5.0, sheet_path, folder / "spread.json")
    result = json.loads((run / "tolerance.json").read_text())
    assert set(result) >= {"r", "lateral", "j5", "end"} and result["end"] in ("batch", "complete")
    meta = next(r for r in map(json.loads, (run / "tolerance.jsonl").read_text().splitlines()) if r["event"] == "meta")
    assert meta["spread_sha256"] and meta["reference"] == spread["reference"]


@pytest.mark.parametrize("field", ["complete", "failures", "incomplete", "yellow_ok"])
def test_tolerance_refuses_unfinished_spread(wired, ready_config, table, tmp_path, field):
    spread = {"complete": True, "failures": 0, "incomplete": 0, "yellow_ok": True, "reference": {"0": [915, 654], "1": [851, 211]}, "cells": {}, "transfer": "verified"}
    spread[field] = {"complete": False, "failures": 1, "incomplete": 1, "yellow_ok": False}[field]
    path = tmp_path / "spread.json"
    path.write_text(json.dumps(spread))
    arm = _arm(ready_config, table)
    with pytest.raises(SystemExit, match="G0d 1차로 가지 않는다"):
        session.run_g0d("tolerance", arm, ready_config, table, tmp_path, "x", 5.0, tmp_path / "sheet.json", path)
    assert not arm.driver.driver.writes


def test_stop_is_marked_and_warns_about_held_block(wired, monkeypatch, ready_config, table, tmp_path):
    monkeypatch.setattr(Camera, "yellow_value", 500)  # 판정 실패. 답은 y라 miss가 아니므로 쓰기 없이 정지
    folder = tmp_path / "G0d-0"
    folder.mkdir()
    with pytest.raises(SystemExit, match="끝 종류 stop"):
        session.run_g0d("natural-spread", _arm(ready_config, table), ready_config, table, folder, "G0d-0", 5.0, None, None)
    assert json.loads((folder / "spread.json").read_text())["end"] == "stop"


def test_writes_left_reaches_executor(wired, ready_config, table, tmp_path, capsys):
    folder = tmp_path / "G0d-0"
    folder.mkdir()
    arm = _arm(ready_config, table)
    with pytest.raises(SystemExit, match="남은 쓰기"):
        session.run_g0d("natural-spread", arm, ready_config, table, folder, "G0d-0", 5.0, None, None, writes_left=lambda: 10)
    assert not arm.driver.driver.writes
