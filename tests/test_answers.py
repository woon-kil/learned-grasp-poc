from __future__ import annotations

import json

import pytest

from learned_grasp.answers import FileAnswers, answer_once


class Clock:
    def __init__(self):
        self.now = 0.0
        self.events = []  # (시각, 할 일)

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        for when, action in list(self.events):
            if self.now >= when:
                self.events.remove((when, action))
                action()


def _reply(folder, answer, same_id=True):
    def action():
        data = json.loads((folder / "question.json").read_text())
        (folder / "answer.json").write_text(json.dumps({"id": data["id"] if same_id else "other", "answer": answer}))
    return action


def test_answer_in_time_is_returned_and_files_cleaned(tmp_path):
    clock = Clock()
    clock.events.append((1.0, _reply(tmp_path, "miss")))
    ask = FileAnswers(tmp_path, clock=clock, sleep=clock.sleep, log=lambda _: None)
    assert ask("failure", "t0 판정 실패", 10.0) == "miss"
    assert not (tmp_path / "question.json").exists() and not (tmp_path / "answer.json").exists()


def test_answer_after_deadline_is_refused(tmp_path):
    clock = Clock()
    clock.events.append((12.0, _reply(tmp_path, "miss")))
    ask = FileAnswers(tmp_path, clock=clock, sleep=clock.sleep, log=lambda _: None)
    assert ask("failure", "t0", 10.0) is None
    assert clock.now <= 10.5


def test_answer_for_other_question_or_bad_choice_is_ignored(tmp_path):
    clock = Clock()
    clock.events.append((1.0, _reply(tmp_path, "miss", same_id=False)))
    clock.events.append((2.0, _reply(tmp_path, "y")))
    ask = FileAnswers(tmp_path, clock=clock, sleep=clock.sleep, log=lambda _: None)
    assert ask("failure", "t0", 5.0) is None


def test_descent_waits_without_deadline(tmp_path):
    clock = Clock()
    clock.events.append((300.0, _reply(tmp_path, "n")))
    ask = FileAnswers(tmp_path, clock=clock, sleep=clock.sleep, log=lambda _: None)
    assert ask("descent", "t0 하강", None) == "n"


def test_terminal_tool_writes_only_valid_answer(tmp_path):
    assert answer_once(tmp_path, read=lambda _: "y") == "열린 질문이 없다"
    (tmp_path / "question.json").write_text(json.dumps({"id": "q1", "kind": "failure", "prompt": "p", "choices": ["miss", "held", "unknown"]}))
    assert "보내지 않았다" in answer_once(tmp_path, read=lambda _: "y")
    assert not (tmp_path / "answer.json").exists()
    assert answer_once(tmp_path, read=lambda _: " MISS ") == "보냈다: miss"
    assert json.loads((tmp_path / "answer.json").read_text()) == {"id": "q1", "answer": "miss"}


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        FileAnswers(tmp_path, log=lambda _: None)("other", "p", None)
