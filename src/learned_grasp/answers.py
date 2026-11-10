"""G0d 사람 답. 실행기는 질문 파일을 쓰고 답 파일을 기다린다. 사람은 다른 터미널에서 `session answer`로 답한다.

입력 대기와 실행기 대기가 분리돼 닫힘 감시를 막지 않는다. 마감을 넘긴 답과 다른 질문의 답은 받지 않는다.
답을 자동화하지 않는다.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Callable

CHOICES = {"descent": ("y", "n"), "reload": ("y", "n"), "failure": ("miss", "held", "unknown")}


def _write(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


class FileAnswers:
    def __init__(self, folder: Path, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep, log: Callable[[str], None] = print):
        self.folder = Path(folder)
        self.clock, self.sleep, self.log = clock, sleep, log

    @property
    def question(self) -> Path:
        return self.folder / "question.json"

    @property
    def answer(self) -> Path:
        return self.folder / "answer.json"

    def __call__(self, kind: str, prompt: str, deadline_s: float | None) -> str | None:
        if kind not in CHOICES:
            raise ValueError(kind)
        qid = uuid.uuid4().hex
        self.answer.unlink(missing_ok=True)
        _write(self.question, {"id": qid, "kind": kind, "prompt": prompt, "choices": CHOICES[kind], "deadline_s": deadline_s})
        self.log(json.dumps({"need_answer": kind, "prompt": prompt, "deadline_s": deadline_s}, ensure_ascii=False))
        end = None if deadline_s is None else self.clock() + deadline_s
        try:
            while True:
                if self.answer.exists():
                    data = json.loads(self.answer.read_text(encoding="utf-8"))
                    self.answer.unlink(missing_ok=True)
                    if data.get("id") == qid and data.get("answer") in CHOICES[kind]:
                        if end is not None and self.clock() > end:
                            return None
                        return data["answer"]
                if end is not None and self.clock() > end:
                    return None
                self.sleep(0.2)
        finally:
            self.question.unlink(missing_ok=True)


def answer_once(folder: Path, read: Callable[[str], str] = input) -> str:
    """터미널 도구. 지금 열린 질문 하나에 답한다. 질문이 없으면 아무것도 쓰지 않는다."""
    question = Path(folder) / "question.json"
    if not question.exists():
        return "열린 질문이 없다"
    data = json.loads(question.read_text(encoding="utf-8"))
    choices = data["choices"]
    reply = read(f"{data['prompt']} {list(choices)}: ").strip().lower()
    if reply not in choices:
        return f"보내지 않았다: {reply!r}는 {list(choices)} 중 하나가 아니다"
    _write(Path(folder) / "answer.json", {"id": data["id"], "answer": reply})
    return f"보냈다: {reply}"
