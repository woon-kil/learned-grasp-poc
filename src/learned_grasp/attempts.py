"""시도 기록. 줄 단위 증분 저장이고 실패·중단을 지우지 않는다."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class AttemptLedger:
    def __init__(self, path: Path, clock=time.time):
        self.path = Path(path)
        self.clock = clock

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def bind(self, **meta) -> None:
        """첫 실행의 구성을 기록하고, 재개 때 다르면 거부한다."""
        previous = next((record for record in self.records() if record["event"] == "meta"), None)
        if previous is None:
            self._append({"event": "meta", "t": self.clock(), **meta})
            return
        different = {key: (previous.get(key), value) for key, value in meta.items() if previous.get(key) != value}
        if different:
            raise RuntimeError(f"다른 구성으로 같은 기록을 이어 쓸 수 없다: {different}")

    def start(self, trial: dict, **fields) -> None:
        self._append({"event": "start", "t": self.clock(), **trial, **fields})

    def finish(self, trial_id: str, **fields) -> None:
        self._append({"event": "finish", "trial_id": trial_id, "t": self.clock(), **fields})

    def state(self) -> tuple[dict[str, dict], set[str]]:
        """완료 기록과, 시작했지만 끝나지 않은 시도 id를 돌려준다."""
        started: dict[str, dict] = {}
        finished: dict[str, dict] = {}
        for record in self.records():
            if record["event"] == "start":
                started[record["trial_id"]] = record
            elif record["event"] == "finish":
                begin = started.get(record["trial_id"], {})
                finished[record["trial_id"]] = {**begin, **record, "started_at": begin.get("t")}
        return finished, set(started) - set(finished)

    def close_interrupted(self) -> list[str]:
        """이전 세션에서 끝나지 않은 시도를 중단으로 닫는다. 다시 시도하지 않고 분모에 남긴다."""
        _, open_ids = self.state()
        for trial_id in sorted(open_ids):
            self.finish(trial_id, outcome="interrupted", success=False, reason="세션 중단")
        return sorted(open_ids)
