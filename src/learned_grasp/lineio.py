"""하위 프로세스 표준출력에서 한 줄을 마감 시간 안에 읽는다.

select가 읽을 수 있다고 해도 줄 끝이 오지 않으면 readline은 무기한 기다린다. 그래서 파일 기술자를 직접 읽고
줄 끝이 올 때까지 남은 시간만 기다린다.
"""

from __future__ import annotations

import os
import select
import time


class LineReader:
    def __init__(self, stream, clock=time.monotonic):
        self.fd = stream.fileno()
        self.buffer = b""
        self.clock = clock

    def readline(self, timeout_s: float) -> str:
        """한 줄을 돌려준다. 연결이 끝나면 빈 문자열, 마감을 넘기면 TimeoutError다."""
        deadline = self.clock() + timeout_s
        while b"\n" not in self.buffer:
            remaining = deadline - self.clock()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                raise TimeoutError(f"{timeout_s:g}초 안에 한 줄을 받지 못했다")
            chunk = os.read(self.fd, 4096)
            if not chunk:
                return ""
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8") + "\n"
