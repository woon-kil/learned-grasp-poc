#!/usr/bin/env python3
"""H2-1: 팔이 정지한 동안 Gemini 컬러를 N장 연속으로 받아 온다. 로봇 명령은 없다.

사용: uv run python scripts/h2_burst.py <출력디렉터리(체크아웃 안 상대경로)> <이름접두> <장수>
저장: <출력디렉터리>/<이름접두>-NN.color.png (Mac)
"""
import sys
import time

from learned_grasp.camera import P620Camera


def main() -> None:
    out, prefix, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
    started = time.monotonic()
    camera = P620Camera(out)
    ready = time.monotonic()
    try:
        for i in range(count):
            name = f"{prefix}-{i:02d}"
            camera.snapshot(name)
            camera.fetch(name)
    finally:
        camera.close()
    print(f"{count}장 저장: {out}/{prefix}-NN.color.png  서버 시작 {ready - started:.1f}초, 촬영·전송 {time.monotonic() - ready:.1f}초")


if __name__ == "__main__":
    main()
