#!/usr/bin/env python3
"""드라이버 시작 뒤 컬러 밝기·노출·게인의 시간 변화를 잰다. 로봇 명령은 없다.

사용: python3 scripts/gemini_probe.py <driver_log> <output_dir> <duration_sec>
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from gemini_capture import image_array, qos, write_png
from orbbec_camera_msgs.srv import GetInt32
from rclpy.node import Node
from sensor_msgs.msg import Image

BOARD = (570, 820, 210, 390)


class Probe(Node):
    def __init__(self):
        super().__init__("lg_gemini_probe")
        self.latest = None
        self.count = 0
        for reliable in (True, False):
            self.create_subscription(Image, "/camera/color/image_raw", self.on_color, qos(reliable))
        self.exposure = self.create_client(GetInt32, "/camera/get_color_exposure")
        self.gain = self.create_client(GetInt32, "/camera/get_color_gain")

    def on_color(self, message):
        if message.encoding == "rgb8":
            self.latest = message
            self.count += 1

    def read(self, client):
        if not client.service_is_ready():
            return None
        future = client.call_async(GetInt32.Request())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not future.done():
            rclpy.spin_once(self, timeout_sec=0.05)
        return future.result().data if future.done() and future.result().success else None


def main():
    output = Path(sys.argv[2])
    duration = float(sys.argv[4]) if len(sys.argv) > 4 else float(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Probe()
    start = time.monotonic()
    rows = []
    first = None
    next_sample = 0.0
    try:
        while time.monotonic() - start < duration:
            rclpy.spin_once(node, timeout_sec=0.05)
            elapsed = time.monotonic() - start
            if node.latest is None or elapsed < next_sample:
                continue
            image = image_array(node.latest)
            u0, u1, v0, v1 = BOARD
            row = {
                "t": round(elapsed, 2),
                "frames": node.count,
                "mean": round(float(image.mean()), 1),
                "board": round(float(np.median(image[v0:v1, u0:u1].mean(axis=2))), 1),
                "saturated_pct": round(float((image >= 250).any(axis=2).mean() * 100), 3),
                "exposure": node.read(node.exposure),
                "gain": node.read(node.gain),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if first is None:
                first = image
                write_png(output / "first.png", image)
            next_sample = elapsed + 1.0
        if node.latest is not None:
            write_png(output / "last.png", image_array(node.latest))
    finally:
        (output / "series.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
