#!/usr/bin/env python3
"""드라이버를 켜 둔 채 요청마다 저장하는 카메라 서버. 로봇 명령은 없다.

표준입력 JSON 줄: {"op":"snap","name":"x"} / {"op":"diff","a":"x","b":"y","roi":[u0,u1,v0,v1]} / {"op":"bye"}
사용: python3 scripts/gemini_serve.py <driver_log> <output_dir> <settle_sec>
"""

import json
import select
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from gemini_capture import image_array, qos, stamp_sec, write_png
from gemini_serve_core import change_fraction, yellow_pixels
from rclpy.node import Node
from sensor_msgs.msg import Image

FRESH_FRAMES = 3  # 요청 뒤 도착한 프레임만 쓴다. 팔이 멈춘 뒤의 장면을 저장하기 위해서다


class Serve(Node):
    def __init__(self):
        super().__init__("lg_gemini_serve")
        self.color = None
        self.depth = None
        self.color_count = 0
        for reliable in (True, False):
            self.create_subscription(Image, "/camera/color/image_raw", self.on_color, qos(reliable))
            self.create_subscription(Image, "/camera/depth/image_raw", self.on_depth, qos(reliable))

    def on_color(self, message):
        if message.encoding == "rgb8":
            self.color = message
            self.color_count += 1

    def on_depth(self, message):
        if message.encoding == "16UC1":
            self.depth = message


def reply(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    output = Path(sys.argv[2])
    settle = float(sys.argv[3])
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Serve()
    saved = {}
    end = time.monotonic() + settle
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    if node.color is None:
        reply({"ok": False, "error": "컬러 프레임 없음"})
        return
    reply({"ok": True, "ready": True})
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.02)
            if not select.select([sys.stdin], [], [], 0)[0]:
                continue
            line = sys.stdin.readline()
            if not line:
                return
            request = json.loads(line)
            if request["op"] == "bye":
                reply({"ok": True})
                return
            if request["op"] == "snap":
                target = node.color_count + FRESH_FRAMES
                deadline = time.monotonic() + 5.0
                while node.color_count < target and time.monotonic() < deadline:
                    rclpy.spin_once(node, timeout_sec=0.05)
                if node.color_count < target:
                    reply({"ok": False, "error": "새 프레임 없음"})
                    return
                image = image_array(node.color)
                name = request["name"]
                write_png(output / f"{name}.color.png", image)
                if node.depth is not None:
                    write_png(output / f"{name}.depth.png", image_array(node.depth))
                saved[name] = image
                reply({"ok": True, "name": name, "stamp": stamp_sec(node.color.header.stamp), "mean": round(float(np.mean(image)), 1)})
            elif request["op"] == "diff":
                reply({"ok": True, "fraction": change_fraction(saved[request["a"]], saved[request["b"]], request["roi"])})
            elif request["op"] == "yellow":
                reply({"ok": True, "pixels": yellow_pixels(saved[request["name"]], request["roi"])})
            else:
                reply({"ok": False, "error": "알 수 없는 op"})
                return
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
