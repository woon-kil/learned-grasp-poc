#!/usr/bin/env python3
"""DaBai DCW2 정지 중 사양 측정: 해상도, 실제 프레임률, 내부 파라미터(화각), 깊이 통계, 첫 프레임 시간.

Jetson에서 ROS Humble과 dofbot_pro_ws를 source한 뒤 orbbec_camera 런치를 띄운 상태로 실행한다.
로봇 명령도 관절 조회도 없다. 사용법: python3 dabai_probe.py 출력디렉터리 [측정초]
"""

import json
import math
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


def qos():
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10,
                      reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)


def write_png(path, image):
    height, width = image.shape[:2]
    if image.ndim == 2:
        color_type, bit_depth, rows = 0, 16, image.astype(">u2")
    else:
        color_type, bit_depth, rows = 2, 8, image
    raw = b"".join(b"\x00" + np.ascontiguousarray(rows[r]).tobytes() for r in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def arrays(msg):
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if msg.encoding == "16UC1":
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    raise RuntimeError(f"예상 밖 인코딩 {msg.encoding}")


class Probe(Node):
    def __init__(self, ns):
        super().__init__("dabai_probe")
        self.t0 = time.monotonic()
        self.rx = {"color": [], "depth": []}
        self.last = {}
        self.info = {}
        self.first = {}
        self.create_subscription(Image, f"/{ns}/color/image_raw", lambda m: self.on("color", m), qos())
        self.create_subscription(Image, f"/{ns}/depth/image_raw", lambda m: self.on("depth", m), qos())
        self.create_subscription(CameraInfo, f"/{ns}/color/camera_info", lambda m: self.on_info("color", m), qos())
        self.create_subscription(CameraInfo, f"/{ns}/depth/camera_info", lambda m: self.on_info("depth", m), qos())

    def on(self, kind, msg):
        now = time.monotonic()
        self.first.setdefault(kind, now - self.t0)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.rx[kind].append((now, stamp))
        self.last[kind] = (msg.encoding, msg.width, msg.height, msg.header.frame_id, arrays(msg))

    def on_info(self, kind, msg):
        self.info[kind] = {"width": msg.width, "height": msg.height, "k": list(msg.k), "d": list(msg.d),
                           "distortion_model": msg.distortion_model}


def fov(k, width, height):
    fx, fy = k[0], k[4]
    return (math.degrees(2 * math.atan(width / (2 * fx))), math.degrees(2 * math.atan(height / (2 * fy))))


def rate(times):
    if len(times) < 3:
        return None
    arrival = [t[0] for t in times]
    stamps = [t[1] for t in times]
    return {"n": len(times), "arrival_hz": (len(times) - 1) / (arrival[-1] - arrival[0]),
            "stamp_hz": (len(times) - 1) / (stamps[-1] - stamps[0]) if stamps[-1] > stamps[0] else None}


def main():
    out = Path(sys.argv[1])
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0
    ns = sys.argv[3] if len(sys.argv) > 3 else "dabai"
    out.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Probe(ns)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    result = {"robot_commands": 0, "joint_read": 0, "seconds": seconds, "namespace": ns,
              "first_frame_sec": node.first, "rate": {k: rate(v) for k, v in node.rx.items()}, "info": node.info}
    for kind, (enc, w, h, frame, image) in node.last.items():
        result[f"{kind}_image"] = {"encoding": enc, "width": w, "height": h, "frame_id": frame}
        write_png(out / f"{kind}.png", image)
        if kind == "depth":
            nz = image[image > 0]
            result["depth_stats_mm"] = {
                "valid_ratio": float(nz.size / image.size),
                "min": int(nz.min()) if nz.size else None,
                "p05": float(np.percentile(nz, 5)) if nz.size else None,
                "median": float(np.median(nz)) if nz.size else None,
                "p95": float(np.percentile(nz, 95)) if nz.size else None,
                "max": int(nz.max()) if nz.size else None,
            }
    for kind, info in node.info.items():
        if info["k"][0]:
            info["fov_deg_h_v"] = fov(info["k"], info["width"], info["height"])
    (out / "probe.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
