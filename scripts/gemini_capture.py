#!/usr/bin/env python3
"""Gemini 335L 한 장. RGB, color에 정합된 depth, CameraInfo를 저장한다.

simple-vision-tech-poc/scripts/b00_capture.py에서 옮겼다.

로봇·서보·MoveIt 명령은 없다.
사용: python3 scripts/gemini_capture.py <driver_log> <output_dir>
"""

import hashlib
import json
import os
import re
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


PAIR_MAX_SEC = 0.05


def qos(reliable):
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )


def stamp_sec(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def write_png(path, image):
    height, width = image.shape[:2]
    if image.ndim == 2 and image.dtype == np.uint16:
        color_type = 0
        bit_depth = 16
        rows = image.astype(">u2")
    elif image.ndim == 2:
        color_type = 0
        bit_depth = 8
        rows = image
    else:
        color_type = 2
        bit_depth = 8
        rows = image
    raw = b"".join(b"\x00" + np.ascontiguousarray(rows[row]).tobytes() for row in range(height))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    payload = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    path.write_bytes(payload)


def sha256_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def image_array(message):
    if message.encoding == "rgb8":
        return np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.width, 3).copy()
    if message.encoding == "16UC1":
        return np.frombuffer(message.data, dtype=np.uint16).reshape(message.height, message.width).copy()
    raise RuntimeError(f"지원하지 않는 encoding: {message.encoding}")


def info_dict(message):
    return {
        "frame_id": message.header.frame_id,
        "stamp_sec": stamp_sec(message.header.stamp),
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": message.distortion_model,
        "d": [float(value) for value in message.d],
        "k": [float(value) for value in message.k],
        "r": [float(value) for value in message.r],
        "p": [float(value) for value in message.p],
    }


def grab_identity(log_text):
    def grab(pattern):
        match = re.search(pattern, log_text)
        return match.group(1).strip() if match else None

    return {
        "device_name": grab(r"Device (.+) connected"),
        "serial": grab(r"Serial number: (\S+)"),
        "firmware": grab(r"Firmware version: (\S+)"),
        "ros_wrapper_version": grab(r"ROS Wrapper version: (\S+)"),
        "sdk_version": grab(r"SDK version: (\S+)"),
    }


class Grab(Node):
    def __init__(self):
        super().__init__("lg_gemini_capture")
        self.colors = []
        self.depths = []
        self.unaligned = []
        self.color_infos = []
        self.depth_infos = []
        topics = (
            (Image, "/camera/color/image_raw", self.on_color),
            (Image, "/camera/depth/image_raw", self.on_depth),
            (Image, "/camera/depth/image_unaligned", self.on_unaligned),
            (CameraInfo, "/camera/color/camera_info", self.on_color_info),
            (CameraInfo, "/camera/depth/camera_info", self.on_depth_info),
        )
        for reliable in (True, False):
            for msg_type, topic, callback in topics:
                self.create_subscription(msg_type, topic, callback, qos(reliable))

    def _push(self, bucket, item):
        bucket.append(item)
        del bucket[:-40]

    def on_color(self, message):
        if message.encoding == "rgb8" and message.width > 0:
            self._push(
                self.colors,
                (stamp_sec(message.header.stamp), message.header.frame_id, message.encoding, image_array(message)),
            )

    def on_depth(self, message):
        if message.encoding == "16UC1" and message.width > 0:
            self._push(
                self.depths,
                (stamp_sec(message.header.stamp), message.header.frame_id, message.encoding, image_array(message)),
            )

    def on_unaligned(self, message):
        if message.width > 0 and message.height > 0:
            self._push(
                self.unaligned,
                {
                    "stamp_sec": stamp_sec(message.header.stamp),
                    "frame_id": message.header.frame_id,
                    "encoding": message.encoding,
                    "width": int(message.width),
                    "height": int(message.height),
                },
            )

    def on_color_info(self, message):
        self._push(self.color_infos, info_dict(message))

    def on_depth_info(self, message):
        self._push(self.depth_infos, info_dict(message))


def closest(items, stamp, key):
    if not items:
        return None
    return min(items, key=lambda item: abs(key(item) - stamp))


def wait_pair(node, after_stamp, timeout):
    deadline = time.monotonic() + timeout
    node.colors.clear()
    node.depths.clear()
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.2)
        if not node.colors or not node.depths or not node.color_infos:
            continue
        best = None
        for color in node.colors:
            for depth in node.depths:
                delta = abs(color[0] - depth[0])
                if best is None or delta < best[0]:
                    best = (delta, color, depth)
        if best is None or best[0] > PAIR_MAX_SEC:
            continue
        if best[1][3].shape[:2] != best[2][3].shape:
            continue
        if after_stamp is not None and best[1][0] <= after_stamp:
            continue
        return best
    return None


def save_capture(output, log_path, node, pair, repeat_index, repeat_count, interval_sec):
    output.mkdir(parents=True, exist_ok=True)
    delta, color, depth = pair
    color_info = closest(node.color_infos, color[0], lambda item: item["stamp_sec"])
    depth_info = closest(node.depth_infos, depth[0], lambda item: item["stamp_sec"])
    unaligned = closest(node.unaligned, color[0], lambda item: item["stamp_sec"])
    write_png(output / "color.png", color[3])
    write_png(output / "depth.png", depth[3])
    identity = grab_identity(log_path.read_text(errors="replace")) if log_path.exists() else {}
    payload = {
        "scope": "STREAM_SAVE_REPLAY",
        "geometric_registration_accuracy": "NOT_CLAIMED",
        "color_topic": "/camera/color/image_raw",
        "depth_topic": "/camera/depth/image_raw",
        "unaligned_topic": "/camera/depth/image_unaligned",
        "color_encoding": color[2],
        "depth_encoding": depth[2],
        "depth_bit_depth": 16,
        "color_frame_id": color[1],
        "depth_frame_id": depth[1],
        "color_stamp_sec": color[0],
        "depth_stamp_sec": depth[0],
        "stamp_delta_sec": delta,
        "color_shape": [int(value) for value in color[3].shape],
        "depth_shape": [int(value) for value in depth[3].shape],
        "depth_nonzero": int(np.count_nonzero(depth[3])),
        "color": color_info,
        "depth": depth_info,
        "unaligned": unaligned,
        "identity": identity,
        "robot_commands": 0,
        "runtime": {
            "git_commit": os.environ.get("LG_GIT_COMMIT", ""),
            "git_branch": os.environ.get("LG_GIT_BRANCH", ""),
            "host": os.environ.get("LG_HOST", ""),
            "image": os.environ.get("LG_CAMERA_IMAGE", ""),
            "container": os.environ.get("LG_CONTAINER", ""),
            "usb_node": os.environ.get("LG_USB_NODE", ""),
            "project_root": os.environ.get("LG_PROJECT_ROOT", ""),
            "repeat_index": repeat_index,
            "repeat_count": repeat_count,
            "interval_sec": interval_sec,
        },
        "files": {
            "color.png": sha256_file(output / "color.png"),
            "depth.png": sha256_file(output / "depth.png"),
        },
        "alignment": {
            "same_shape": list(color[3].shape[:2]) == list(depth[3].shape),
            "same_frame": color[1] == depth[1] == "camera_color_optical_frame",
            "color_info_matches_image": bool(
                color_info
                and color_info["width"] == color[3].shape[1]
                and color_info["height"] == color[3].shape[0]
            ),
        },
    }
    (output / "camera_info.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ok = all(payload["alignment"].values()) and payload["depth_nonzero"] > 0
    print(
        "GEMINI_CAPTURE_SAVED",
        "aligned" if ok else "NOT_ALIGNED",
        color[1],
        payload["color_shape"],
        identity.get("serial"),
    )
    if not ok:
        raise SystemExit("정렬 계약 실패")
    return color[0]


def main():
    if len(sys.argv) not in (3, 5):
        raise SystemExit("사용: gemini_capture.py <driver_log> <output_dir> [<repeat> <interval_sec>]")
    log_path = Path(sys.argv[1])
    output = Path(sys.argv[2]).resolve()
    repeat = 1 if len(sys.argv) == 3 else int(sys.argv[3])
    interval = 0.0 if len(sys.argv) == 3 else float(sys.argv[4])
    if repeat < 1:
        raise SystemExit("repeat는 1 이상이어야 한다")
    output.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = Grab()
    after = None
    try:
        for index in range(1, repeat + 1):
            if index > 1 and interval > 0:
                end = time.monotonic() + interval
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.2)
            pair = wait_pair(node, after, 40.0)
            if pair is None:
                raise SystemExit(f"정합 depth와 RGB를 같은 시각에 받지 못했다. index={index}")
            target = output if repeat == 1 else output / f"s{index:02d}"
            after = save_capture(target, log_path, node, pair, index, repeat, interval)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
