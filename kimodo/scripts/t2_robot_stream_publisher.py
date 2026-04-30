#!/usr/bin/env python3
"""Publish streamed T2 arm frames to dual Nero ROS 2 JointState topics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterator


ROBOT_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def wait_for_subscribers(right_pub, left_pub, timeout_sec: float) -> None:
    if timeout_sec <= 0.0:
        return

    start = time.monotonic()
    while time.monotonic() - start < timeout_sec:
        if right_pub.get_subscription_count() > 0 and left_pub.get_subscription_count() > 0:
            return
        time.sleep(0.1)

    raise TimeoutError(
        "Timed out waiting for subscribers on both arm topics. "
        "Start the robot-side dual Nero control launch first."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream current visualizer T2 arm frames to AGX Nero ROS 2 JointState topics."
    )
    parser.add_argument("--right-topic", default="/right_arm/control/move_j")
    parser.add_argument("--left-topic", default="/left_arm/control/move_j")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read streamed visualizer frames and print validated arm data without importing ROS or publishing.",
    )
    parser.add_argument(
        "--wait-for-subscribers",
        type=_non_negative_float,
        default=5.0,
        help="Seconds to wait for both robot-side subscribers before accepting frames. Defaults to 5 seconds.",
    )
    return parser.parse_args(argv)


def _validate_positions(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != len(ROBOT_JOINT_NAMES):
        raise ValueError(f"{label} must be a list with {len(ROBOT_JOINT_NAMES)} values")
    return [float(item) for item in value]


def _iter_stream_frames() -> Iterator[tuple[int, list[float], list[float]]]:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        frame_index = int(payload.get("frame_index", -1))
        right = _validate_positions(payload.get("right"), "right")
        left = _validate_positions(payload.get("left"), "left")
        yield frame_index, right, left


def _format_positions(values: list[float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in zip(ROBOT_JOINT_NAMES, values))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.dry_run:
        print("[READY] Dry-running streamed visualizer frames", flush=True)
        try:
            for frame_index, right, left in _iter_stream_frames():
                print(f"[DRY-RUN] frame={frame_index}", flush=True)
                print(f"[DRY-RUN]   right: {_format_positions(right)}", flush=True)
                print(f"[DRY-RUN]   left:  {_format_positions(left)}", flush=True)
        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.", flush=True)
            return 130
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
            return 1
        return 0

    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        print(
            "[ERROR] ROS 2 Python packages are not available. Source ROS 2 and the robot workspace.",
            file=sys.stderr,
        )
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    rclpy.init()
    node = Node("t2_robot_stream_publisher")
    right_pub = node.create_publisher(JointState, args.right_topic, 10)
    left_pub = node.create_publisher(JointState, args.left_topic, 10)

    try:
        wait_for_subscribers(right_pub, left_pub, args.wait_for_subscribers)
        print(
            f"[READY] Streaming visualizer frames to {args.right_topic} and {args.left_topic}",
            flush=True,
        )

        for _frame_index, right, left in _iter_stream_frames():
            now = node.get_clock().now().to_msg()

            right_msg = JointState()
            right_msg.header.stamp = now
            right_msg.name = ROBOT_JOINT_NAMES
            right_msg.position = right

            left_msg = JointState()
            left_msg.header.stamp = now
            left_msg.name = ROBOT_JOINT_NAMES
            left_msg.position = left

            right_pub.publish(right_msg)
            left_pub.publish(left_msg)
            rclpy.spin_once(node, timeout_sec=0.0)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.", flush=True)
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
