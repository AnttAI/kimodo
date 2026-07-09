#!/usr/bin/env python3
"""Publish streamed T2 arm frames and optional gripper widths to ROS 2."""

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


def wait_for_subscribers(right_pub, left_pub, base_pub, timeout_sec: float, require_base: bool) -> None:
    if timeout_sec <= 0.0:
        return

    start = time.monotonic()
    while time.monotonic() - start < timeout_sec:
        arms_ready = right_pub.get_subscription_count() > 0 and left_pub.get_subscription_count() > 0
        base_ready = not require_base or base_pub.get_subscription_count() > 0
        if arms_ready and base_ready:
            return
        time.sleep(0.1)

    raise TimeoutError(
        "Timed out waiting for the arm/base ROS subscribers. "
        "Start the robot-side dual Nero + TaraBase launch first."
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream current visualizer T2 arm frames to AGX Nero ROS 2 JointState topics."
    )
    parser.add_argument("--right-topic", default="/right_arm/control/move_j")
    parser.add_argument("--left-topic", default="/left_arm/control/move_j")
    parser.add_argument("--gripper-topic", default="/right_arm/control/joint_states")
    parser.add_argument("--base-topic", default="/base/cmd_wheel_rpm")
    parser.add_argument("--base-csv-topic", default="/base/play_csv")
    parser.add_argument(
        "--require-base-subscriber",
        action="store_true",
        help="Wait for the TaraBase ROS subscriber before accepting streamed frames.",
    )
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


def _iter_stream_frames() -> Iterator[
    tuple[
        int,
        list[float] | None,
        list[float] | None,
        float | None,
        float,
        tuple[float, float] | None,
        dict[str, object] | None,
        str | None,
    ]
]:
    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        frame_index = int(payload.get("frame_index", -1))
        right_value = payload.get("right")
        left_value = payload.get("left")
        if right_value is None and left_value is None:
            right = None
            left = None
        elif right_value is None or left_value is None:
            raise ValueError("right and left arm positions must either both be present or both be omitted")
        else:
            right = _validate_positions(right_value, "right")
            left = _validate_positions(left_value, "left")
        gripper_value = payload.get("gripper")
        gripper = None if gripper_value is None else float(gripper_value)
        effort = float(payload.get("gripper_effort", 1.0))
        base_value = payload.get("base_wheel_rpm")
        if base_value is None:
            base_wheel_rpm = None
        elif not isinstance(base_value, list) or len(base_value) != 2:
            raise ValueError("base_wheel_rpm must be [left_rpm, right_rpm]")
        else:
            base_wheel_rpm = (float(base_value[0]), float(base_value[1]))
        base_csv_value = payload.get("base_csv")
        if base_csv_value is not None and not isinstance(base_csv_value, dict):
            raise ValueError("base_csv must be an object containing csv_text and options")
        base_csv = dict(base_csv_value) if isinstance(base_csv_value, dict) else None
        if base_csv is not None:
            if not isinstance(base_csv.get("csv_text"), str):
                raise ValueError("base_csv.csv_text must be a string")
            if not isinstance(base_csv.get("options", {}), dict):
                raise ValueError("base_csv.options must be an object")
        base_csv_action_value = payload.get("base_csv_action")
        base_csv_action = None if base_csv_action_value is None else str(base_csv_action_value)
        yield frame_index, right, left, gripper, effort, base_wheel_rpm, base_csv, base_csv_action


def _format_positions(values: list[float]) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in zip(ROBOT_JOINT_NAMES, values))


def _combined_right_gripper_msg(joint_state_cls, stamp, right: list[float], gripper: float, effort: float):
    msg = joint_state_cls()
    msg.header.stamp = stamp
    msg.name = [*ROBOT_JOINT_NAMES, "gripper"]
    msg.position = [*right, gripper]
    msg.velocity = []
    msg.effort = [0.0] * len(ROBOT_JOINT_NAMES) + [effort]
    return msg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.dry_run:
        print("[READY] Dry-running streamed visualizer frames", flush=True)
        try:
            for (
                frame_index,
                right,
                left,
                gripper,
                effort,
                base_wheel_rpm,
                base_csv,
                base_csv_action,
            ) in _iter_stream_frames():
                print(f"[DRY-RUN] frame={frame_index}", flush=True)
                if right is not None and left is not None:
                    print(f"[DRY-RUN]   right: {_format_positions(right)}", flush=True)
                    print(f"[DRY-RUN]   left:  {_format_positions(left)}", flush=True)
                if gripper is not None and right is not None:
                    print(
                        "[DRY-RUN]   combined gripper msg: "
                        f"name={[*ROBOT_JOINT_NAMES, 'gripper']} "
                        f"position={[*right, gripper]} "
                        f"effort={[0.0] * len(ROBOT_JOINT_NAMES) + [effort]}",
                        flush=True,
                    )
                if base_wheel_rpm is not None:
                    print(f"[DRY-RUN]   base wheel RPM: {list(base_wheel_rpm)}", flush=True)
                if base_csv is not None:
                    print("[DRY-RUN]   base CSV playback request", flush=True)
                if base_csv_action is not None:
                    print(f"[DRY-RUN]   base CSV action: {base_csv_action}", flush=True)
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
        from std_msgs.msg import Float64MultiArray, String
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
    gripper_pub = node.create_publisher(JointState, args.gripper_topic, 10)
    base_pub = node.create_publisher(Float64MultiArray, args.base_topic, 10)
    base_csv_pub = node.create_publisher(String, args.base_csv_topic, 10)

    try:
        wait_for_subscribers(
            right_pub,
            left_pub,
            base_pub,
            args.wait_for_subscribers,
            args.require_base_subscriber,
        )
        print(
            f"[READY] Streaming visualizer frames to {args.right_topic}, {args.left_topic}, "
            f"gripper commands to {args.gripper_topic}, and base RPM to {args.base_topic}",
            flush=True,
        )

        for (
            _frame_index,
            right,
            left,
            gripper,
            effort,
            base_wheel_rpm,
            base_csv,
            base_csv_action,
        ) in _iter_stream_frames():
            now = node.get_clock().now().to_msg()

            if right is not None and left is not None:
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
            if gripper is not None and right is not None:
                gripper_pub.publish(_combined_right_gripper_msg(JointState, now, right, gripper, effort))
            if base_wheel_rpm is not None:
                base_msg = Float64MultiArray()
                base_msg.data = [base_wheel_rpm[0], base_wheel_rpm[1]]
                base_pub.publish(base_msg)
            if base_csv is not None or base_csv_action is not None:
                base_csv_msg = String()
                if base_csv is not None:
                    base_csv_msg.data = json.dumps(base_csv, separators=(",", ":"))
                else:
                    base_csv_msg.data = json.dumps(
                        {"action": base_csv_action}, separators=(",", ":")
                    )
                base_csv_pub.publish(base_csv_msg)
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
