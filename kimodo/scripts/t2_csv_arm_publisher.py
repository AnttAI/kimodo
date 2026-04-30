#!/usr/bin/env python3
"""Publish only left/right Nero arm joints from a T2 retargeted CSV."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROBOT_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
RIGHT_T2_COLUMNS = [f"right_joint{i}_dof" for i in range(1, 8)]
LEFT_T2_COLUMNS = [f"left_joint{i}_dof" for i in range(1, 8)]


@dataclass(frozen=True)
class ArmFrame:
    frame_index: int
    right: list[float]
    left: list[float]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _offsets(value: str) -> list[float]:
    try:
        offsets = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("offsets must be comma-separated radian values") from exc
    if len(offsets) != len(ROBOT_JOINT_NAMES):
        raise argparse.ArgumentTypeError(f"expected {len(ROBOT_JOINT_NAMES)} comma-separated offsets")
    return offsets


def _iter_csv_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        non_comment_lines = (line for line in f if not line.lstrip().startswith("#"))
        reader = csv.DictReader(non_comment_lines)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {csv_path}")

        missing = [column for column in [*RIGHT_T2_COLUMNS, *LEFT_T2_COLUMNS] if column not in reader.fieldnames]
        if missing:
            raise ValueError("CSV is not a T2 arm CSV. Missing columns: " + ", ".join(missing))

        for row in reader:
            if row:
                yield row


def load_arm_frames(
    csv_path: Path,
    start_frame: int,
    max_frames: int | None,
    frame_stride: int,
) -> list[ArmFrame]:
    frames: list[ArmFrame] = []
    for source_index, row in enumerate(_iter_csv_rows(csv_path)):
        if source_index < start_frame:
            continue
        if (source_index - start_frame) % frame_stride != 0:
            continue

        try:
            frame_index = int(float(row.get("Frame", source_index)))
            right = [math.radians(float(row[column])) for column in RIGHT_T2_COLUMNS]
            left = [math.radians(float(row[column])) for column in LEFT_T2_COLUMNS]
        except ValueError as exc:
            raise ValueError(f"Invalid numeric data near source row {source_index + 2}") from exc

        frames.append(ArmFrame(frame_index=frame_index, right=right, left=left))
        if max_frames is not None and len(frames) >= max_frames:
            break

    if not frames:
        raise ValueError("No frames selected from CSV.")
    return frames


def apply_offsets(
    frames: list[ArmFrame],
    right_offsets: list[float],
    left_offsets: list[float],
) -> list[ArmFrame]:
    if not any(right_offsets) and not any(left_offsets):
        return frames

    return [
        ArmFrame(
            frame_index=frame.frame_index,
            right=[value + offset for value, offset in zip(frame.right, right_offsets)],
            left=[value + offset for value, offset in zip(frame.left, left_offsets)],
        )
        for frame in frames
    ]


def print_dry_run(frames: list[ArmFrame], dry_run_frames: int) -> None:
    print(f"[DRY-RUN] selected_frames={len(frames)}")
    print(f"[DRY-RUN] ros_joint_names={ROBOT_JOINT_NAMES}")
    for frame in frames[:dry_run_frames]:
        right = ", ".join(f"{name}={value:.4f}" for name, value in zip(ROBOT_JOINT_NAMES, frame.right))
        left = ", ".join(f"{name}={value:.4f}" for name, value in zip(ROBOT_JOINT_NAMES, frame.left))
        print(f"[DRY-RUN] frame={frame.frame_index}")
        print(f"[DRY-RUN]   right: {right}")
        print(f"[DRY-RUN]   left:  {left}")


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


def publish_frames(args: argparse.Namespace, frames: list[ArmFrame]) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
    except ImportError as exc:
        raise RuntimeError(
            "ROS 2 Python packages are not available. Source ROS 2 and the robot workspace, "
            "or run with --dry-run."
        ) from exc

    rclpy.init()
    node = Node("t2_csv_arm_publisher")
    right_pub = node.create_publisher(JointState, args.right_topic, 10)
    left_pub = node.create_publisher(JointState, args.left_topic, 10)

    try:
        wait_for_subscribers(right_pub, left_pub, args.wait_for_subscribers)

        interval = 1.0 / (args.rate * args.playback_speed)
        loop_count = 0
        print(
            f"[INFO] Publishing {len(frames)} frames to "
            f"{args.right_topic} and {args.left_topic} at {1.0 / interval:.2f} Hz"
        )

        while rclpy.ok():
            loop_count += 1
            for frame in frames:
                now = node.get_clock().now().to_msg()

                right_msg = JointState()
                right_msg.header.stamp = now
                right_msg.name = ROBOT_JOINT_NAMES
                right_msg.position = frame.right

                left_msg = JointState()
                left_msg.header.stamp = now
                left_msg.name = ROBOT_JOINT_NAMES
                left_msg.position = frame.left

                right_pub.publish(right_msg)
                left_pub.publish(left_msg)
                rclpy.spin_once(node, timeout_sec=0.0)
                time.sleep(interval)

            if not args.loop:
                break
            if args.loop_count is not None and loop_count >= args.loop_count:
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish only T2 CSV arm joints to AGX Nero ROS 2 JointState topics."
    )
    parser.add_argument("--csv", required=True, type=Path, help="T2 retargeted CSV file.")
    parser.add_argument("--right-topic", default="/right_arm/control/move_j")
    parser.add_argument("--left-topic", default="/left_arm/control/move_j")
    parser.add_argument("--rate", type=_positive_float, default=20.0, help="CSV row publish rate in Hz.")
    parser.add_argument(
        "--playback-speed",
        type=_positive_float,
        default=1.0,
        help="Playback speed multiplier. Values below 1 slow motion down.",
    )
    parser.add_argument("--start-frame", type=_non_negative_int, default=0)
    parser.add_argument("--max-frames", type=_positive_int)
    parser.add_argument("--frame-stride", type=_positive_int, default=1)
    parser.add_argument("--dry-run-frames", type=_positive_int, default=5)
    parser.add_argument(
        "--right-offsets",
        type=_offsets,
        default=[0.0] * len(ROBOT_JOINT_NAMES),
        help="Comma-separated radian offsets added to right joint1..joint7.",
    )
    parser.add_argument(
        "--left-offsets",
        type=_offsets,
        default=[0.0] * len(ROBOT_JOINT_NAMES),
        help="Comma-separated radian offsets added to left joint1..joint7.",
    )
    parser.add_argument("--loop", action="store_true", help="Repeat the selected CSV frames.")
    parser.add_argument("--loop-count", type=_positive_int)
    parser.add_argument(
        "--wait-for-subscribers",
        type=float,
        default=5.0,
        help="Seconds to wait for both robot-side subscribers before publishing.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print selected arm data without ROS.")
    mode.add_argument("--execute", action="store_true", help="Actually publish to ROS 2.")

    args = parser.parse_args(argv)
    if args.loop_count is not None and not args.loop:
        parser.error("--loop-count requires --loop")
    if not args.csv.is_file():
        parser.error(f"CSV file not found: {args.csv}")
    if not args.execute:
        args.dry_run = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        frames = load_arm_frames(
            csv_path=args.csv,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            frame_stride=args.frame_stride,
        )
        frames = apply_offsets(frames, args.right_offsets, args.left_offsets)
        if args.dry_run:
            print_dry_run(frames, args.dry_run_frames)
            return 0

        publish_frames(args, frames)
        return 0
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
