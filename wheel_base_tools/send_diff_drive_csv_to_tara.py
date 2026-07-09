#!/usr/bin/env python3
"""Stream diff-drive CSV RPM commands to TaraBase over MODBUS."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
TARA_SDK_SRC = REPO_ROOT / "tara_sdk" / "src"
if str(TARA_SDK_SRC) not in sys.path:
    sys.path.insert(0, str(TARA_SDK_SRC))


DEFAULT_CSV = (
    REPO_ROOT
    / "robot_demo_outputs"
    / "wheel_base_robot"
    / "kimodo_picking_item_from_the_shelf_office_diff_drive.csv"
)


@dataclass(frozen=True)
class WheelCommand:
    frame: int
    left_rpm: float
    right_rpm: float
    action: str = ""
    encoder_target_wheel_revolutions: float | None = None


DEFAULT_WHEEL_RADIUS_M = 0.10
TARA_ENCODER_COUNTS_PER_WHEEL_REVOLUTION = 4096
ENCODER_TURN_TIMEOUT_S = 8.0


ProgressCallback = Callable[
    [int, int, float, float, float, float, float, float, float, float, float, float],
    None,
]
StatusCallback = Callable[[str], None]


def load_wheel_commands(csv_path: Path) -> list[WheelCommand]:
    csv_path = csv_path.expanduser().resolve()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header")

        left_column = "left_motor_rpm" if "left_motor_rpm" in reader.fieldnames else "left_wheel_rpm"
        right_column = "right_motor_rpm" if "right_motor_rpm" in reader.fieldnames else "right_wheel_rpm"
        missing = [name for name in (left_column, right_column) if name not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is missing {', '.join(missing)}. "
                "Expected left_motor_rpm/right_motor_rpm columns."
            )

        commands: list[WheelCommand] = []
        for row_index, row in enumerate(reader):
            frame_text = row.get("Frame", row_index)
            commands.append(
                WheelCommand(
                    frame=int(float(frame_text)),
                    left_rpm=float(row[left_column]),
                    right_rpm=float(row[right_column]),
                    action=str(row.get("base_action", "")).strip(),
                    encoder_target_wheel_revolutions=(
                        float(row["encoder_target_wheel_revolutions"])
                        if row.get("encoder_target_wheel_revolutions", "").strip()
                        else None
                    ),
                )
            )

    if not commands:
        raise ValueError(f"{csv_path} has no command rows")
    return commands


def reverse_wheel_commands(commands: list[WheelCommand]) -> list[WheelCommand]:
    """Return commands that retrace the same wheel path backward."""
    total = len(commands)
    return [
        WheelCommand(
            frame=total - 1 - index,
            left_rpm=-command.left_rpm,
            right_rpm=-command.right_rpm,
            action=command.action,
            # Encoder termination is intentionally only enabled for normal
            # forward playback during this left-90 prototype.
            encoder_target_wheel_revolutions=None,
        )
        for index, command in enumerate(reversed(commands))
    ]


def _clamp(value: float, max_abs_rpm: float) -> float:
    return max(-max_abs_rpm, min(max_abs_rpm, value))


def _signed_32_bit_delta(current: int, start: int) -> int:
    """Return a wrap-safe signed delta for the controller's position counter."""
    return ((current - start + 0x80000000) % 0x100000000) - 0x80000000


def playback_speed_fit_to_rpm_limit(
    commands: list[WheelCommand],
    *,
    speed_scale: float,
    playback_speed: float,
    max_abs_rpm: float,
) -> float:
    max_command_rpm = 0.0
    for command in commands:
        max_command_rpm = max(
            max_command_rpm,
            abs(command.left_rpm * speed_scale * playback_speed),
            abs(command.right_rpm * speed_scale * playback_speed),
        )
    if max_command_rpm <= max_abs_rpm or max_command_rpm == 0.0:
        return playback_speed
    return playback_speed * (max_abs_rpm / max_command_rpm)


def estimate_travel_distance_m(
    commands: list[WheelCommand],
    *,
    fps: float,
    speed_scale: float,
    playback_speed: float,
    max_abs_rpm: float,
    fit_to_rpm_limit: bool = False,
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
    start_index: int = 0,
) -> float:
    if fps <= 0 or playback_speed <= 0:
        return 0.0
    effective_playback_speed = (
        playback_speed_fit_to_rpm_limit(
            commands,
            speed_scale=speed_scale,
            playback_speed=playback_speed,
            max_abs_rpm=max_abs_rpm,
        )
        if fit_to_rpm_limit
        else playback_speed
    )
    dt = 1.0 / (fps * effective_playback_speed)
    distance_m = 0.0
    for command in commands[start_index:]:
        left = _clamp(command.left_rpm * speed_scale * effective_playback_speed, max_abs_rpm)
        right = _clamp(command.right_rpm * speed_scale * effective_playback_speed, max_abs_rpm)
        avg_rpm = 0.5 * (left + right)
        linear_m_s = avg_rpm * (2.0 * math.pi / 60.0) * wheel_radius_m
        distance_m += abs(linear_m_s) * dt
    return distance_m


def stream_wheel_commands(
    *,
    csv_path: Path = DEFAULT_CSV,
    port: str = "/dev/ttyUSB0",
    slave_id: int = 1,
    baudrate: int = 115200,
    fps: float = 30.0,
    speed_scale: float = 1.0,
    playback_speed: float = 1.0,
    max_abs_rpm: float = 50.0,
    left_motor_sign: int = 1,
    right_motor_sign: int = -1,
    invert_turn_direction: bool = True,
    fit_to_rpm_limit: bool = False,
    reverse_playback: bool = False,
    dry_run: bool = False,
    debug: bool = False,
    start_index: int = 0,
    stop_event: Event | None = None,
    progress_callback: ProgressCallback | None = None,
    status_callback: StatusCallback | None = None,
) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if speed_scale <= 0:
        raise ValueError(f"speed_scale must be positive, got {speed_scale}")
    if playback_speed <= 0:
        raise ValueError(f"playback_speed must be positive, got {playback_speed}")
    if max_abs_rpm <= 0 or max_abs_rpm > 3000:
        raise ValueError(f"max_abs_rpm must be in (0, 3000], got {max_abs_rpm}")
    if left_motor_sign not in (-1, 1):
        raise ValueError(f"left_motor_sign must be -1 or 1, got {left_motor_sign}")
    if right_motor_sign not in (-1, 1):
        raise ValueError(f"right_motor_sign must be -1 or 1, got {right_motor_sign}")

    def status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)
        print(message)

    commands = load_wheel_commands(csv_path)
    if reverse_playback:
        commands = reverse_wheel_commands(commands)
    if start_index < 0:
        raise ValueError(f"start_index must be non-negative, got {start_index}")
    if start_index >= len(commands):
        status(f"start_index {start_index} is past the end of {len(commands)} commands")
        return
    total = len(commands)
    effective_playback_speed = (
        playback_speed_fit_to_rpm_limit(
            commands,
            speed_scale=speed_scale,
            playback_speed=playback_speed,
            max_abs_rpm=max_abs_rpm,
        )
        if fit_to_rpm_limit
        else playback_speed
    )
    total_distance_m = estimate_travel_distance_m(
        commands,
        fps=fps,
        speed_scale=speed_scale,
        playback_speed=effective_playback_speed,
        max_abs_rpm=max_abs_rpm,
    )
    sent_distance_m = estimate_travel_distance_m(
        commands,
        fps=fps,
        speed_scale=speed_scale,
        playback_speed=effective_playback_speed,
        max_abs_rpm=max_abs_rpm,
        start_index=0,
    ) - estimate_travel_distance_m(
        commands,
        fps=fps,
        speed_scale=speed_scale,
        playback_speed=effective_playback_speed,
        max_abs_rpm=max_abs_rpm,
        start_index=start_index,
    )
    bot = None
    encoder_start_positions: tuple[int, int] | None = None
    encoder_target_counts: float | None = None
    encoder_turn_started_at: float | None = None
    encoder_turn_reached = False
    encoder_drive_csv_rpm: tuple[float, float] | None = None
    encoder_last_revolutions = 0.0
    encoder_action: str | None = None

    direction_label = "reverse" if reverse_playback else "forward"
    status(f"Loaded {total} {direction_label} wheel commands from {csv_path}")
    status(
        f"Streaming from row {start_index + 1}/{total} at {fps:g} FPS, "
        f"playback speed {effective_playback_speed:g}x, RPM scale {speed_scale:g}, "
        f"clamp +/-{max_abs_rpm:g} RPM, estimated distance {total_distance_m:.3f} m"
    )
    if fit_to_rpm_limit and effective_playback_speed != playback_speed:
        status(
            f"Fit-to-limit enabled: slowed timing by {playback_speed / effective_playback_speed:.2f}x "
            f"to preserve the path under +/-{max_abs_rpm:g} RPM."
        )

    try:
        if not dry_run:
            from tara_sdk import TaraBase

            bot = TaraBase(port=port, slave_id=slave_id, baudrate=baudrate, debug=debug)
            bot.connect()
            if not bot.connected:
                raise RuntimeError(f"Could not connect to TaraBase on {port}")
            bot.clear_fault()
            bot.enable_motors()

        start_time = time.monotonic()
        for index in range(start_index, total):
            command = commands[index]
            if stop_event is not None and stop_event.is_set():
                status("Stop requested; stopping stream.")
                break

            csv_left = command.left_rpm
            csv_right = command.right_rpm
            encoder_controlled = (
                not reverse_playback
                and command.action in {"turn_left_90", "turn_right_90"}
                and command.encoder_target_wheel_revolutions is not None
            )
            if encoder_controlled and not dry_run:
                assert bot is not None
                if encoder_action != command.action:
                    encoder_start_positions = None
                    encoder_target_counts = None
                    encoder_turn_started_at = None
                    encoder_turn_reached = False
                    encoder_drive_csv_rpm = None
                    encoder_last_revolutions = 0.0
                    encoder_action = command.action
                encoder_action_label = command.action.replace("turn_", "").replace("_", " ")
                if encoder_start_positions is None:
                    positions = bot.get_motor_positions()
                    if positions is None:
                        raise RuntimeError("Could not read TaraBase encoder positions")
                    encoder_start_positions = (
                        int(positions["left_motor_position"]),
                        int(positions["right_motor_position"]),
                    )
                    encoder_target_counts = (
                        command.encoder_target_wheel_revolutions
                        * TARA_ENCODER_COUNTS_PER_WHEEL_REVOLUTION
                    )
                    encoder_turn_started_at = time.monotonic()
                    status(
                        f"{encoder_action_label.title()} encoder control active: stopping at "
                        f"{command.encoder_target_wheel_revolutions:.3f} wheel rev "
                        f"({encoder_target_counts:.0f} counts average)."
                    )
                else:
                    positions = bot.get_motor_positions()
                    if positions is None:
                        raise RuntimeError(
                            f"Lost TaraBase encoder feedback during {encoder_action_label} turn"
                        )
                    left_counts = abs(
                        _signed_32_bit_delta(
                            int(positions["left_motor_position"]), encoder_start_positions[0]
                        )
                    )
                    right_counts = abs(
                        _signed_32_bit_delta(
                            int(positions["right_motor_position"]), encoder_start_positions[1]
                        )
                    )
                    average_counts = 0.5 * (left_counts + right_counts)
                    encoder_last_revolutions = (
                        average_counts / TARA_ENCODER_COUNTS_PER_WHEEL_REVOLUTION
                    )
                    assert encoder_target_counts is not None
                    if average_counts >= encoder_target_counts:
                        encoder_turn_reached = True
                    assert encoder_turn_started_at is not None
                    if (
                        not encoder_turn_reached
                        and time.monotonic() - encoder_turn_started_at > ENCODER_TURN_TIMEOUT_S
                    ):
                        raise RuntimeError(
                            f"{encoder_action_label.title()} encoder target was not reached within "
                            f"{ENCODER_TURN_TIMEOUT_S:.1f}s (reached "
                            f"{encoder_last_revolutions:.3f}/"
                            f"{command.encoder_target_wheel_revolutions:.3f} rev)"
                        )

                if not encoder_turn_reached and (csv_left != 0.0 or csv_right != 0.0):
                    encoder_drive_csv_rpm = (csv_left, csv_right)
                if encoder_turn_reached:
                    csv_left = 0.0
                    csv_right = 0.0
                elif encoder_drive_csv_rpm is not None:
                    # If the timeline's old 2.90 s command expires first, keep
                    # turning until encoder travel reaches the requested value.
                    csv_left, csv_right = encoder_drive_csv_rpm
            scaled_left = _clamp(csv_left * speed_scale * effective_playback_speed, max_abs_rpm)
            scaled_right = _clamp(csv_right * speed_scale * effective_playback_speed, max_abs_rpm)
            left = scaled_left
            right = scaled_right
            if invert_turn_direction:
                forward = 0.5 * (left + right)
                turn = 0.5 * (right - left)
                left = forward + turn
                right = forward - turn
            sent_left = left * left_motor_sign
            sent_right = right * right_motor_sign
            dt = 1.0 / (fps * effective_playback_speed)
            avg_rpm = 0.5 * (left + right)
            linear_m_s = avg_rpm * (2.0 * math.pi / 60.0) * DEFAULT_WHEEL_RADIUS_M
            sent_distance_m += abs(linear_m_s) * dt

            if dry_run:
                print(
                    f"frame={command.frame} "
                    f"csv_left={csv_left:.2f} csv_right={csv_right:.2f} "
                    f"scaled_left={scaled_left:.2f} scaled_right={scaled_right:.2f} "
                    f"path_left={left:.2f} path_right={right:.2f} "
                    f"sdk_left={sent_left:.2f} sdk_right={sent_right:.2f} "
                    f"distance={sent_distance_m:.3f}/{total_distance_m:.3f} m"
                )
            else:
                assert bot is not None
                if not bot.set_velocity(int(round(sent_left)), int(round(sent_right))):
                    raise RuntimeError(f"set_velocity failed at frame {command.frame}")

            if progress_callback is not None:
                progress_callback(
                    index + 1,
                    total,
                    csv_left,
                    csv_right,
                    scaled_left,
                    scaled_right,
                    left,
                    right,
                    sent_left,
                    sent_right,
                    sent_distance_m,
                    total_distance_m,
                )

            target_time = start_time + ((index - start_index + 1) * dt)
            sleep_seconds = target_time - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        # A 90-degree prompt may contain no hold rows after its old elapsed-time
        # endpoint. Continue the same turn command until encoder travel—not CSV
        # duration—reaches 0.475 wheel revolutions.
        while (
            bot is not None
            and encoder_start_positions is not None
            and encoder_target_counts is not None
            and encoder_drive_csv_rpm is not None
            and not encoder_turn_reached
        ):
            if stop_event is not None and stop_event.is_set():
                status("Stop requested during encoder-controlled 90-degree turn.")
                break
            positions = bot.get_motor_positions()
            if positions is None:
                raise RuntimeError("Lost TaraBase encoder feedback during 90-degree turn")
            left_counts = abs(
                _signed_32_bit_delta(
                    int(positions["left_motor_position"]), encoder_start_positions[0]
                )
            )
            right_counts = abs(
                _signed_32_bit_delta(
                    int(positions["right_motor_position"]), encoder_start_positions[1]
                )
            )
            average_counts = 0.5 * (left_counts + right_counts)
            encoder_last_revolutions = average_counts / TARA_ENCODER_COUNTS_PER_WHEEL_REVOLUTION
            if average_counts >= encoder_target_counts:
                encoder_turn_reached = True
                break
            assert encoder_turn_started_at is not None
            if time.monotonic() - encoder_turn_started_at > ENCODER_TURN_TIMEOUT_S:
                raise RuntimeError(
                    "90-degree encoder target was not reached within "
                    f"{ENCODER_TURN_TIMEOUT_S:.1f}s (reached {encoder_last_revolutions:.3f}/0.475 rev)"
                )
            time.sleep(0.02)

        if bot is not None and encoder_turn_reached:
            bot.set_velocity(0, 0)
            status(
                f"{(encoder_action or '90-degree turn').replace('turn_', '').replace('_', ' ').title()} "
                "encoder target reached at "
                f"{encoder_last_revolutions:.3f} wheel rev; motors stopped."
            )

    finally:
        if bot is not None:
            try:
                bot.stop_motors()
            finally:
                bot.disconnect()
        status("TaraBase stream finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a diff-drive CSV to TaraBase wheel motors.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV with left_motor_rpm/right_motor_rpm columns.")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="USB serial port. Default: /dev/ttyUSB0.")
    parser.add_argument("--slave-id", type=int, default=1, help="MODBUS slave id. Default: 1.")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate. Default: 115200.")
    parser.add_argument("--fps", type=float, default=30.0, help="CSV frame rate. Default: 30.")
    parser.add_argument("--speed-scale", type=float, default=1.0, help="Multiplier applied to CSV RPM values.")
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Motion speed multiplier. Scales RPM and frame timing together. Default: 1.",
    )
    parser.add_argument("--max-abs-rpm", type=float, default=50.0, help="Safety clamp. Default: +/-50 RPM.")
    parser.add_argument("--left-motor-sign", type=int, choices=(-1, 1), default=1, help="Hardware sign for left motor. Default: 1.")
    parser.add_argument("--right-motor-sign", type=int, choices=(-1, 1), default=-1, help="Hardware sign for right motor. Default: -1.")
    parser.add_argument(
        "--no-invert-turn-direction",
        action="store_true",
        help="Disable real-robot turn-direction inversion.",
    )
    parser.add_argument(
        "--fit-to-rpm-limit",
        action="store_true",
        help="Slow playback so the full CSV fits under --max-abs-rpm without flattening turns.",
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Play the CSV backward by reversing row order and negating wheel RPMs.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based CSV row to start from. Default: 0.")
    parser.add_argument("--debug", action="store_true", help="Enable minimalmodbus debug logs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without connecting to hardware.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stream_wheel_commands(
        csv_path=args.csv,
        port=args.port,
        slave_id=args.slave_id,
        baudrate=args.baudrate,
        fps=args.fps,
        speed_scale=args.speed_scale,
        playback_speed=args.playback_speed,
        max_abs_rpm=args.max_abs_rpm,
        left_motor_sign=args.left_motor_sign,
        right_motor_sign=args.right_motor_sign,
        invert_turn_direction=not args.no_invert_turn_direction,
        fit_to_rpm_limit=args.fit_to_rpm_limit,
        reverse_playback=args.reverse,
        dry_run=args.dry_run,
        debug=args.debug,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
