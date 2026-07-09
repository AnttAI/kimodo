"""Generate frame-by-frame TaraBase CSVs for dedicated movement prompts."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Sequence


FORWARD_PROMPT = "move forward"
BACKWARD_PROMPT = "move backward"
DEFAULT_LINEAR_SPEED_M_S = 0.20
DEFAULT_WHEEL_RADIUS_M = 0.10
DEFAULT_WHEEL_SEPARATION_M = 0.38
DEFAULT_TURN_WHEEL_RPM = 10.0
DEFAULT_TURN_SECONDS_PER_90_DEG = 2.90
NINETY_DEG_ENCODER_TARGET_WHEEL_REVOLUTIONS = 0.475
# Backward-compatible name for code that imported the first prototype value.
LEFT_90_ENCODER_TARGET_WHEEL_REVOLUTIONS = NINETY_DEG_ENCODER_TARGET_WHEEL_REVOLUTIONS
# Match known-good Tara CSVs: positive CSV RPM is hardware forward and
# negative CSV RPM is hardware backward. The standalone sender applies the
# final per-motor wiring signs. Visual wheel direction is stored separately.
TARA_HARDWARE_DIRECTION_SIGN = 1
# Raw wheel CSV headings are corrected by 180 degrees in robot_app's viewer.
HUMAN_FORWARD_RAW_YAW_RAD = -math.pi / 2.0


def base_prompt_action(prompt: str) -> str | None:
    """Parse an exact dedicated translation or 90/180-degree turn prompt."""
    normalized = " ".join(prompt.strip().lower().split())
    if normalized == FORWARD_PROMPT:
        return "forward"
    if normalized in {BACKWARD_PROMPT, "move backwards"}:
        return "backward"
    turn_match = re.fullmatch(r"turn (left|right) by (90|180) (?:degrees?|degress)", normalized)
    if turn_match is not None:
        # Keep the calibrated TaraBase motion exactly the same, but swap the
        # user-facing turn names so the prompt direction matches the real base.
        turn_side = "right" if turn_match.group(1) == "left" else "left"
        return f"turn_{turn_side}_{turn_match.group(2)}"
    return None


def base_prompt_direction(prompt: str) -> int | None:
    """Backward-compatible direction helper for translation prompts."""
    action = base_prompt_action(prompt)
    return 1 if action == "forward" else -1 if action == "backward" else None


def write_prompt_base_csv(
    output_csv: Path,
    segments: Sequence[tuple[str, int]],
    *,
    fps: float,
    initial_x_m: float = 0.0,
    initial_z_m: float = 0.0,
    initial_yaw_rad: float = HUMAN_FORWARD_RAW_YAW_RAD,
    speed_m_s: float = DEFAULT_LINEAR_SPEED_M_S,
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
    wheel_separation_m: float = DEFAULT_WHEEL_SEPARATION_M,
    turn_wheel_rpm: float = DEFAULT_TURN_WHEEL_RPM,
    turn_seconds_per_90_deg: float = DEFAULT_TURN_SECONDS_PER_90_DEG,
) -> Path:
    """Write base commands with a zero-based timeline and an optional world start pose."""
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if speed_m_s <= 0.0:
        raise ValueError("speed_m_s must be positive")
    if wheel_radius_m <= 0.0:
        raise ValueError("wheel_radius_m must be positive")
    if wheel_separation_m <= 0.0:
        raise ValueError("wheel_separation_m must be positive")
    if turn_wheel_rpm <= 0.0:
        raise ValueError("turn_wheel_rpm must be positive")
    if turn_seconds_per_90_deg <= 0.0:
        raise ValueError("turn_seconds_per_90_deg must be positive")
    valid_actions = {
        "forward",
        "backward",
        "turn_left_90",
        "turn_left_180",
        "turn_right_90",
        "turn_right_180",
    }
    if not segments or any(action not in valid_actions or frames <= 0 for action, frames in segments):
        raise ValueError("segments contain an invalid action or frame count")

    rpm_per_m_s = 60.0 / (2.0 * math.pi * wheel_radius_m)
    rad_s_per_m_s = 1.0 / wheel_radius_m
    if not all(math.isfinite(value) for value in (initial_x_m, initial_z_m, initial_yaw_rad)):
        raise ValueError("initial base pose must contain finite values")
    x_m = float(initial_x_m)
    z_m = float(initial_z_m)
    distance_m = 0.0
    raw_yaw_rad = float(initial_yaw_rad)
    output_csv = output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "Frame",
                "time_s",
                "root_x_m",
                "root_y_m",
                "root_z_m",
                "root_yaw_rad",
                "root_yaw_deg",
                "step_ground_distance_m",
                "distance_from_start_m",
                "forward_velocity_m_s",
                "yaw_rate_rad_s",
                "left_wheel_rad_s",
                "right_wheel_rad_s",
                "left_motor_rpm",
                "right_motor_rpm",
                "base_action",
                "encoder_target_wheel_revolutions",
            ]
        )
        frame_idx = 0
        for action, frames in segments:
            is_turn = action.startswith("turn_")
            direction = 1 if action == "forward" else -1 if action == "backward" else 0
            hardware_turn_sign = -1.0 if "left" in action else 1.0
            turn_degrees = 180.0 if action.endswith("180") else 90.0
            # The Tara's established motor/wiring mapping makes the physical
            # 90-degree direction opposite to the logical prompt direction.
            # Mirror only the Viser motion so it matches the real base; keep
            # hardware RPM polarity exactly as calibrated.
            visual_turn_sign = (
                -hardware_turn_sign if is_turn and turn_degrees == 90.0 else hardware_turn_sign
            )
            turn_angle_rad = math.radians(turn_degrees)
            turn_duration_s = turn_seconds_per_90_deg * (turn_degrees / 90.0)
            turn_command_frames = max(1, int(round(turn_duration_s * fps)))
            if is_turn and frames < turn_command_frames:
                raise ValueError(
                    f"{action.replace('_', ' ')} needs at least {turn_duration_s:.2f}s"
                )
            visual_yaw_rate_rad_s = (
                visual_turn_sign * turn_angle_rad / turn_duration_s if is_turn else 0.0
            )
            hardware_yaw_rate_rad_s = (
                hardware_turn_sign * turn_angle_rad / turn_duration_s if is_turn else 0.0
            )
            turn_start_yaw_rad = raw_yaw_rad
            for segment_frame in range(frames):
                final_turn_frame = is_turn and segment_frame >= turn_command_frames
                if is_turn:
                    visual_turn_denominator = max(1, turn_command_frames - 1)
                    visual_progress = min(segment_frame / visual_turn_denominator, 1.0)
                    raw_yaw_rad = (
                        turn_start_yaw_rad
                        + visual_turn_sign * turn_angle_rad * visual_progress
                    )
                velocity_m_s = direction * speed_m_s
                step_m = velocity_m_s / fps
                active_visual_yaw_rate = (
                    0.0 if final_turn_frame else visual_yaw_rate_rad_s
                )
                active_hardware_yaw_rate = (
                    0.0 if final_turn_frame else hardware_yaw_rate_rad_s
                )
                if is_turn:
                    visual_left_linear_m_s = (
                        active_visual_yaw_rate * wheel_separation_m / 2.0
                    )
                    visual_right_linear_m_s = -visual_left_linear_m_s
                    hardware_left_linear_m_s = (
                        active_hardware_yaw_rate * wheel_separation_m / 2.0
                    )
                    hardware_right_linear_m_s = -hardware_left_linear_m_s
                else:
                    visual_left_linear_m_s = velocity_m_s
                    visual_right_linear_m_s = velocity_m_s
                    hardware_left_linear_m_s = velocity_m_s
                    hardware_right_linear_m_s = velocity_m_s
                left_wheel_rad_s = visual_left_linear_m_s * rad_s_per_m_s
                right_wheel_rad_s = visual_right_linear_m_s * rad_s_per_m_s
                left_motor_rpm = int(
                    round(TARA_HARDWARE_DIRECTION_SIGN * hardware_left_linear_m_s * rpm_per_m_s)
                )
                right_motor_rpm = int(
                    round(TARA_HARDWARE_DIRECTION_SIGN * hardware_right_linear_m_s * rpm_per_m_s)
                )
                writer.writerow(
                    [
                        frame_idx,
                        frame_idx / fps,
                        x_m,
                        z_m,
                        0.0,
                        raw_yaw_rad,
                        math.degrees(raw_yaw_rad),
                        0.0 if frame_idx == 0 or is_turn else abs(step_m),
                        distance_m,
                        velocity_m_s,
                        active_visual_yaw_rate,
                        left_wheel_rad_s,
                        right_wheel_rad_s,
                        left_motor_rpm,
                        right_motor_rpm,
                        action,
                        (
                            NINETY_DEG_ENCODER_TARGET_WHEEL_REVOLUTIONS
                            if action in {"turn_left_90", "turn_right_90"}
                            else ""
                        ),
                    ]
                )
                if not is_turn:
                    display_yaw_rad = raw_yaw_rad + math.pi
                    x_m += step_m * math.cos(display_yaw_rad)
                    z_m += step_m * math.sin(display_yaw_rad)
                    distance_m += abs(step_m)
                frame_idx += 1
    return output_csv


def move_forward(
    output_csv: Path,
    num_frames: int,
    *,
    fps: float,
    speed_m_s: float = DEFAULT_LINEAR_SPEED_M_S,
) -> Path:
    """Generate a forward TaraBase command CSV for ``num_frames`` frames."""
    return write_prompt_base_csv(output_csv, [("forward", num_frames)], fps=fps, speed_m_s=speed_m_s)


def move_backward(
    output_csv: Path,
    num_frames: int,
    *,
    fps: float,
    speed_m_s: float = DEFAULT_LINEAR_SPEED_M_S,
) -> Path:
    """Generate a backward TaraBase command CSV for ``num_frames`` frames."""
    return write_prompt_base_csv(output_csv, [("backward", num_frames)], fps=fps, speed_m_s=speed_m_s)
