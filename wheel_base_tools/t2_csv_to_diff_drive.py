#!/usr/bin/env python3
"""Convert a retargeted T2 CSV root trajectory to differential-drive commands."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT_TRANSLATE_COLUMNS = ("root_translateX", "root_translateY", "root_translateZ")
ROOT_ROTATE_COLUMNS = ("root_rotateX", "root_rotateY", "root_rotateZ")

# Raw T2 CSV is MuJoCo-style z-up, x/y ground. The combined viewer shows T2
# after Kimodo's T2 frame conversion plus the 180-degree yaw correction used to
# compare the humanoid and wheel-base trajectories. This direct mapping is:
# raw (x, y, z) meters -> scene (x, y-up, z) = (-x, z, y).
RAW_T2_TO_VIEWER_SCENE = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
CSV_TO_KIMODO_YAW = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
T2_VIEWER_YAW_CORRECTION = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()
T2_URDF_TO_SCENE_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
T2_SCENE_YAW_CORRECTION = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()
T2_DEFAULT_ROT = T2_SCENE_YAW_CORRECTION @ T2_URDF_TO_SCENE_ROT


def _column_indices(header: list[str], columns: tuple[str, ...]) -> list[int]:
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"Missing required CSV column(s): {', '.join(missing)}")
    return [header.index(column) for column in columns]


def _read_t2_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[0] < 2:
        raise ValueError("Need at least two frames to compute velocity")
    return header, data


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-8, None)


def _smooth_angles(angles: np.ndarray, window_size: int = 9) -> np.ndarray:
    if window_size <= 1 or angles.shape[0] < 3:
        return angles
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, angles.shape[0] if angles.shape[0] % 2 == 1 else angles.shape[0] - 1)
    if window_size <= 1:
        return angles
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    pad = window_size // 2
    sin_values = np.pad(np.sin(angles), (pad, pad), mode="edge")
    cos_values = np.pad(np.cos(angles), (pad, pad), mode="edge")
    smooth_sin = np.convolve(sin_values, kernel, mode="valid")
    smooth_cos = np.convolve(cos_values, kernel, mode="valid")
    return np.unwrap(np.arctan2(smooth_sin, smooth_cos))


def _smooth_positions(positions: np.ndarray, window_size: int = 31) -> np.ndarray:
    if window_size <= 1 or positions.shape[0] < 3:
        return positions
    if window_size % 2 == 0:
        window_size += 1
    window_size = min(window_size, positions.shape[0] if positions.shape[0] % 2 == 1 else positions.shape[0] - 1)
    if window_size <= 1:
        return positions
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    pad = window_size // 2
    return np.stack(
        [
            np.convolve(np.pad(positions[:, axis], (pad, pad), mode="edge"), kernel, mode="valid")
            for axis in range(positions.shape[1])
        ],
        axis=1,
    )


def _path_headings(ground_pos_m: np.ndarray) -> np.ndarray:
    ground_pos_m = _smooth_positions(ground_pos_m, window_size=241)
    n = ground_pos_m.shape[0]
    vectors = np.zeros((n, 2), dtype=np.float64)
    if n == 2:
        vectors[:] = ground_pos_m[1] - ground_pos_m[0]
    else:
        vectors[0] = ground_pos_m[1] - ground_pos_m[0]
        vectors[-1] = ground_pos_m[-1] - ground_pos_m[-2]
        vectors[1:-1] = ground_pos_m[2:] - ground_pos_m[:-2]

    headings = np.zeros(n, dtype=np.float64)
    valid_heading = 0.0
    for idx, vector in enumerate(vectors):
        if float(np.linalg.norm(vector)) > 1e-5:
            valid_heading = math.atan2(float(vector[1]), float(vector[0]))
        headings[idx] = valid_heading
    return _smooth_angles(np.unwrap(headings), window_size=61)


def _t2_viewer_root_yaw(root_rot_raw: Rotation) -> np.ndarray:
    root_rot_mujoco = root_rot_raw.as_matrix()
    root_rot_scene = (
        T2_VIEWER_YAW_CORRECTION[None, ...]
        @ CSV_TO_KIMODO_YAW[None, ...]
        @ MUJOCO_TO_KIMODO[None, ...]
        @ root_rot_mujoco
        @ MUJOCO_TO_KIMODO.T[None, ...]
    )
    t2_visual_rot = root_rot_scene @ T2_DEFAULT_ROT[None, ...]
    forward_ground = t2_visual_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return _smooth_angles(np.unwrap(np.arctan2(forward_ground[:, 2], forward_ground[:, 0])), window_size=15)


def _wheel_base_yaw(
    path_yaw_rad: np.ndarray,
    t2_yaw_rad: np.ndarray,
    speed_m_s: np.ndarray,
    total_distance_m: float,
) -> np.ndarray:
    """Use path-facing yaw normally, while preserving intentional T2 turns."""
    if total_distance_m < 0.5:
        return t2_yaw_rad.copy()

    # Backward-walk clips translate opposite the humanoid's facing direction.
    # Keep the wheel base front/logo aligned with T2's front instead of pointing
    # the wheel base along the reverse travel vector.
    mean_facing_alignment = float(np.mean(np.cos(path_yaw_rad - t2_yaw_rad)))
    if mean_facing_alignment < -0.8:
        return t2_yaw_rad.copy()

    path_turn = float(path_yaw_rad[-1] - path_yaw_rad[0])
    t2_turn = float(t2_yaw_rad[-1] - t2_yaw_rad[0])
    min_t2_turn = math.radians(100.0)

    follows_t2_turn = abs(t2_turn) >= min_t2_turn and (
        abs(path_turn) < 0.8 * abs(t2_turn) or math.copysign(1.0, path_turn) != math.copysign(1.0, t2_turn)
    )
    if follows_t2_turn:
        yaw_rad = t2_yaw_rad.copy()
    else:
        yaw_rad = path_yaw_rad.copy()

    yaw_rad = yaw_rad.copy()
    if not follows_t2_turn:
        t2_step_yaw = np.abs(np.diff(t2_yaw_rad, prepend=t2_yaw_rad[0]))
        for frame_idx in range(1, yaw_rad.shape[0]):
            if speed_m_s[frame_idx] < 0.10 and t2_step_yaw[frame_idx] < math.radians(0.60):
                yaw_rad[frame_idx] = yaw_rad[frame_idx - 1]
    return yaw_rad


def convert_t2_csv_to_diff_drive(
    input_csv: Path,
    output_csv: Path,
    fps: float,
    wheel_radius_m: float,
    wheel_separation_m: float,
    max_forward_speed: float | None,
    max_yaw_rate: float | None,
    rpm_output_csv: Path | None = None,
    max_motor_rpm: int | None = 3000,
    yaw_source: str = "auto",
) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if wheel_radius_m <= 0:
        raise ValueError(f"wheel radius must be positive, got {wheel_radius_m}")
    if wheel_separation_m <= 0:
        raise ValueError(f"wheel separation must be positive, got {wheel_separation_m}")
    if max_motor_rpm is not None and max_motor_rpm <= 0:
        raise ValueError(f"max motor RPM must be positive, got {max_motor_rpm}")
    if yaw_source not in {"auto", "path", "t2"}:
        raise ValueError("yaw_source must be one of: auto, path, t2")

    header, data = _read_t2_csv(input_csv)
    translate_indices = _column_indices(header, ROOT_TRANSLATE_COLUMNS)
    rotate_indices = _column_indices(header, ROOT_ROTATE_COLUMNS)

    dt = 1.0 / fps

    # Retargeted T2 CSV root translation is centimeters; convert to meters.
    root_pos_m = data[:, translate_indices] * 0.01
    scene_root_pos_m = (RAW_T2_TO_VIEWER_SCENE @ root_pos_m[..., None]).squeeze(-1)

    # Retargeted T2 CSV root rotation is XYZ Euler in degrees.
    root_rot = Rotation.from_euler("xyz", data[:, rotate_indices], degrees=True)

    # Use the same ground-plane trajectory that the T2 viewer displays.
    ground_pos = scene_root_pos_m[:, [0, 2]]
    ground_delta = ground_pos[1:] - ground_pos[:-1]
    step_distance_m = np.linalg.norm(ground_delta, axis=1)
    distance_from_start_m = np.concatenate([[0.0], np.cumsum(step_distance_m)])

    path_speed_m_s = step_distance_m / dt
    frame_speed_m_s = np.concatenate([path_speed_m_s, path_speed_m_s[-1:]], axis=0)

    # A wheel base should face along its ground trajectory. The humanoid root
    # yaw includes torso/gait pose changes, so using it makes straight walking
    # look like steering. For large turning motions, match the displayed T2
    # orientation so the wheel base turns on the same side and ends aligned.
    t2_yaw_rad = _t2_viewer_root_yaw(root_rot)
    path_yaw_rad = _path_headings(ground_pos)
    if yaw_source == "path":
        yaw_rad = path_yaw_rad
    elif yaw_source == "t2":
        yaw_rad = t2_yaw_rad
    else:
        yaw_rad = _wheel_base_yaw(
            path_yaw_rad,
            t2_yaw_rad,
            frame_speed_m_s,
            float(distance_from_start_m[-1]),
        )

    heading = np.stack([np.cos(yaw_rad[:-1]), np.sin(yaw_rad[:-1])], axis=1)
    forward_velocity_m_s = np.sum(ground_delta * heading, axis=1) / dt
    yaw_rate_rad_s = np.diff(yaw_rad) / dt

    if max_forward_speed is not None:
        if max_forward_speed <= 0:
            raise ValueError("--max-forward-speed must be positive")
        forward_velocity_m_s = np.clip(forward_velocity_m_s, -max_forward_speed, max_forward_speed)

    if max_yaw_rate is not None:
        if max_yaw_rate <= 0:
            raise ValueError("--max-yaw-rate must be positive")
        yaw_rate_rad_s = np.clip(yaw_rate_rad_s, -max_yaw_rate, max_yaw_rate)

    left_linear_m_s = forward_velocity_m_s - yaw_rate_rad_s * wheel_separation_m / 2.0
    right_linear_m_s = forward_velocity_m_s + yaw_rate_rad_s * wheel_separation_m / 2.0

    left_rad_s = left_linear_m_s / wheel_radius_m
    right_rad_s = right_linear_m_s / wheel_radius_m

    rad_s_to_rpm = 60.0 / (2.0 * math.pi)
    left_rpm = left_rad_s * rad_s_to_rpm
    right_rpm = right_rad_s * rad_s_to_rpm

    # Commands are interval values for frame i -> i+1. Duplicate the last value
    # so the output has one row per input frame.
    def pad_last(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, values[-1:]], axis=0)

    forward_velocity_m_s = pad_last(forward_velocity_m_s)
    yaw_rate_rad_s = pad_last(yaw_rate_rad_s)
    left_linear_m_s = pad_last(left_linear_m_s)
    right_linear_m_s = pad_last(right_linear_m_s)
    left_rad_s = pad_last(left_rad_s)
    right_rad_s = pad_last(right_rad_s)
    left_rpm = pad_last(left_rpm)
    right_rpm = pad_last(right_rpm)
    step_distance_m = np.concatenate([[0.0], step_distance_m])
    time_s = np.arange(root_pos_m.shape[0], dtype=np.float64) * dt

    left_motor_rpm = np.rint(left_rpm).astype(np.int64)
    right_motor_rpm = np.rint(right_rpm).astype(np.int64)
    if max_motor_rpm is not None:
        left_motor_rpm = np.clip(left_motor_rpm, -max_motor_rpm, max_motor_rpm)
        right_motor_rpm = np.clip(right_motor_rpm, -max_motor_rpm, max_motor_rpm)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Frame",
                "root_x_m",
                "root_y_m",
                "root_z_m",
                "root_yaw_rad",
                "root_yaw_deg",
                "step_ground_distance_m",
                "distance_from_start_m",
                "left_motor_rpm",
                "right_motor_rpm",
            ]
        )
        for frame_idx in range(root_pos_m.shape[0]):
            writer.writerow(
                [
                    int(data[frame_idx, 0]) if "Frame" in header else frame_idx,
                    scene_root_pos_m[frame_idx, 0],
                    scene_root_pos_m[frame_idx, 2],
                    scene_root_pos_m[frame_idx, 1],
                    yaw_rad[frame_idx],
                    math.degrees(yaw_rad[frame_idx]),
                    step_distance_m[frame_idx],
                    distance_from_start_m[frame_idx],
                    int(left_motor_rpm[frame_idx]),
                    int(right_motor_rpm[frame_idx]),
                ]
            )

    if rpm_output_csv is not None:
        rpm_output_csv.parent.mkdir(parents=True, exist_ok=True)
        with rpm_output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame", "time_s", "left_motor_rpm", "right_motor_rpm"])
            for frame_idx in range(root_pos_m.shape[0]):
                writer.writerow(
                    [
                        int(data[frame_idx, 0]) if "Frame" in header else frame_idx,
                        time_s[frame_idx],
                        int(left_motor_rpm[frame_idx]),
                        int(right_motor_rpm[frame_idx]),
                    ]
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert retargeted T2 CSV root motion to differential-drive wheel commands."
    )
    parser.add_argument("input_csv", type=Path, help="Retargeted T2 CSV with root_translate/root_rotate columns.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output command CSV. Defaults to '<input_stem>_diff_drive.csv' beside the input.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="CSV frame rate in Hz. Default: 30.")
    parser.add_argument(
        "--wheel-diameter",
        type=float,
        default=0.20,
        help="Wheel diameter in meters. Default: 0.20 for 200 mm wheels.",
    )
    parser.add_argument(
        "--wheel-separation",
        type=float,
        default=0.38,
        help="Distance between left and right wheels in meters. Default: 0.38.",
    )
    parser.add_argument(
        "--max-forward-speed",
        type=float,
        default=None,
        help="Optional absolute clamp for forward velocity in m/s.",
    )
    parser.add_argument(
        "--max-yaw-rate",
        type=float,
        default=None,
        help="Optional absolute clamp for yaw rate in rad/s.",
    )
    parser.add_argument(
        "--yaw-source",
        choices=("auto", "path", "t2"),
        default="auto",
        help="Choose wheel-base yaw from path direction, T2 body yaw, or auto heuristic. Default: auto.",
    )
    parser.add_argument(
        "--rpm-output",
        type=Path,
        default=None,
        help="Optional hardware-ready CSV with only Frame, time_s, left_motor_rpm, right_motor_rpm.",
    )
    parser.add_argument(
        "--max-motor-rpm",
        type=int,
        default=3000,
        help="Clamp rounded motor RPM commands to this absolute value. Default: 3000.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_csv = args.output
    if output_csv is None:
        output_csv = input_csv.with_name(f"{input_csv.stem}_diff_drive.csv")
    else:
        output_csv = output_csv.expanduser().resolve()

    convert_t2_csv_to_diff_drive(
        input_csv=input_csv,
        output_csv=output_csv,
        fps=args.fps,
        wheel_radius_m=args.wheel_diameter / 2.0,
        wheel_separation_m=args.wheel_separation,
        max_forward_speed=args.max_forward_speed,
        max_yaw_rate=args.max_yaw_rate,
        rpm_output_csv=args.rpm_output.expanduser().resolve() if args.rpm_output is not None else None,
        max_motor_rpm=args.max_motor_rpm,
        yaw_source=args.yaw_source,
    )
    print(output_csv)


if __name__ == "__main__":
    main()