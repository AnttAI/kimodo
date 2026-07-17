# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-focused Kimodo demo.

This app keeps the regular Kimodo authoring UI intact and adds a separate
SOMA -> T2 -> real robot workflow panel.
"""

from __future__ import annotations

import argparse
import base64
import csv
import errno
import hashlib
import http.server
import json
import math
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import viser
import viser.transforms as tf
from plyfile import PlyData

from kimodo.demo.app import Demo
from kimodo.demo.config import DEFAULT_CUR_DURATION, DEFAULT_MODEL, NB_TRANSITION_FRAMES
from kimodo.demo.warehouse_ui.rack_motion_planner import (
    cardinal_rack_route_required_seconds,
    cardinal_route_primitives,
    choose_safer_turn_side,
    plan_cardinal_rack_route,
    plan_cardinal_return_route,
    rack_walk_model_prompt,
    rack_return_model_prompt,
    rack_width_side_approach_pose,
    requested_rack_pick,
    rack_shelf_object_position,
    requested_base_rack_name,
    requested_base_return_rack_name,
    requested_human_return_rack_name,
    requested_rack_name,
)
from kimodo.exports.bvh import read_bvh_frame_time_seconds, save_motion_bvh
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import DEFAULT_TEXT_ENCODER_URL, resolve_model_name
from kimodo.motion_io import load_motion_file
from kimodo.skeleton import global_rots_to_local_rots
from kimodo.tools import seed_everything
from kimodo.retarget.soma_t2 import SomaT2RetargetJob, default_soma_retargeter_root
from kimodo.retarget.soma_t3 import SomaT3RetargetJob
from kimodo.robot.t2_nero import T2NeroConnection, default_t2_stream_script
from kimodo.scripts.t2_csv_arm_publisher import ArmFrame, load_arm_frames
from kimodo.viz.tara_rig import T2ViewerMotion, load_tara_motion_csv
from wheel_base_tools.send_diff_drive_csv_to_tara import stream_wheel_commands
from wheel_base_tools.prompt_base_motion import (
    DEFAULT_LINEAR_SPEED_M_S,
    DEFAULT_TURN_SECONDS_PER_90_DEG,
    base_prompt_action,
    write_prompt_base_csv,
)
from wheel_base_tools.view_two_wheel_base_tara_send import (
    DEFAULT_WHEEL_URDF_PATH,
    WheelBasePlayback,
    _read_diff_drive_csv,
)
from wheel_base_tools.view_t3_robot import DEFAULT_T3_URDF_PATH, T3Playback


MEMORIES_ROOT = Path("/home/jony/Downloads/soma-retargeter/assets/motions")
T3_COMBINED_MOTION_PATHS = {
    "pick_item": Path("/home/jony/Downloads/soma-retargeter/assets/pick_item.csv"),
    "drop_item": Path("/home/jony/Downloads/soma-retargeter/assets/drop_item.csv"),
}
WAREHOUSE_PLANNER_ROOT = Path(
    os.environ.get("KIMODO_WAREHOUSE_PLANNER_ROOT", str(Path(__file__).with_name("warehouse_ui")))
)
WAREHOUSE_PACKING_PLANNER = WAREHOUSE_PLANNER_ROOT / "packing_planner.py"
WAREHOUSE_PLANNER_PYTHON = Path(
    os.environ.get(
        "KIMODO_WAREHOUSE_PLANNER_PYTHON",
        "/home/jony/Downloads/warehouse_fulfillment/.venv/bin/python",
    )
)
WORLD_SCENES_ROOT = Path("/home/jony/Downloads/worlds")
DEFAULT_WORLD_SCENE_PATH = WORLD_SCENES_ROOT / "scene.ply"
TEXT_ENCODER_SERVER_COMMAND = "python -m kimodo.scripts.run_text_encoder_server"
GAUSSIAN_SH_C0 = 0.28209479177387814
WORLD_SCENE_ROTATION_DEG = np.array([-90.0, 90.0, 0.0], dtype=np.float64)
WORLD_SCENE_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WORLD_SCENE_SCALE = 1.0
WORLD_SCENE_SCALE_LIMITS = (0.01, 10.0)
WORLD_SCENE_MOVE_X_LIMITS = (-5.0, 5.0)
WORLD_SCENE_HEIGHT_LIMITS = (-5.0, 5.0)
WORLD_SCENE_MOVE_Z_LIMITS = (-5.0, 15.0)
OFFICE_WORLD_SCENE_MOVE_LIMITS = (-50.0, 50.0)
OFFICE_WORLD_SCENE_HEIGHT_LIMITS = (-50.0, 50.0)
WORLD_OBJECT_PICK_MAX_DISTANCE = 0.18

# Physical shelving shown in the hand-drawn floor plan. The bold Viser grid
# sections are 0.60 x 0.60 m, and the robot/base marker is the map origin.
# Racks 1 and 2 sit along the long X=-1.80 edge. Racks 3 and 4 sit along the
# extended Z=-3.60 edge. Rack 4 stays on X=0.00; rack 3 is 10 cm toward it
# from the X=-1.20 bold grid line.
# Centers are inset by 17 cm so each rack's 34 cm face stays inside the area.
# Negative Z is behind the human.
RACK_WIDTH_M = 0.34
RACK_DEPTH_M = 0.72
RACK_HEIGHT_M = 1.65
RACK_FIRST_SHELF_HEIGHT_M = 0.10
RACK_SHELF_SPACING_M = 0.3048  # one foot
RACK_SHELF_THICKNESS_M = 0.018
RACK_POST_SIZE_M = 0.025
RACK_SHELF_COUNT = 5
RACK_PICK_OBJECT_SIZE_M = 0.06
RACK_PICK_OBJECT_COLORS = ((220, 70, 70), (70, 180, 90), (70, 120, 220))
RACK_SHELF_PRODUCTS = {
    "rack_1": ("Coconuts", "Tomatoes", "Potatoes", "Water", "Curry leaves"),
    "rack_2": ("Rice", "Red gram", "Ice cream", "Coke", "Chips packets"),
}
RACK_MAP_POSITIONS = {
    "rack_1": np.array([-1.63, 0.0, -1.20], dtype=np.float64),
    "rack_2": np.array([-1.63, 0.0, -2.40], dtype=np.float64),
    "rack_3": np.array([-1.10, 0.0, -3.43], dtype=np.float64),
    "rack_4": np.array([0.00, 0.0, -3.43], dtype=np.float64),
}
RACK_MAP_YAWS_RAD = {
    # Keep the designated local +X broad face pointing inward, toward the
    # orange work area, just like Racks 1 and 2 on the left boundary.
    "rack_3": -np.pi / 2.0,
    "rack_4": -np.pi / 2.0,
}
# Stop in front of the shelf-opening (width) face, measured outward from the
# rack edge rather than from its center.
RACK_HUMAN_APPROACH_CLEARANCE_M = 0.45
RACK_HUMAN_SLOW_WALK_SPEED_M_S = 0.60
RACK_HUMAN_FAST_WALK_SPEED_M_S = 0.90
WORK_AREA_GRID_SECTION_M = 0.60
WORK_AREA_GRID_SHAPE = (4, 6)  # 4 x 6 bold sections = 24 squares.
WORK_AREA_SIDE_SHIFT_M = 0.60  # Put the robot on the edge, one section from the corner.
WORK_AREA_BOUNDARY_COLOR = (255, 128, 0)
COUNTER_TABLE_LENGTH_M = 0.80
COUNTER_TABLE_WIDTH_M = 0.40
COUNTER_TABLE_HEIGHT_M = 1.03
COUNTER_TABLE_DISTANCE_FROM_ORIGIN_M = 0.30
COUNTER_TABLE_TOP_THICKNESS_M = 0.06
COUNTER_TABLE_LEG_SIZE_M = 0.045
WORLD_SCENE_PRESETS = {
    "scene.ply": {
        "scale": 0.75,
        "rotation_deg": np.array([-178.0, 84.0, 0.0], dtype=np.float64),
        "position": np.array([-2.39, 1.17, 3.77], dtype=np.float64),
    },
    "office_world.ply": {
        "scale": 1.0,
        "rotation_deg": np.array([0.0, 70.0, -180.0], dtype=np.float64),
        "position": np.array([16.64, -10.91, 21.29], dtype=np.float64),
    }
}
WORLD_SCENE_ALIASES = {
    "office": "office_world.ply",
    "office_world": "office_world.ply",
    "office-world": "office_world.ply",
    "home": "home.ply",
    "scene": "scene.ply",
    "default": "scene.ply",
}


def _rack_shelf_surface_heights() -> tuple[float, ...]:
    """Return the five shelf-top heights, numbered from bottom to top."""
    return tuple(
        RACK_FIRST_SHELF_HEIGHT_M + index * RACK_SHELF_SPACING_M
        for index in range(RACK_SHELF_COUNT)
    )


def _add_pick_objects_to_scene(
    client: viser.ClientHandle,
) -> tuple[
    dict[tuple[str, int, int], viser.SceneHandle],
    dict[tuple[str, int, int], np.ndarray],
]:
    """Place three selectable objects on every shelf of every physical rack."""
    handles: dict[tuple[str, int, int], viser.SceneHandle] = {}
    home_positions: dict[tuple[str, int, int], np.ndarray] = {}
    for rack_name, rack_center in RACK_MAP_POSITIONS.items():
        rack_yaw = RACK_MAP_YAWS_RAD.get(rack_name, 0.0)
        for shelf_number, shelf_height in enumerate(_rack_shelf_surface_heights(), start=1):
            for object_index in range(1, 4):
                key = (rack_name, shelf_number, object_index)
                position = np.asarray(
                    rack_shelf_object_position(
                        rack_center,
                        rack_yaw,
                        shelf_height,
                        object_index,
                        face_normal_half_extent_m=RACK_WIDTH_M / 2.0,
                        object_height_m=RACK_PICK_OBJECT_SIZE_M,
                    ),
                    dtype=np.float64,
                )
                handle = client.scene.add_box(
                    # Keep these outside the transformed rack frame: their stored
                    # positions are world coordinates and later become hand coordinates.
                    f"/physical_world/rack_pick_objects/{rack_name}/shelf_{shelf_number}/object_{object_index}",
                    dimensions=(RACK_PICK_OBJECT_SIZE_M,) * 3,
                    color=RACK_PICK_OBJECT_COLORS[object_index - 1],
                    position=position,
                )
                handles[key] = handle
                home_positions[key] = position
    return handles, home_positions


def _rotation_between_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a stable 3x3 rotation mapping source direction onto target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source /= max(float(np.linalg.norm(source)), 1e-9)
    target /= max(float(np.linalg.norm(target)), 1e-9)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis_seed = np.array([1.0, 0.0, 0.0])
        if abs(source[0]) > 0.8:
            axis_seed = np.array([0.0, 1.0, 0.0])
        axis = np.cross(source, axis_seed)
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = cross / sine
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + sine * skew + (1.0 - dot) * (skew @ skew)


def _smooth_waypoint_path(
    total_frames: int,
    waypoints: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    """Interpolate a C1-smooth Cartesian path through ordered hand waypoints."""
    if total_frames < 2 or not waypoints:
        raise ValueError("A waypoint path needs at least two frames and one waypoint")
    ordered = sorted((int(frame), np.asarray(point, dtype=np.float64)) for frame, point in waypoints)
    if ordered[0][0] != 0 or ordered[-1][0] != total_frames - 1:
        raise ValueError("Waypoint path must include the first and final frames")
    path = np.empty((total_frames, 3), dtype=np.float64)
    for (start_frame, start), (end_frame, end) in zip(ordered[:-1], ordered[1:]):
        span = max(1, end_frame - start_frame)
        for frame in range(start_frame, end_frame + 1):
            progress = (frame - start_frame) / span
            progress = progress * progress * (3.0 - 2.0 * progress)
            path[frame] = start + progress * (end - start)
    return path


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _blend_rotations(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Geodesically interpolate two 3x3 rotation matrices."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    delta = end @ start.T
    cos_angle = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return start.copy()
    axis = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= max(2.0 * np.sin(angle), 1e-9)
    return _rotation_from_axis_angle(axis, alpha * angle) @ start


def _smooth_rotation_waypoint_path(
    total_frames: int,
    waypoints: list[tuple[int, np.ndarray]],
) -> np.ndarray:
    if total_frames < 2 or not waypoints:
        raise ValueError("A rotation waypoint path needs at least two frames and one waypoint")
    ordered = sorted((int(frame), np.asarray(rotation, dtype=np.float64)) for frame, rotation in waypoints)
    if ordered[0][0] != 0 or ordered[-1][0] != total_frames - 1:
        raise ValueError("Rotation waypoint path must include the first and final frames")
    rotations = np.empty((total_frames, 3, 3), dtype=np.float64)
    for (start_frame, start), (end_frame, end) in zip(ordered[:-1], ordered[1:]):
        span = max(1, end_frame - start_frame)
        for frame in range(start_frame, end_frame + 1):
            progress = (frame - start_frame) / span
            progress = progress * progress * (3.0 - 2.0 * progress)
            rotations[frame] = _blend_rotations(start, end, progress)
    return rotations


def _enforce_right_arm_reach(
    motion,
    reach_start_frame: int,
    hold_start_frame: int,
    hand_target_world: np.ndarray,
    preserve_motion_path: bool = False,
    target_path_world: np.ndarray | None = None,
    target_hand_rotations_world: np.ndarray | None = None,
    target_palm_normals_world: np.ndarray | None = None,
    palm_normal_local: np.ndarray | None = None,
    elbow_bend_hint_world: np.ndarray | None = None,
    refresh_cache: bool = True,
    side: str = "right",
) -> float:
    """Apply smooth analytic two-bone IK and return final wrist error in meters."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    skeleton = motion.skeleton
    names = skeleton.bone_order_names
    prefix = "Left" if side == "left" else "Right"
    shoulder_idx = names.index(f"{prefix}Arm")
    elbow_idx = names.index(f"{prefix}ForeArm")
    wrist_idx = names.index(f"{prefix}Hand")
    hand_names = skeleton.left_hand_joint_names if side == "left" else skeleton.right_hand_joint_names
    hand_indices = [names.index(name) for name in hand_names]

    positions = motion.joints_pos.detach().cpu().numpy().copy()
    global_rotations = motion.joints_rot.detach().cpu().numpy().copy()
    target = np.asarray(hand_target_world, dtype=np.float64)
    wrist_start = positions[reach_start_frame, wrist_idx].copy()
    wrist_end = positions[-1, wrist_idx].copy()
    reach_span = max(1, hold_start_frame - reach_start_frame)

    for frame in range(reach_start_frame, motion.length):
        progress = min(1.0, (frame - reach_start_frame) / reach_span)
        progress = progress * progress * (3.0 - 2.0 * progress)
        if target_path_world is not None:
            frame_target = np.asarray(target_path_world[frame], dtype=np.float64)
        elif preserve_motion_path:
            frame_target = positions[frame, wrist_idx] + progress * (target - wrist_end)
        else:
            frame_target = wrist_start + progress * (target - wrist_start)
        shoulder = positions[frame, shoulder_idx]
        elbow = positions[frame, elbow_idx]
        wrist = positions[frame, wrist_idx]
        upper = elbow - shoulder
        lower = wrist - elbow
        upper_length = float(np.linalg.norm(upper))
        lower_length = float(np.linalg.norm(lower))
        shoulder_to_target = frame_target - shoulder
        distance = float(np.linalg.norm(shoulder_to_target))
        if upper_length < 1e-6 or lower_length < 1e-6 or distance < 1e-6:
            continue
        max_reach = max(1e-6, upper_length + lower_length - 1e-4)
        distance = min(distance, max_reach)
        direction = shoulder_to_target / max(float(np.linalg.norm(shoulder_to_target)), 1e-9)
        along = (upper_length**2 - lower_length**2 + distance**2) / (2.0 * distance)
        height = np.sqrt(max(upper_length**2 - along**2, 0.0))
        bend = elbow - (shoulder + np.dot(elbow - shoulder, direction) * direction)
        if elbow_bend_hint_world is not None:
            hinted_bend = np.asarray(elbow_bend_hint_world, dtype=np.float64)
            hinted_bend = hinted_bend - np.dot(hinted_bend, direction) * direction
            if np.linalg.norm(hinted_bend) >= 1e-6:
                bend = hinted_bend
        if np.linalg.norm(bend) < 1e-6:
            bend = np.cross(direction, np.array([0.0, 1.0, 0.0]))
        if np.linalg.norm(bend) < 1e-6:
            bend = np.cross(direction, np.array([1.0, 0.0, 0.0]))
        bend /= max(float(np.linalg.norm(bend)), 1e-9)
        solved_elbow = shoulder + along * direction + height * bend
        solved_wrist = shoulder + distance * direction

        upper_delta = _rotation_between_vectors(upper, solved_elbow - shoulder)
        lower_delta = _rotation_between_vectors(lower, solved_wrist - solved_elbow)
        global_rotations[frame, shoulder_idx] = upper_delta @ global_rotations[frame, shoulder_idx]
        global_rotations[frame, elbow_idx] = lower_delta @ global_rotations[frame, elbow_idx]
        for hand_idx in hand_indices:
            global_rotations[frame, hand_idx] = lower_delta @ global_rotations[frame, hand_idx]
        if target_palm_normals_world is not None and palm_normal_local is not None:
            forearm_axis = solved_wrist - solved_elbow
            forearm_axis /= max(float(np.linalg.norm(forearm_axis)), 1e-9)
            desired_palm = np.asarray(target_palm_normals_world[frame], dtype=np.float64)
            desired_palm = desired_palm - np.dot(desired_palm, forearm_axis) * forearm_axis
            current_palm = global_rotations[frame, wrist_idx] @ np.asarray(
                palm_normal_local,
                dtype=np.float64,
            )
            current_palm = current_palm - np.dot(current_palm, forearm_axis) * forearm_axis
            if np.linalg.norm(desired_palm) >= 1e-6 and np.linalg.norm(current_palm) >= 1e-6:
                desired_palm /= max(float(np.linalg.norm(desired_palm)), 1e-9)
                current_palm /= max(float(np.linalg.norm(current_palm)), 1e-9)
                roll_delta = _rotation_between_vectors(current_palm, desired_palm)
                # Apply roll to the forearm and rigid hand together. Because the
                # roll axis is the forearm direction, wrist position stays fixed
                # and the wrist remains straight relative to the forearm.
                global_rotations[frame, elbow_idx] = roll_delta @ global_rotations[frame, elbow_idx]
                for hand_idx in hand_indices:
                    global_rotations[frame, hand_idx] = roll_delta @ global_rotations[frame, hand_idx]
        if target_hand_rotations_world is not None:
            desired_wrist = np.asarray(target_hand_rotations_world[frame], dtype=np.float64)
            if np.all(np.isfinite(desired_wrist)):
                wrist_delta = desired_wrist @ global_rotations[frame, wrist_idx].T
                # Rotate the hand/gripper as one rigid assembly. This preserves the
                # neutral finger/gripper shape while the forearm + wrist base orient
                # the tool toward the object.
                for hand_idx in hand_indices:
                    global_rotations[frame, hand_idx] = wrist_delta @ global_rotations[frame, hand_idx]

    device = motion.joints_rot.device
    dtype = motion.joints_rot.dtype
    adjusted_global = torch.as_tensor(global_rotations, device=device, dtype=dtype)
    local_rotations = global_rots_to_local_rots(adjusted_global, skeleton)
    root_positions = motion.joints_pos[:, skeleton.root_idx].clone()
    solved_global, solved_positions, _ = skeleton.fk(local_rotations, root_positions)
    motion.joints_rot = solved_global
    motion.joints_pos = solved_positions
    motion.joints_local_rot = local_rotations
    if refresh_cache:
        motion.precompute_mesh_info()
    final_wrist = solved_positions[-1, wrist_idx].detach().cpu().numpy()
    return float(np.linalg.norm(final_wrist - target))


def _freeze_body_except_right_arm(
    motion,
    base_global_rotations: np.ndarray,
    base_root_position: np.ndarray,
    side: str = "right",
) -> None:
    """Keep the outbound standing pose everywhere except the right arm chain."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    skeleton = motion.skeleton
    device = motion.joints_rot.device
    dtype = motion.joints_rot.dtype
    generated_local = global_rots_to_local_rots(motion.joints_rot, skeleton)
    base_global = torch.as_tensor(base_global_rotations, device=device, dtype=dtype)
    base_local = global_rots_to_local_rots(base_global, skeleton)
    prefix = "Left" if side == "left" else "Right"
    moving_names = {f"{prefix}Shoulder", f"{prefix}Arm", f"{prefix}ForeArm"}
    for joint_index, joint_name in enumerate(skeleton.bone_order_names):
        if joint_name not in moving_names:
            generated_local[:, joint_index] = base_local[joint_index]
    root_positions = torch.as_tensor(base_root_position, device=device, dtype=dtype)[None].repeat(
        motion.length, 1
    )
    solved_global, solved_positions, _ = skeleton.fk(generated_local, root_positions)
    motion.joints_local_rot = generated_local
    motion.joints_rot = solved_global
    motion.joints_pos = solved_positions


def _freeze_body_except_arm_sides(
    motion,
    base_global_rotations: np.ndarray,
    base_root_position: np.ndarray,
    sides: set[str],
) -> None:
    """Keep the base pose everywhere except the requested arm chains."""
    if not sides <= {"left", "right"}:
        raise ValueError("sides must contain only 'left' and/or 'right'")
    skeleton = motion.skeleton
    device = motion.joints_rot.device
    dtype = motion.joints_rot.dtype
    generated_local = global_rots_to_local_rots(motion.joints_rot, skeleton)
    base_global = torch.as_tensor(base_global_rotations, device=device, dtype=dtype)
    base_local = global_rots_to_local_rots(base_global, skeleton)
    moving_names: set[str] = set()
    for side in sides:
        prefix = "Left" if side == "left" else "Right"
        moving_names.update({f"{prefix}Shoulder", f"{prefix}Arm", f"{prefix}ForeArm"})
    for joint_index, joint_name in enumerate(skeleton.bone_order_names):
        if joint_name not in moving_names:
            generated_local[:, joint_index] = base_local[joint_index]
    root_positions = torch.as_tensor(base_root_position, device=device, dtype=dtype)[None].repeat(
        motion.length, 1
    )
    solved_global, solved_positions, _ = skeleton.fk(generated_local, root_positions)
    motion.joints_local_rot = generated_local
    motion.joints_rot = solved_global
    motion.joints_pos = solved_positions


def _hold_right_arm_local_pose(motion, hold_global_rotations: np.ndarray, side: str = "right") -> None:
    """Keep the carried object pose fixed relative to the moving body."""
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")
    skeleton = motion.skeleton
    device = motion.joints_rot.device
    dtype = motion.joints_rot.dtype
    hold_global = torch.as_tensor(hold_global_rotations, device=device, dtype=dtype)
    hold_local = global_rots_to_local_rots(hold_global, skeleton)
    local_rotations = global_rots_to_local_rots(motion.joints_rot, skeleton)

    prefix = "Left" if side == "left" else "Right"
    hand_names = skeleton.left_hand_joint_names if side == "left" else skeleton.right_hand_joint_names
    hold_names = {
        f"{prefix}Shoulder",
        f"{prefix}Arm",
        f"{prefix}ForeArm",
        *hand_names,
    }
    for joint_index, joint_name in enumerate(skeleton.bone_order_names):
        if joint_name in hold_names:
            local_rotations[:, joint_index] = hold_local[joint_index]

    root_positions = motion.joints_pos[:, skeleton.root_idx].clone()
    solved_global, solved_positions, _ = skeleton.fk(local_rotations, root_positions)
    motion.joints_local_rot = local_rotations
    motion.joints_rot = solved_global
    motion.joints_pos = solved_positions
    motion.precompute_mesh_info()


def _add_rack_to_scene(
    client: viser.ClientHandle,
    name: str,
    floor_center: np.ndarray,
    yaw_rad: float = 0.0,
) -> list[viser.SceneHandle]:
    """Build one dimensionally accurate rack from simple Viser boxes."""
    root = f"/physical_world/racks/{name}"
    handles: list[viser.SceneHandle] = [
        client.scene.add_frame(
            root,
            show_axes=False,
            position=floor_center,
            wxyz=np.array(
                [np.cos(yaw_rad / 2.0), 0.0, np.sin(yaw_rad / 2.0), 0.0],
                dtype=np.float64,
            ),
        )
    ]
    half_x = (RACK_WIDTH_M - RACK_POST_SIZE_M) / 2.0
    half_z = (RACK_DEPTH_M - RACK_POST_SIZE_M) / 2.0

    for index, (x_offset, z_offset) in enumerate(
        ((-half_x, -half_z), (-half_x, half_z), (half_x, -half_z), (half_x, half_z)),
        start=1,
    ):
        handles.append(
            client.scene.add_box(
                f"{root}/post_{index}",
                dimensions=(RACK_POST_SIZE_M, RACK_HEIGHT_M, RACK_POST_SIZE_M),
                color=(70, 78, 86),
                position=np.array([x_offset, RACK_HEIGHT_M / 2.0, z_offset], dtype=np.float64),
            )
        )

    for index, surface_height in enumerate(_rack_shelf_surface_heights(), start=1):
        handles.append(
            client.scene.add_box(
                f"{root}/shelf_{index}",
                dimensions=(RACK_WIDTH_M, RACK_SHELF_THICKNESS_M, RACK_DEPTH_M),
                color=(150, 158, 166),
                position=np.array(
                    [0.0, surface_height - RACK_SHELF_THICKNESS_M / 2.0, 0.0],
                    dtype=np.float64,
                ),
            )
        )
        product_names = RACK_SHELF_PRODUCTS.get(name, ())
        if index <= len(product_names):
            handles.append(
                client.scene.add_label(
                    f"{root}/shelf_{index}_product",
                    text=f"S{index}: {product_names[index - 1]}",
                    position=np.array(
                        [0.0, surface_height + 0.08, RACK_DEPTH_M / 2.0 + 0.03],
                        dtype=np.float64,
                    ),
                )
            )

    handles.append(
        client.scene.add_label(
            f"{root}/label",
            text=name.replace("_", " ").title(),
            position=np.array([0.0, RACK_HEIGHT_M + 0.08, 0.0], dtype=np.float64),
        )
    )
    return handles


def _add_counter_table_to_scene(client: viser.ClientHandle) -> list[viser.SceneHandle]:
    """Build the counter table in front of the robot origin."""
    root = "/physical_world/counter_table"
    center_z = COUNTER_TABLE_DISTANCE_FROM_ORIGIN_M + COUNTER_TABLE_WIDTH_M / 2.0
    handles: list[viser.SceneHandle] = [
        client.scene.add_frame(
            root,
            show_axes=False,
            position=np.array([0.0, 0.0, center_z], dtype=np.float64),
        )
    ]

    top_center_y = COUNTER_TABLE_HEIGHT_M - COUNTER_TABLE_TOP_THICKNESS_M / 2.0
    leg_height = COUNTER_TABLE_HEIGHT_M - COUNTER_TABLE_TOP_THICKNESS_M
    handles.append(
        client.scene.add_box(
            f"{root}/top",
            dimensions=(
                COUNTER_TABLE_LENGTH_M,
                COUNTER_TABLE_TOP_THICKNESS_M,
                COUNTER_TABLE_WIDTH_M,
            ),
            color=(128, 92, 58),
            position=np.array([0.0, top_center_y, 0.0], dtype=np.float64),
        )
    )

    half_x = COUNTER_TABLE_LENGTH_M / 2.0 - COUNTER_TABLE_LEG_SIZE_M / 2.0
    half_z = COUNTER_TABLE_WIDTH_M / 2.0 - COUNTER_TABLE_LEG_SIZE_M / 2.0
    for index, (x_offset, z_offset) in enumerate(
        ((-half_x, -half_z), (-half_x, half_z), (half_x, -half_z), (half_x, half_z)),
        start=1,
    ):
        handles.append(
            client.scene.add_box(
                f"{root}/leg_{index}",
                dimensions=(COUNTER_TABLE_LEG_SIZE_M, leg_height, COUNTER_TABLE_LEG_SIZE_M),
                color=(92, 64, 45),
                position=np.array([x_offset, leg_height / 2.0, z_offset], dtype=np.float64),
            )
        )

    handles.append(
        client.scene.add_box(
            f"{root}/front_panel",
            dimensions=(COUNTER_TABLE_LENGTH_M, leg_height * 0.70, 0.025),
            color=(116, 78, 50),
            position=np.array([0.0, leg_height * 0.40, half_z], dtype=np.float64),
        )
    )
    return handles


def _add_work_area_boundary(client: viser.ClientHandle) -> viser.SceneHandle:
    """Outline the 24-square work area without adding any floor fill."""
    width = WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
    depth = WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
    x_near = WORK_AREA_SIDE_SHIFT_M
    x_far = x_near - width
    y = 0.006  # Slightly above the grid to prevent z-fighting.
    corners = np.array(
        [
            [x_near, y, 0.0],
            [x_far, y, 0.0],
            [x_far, y, -depth],
            [x_near, y, -depth],
        ],
        dtype=np.float32,
    )
    points = np.stack((corners, np.roll(corners, -1, axis=0)), axis=1)
    return client.scene.add_line_segments(
        "/physical_world/work_area_boundary",
        points=points,
        colors=WORK_AREA_BOUNDARY_COLOR,
        line_width=4.0,
    )


def _world_scene_move_limits_for_path(path: Path | None) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if path is not None and path.name == "office_world.ply":
        return (
            OFFICE_WORLD_SCENE_MOVE_LIMITS,
            OFFICE_WORLD_SCENE_HEIGHT_LIMITS,
            OFFICE_WORLD_SCENE_MOVE_LIMITS,
        )
    return WORLD_SCENE_MOVE_X_LIMITS, WORLD_SCENE_HEIGHT_LIMITS, WORLD_SCENE_MOVE_Z_LIMITS


def _world_scene_wxyz(rotation_deg: np.ndarray) -> np.ndarray:
    rotation_rad = np.deg2rad(rotation_deg)
    return (
        tf.SO3.from_y_radians(float(rotation_rad[1]))
        @ tf.SO3.from_x_radians(float(rotation_rad[0]))
        @ tf.SO3.from_z_radians(float(rotation_rad[2]))
    ).wxyz


def _world_scene_transform_for_path(path: Path | None) -> tuple[float, np.ndarray, np.ndarray]:
    preset = WORLD_SCENE_PRESETS.get(path.name if path is not None else "")
    if preset is None:
        return WORLD_SCENE_SCALE, WORLD_SCENE_ROTATION_DEG.copy(), WORLD_SCENE_POSITION.copy()
    return (
        float(preset["scale"]),
        np.asarray(preset["rotation_deg"], dtype=np.float64).copy(),
        np.asarray(preset["position"], dtype=np.float64).copy(),
    )


def _configure_text_encoder_runtime(text_encoder_mode: str | None, text_encoder_url: str | None) -> str:
    if text_encoder_url:
        os.environ["TEXT_ENCODER_URL"] = text_encoder_url

    selected_mode = text_encoder_mode or os.environ.get("TEXT_ENCODER_MODE") or "api"
    os.environ["TEXT_ENCODER_MODE"] = selected_mode
    return selected_mode


def _text_encoder_startup_error(mode: str) -> RuntimeError:
    url = os.environ.get("TEXT_ENCODER_URL", DEFAULT_TEXT_ENCODER_URL)
    if mode != "api":
        return RuntimeError(f"Failed to start Kimodo robot demo with TEXT_ENCODER_MODE={mode!r}.")
    return RuntimeError(
        "The Kimodo robot demo needs the text encoder server, but it is not reachable.\n\n"
        f"Start it in another terminal with:\n  {TEXT_ENCODER_SERVER_COMMAND}\n\n"
        f"Then run this app again. Current TEXT_ENCODER_URL is {url!r}.\n\n"
        "If you intentionally want the old in-process 8B LLM2Vec fallback, run with "
        "`--text-encoder-mode auto` or set `TEXT_ENCODER_MODE=local`, but that can be killed by the OS "
        "on machines without enough free memory."
    )


def _safe_clip_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "kimodo_motion"


def _jsonable_array(value: np.ndarray | list | tuple) -> list:
    return np.asarray(value, dtype=np.float64).tolist()


def _save_pick_constraint_debug_json(
    *,
    rack_name: str,
    object_index: int,
    shelf_number: int,
    total_frames: int,
    fps: float,
    rack_target: np.ndarray,
    rack_facing_heading: float,
    object_position: np.ndarray,
    outward_normal: np.ndarray,
    pregrasp_frame: int,
    grasp_frame: int,
    lift_frame: int,
    chest_frame: int,
    lineup_target: np.ndarray | None = None,
    pregrasp_target: np.ndarray,
    grasp_target: np.ndarray,
    lift_target: np.ndarray,
    chest_target: np.ndarray,
    root_path: np.ndarray,
    right_hand_path: np.ndarray,
    right_hand_start_position: np.ndarray,
    hand_specs: dict[str, dict[str, object]] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Save the generated shelf-pick constraints in a human-readable JSON file."""
    output_dir = output_dir or (Path.cwd() / "robot_demo_outputs" / "pick_constraints")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{rack_name}_shelf_{shelf_number}_object_{object_index}_pick_constraints"
    )
    payload = {
        "task": "rack_shelf_pick",
        "rack": rack_name,
        "object_index": int(object_index),
        "shelf_number": int(shelf_number),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "duration_seconds": float(total_frames / max(fps, 1e-9)),
        "policy": {
            "root": "stationary at rack approach pose",
            "body": "frozen from saved move-to-rack final pose",
            "right_arm": "direct reach to object, pregrasp, grasp, lift 5cm, chest hold",
            "wrist": "kept straight in line with forearm; no extra wrist curl",
            "forearm_roll": "rolls forearm so palm faces upward/support direction during pick",
            "left_arm_feet_body": "held still",
        },
        "rack_target_xyz": _jsonable_array(rack_target),
        "rack_facing_heading_rad": float(rack_facing_heading),
        "rack_facing_heading_deg": float(np.rad2deg(rack_facing_heading)),
        "object_position_xyz": _jsonable_array(object_position),
        "rack_outward_normal_xyz": _jsonable_array(outward_normal),
        "keyframes": {
            "start": 0,
            "pregrasp": int(pregrasp_frame),
            "grasp": int(grasp_frame),
            "lift": int(lift_frame),
            "chest_hold": int(chest_frame),
            "end": int(total_frames - 1),
        },
        "targets_xyz": {
            "right_hand_start": _jsonable_array(right_hand_start_position),
            **(
                {"lineup": _jsonable_array(lineup_target)}
                if lineup_target is not None
                else {}
            ),
            "pregrasp": _jsonable_array(pregrasp_target),
            "grasp": _jsonable_array(grasp_target),
            "lift": _jsonable_array(lift_target),
            "chest_hold": _jsonable_array(chest_target),
        },
        "root_path_xyz_per_frame": _jsonable_array(root_path),
        "right_hand_path_xyz_per_frame": _jsonable_array(right_hand_path),
    }
    if hand_specs:
        hands_payload: dict[str, dict[str, object]] = {}
        for side, spec in hand_specs.items():
            object_key = spec.get("object_key")
            hand_start = spec.get("hand_start_position")
            if hand_start is None and "hand_root_index" in spec:
                hand_start = right_hand_start_position if side == "right" else None
            hands_payload[side] = {
                "object_key": list(object_key) if object_key is not None else None,
                "object_index": int(spec["object_index"]),
                "object_position_xyz": _jsonable_array(spec["object_position"]),
                "hand_start_xyz": _jsonable_array(hand_start) if hand_start is not None else None,
                "targets_xyz": {
                    "pregrasp": _jsonable_array(spec["pregrasp_target"]),
                    "diagonal_approach": _jsonable_array(spec["diagonal_approach_target"]),
                    "grasp": _jsonable_array(spec["grasp_target_world"]),
                    "lift": _jsonable_array(spec["lift_target"]),
                    "chest_hold": _jsonable_array(spec["chest_target"]),
                },
                "hand_path_xyz_per_frame": _jsonable_array(spec["hand_target_path"]),
                "target_palm_normals_xyz_per_frame": _jsonable_array(spec["target_palm_normals"]),
                "elbow_bend_hint_xyz": _jsonable_array(spec["elbow_bend_hint"]),
            }
        payload["hands"] = hands_payload
        if "left" in hand_specs:
            payload["left_hand_path_xyz_per_frame"] = _jsonable_array(
                hand_specs["left"]["hand_target_path"]
            )
        if "right" in hand_specs:
            payload["right_hand_path_xyz_per_frame"] = _jsonable_array(
                hand_specs["right"]["hand_target_path"]
            )
    path = output_dir / f"{stem}.json"
    latest_path = output_dir / "latest_pick_constraints.json"
    text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return path


def _save_counter_place_constraint_debug_json(
    *,
    object_key: tuple[str, int, int],
    total_frames: int,
    fps: float,
    root_position: np.ndarray,
    pre_release_frame: int,
    release_frame: int,
    release_hold_end_frame: int,
    retreat_frame: int,
    normal_frame: int,
    object_release_position: np.ndarray,
    pre_release_target: np.ndarray,
    release_target: np.ndarray,
    retreat_target: np.ndarray,
    normal_target: np.ndarray,
    root_path: np.ndarray,
    right_hand_path: np.ndarray,
    right_hand_start_position: np.ndarray,
    output_dir: Path | None = None,
) -> Path:
    """Save generated counter-placement constraints in a human-readable JSON file."""
    output_dir = output_dir or (Path.cwd() / "robot_demo_outputs" / "place_constraints")
    output_dir.mkdir(parents=True, exist_ok=True)
    rack_name, shelf_number, object_index = object_key
    stem = (
        f"{time.strftime('%Y%m%d_%H%M%S')}_"
        f"{rack_name}_shelf_{shelf_number}_object_{object_index}_counter_place_constraints"
    )
    payload = {
        "task": "counter_place_release",
        "rack": rack_name,
        "object_index": int(object_index),
        "shelf_number": int(shelf_number),
        "total_frames": int(total_frames),
        "fps": float(fps),
        "duration_seconds": float(total_frames / max(fps, 1e-9)),
        "policy": {
            "root": "stationary at counter/origin",
            "body": "frozen from carry-return final pose",
            "right_arm": "pre-release, release, wait 1s, slight backward retreat, normal pose",
            "object": "attached to right hand until release frame, then left safely on counter",
            "left_arm_feet_body": "held still",
        },
        "root_position_xyz": _jsonable_array(root_position),
        "object_release_position_xyz": _jsonable_array(object_release_position),
        "keyframes": {
            "start": 0,
            "pre_release": int(pre_release_frame),
            "release": int(release_frame),
            "release_hold_end": int(release_hold_end_frame),
            "retreat": int(retreat_frame),
            "normal": int(normal_frame),
            "end": int(total_frames - 1),
        },
        "targets_xyz": {
            "right_hand_start": _jsonable_array(right_hand_start_position),
            "pre_release": _jsonable_array(pre_release_target),
            "release": _jsonable_array(release_target),
            "retreat": _jsonable_array(retreat_target),
            "normal": _jsonable_array(normal_target),
        },
        "root_path_xyz_per_frame": _jsonable_array(root_path),
        "right_hand_path_xyz_per_frame": _jsonable_array(right_hand_path),
    }
    path = output_dir / f"{stem}.json"
    latest_path = output_dir / "latest_place_constraints.json"
    text = json.dumps(payload, indent=2)
    path.write_text(text, encoding="utf-8")
    latest_path.write_text(text, encoding="utf-8")
    return path


def _default_output_root() -> Path:
    return MEMORIES_ROOT.expanduser().resolve()


def _scan_memory_stems(memories_root: Path) -> list[str]:
    bvh_root = memories_root / "bvh"
    t2_csv_root = memories_root / "t2_csv"
    t3_csv_root = memories_root / "t3_csv"
    stems: set[str] = set()
    if bvh_root.is_dir():
        stems.update(str(path.relative_to(bvh_root).with_suffix("")) for path in bvh_root.rglob("*.bvh"))
    if t3_csv_root.is_dir():
        stems.update(str(path.relative_to(t3_csv_root).with_suffix("")) for path in t3_csv_root.rglob("*.csv"))
    if t2_csv_root.is_dir():
        stems.update(str(path.relative_to(t2_csv_root).with_suffix("")) for path in t2_csv_root.rglob("*.csv"))
    return sorted(stems)


def _scan_base_memory_stems(memories_root: Path) -> list[str]:
    wheel_csv_root = memories_root / "wheel_csv"
    stems: set[str] = set()
    if wheel_csv_root.is_dir():
        for path in wheel_csv_root.rglob("*_diff_drive.csv"):
            stem = str(path.relative_to(wheel_csv_root).with_suffix(""))
            stems.add(stem.removesuffix("_diff_drive"))
    return sorted(stems)


def _memory_bvh_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "bvh" / Path(stem).with_suffix(".bvh")


def _memory_csv_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "t2_csv" / Path(stem).with_suffix(".csv")


def _memory_t3_csv_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "t3_csv" / Path(stem).with_suffix(".csv")


def _base_memory_csv_path(memories_root: Path, stem: str) -> Path:
    stem_path = Path(stem.removesuffix("_diff_drive"))
    return memories_root / "wheel_csv" / stem_path.with_name(f"{stem_path.name}_diff_drive.csv")


def _resolve_base_csv_reference(requested: str, memories_root: Path) -> Path:
    """Resolve an absolute path, relative CSV path, or base-memory stem."""
    requested = _stem_from_memory_label(requested).strip()
    if not requested:
        raise ValueError("Continuation Base CSV is empty")

    requested_path = Path(requested).expanduser()
    wheel_csv_root = memories_root.expanduser().resolve() / "wheel_csv"
    candidates: list[Path] = []
    if requested_path.is_absolute():
        candidates.append(requested_path)
    else:
        candidates.append(wheel_csv_root / requested_path)
        stem_path = Path(requested.removesuffix(".csv").removesuffix("_diff_drive"))
        candidates.append(
            wheel_csv_root / stem_path.with_name(f"{stem_path.name}_diff_drive.csv")
        )

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not find continuation base CSV {requested!r} under {wheel_csv_root}"
    )


def _resolve_base_memory_stem(requested: str, memories_root: Path) -> str:
    """Resolve a base-memory label, filename, or stem to a known stem."""
    requested = _stem_from_memory_label(requested).strip()
    requested = requested.removesuffix(".csv").removesuffix("_diff_drive")
    stems = _scan_base_memory_stems(memories_root)
    exact = {stem: stem for stem in stems}
    lowered = {stem.lower(): stem for stem in stems}
    names = {Path(stem).name.lower(): stem for stem in stems}
    if requested in exact:
        return exact[requested]
    if requested.lower() in lowered:
        return lowered[requested.lower()]
    if requested.lower() in names:
        return names[requested.lower()]
    raise FileNotFoundError(f"Unknown base memory {requested!r} under {memories_root / 'wheel_csv'}")


def _memory_label(memories_root: Path, stem: str) -> str:
    has_bvh = _memory_bvh_path(memories_root, stem).is_file()
    has_t3_csv = _memory_t3_csv_path(memories_root, stem).is_file()
    has_t2_csv = _memory_csv_path(memories_root, stem).is_file()
    if has_bvh and has_t3_csv:
        state = "bvh+t3"
    elif has_bvh and has_t2_csv:
        state = "bvh+t2"
    elif has_bvh:
        state = "needs robot csv"
    elif has_t3_csv:
        state = "t3 only"
    elif has_t2_csv:
        state = "t2 only"
    else:
        state = "missing"
    return f"[{state}] {stem}"


def _base_memory_label(memories_root: Path, stem: str) -> str:
    return f"[base csv] {stem}"


def _outbound_rack_from_memory_stem(stem: str) -> str | None:
    """Best-effort rack detection for loaded outbound human memories."""
    normalized = " ".join(str(stem).lower().replace("_", " ").replace("-", " ").split())
    if any(word in normalized for word in ("return", "pick", "grab", "take", "base")):
        return None
    if not any(word in normalized for word in ("move", "walk", "rack")):
        return None
    return requested_rack_name([str(stem)])


def _pick_request_from_memory_stem(stem: str):
    """Best-effort pick request detection for loaded pick memories."""
    normalized = " ".join(str(stem).lower().replace("_", " ").replace("-", " ").split())
    return requested_rack_pick([normalized])


@dataclass(frozen=True)
class DualRackPickRequest:
    rack_name: str
    shelf_number: int
    object_indices: tuple[int, int] = (1, 3)


def _requested_dual_rack_pick(prompts: list[str]) -> DualRackPickRequest | None:
    """Parse two-hand shelf picks for objects 1 and 3."""
    if len(prompts) != 1:
        return None
    prompt = prompts[0]
    normalized = " ".join(prompt.strip().lower().replace("_", " ").split())
    if not re.search(r"\b(?:pick|take|grab)\b", normalized):
        return None
    object_numbers = {int(value) for value in re.findall(r"\b(?:objects?|items?|objs?)\s*[_-]?(\d+)\b", normalized)}
    pair_match = re.search(
        r"\b(?:objects?|items?|objs?)\s*[_-]?(\d+)\s*(?:and|&|,)\s*(?:(?:objects?|items?|objs?)\s*[_-]?)?(\d+)\b",
        normalized,
    )
    if pair_match is not None:
        object_numbers.update(int(value) for value in pair_match.groups())
    if object_numbers != {1, 3}:
        return None
    if "both hand" not in normalized and "both arms" not in normalized and "two object" not in normalized and "2 object" not in normalized:
        return None
    rack_name = requested_rack_name(prompts)
    shelf_match = re.search(r"\bshelf\s*[_-]?(\d+)\b", normalized)
    if rack_name is None or shelf_match is None:
        return None
    shelf_number = int(shelf_match.group(1))
    if shelf_number not in {1, 2, 3, 4, 5}:
        raise ValueError("Rack shelf number must be 1, 2, 3, 4, or 5")
    return DualRackPickRequest(rack_name=rack_name, shelf_number=shelf_number)


@dataclass(frozen=True)
class RackTransferRequest:
    source_rack: str
    target_rack: str


@dataclass(frozen=True)
class RackTransferRoute:
    positions: tuple[tuple[float, float, float], ...]
    headings: tuple[float, ...]
    final_heading: float
    turn_degrees: tuple[int, ...]
    reverse_end_frame: int = 0


def _requested_rack_transfer(prompts: list[str]) -> RackTransferRequest | None:
    """Parse explicit human routes like ``move from rack 1 to rack 2``."""
    if len(prompts) != 1:
        return None
    normalized = " ".join(prompts[0].strip().lower().replace("_", " ").split())
    if "base" in normalized or not re.search(r"\b(?:move|walk|go|navigate)\b", normalized):
        return None
    match = re.search(
        r"\bfrom\s+rack\s*(\d+)\b.*\bto\s+rack\s*(\d+)\b",
        normalized,
        re.IGNORECASE,
    )
    if match is None:
        return None
    source_rack = f"rack_{int(match.group(1))}"
    target_rack = f"rack_{int(match.group(2))}"
    if source_rack == target_rack:
        raise ValueError("Rack-to-rack motion needs two different racks.")
    return RackTransferRequest(source_rack, target_rack)


def _requested_counter_place(prompts: list[str]) -> bool:
    """Return True for explicit object release/place-on-counter prompts."""
    if len(prompts) != 1:
        return False
    normalized = " ".join(prompts[0].strip().lower().replace("_", " ").split())
    if not re.search(r"\b(?:place|release|drop|put)\b", normalized):
        return False
    return "counter" in normalized or "conter" in normalized or "table" in normalized


def _minimum_generate_duration_seconds(prompts: list[str], fps: float) -> float | None:
    if _requested_dual_rack_pick(prompts) is not None or requested_rack_pick(prompts) is not None:
        return 7.0
    transfer_request = _requested_rack_transfer(prompts)
    if transfer_request is not None:
        for rack_name in (transfer_request.source_rack, transfer_request.target_rack):
            if rack_name not in RACK_MAP_POSITIONS:
                return None
        x_near = WORK_AREA_SIDE_SHIFT_M
        x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
        z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
        source_target, source_heading = rack_width_side_approach_pose(
            RACK_MAP_POSITIONS[transfer_request.source_rack],
            RACK_MAP_YAWS_RAD.get(transfer_request.source_rack, 0.0),
            RACK_WIDTH_M,
            RACK_HUMAN_APPROACH_CLEARANCE_M,
            (x_far, x_near),
            (z_far, 0.0),
        )
        target_position, target_heading = rack_width_side_approach_pose(
            RACK_MAP_POSITIONS[transfer_request.target_rack],
            RACK_MAP_YAWS_RAD.get(transfer_request.target_rack, 0.0),
            RACK_WIDTH_M,
            RACK_HUMAN_APPROACH_CLEARANCE_M,
            (x_far, x_near),
            (z_far, 0.0),
        )
        distance = abs(float(target_position[0]) - float(source_target[0])) + abs(
            float(target_position[2]) - float(source_target[2])
        )
        heading_delta = abs(_normalize_heading(target_heading - source_heading))
        return 1.0 + distance / _rack_human_walk_speed_m_s(transfer_request.target_rack) + (
            0.90 * heading_delta / (math.pi / 2.0)
        )

    rack_name = requested_rack_name(prompts)
    if rack_name is None:
        return None

    normalized = " | ".join(prompts).lower()
    if not re.search(r"\b(move|walk|go|navigate)\b", normalized):
        return None
    if rack_name not in RACK_MAP_POSITIONS:
        return None

    x_near = WORK_AREA_SIDE_SHIFT_M
    x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
    z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
    rack_target, rack_facing_heading = rack_width_side_approach_pose(
        RACK_MAP_POSITIONS[rack_name],
        RACK_MAP_YAWS_RAD.get(rack_name, 0.0),
        RACK_WIDTH_M,
        RACK_HUMAN_APPROACH_CLEARANCE_M,
        (x_far, x_near),
        (z_far, 0.0),
    )
    required_seconds = cardinal_rack_route_required_seconds(
        approach_position=rack_target,
        final_heading=rack_facing_heading,
        fps=fps,
        walk_speed_m_s=_rack_human_walk_speed_m_s(rack_name),
        first_axis="z" if rack_name in {"rack_1", "rack_2", "rack_3"} else "x",
    )
    return required_seconds + 1.0


def _rack_human_walk_speed_m_s(rack_name: str) -> float:
    if rack_name in {"rack_3", "rack_4"}:
        return RACK_HUMAN_FAST_WALK_SPEED_M_S
    return RACK_HUMAN_SLOW_WALK_SPEED_M_S


def _heading_between_positions(start: tuple[float, float, float], end: tuple[float, float, float]) -> float:
    delta_x = float(end[0]) - float(start[0])
    delta_z = float(end[2]) - float(start[2])
    if math.isclose(delta_x, 0.0, abs_tol=1e-9) and math.isclose(delta_z, 0.0, abs_tol=1e-9):
        raise ValueError("Route points must be different to compute a heading.")
    return math.atan2(delta_x, delta_z)


def _normalize_heading(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _append_route_point(
    points: list[tuple[float, float, float]],
    point: tuple[float, float, float],
) -> None:
    if not points or any(
        not math.isclose(a, b, abs_tol=1e-9)
        for a, b in zip(points[-1], point)
    ):
        points.append(point)


def _plan_cardinal_rack_transfer_route(
    *,
    start_position: tuple[float, float, float],
    start_heading: float,
    target_position: tuple[float, float, float],
    final_heading: float,
    total_frames: int,
    fps: float,
    walk_speed_m_s: float,
    turn_seconds_per_90: float = 0.90,
    first_axis: str = "x",
    reverse_distance: float = 0.12,
    reverse_seconds: float = 0.50,
) -> RackTransferRoute:
    """Plan a cardinal route from one rack approach pose to another."""
    if (
        total_frames < 2
        or fps <= 0.0
        or walk_speed_m_s <= 0.0
        or reverse_distance < 0.0
        or (reverse_distance > 0.0 and reverse_seconds <= 0.0)
    ):
        raise ValueError("total_frames, fps, walk_speed_m_s, and reverse settings must be positive")
    if first_axis not in {"x", "z"}:
        raise ValueError("first_axis must be x or z")

    start = tuple(float(value) for value in start_position)
    target = tuple(float(value) for value in target_position)
    heading_vector = (math.sin(float(start_heading)), math.cos(float(start_heading)))
    target_heading_vector = (math.sin(float(final_heading)), math.cos(float(final_heading)))
    backed = (
        start[0] - reverse_distance * heading_vector[0],
        0.0,
        start[2] - reverse_distance * heading_vector[1],
    )
    target_backed = (
        target[0] - reverse_distance * target_heading_vector[0],
        0.0,
        target[2] - reverse_distance * target_heading_vector[1],
    )
    points = [backed if reverse_distance > 0.0 else start]
    if first_axis == "x":
        _append_route_point(points, (target_backed[0], 0.0, points[0][2]))
    else:
        _append_route_point(points, (points[0][0], 0.0, target_backed[2]))
    _append_route_point(points, target_backed)

    actions: list[tuple[str, object]] = []
    current_heading = float(start_heading)
    turn_values: list[float] = []
    if reverse_distance > 0.0:
        actions.append(("reverse", (start, backed, float(start_heading))))
    for segment_start, segment_end in zip(points[:-1], points[1:]):
        move_heading = _heading_between_positions(segment_start, segment_end)
        turn_delta = _normalize_heading(move_heading - current_heading)
        if not math.isclose(turn_delta, 0.0, abs_tol=1e-8):
            actions.append(("turn", turn_delta))
            turn_values.append(turn_delta)
        actions.append(("move", (segment_start, segment_end)))
        current_heading = move_heading

    final_turn = _normalize_heading(final_heading - current_heading)
    if not math.isclose(final_turn, 0.0, abs_tol=1e-8):
        actions.append(("turn", final_turn))
        turn_values.append(final_turn)
    if reverse_distance > 0.0:
        actions.append(("move", (target_backed, target)))

    turn_steps = [
        max(1, int(round(turn_seconds_per_90 * fps * abs(float(value)) / (math.pi / 2.0))))
        for kind, value in actions
        if kind == "turn"
    ]
    move_distances = [
        math.hypot(value[1][0] - value[0][0], value[1][2] - value[0][2])
        for kind, value in actions
        if kind in {"move", "reverse"}
    ]
    move_steps = []
    for kind, value in actions:
        if kind == "reverse":
            move_steps.append(max(1, int(round(reverse_seconds * fps))))
        elif kind == "move":
            distance = math.hypot(value[1][0] - value[0][0], value[1][2] - value[0][2])
            move_steps.append(max(1, int(round(distance / walk_speed_m_s * fps))))
    required_steps = sum(turn_steps) + sum(move_steps)
    if required_steps > total_frames - 1:
        required_seconds = (required_steps + 1) / max(fps, 1e-9)
        raise ValueError(
            f"Rack-to-rack motion is too short; use at least {math.ceil(required_seconds)} seconds."
        )

    positions = [start]
    headings = [float(start_heading)]
    position = start
    heading = float(start_heading)
    turn_index = 0
    move_index = 0
    reverse_end_frame = 0
    for kind, value in actions:
        if kind == "turn":
            steps = turn_steps[turn_index]
            turn_index += 1
            delta = float(value)
            for step in range(1, steps + 1):
                positions.append(position)
                headings.append(_normalize_heading(heading + delta * step / steps))
            heading = _normalize_heading(heading + delta)
            continue

        steps = move_steps[move_index]
        move_index += 1
        if kind == "reverse":
            segment_start, segment_end, fixed_heading = value
            heading = float(fixed_heading)
        else:
            segment_start, segment_end = value
            heading = _heading_between_positions(segment_start, segment_end)
        for step in range(1, steps + 1):
            ratio = step / steps
            position = tuple(
                float(segment_start[axis] + ratio * (segment_end[axis] - segment_start[axis]))
                for axis in range(3)
            )
            positions.append(position)
            headings.append(heading)
        if kind == "reverse":
            reverse_end_frame = len(positions) - 1

    while len(positions) < total_frames:
        positions.append(target)
        headings.append(float(final_heading))

    if len(positions) != total_frames:
        raise AssertionError(f"Expected {total_frames} route frames, generated {len(positions)}")
    return RackTransferRoute(
        positions=tuple(positions),
        headings=tuple(headings),
        final_heading=_normalize_heading(final_heading),
        turn_degrees=tuple(int(round(abs(math.degrees(value)))) for value in turn_values),
        reverse_end_frame=reverse_end_frame,
    )


def _resolve_generate_duration_seconds(prompts: list[str], requested_duration: float, fps: float) -> float:
    tool_duration = _minimum_generate_duration_seconds(prompts, fps)
    if tool_duration is None:
        return requested_duration
    return tool_duration


def _stem_from_memory_label(label: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", str(label)).strip()


def _normalize_memory_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_memory_stem(requested: str, memories_root: Path) -> str:
    requested = _stem_from_memory_label(requested).strip().removesuffix(".bvh")
    if not requested:
        raise ValueError("Missing memory name")

    stems = _scan_memory_stems(memories_root)
    if not stems:
        raise FileNotFoundError(f"No BVH memories found in {memories_root / 'bvh'}")

    candidates = [requested]
    if requested.startswith("kimodo_"):
        candidates.append(f"generated/{requested}")

    exact = {stem: stem for stem in stems}
    lower = {stem.lower(): stem for stem in stems}
    names = {Path(stem).name.lower(): stem for stem in stems}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        lowered = candidate.lower()
        if lowered in lower:
            return lower[lowered]
        if lowered in names:
            return names[lowered]

    normalized_requested = _normalize_memory_search(requested)
    normalized_matches = [
        stem
        for stem in stems
        if normalized_requested
        and (
            normalized_requested == _normalize_memory_search(stem)
            or normalized_requested == _normalize_memory_search(Path(stem).name)
        )
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    contains_matches = [
        stem
        for stem in stems
        if requested.lower() in stem.lower()
        or requested.lower() in Path(stem).name.lower()
        or (
            normalized_requested
            and normalized_requested in _normalize_memory_search(stem)
        )
    ]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1 or len(normalized_matches) > 1:
        matches = sorted(set(normalized_matches + contains_matches))
        raise ValueError(f"Ambiguous memory name. Matches: {', '.join(matches)}")

    available = ", ".join(stems) or "<none>"
    raise FileNotFoundError(f"Could not find memory {requested!r}. Available memories: {available}")


def _new_generated_stem() -> str:
    return f"generated/kimodo_{uuid.uuid4().hex[:10]}"


def _model_native_fps(demo: Demo, model_name: str, fallback: float) -> float:
    bundle = demo.models.get(model_name)
    if bundle is not None and bundle.model_fps and bundle.model_fps > 0.0:
        return float(bundle.model_fps)
    return float(fallback)


def _scan_world_scene_paths(worlds_root: Path) -> list[Path]:
    worlds_root = worlds_root.expanduser().resolve()
    if not worlds_root.is_dir():
        return []
    return sorted(path for path in worlds_root.rglob("*.ply") if path.is_file())


def _world_scene_label(path: Path, worlds_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(worlds_root.expanduser().resolve()))
    except ValueError:
        return str(path)


def _resolve_world_scene_path(requested: str, worlds_root: Path = WORLD_SCENES_ROOT) -> Path:
    worlds_root = worlds_root.expanduser().resolve()
    paths = _scan_world_scene_paths(worlds_root)
    labels = {_world_scene_label(path, worlds_root): path for path in paths}
    names = {path.name: path for path in paths}

    normalized = requested.strip()
    alias = WORLD_SCENE_ALIASES.get(normalized.lower(), normalized)
    candidates = [alias]
    if not alias.endswith(".ply"):
        candidates.append(f"{alias}.ply")

    for candidate in candidates:
        if candidate in labels:
            return labels[candidate]
        if candidate in names:
            return names[candidate]

    contains = [
        path
        for label, path in labels.items()
        if normalized.lower() in label.lower() or alias.lower() in label.lower()
    ]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        matches = ", ".join(_world_scene_label(path, worlds_root) for path in contains)
        raise ValueError(f"Ambiguous world name. Matches: {matches}")

    available = ", ".join(sorted(labels)) or "<none>"
    raise ValueError(f"World not found: {requested!r}. Available: {available}")


def _load_world_scene(path: Path) -> dict[str, np.ndarray | str]:
    """Load a world PLY without touching the human/robot scene nodes."""
    path = path.expanduser().resolve()
    plydata = PlyData.read(path)
    vertex_data = plydata["vertex"]
    vertex_fields = set(vertex_data.data.dtype.names or ())
    positions = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=-1).astype(np.float32)

    gaussian_fields = {
        "scale_0",
        "scale_1",
        "scale_2",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    if gaussian_fields.issubset(vertex_fields):
        scales = np.exp(
            np.stack([vertex_data["scale_0"], vertex_data["scale_1"], vertex_data["scale_2"]], axis=-1)
        ).astype(np.float32)
        wxyzs = np.stack(
            [vertex_data["rot_0"], vertex_data["rot_1"], vertex_data["rot_2"], vertex_data["rot_3"]],
            axis=1,
        ).astype(np.float32)
        colors = (
            0.5
            + GAUSSIAN_SH_C0
            * np.stack([vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"]], axis=1)
        ).astype(np.float32)
        colors = np.clip(colors, 0.0, 1.0)
        opacities = (1.0 / (1.0 + np.exp(-vertex_data["opacity"][:, None]))).astype(np.float32)
        rotations = tf.SO3(wxyzs).as_matrix().astype(np.float32)
        covariances = np.einsum(
            "nij,njk,nlk->nil",
            rotations,
            np.eye(3, dtype=np.float32)[None, :, :] * scales[:, None, :] ** 2,
            rotations,
            optimize=True,
        ).astype(np.float32)
        return {
            "kind": "gaussian_splats",
            "centers": positions,
            "rgbs": colors,
            "opacities": opacities,
            "covariances": covariances,
        }

    rgb_fields = {"red", "green", "blue"}
    if rgb_fields.issubset(vertex_fields):
        colors = np.stack([vertex_data["red"], vertex_data["green"], vertex_data["blue"]], axis=-1).astype(np.uint8)
    else:
        colors = np.array([170, 170, 170], dtype=np.uint8)
    return {"kind": "point_cloud", "points": positions, "colors": colors}


def _csv_motion_summary(csv_path: Path) -> str:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "Frames: `0`"

    def span(columns: list[str]) -> float:
        values = []
        for row in rows:
            for column in columns:
                if column in row:
                    values.append(float(row[column]))
        if not values:
            return 0.0
        return max(values) - min(values)

    root_spans_cm = [
        span(["root_translateX"]),
        span(["root_translateY"]),
        span(["root_translateZ"]),
    ]
    right_span_deg = span([f"right_joint{i}_dof" for i in range(1, 8)])
    left_span_deg = span([f"left_joint{i}_dof" for i in range(1, 8)])
    body_span_deg = span(
        [
            "waist_yaw_joint_dof",
            "waist_roll_joint_dof",
            "waist_pitch_joint_dof",
            "left_hip_pitch_joint_dof",
            "left_hip_roll_joint_dof",
            "left_hip_yaw_joint_dof",
            "left_knee_joint_dof",
            "left_ankle_roll_joint_dof",
            "left_ankle_pitch_joint_dof",
            "right_hip_pitch_joint_dof",
            "right_hip_roll_joint_dof",
            "right_hip_yaw_joint_dof",
            "right_knee_joint_dof",
            "right_ankle_roll_joint_dof",
            "right_ankle_pitch_joint_dof",
        ]
    )
    return (
        f"Frames: `{len(rows)}`\n\n"
        "Root span XYZ: "
        f"`{root_spans_cm[0] / 100.0:.2f}, "
        f"{root_spans_cm[1] / 100.0:.2f}, "
        f"{root_spans_cm[2] / 100.0:.2f} m`\n\n"
        f"Right arm span: `{right_span_deg:.1f} deg`\n\n"
        f"Left arm span: `{left_span_deg:.1f} deg`\n\n"
        f"Body/leg span: `{body_span_deg:.1f} deg`"
    )


def _load_right_gripper_widths(csv_path: Path) -> list[float]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if "right_gripper_joint1_dof" not in reader.fieldnames or "right_gripper_joint2_dof" not in reader.fieldnames:
            return []

        widths: list[float] = []
        for row in reader:
            if not row:
                continue
            joint1 = abs(float(row["right_gripper_joint1_dof"]))
            joint2 = abs(float(row["right_gripper_joint2_dof"]))
            widths.append(max(joint1, joint2) * 2.0)
    return widths


def _load_base_wheel_rpms(csv_path: Path) -> list[tuple[float, float]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "left_motor_rpm" not in fieldnames or "right_motor_rpm" not in fieldnames:
            return []
        return [
            (float(row["left_motor_rpm"]), float(row["right_motor_rpm"]))
            for row in reader
            if row
        ]


def _load_base_encoder_targets(csv_path: Path) -> list[float | None]:
    """Load optional per-row wheel-revolution targets used by calibrated turns."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "left_motor_rpm" not in fieldnames or "right_motor_rpm" not in fieldnames:
            return []
        has_target = "encoder_target_wheel_revolutions" in fieldnames
        targets: list[float | None] = []
        for row in reader:
            if not row:
                continue
            value = str(row.get("encoder_target_wheel_revolutions", "")).strip() if has_target else ""
            targets.append(float(value) if value else None)
        return targets


def _csv_has_columns(csv_path: Path, required: set[str]) -> bool:
    with csv_path.open(newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    return required.issubset(fieldnames)


def _csv_frame_rate(csv_path: Path, fallback: float) -> float:
    """Read a stable frame rate from time_s while tolerating older CSVs."""
    times: list[float] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "time_s" not in reader.fieldnames:
            return float(fallback)
        for row in reader:
            try:
                times.append(float(row["time_s"]))
            except (KeyError, TypeError, ValueError):
                continue
            if len(times) >= 200:
                break
    if len(times) < 2:
        return float(fallback)
    intervals = np.diff(np.asarray(times, dtype=np.float64))
    intervals = intervals[np.isfinite(intervals) & (intervals > 0.0)]
    if not len(intervals):
        return float(fallback)
    return float(1.0 / np.median(intervals))


@dataclass
class RobotWorkflowState:
    output_root: Path = field(default_factory=_default_output_root)
    memory_stem: str | None = None
    bvh_path: Path | None = None
    npz_path: Path | None = None
    csv_path: Path | None = None
    t2_motion: T2ViewerMotion | None = None
    t3_motion: T3Playback | None = None
    wheel_base: WheelBasePlayback | None = None
    wheel_csv_path: Path | None = None
    arm_frames: list[ArmFrame] = field(default_factory=list)
    gripper_widths: list[float] = field(default_factory=list)
    base_wheel_rpms: list[tuple[float, float]] = field(default_factory=list)
    base_encoder_targets: list[float | None] = field(default_factory=list)
    connection: T2NeroConnection = field(default_factory=T2NeroConnection)
    status_markdown: viser.GuiMarkdownHandle | None = None
    robot_markdown: viser.GuiMarkdownHandle | None = None
    retarget_running: bool = False
    placing_object: bool = False
    placed_object_handles: list[viser.SceneHandle] = field(default_factory=list)
    placed_object_count: int = 0
    stream_real_robot_playback: bool = False
    real_robot_approval_pending: bool = False
    real_robot_previewed_memory_stem: str | None = None
    hardware_previewed_csv_paths: set[Path] = field(default_factory=set)
    tara_stop_event: threading.Event = field(default_factory=threading.Event)
    tara_thread: threading.Thread | None = None
    tara_next_index: int = 0
    tara_active_direction: str = "forward"
    base_prompt_segments: list[tuple[str, int]] = field(default_factory=list)
    base_prompt_fps: float = 30.0
    base_prompt_stem: str | None = None
    base_prompt_start_pose: tuple[float, float, float] | None = None
    continue_base_from_last_pose: bool = False
    continuation_base_csv_reference: str = ""
    chain_loaded_base_memories: bool = True
    loop_base_memory_preview: bool = False
    freeze_t3_base: bool = False
    base_route_rack: str | None = None
    base_route_kind: str | None = None
    rack_object_handles: dict[tuple[str, int, int], viser.SceneHandle] = field(default_factory=dict)
    rack_object_home_positions: dict[tuple[str, int, int], np.ndarray] = field(default_factory=dict)
    picked_rack_object: tuple[str, int, int] | None = None
    picked_rack_objects_by_hand: dict[str, tuple[str, int, int]] = field(default_factory=dict)
    pick_hand_side: str = "right"
    pick_hold_start_frame: int | None = None
    pick_hold_start_frames_by_side: dict[str, int] = field(default_factory=dict)
    pick_object_hand_offset: np.ndarray | None = None
    pick_object_hand_offsets_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    pick_reach_start_frame: int | None = None
    pick_hand_target_world: np.ndarray | None = None
    pick_target_path_world: np.ndarray | None = None
    pick_grasp_target_world: np.ndarray | None = None
    pick_grasp_verified: bool = False
    pick_elbow_bend_hint_world: np.ndarray | None = None
    place_release_frame: int | None = None
    place_object_position_world: np.ndarray | None = None
    place_object_positions_by_hand: dict[str, np.ndarray] = field(default_factory=dict)
    place_reach_start_frame: int | None = None
    place_hand_target_world: np.ndarray | None = None
    place_hand_targets_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    place_target_path_world: np.ndarray | None = None
    place_target_paths_by_side: dict[str, np.ndarray] = field(default_factory=dict)
    place_elbow_bend_hint_world: np.ndarray | None = None
    place_elbow_bend_hints_by_side: dict[str, np.ndarray] = field(default_factory=dict)

    def clear_t2_preview(
        self,
        *,
        clear_wheel_base: bool = True,
        clear_t3: bool = True,
    ) -> None:
        self.tara_stop_event.set()
        if self.t2_motion is not None:
            self.t2_motion.clear()
            self.t2_motion = None
        if clear_t3 and self.t3_motion is not None:
            self.t3_motion.clear()
            self.t3_motion = None
        if clear_wheel_base and self.wheel_base is not None:
            self.wheel_base.clear()
            self.wheel_base = None
        elif self.wheel_base is not None:
            # The wheel URDF points along +X while the neutral human faces +Z.
            self.wheel_base.visual_yaw_offset_rad = float(np.pi / 2.0)
            self.wheel_base.set_stationary()
        self.wheel_csv_path = None
        self.base_prompt_segments.clear()
        self.base_prompt_stem = None
        self.base_prompt_start_pose = None
        self.arm_frames.clear()
        self.gripper_widths.clear()
        self.base_wheel_rpms.clear()
        self.base_encoder_targets.clear()
        self.csv_path = None

    def clear_placed_objects(self) -> None:
        for handle in self.placed_object_handles:
            handle.remove()
        self.placed_object_handles.clear()
        self.placed_object_count = 0


@dataclass
class RemoteT2RetargetJob:
    output_root: Path
    relative_stem: Path
    csv_path: Path


@dataclass
class RemoteT3RetargetJob:
    output_root: Path
    relative_stem: Path
    t3_csv_path: Path
    wheel_csv_path: Path


class RobotDemo(Demo):
    """Kimodo demo with a robot production workflow mounted as a separate panel."""

    def __init__(
        self,
        default_model_name: str = DEFAULT_MODEL,
        world_scene_path: Path | None = DEFAULT_WORLD_SCENE_PATH,
        *,
        tara_remote_url: str | None = None,
        tara_port: str = "/dev/ttyUSB0",
        tara_slave_id: int = 1,
        tara_baudrate: int = 115200,
        tara_fps: float = 30.0,
        tara_max_rpm: float = 30.0,
        tara_rpm_scale: float = 1.0,
        tara_debug: bool = False,
        model_server_url: str | None = None,
        retarget_server_url: str | None = None,
    ):
        super().__init__(default_model_name=default_model_name, model_server_url=model_server_url)
        self.robot_workflows: dict[int, RobotWorkflowState] = {}
        self.world_scene_path = world_scene_path.expanduser().resolve() if world_scene_path is not None else None
        self.tara_remote_url = tara_remote_url.rstrip("/") if tara_remote_url else None
        self.tara_port = tara_port
        self.tara_slave_id = tara_slave_id
        self.tara_baudrate = tara_baudrate
        self.tara_fps = tara_fps
        self.tara_max_rpm = tara_max_rpm
        self.tara_rpm_scale = tara_rpm_scale
        self.tara_debug = tara_debug
        self.retarget_server_url = retarget_server_url.rstrip("/") if retarget_server_url else None
        self._world_scene_data: dict[str, np.ndarray | str] | None = None
        self._world_scene_data_path: Path | None = None
        self._world_scene_lock = threading.Lock()
        self.world_scene_handles: dict[int, viser.SceneHandle] = {}
        self.world_scene_client_transforms: dict[int, tuple[Path, float, np.ndarray, np.ndarray]] = {}
        self.rack_scene_handles: dict[int, list[viser.SceneHandle]] = {}
        self.work_area_boundary_handles: dict[int, viser.SceneHandle] = {}
        self._world_scene_sync_callbacks: dict[int, Callable[[Path], None]] = {}
        self._memory_sync_callbacks: dict[int, Callable[[str, bool], str]] = {}
        self._memory_list_callbacks: dict[int, Callable[[], dict[str, object]]] = {}
        self._base_memory_control_callbacks: dict[
            int, Callable[[dict[str, object]], dict[str, object]]
        ] = {}
        self._base_memory_list_callbacks: dict[int, Callable[[], dict[str, object]]] = {}
        self._generate_retarget_callbacks: dict[int, Callable[[dict[str, object]], dict[str, object]]] = {}
        self._robot_control_callbacks: dict[int, Callable[[dict[str, object]], dict[str, object]]] = {}
        self._control_jobs: dict[str, dict[str, object]] = {}
        self._control_jobs_lock = threading.Lock()
        self._control_server: http.server.ThreadingHTTPServer | None = None

    def _setup_demo_for_client(self, client: viser.ClientHandle) -> None:
        super()._setup_demo_for_client(client)
        # A browser can reconnect while the initial model/mesh cache is still
        # being built. In that case the disconnect callback removes the old
        # session before the base setup returns; do not build GUI state for the
        # now-stale client handle.
        if client.client_id not in self.client_sessions:
            return
        self.work_area_boundary_handles[client.client_id] = _add_work_area_boundary(client)
        self.rack_scene_handles[client.client_id] = [
            handle
            for name, position in RACK_MAP_POSITIONS.items()
            for handle in _add_rack_to_scene(
                client,
                name,
                position,
                RACK_MAP_YAWS_RAD.get(name, 0.0),
            )
        ]
        self.rack_scene_handles[client.client_id].extend(_add_counter_table_to_scene(client))
        self._create_world_scene_gui(client)
        self._hide_examples_folder(client)
        workflow = RobotWorkflowState()
        object_handles, object_home_positions = _add_pick_objects_to_scene(client)
        workflow.rack_object_handles = object_handles
        workflow.rack_object_home_positions = object_home_positions
        if client.client_id not in self.client_sessions or client.client_id not in self.rack_scene_handles:
            return
        self.rack_scene_handles[client.client_id].extend(object_handles.values())
        self.robot_workflows[client.client_id] = workflow
        if DEFAULT_WHEEL_URDF_PATH.is_file():
            workflow.wheel_base = WheelBasePlayback(
                client,
                DEFAULT_WHEEL_URDF_PATH,
                None,
                self.tara_fps,
                root_node_name=f"/tara_wheel_base_{client.client_id}",
                visual_yaw_offset_rad=float(np.pi / 2.0),
            )
            workflow.wheel_base.set_visible(False)
        else:
            print(f"Tara wheel-base URDF not found: {DEFAULT_WHEEL_URDF_PATH}")
        self._create_robot_pipeline_gui(client)

    def start_control_server(self, host: str, port: int) -> None:
        demo = self

        class ControlHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_planner_file(self, relative_path: str) -> None:
                planner_root = Path(__file__).with_name("warehouse_ui")
                requested = (planner_root / relative_path).resolve()
                try:
                    requested.relative_to(planner_root.resolve())
                except ValueError:
                    self._send_json(404, {"ok": False, "error": "Unknown planner asset"})
                    return
                if not requested.is_file():
                    self._send_json(404, {"ok": False, "error": f"Missing planner asset: {relative_path}"})
                    return
                data = requested.read_bytes()
                content_type = mimetypes.guess_type(requested.name)[0] or "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _request_payload(self) -> dict[str, object]:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                payload: dict[str, object] = {
                    key: values[-1] for key, values in query.items() if values
                }
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    body = self.rfile.read(length)
                    if body:
                        body_payload = json.loads(body.decode("utf-8"))
                        if isinstance(body_payload, dict):
                            payload.update(body_payload)
                return payload

            def _packing_plan(self) -> None:
                payload = self._request_payload()
                raw_order = payload.get("order")
                if not isinstance(raw_order, dict) or not raw_order:
                    self._send_json(400, {"ok": False, "error": "A non-empty order object is required."})
                    return
                order_arguments: list[str] = []
                for product_id, raw_quantity in raw_order.items():
                    product_id = str(product_id).strip().lower()
                    if not re.fullmatch(r"[a-z][a-z0-9_]*", product_id):
                        self._send_json(400, {"ok": False, "error": f"Invalid product id: {product_id}"})
                        return
                    if isinstance(raw_quantity, bool):
                        self._send_json(400, {"ok": False, "error": f"Invalid quantity for {product_id}."})
                        return
                    try:
                        quantity = int(raw_quantity)
                    except (TypeError, ValueError):
                        self._send_json(400, {"ok": False, "error": f"Invalid quantity for {product_id}."})
                        return
                    if quantity <= 0:
                        self._send_json(400, {"ok": False, "error": f"Quantity for {product_id} must be positive."})
                        return
                    order_arguments.append(f"{product_id}={quantity}")

                if not WAREHOUSE_PACKING_PLANNER.is_file():
                    self._send_json(503, {"ok": False, "error": f"Packing planner not found: {WAREHOUSE_PACKING_PLANNER}"})
                    return
                planner_python = WAREHOUSE_PLANNER_PYTHON if WAREHOUSE_PLANNER_PYTHON.is_file() else Path(sys.executable)
                try:
                    result = subprocess.run(
                        [str(planner_python), str(WAREHOUSE_PACKING_PLANNER), *order_arguments, "--json"],
                        cwd=WAREHOUSE_PLANNER_ROOT,
                        capture_output=True,
                        text=True,
                        timeout=10.0,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    self._send_json(503, {"ok": False, "error": f"OR-Tools planner could not run: {exc}"})
                    return
                if result.returncode != 0:
                    error_lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
                    error = error_lines[-1].removeprefix("packing_planner.py: error: ") if error_lines else "OR-Tools planning failed."
                    self._send_json(400, {"ok": False, "error": error})
                    return
                try:
                    plan = json.loads(result.stdout)
                except json.JSONDecodeError:
                    self._send_json(500, {"ok": False, "error": "OR-Tools planner returned invalid JSON."})
                    return
                self._send_json(200, {"ok": True, "plan": plan})

            def _load_memory(self) -> None:
                payload = self._request_payload()
                requested = str(payload.get("memory") or payload.get("stem") or "").strip()
                preserve_base_pose = bool(payload.get("preserve_base_pose", False))
                if not requested:
                    self._send_json(400, {"ok": False, "error": "Missing memory name"})
                    return
                callbacks = list(demo._memory_sync_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                errors: list[str] = []
                loaded_clients: list[int] = []
                loaded_stems: list[str] = []
                for client_id, sync_memory in callbacks:
                    try:
                        loaded_stems.append(sync_memory(requested, preserve_base_pose))
                        loaded_clients.append(client_id)
                    except Exception as exc:
                        errors.append(f"{client_id}: {exc}")
                status = 200 if not errors else 500
                self._send_json(
                    status,
                    {
                        "ok": not errors,
                        "stem": loaded_stems[0] if loaded_stems else requested,
                        "requested": requested,
                        "loaded_stems": loaded_stems,
                        "loaded_clients": loaded_clients,
                        "errors": errors,
                    },
                )

            def _load_world(self) -> None:
                payload = self._request_payload()
                requested = str(payload.get("world") or payload.get("path") or "").strip()
                if not requested:
                    self._send_json(400, {"ok": False, "error": "Missing world name"})
                    return
                try:
                    path = _resolve_world_scene_path(requested)
                except Exception as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                callbacks = list(demo._world_scene_sync_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                errors: list[str] = []
                loaded_clients: list[int] = []
                for client_id, sync_world in callbacks:
                    try:
                        sync_world(path)
                        loaded_clients.append(client_id)
                    except Exception as exc:
                        errors.append(f"{client_id}: {exc}")
                status = 200 if not errors else 500
                self._send_json(
                    status,
                    {
                        "ok": not errors,
                        "world": _world_scene_label(path, WORLD_SCENES_ROOT),
                        "path": str(path),
                        "loaded_clients": loaded_clients,
                        "errors": errors,
                    },
                )

            def _worlds(self) -> None:
                worlds_root = WORLD_SCENES_ROOT.expanduser().resolve()
                worlds = [
                    _world_scene_label(path, worlds_root)
                    for path in _scan_world_scene_paths(worlds_root)
                ]
                self._send_json(200, {"ok": True, "worlds_root": str(worlds_root), "worlds": worlds})

            def _memories(self) -> None:
                callbacks = list(demo._memory_list_callbacks.items())
                if callbacks:
                    clients = []
                    for client_id, list_memories in callbacks:
                        payload = list_memories()
                        payload["client_id"] = client_id
                        clients.append(payload)
                    self._send_json(200, {"ok": True, "clients": clients})
                    return

                memories_root = MEMORIES_ROOT.expanduser().resolve()
                stems = _scan_memory_stems(memories_root)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "memories_root": str(memories_root),
                        "stems": stems,
                        "labels": [_memory_label(memories_root, stem) for stem in stems],
                    },
                )

            def _base_memories(self) -> None:
                callbacks = list(demo._base_memory_list_callbacks.items())
                if callbacks:
                    clients = []
                    for client_id, list_base_memories in callbacks:
                        payload = list_base_memories()
                        payload["client_id"] = client_id
                        clients.append(payload)
                    self._send_json(200, {"ok": True, "clients": clients})
                    return
                root = MEMORIES_ROOT.expanduser().resolve()
                stems = _scan_base_memory_stems(root)
                self._send_json(200, {"ok": True, "memories_root": str(root), "stems": stems})

            def _base_control(self, default_action: str | None = None) -> None:
                payload = self._request_payload()
                action = str(payload.get("action") or default_action or "status").strip().lower()
                payload["action"] = action
                callbacks = list(demo._base_memory_control_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                responses: list[dict[str, object]] = []
                errors: list[str] = []
                for client_id, control_base in callbacks:
                    try:
                        client_response = control_base(payload)
                        client_response["client_id"] = client_id
                        responses.append(client_response)
                    except Exception as exc:
                        errors.append(f"{client_id}: {exc}")
                if not responses:
                    self._send_json(500, {"ok": False, "error": "; ".join(errors)})
                    return

                # Use the newest connected Viser client as the top-level status
                # consumed by the planner, while applying the action to every
                # client so embedded and separately opened viewers stay synced.
                response = responses[-1]
                self._send_json(
                    200,
                    {
                        "ok": not errors,
                        **response,
                        "clients": responses,
                        "errors": errors,
                    },
                )

            def _generate_retarget(self) -> None:
                payload = self._request_payload()
                prompt = str(payload.get("prompt") or "").strip()
                if not prompt:
                    self._send_json(400, {"ok": False, "error": "Missing prompt"})
                    return
                callbacks = list(demo._generate_retarget_callbacks.items())
                if callbacks:
                    client_id, generate_retarget = callbacks[0]
                    try:
                        response = generate_retarget(payload)
                    except Exception as exc:
                        self._send_json(500, {"ok": False, "error": str(exc), "client_id": client_id})
                        return
                    response["client_id"] = client_id
                    response["headless"] = False
                    self._send_json(202, {"ok": True, **response})
                    return

                try:
                    response = demo._generate_retarget_headless(payload)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc), "headless": True})
                    return
                self._send_json(202, {"ok": True, "headless": True, **response})

            def _robot_control(self, default_action: str | None = None) -> None:
                payload = self._request_payload()
                action = str(payload.get("action") or default_action or "").strip().lower()
                if not action:
                    self._send_json(400, {"ok": False, "error": "Missing robot action"})
                    return
                payload["action"] = action
                callbacks = list(demo._robot_control_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                client_id, robot_control = callbacks[0]
                try:
                    response = robot_control(payload)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc), "client_id": client_id})
                    return
                response["client_id"] = client_id
                self._send_json(200, {"ok": True, **response})

            def _job(self) -> None:
                payload = self._request_payload()
                job_id = str(payload.get("job_id") or payload.get("id") or "").strip()
                if not job_id:
                    self._send_json(400, {"ok": False, "error": "Missing job_id"})
                    return
                with demo._control_jobs_lock:
                    job = dict(demo._control_jobs.get(job_id) or {})
                if not job:
                    self._send_json(404, {"ok": False, "error": f"Unknown job_id: {job_id}"})
                    return
                self._send_json(200, {"ok": True, "job": job})

            def _motion_files(self) -> None:
                payload = self._request_payload()
                job_id = str(payload.get("job_id") or payload.get("id") or "").strip()
                stem = str(payload.get("stem") or "").strip()

                job: dict[str, object] = {}
                if job_id:
                    with demo._control_jobs_lock:
                        job = dict(demo._control_jobs.get(job_id) or {})
                    if not job:
                        self._send_json(404, {"ok": False, "error": f"Unknown job_id: {job_id}"})
                        return
                    stem = str(job.get("stem") or stem).strip()

                if not stem:
                    self._send_json(400, {"ok": False, "error": "Missing job_id or stem"})
                    return

                bvh_path = Path(str(job.get("bvh_path") or _memory_bvh_path(MEMORIES_ROOT, stem))).expanduser().resolve()
                csv_path = Path(str(job.get("csv_path") or _memory_csv_path(MEMORIES_ROOT, stem))).expanduser().resolve()

                bvh_text = bvh_path.read_text(encoding="utf-8") if bvh_path.is_file() else None
                csv_text = csv_path.read_text(encoding="utf-8") if csv_path.is_file() else None
                if bvh_text is None and csv_text is None:
                    self._send_json(404, {
                        "ok": False,
                        "error": f"No generated files found for stem: {stem}",
                        "bvh_path": str(bvh_path),
                        "csv_path": str(csv_path),
                    })
                    return

                self._send_json(200, {
                    "ok": True,
                    "job_id": job_id or None,
                    "stem": stem,
                    "bvh_path": str(bvh_path),
                    "csv_path": str(csv_path),
                    "bvh": bvh_text,
                    "csv": csv_text,
                })

            def _client_status(self) -> list[dict[str, object]]:
                clients: list[dict[str, object]] = []
                for client_id in sorted(demo.client_sessions.keys()):
                    session = demo.client_sessions.get(client_id)
                    workflow = demo.robot_workflows.get(client_id)
                    world_transform = demo.world_scene_client_transforms.get(client_id)
                    current_world = None
                    if world_transform is not None:
                        current_world = _world_scene_label(world_transform[0], WORLD_SCENES_ROOT)
                    if session is None:
                        continue
                    robot_status = None
                    real_robot_status = None
                    if workflow is not None:
                        robot_status = {
                            "playing": bool(session.playing),
                            "frame": int(session.frame_idx),
                            "max_frame": int(session.max_frame_idx),
                            "model_fps": float(session.model_fps),
                            "duration_seconds": float(session.cur_duration),
                            "has_t2_preview": workflow.t2_motion is not None,
                            "has_robot_frames": bool(workflow.arm_frames),
                            "robot_frame_count": len(workflow.arm_frames),
                            "csv_path": str(workflow.csv_path) if workflow.csv_path is not None else None,
                            "memory_stem": workflow.memory_stem,
                        }
                        real_robot_status = {
                            "connected": workflow.connection.is_connected(),
                            "dry_run": bool(workflow.connection.dry_run),
                            "playing": (
                                bool(session.playing)
                                and workflow.connection.is_connected()
                                and workflow.stream_real_robot_playback
                            ),
                            "frame": int(session.frame_idx),
                            "max_frame": int(session.max_frame_idx),
                            "action_status": (
                                "playing"
                                if bool(session.playing)
                                and workflow.connection.is_connected()
                                and workflow.stream_real_robot_playback
                                else "connected"
                                if workflow.connection.is_connected()
                                else "disconnected"
                            ),
                            "approval_pending": bool(workflow.real_robot_approval_pending),
                            "streaming_enabled": bool(workflow.stream_real_robot_playback),
                            "previewed_for_current_memory": bool(
                                workflow.memory_stem is not None
                                and workflow.real_robot_previewed_memory_stem == workflow.memory_stem
                            ),
                            "has_robot_frames": bool(workflow.arm_frames),
                            "robot_frame_count": len(workflow.arm_frames),
                            "csv_path": str(workflow.csv_path) if workflow.csv_path is not None else None,
                            "memory_stem": workflow.memory_stem,
                            "last_output": workflow.connection.last_output[-6:],
                        }
                    clients.append(
                        {
                            "client_id": client_id,
                            "current_memory": workflow.memory_stem if workflow is not None else None,
                            "current_world": current_world,
                            "playing": bool(session.playing),
                            "frame": int(session.frame_idx),
                            "max_frame": int(session.max_frame_idx),
                            "model_fps": float(session.model_fps),
                            "duration_seconds": float(session.cur_duration),
                            "has_t2_preview": bool(workflow is not None and workflow.t2_motion is not None),
                            "has_robot_frames": bool(workflow is not None and workflow.arm_frames),
                            "robot": robot_status,
                            "real_robot": real_robot_status,
                        }
                    )
                return clients

            def do_GET(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                if path in {"/planner", "/planner/"}:
                    self._send_planner_file("index.html")
                    return
                if path.startswith("/planner/"):
                    self._send_planner_file(path.removeprefix("/planner/"))
                    return
                if path == "/status":
                    clients = self._client_status()
                    memory_clients = sorted(demo._memory_sync_callbacks.keys())
                    world_clients = sorted(demo._world_scene_sync_callbacks.keys())
                    status = {
                        "memory_clients": memory_clients,
                        "world_clients": world_clients,
                        "clients": clients,
                    }
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "status": status,
                        },
                    )
                    return
                if path == "/worlds":
                    self._worlds()
                    return
                if path == "/memories":
                    self._memories()
                    return
                if path == "/base-memories":
                    self._base_memories()
                    return
                if path == "/base-control":
                    self._base_control()
                    return
                if path == "/job":
                    self._job()
                    return
                if path == "/motion-files":
                    self._motion_files()
                    return
                if path == "/robot-status":
                    self._robot_control(default_action="status")
                    return
                if path == "/load-memory":
                    self._load_memory()
                    return
                if path == "/load-world":
                    self._load_world()
                    return
                if path == "/generate-retarget":
                    self._generate_retarget()
                    return
                if path == "/motion-files":
                    self._motion_files()
                    return
                if path == "/robot-control":
                    self._robot_control()
                    return
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

            def do_POST(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                if path == "/load-memory":
                    self._load_memory()
                    return
                if path == "/load-world":
                    self._load_world()
                    return
                if path == "/generate-retarget":
                    self._generate_retarget()
                    return
                if path == "/robot-control":
                    self._robot_control()
                    return
                if path == "/base-control":
                    self._base_control()
                    return
                if path == "/packing-plan":
                    self._packing_plan()
                    return
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

            def do_OPTIONS(self) -> None:
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()

        class ReusableThreadingHTTPServer(http.server.ThreadingHTTPServer):
            allow_reuse_address = True

        try:
            self._control_server = ReusableThreadingHTTPServer((host, port), ControlHandler)
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                raise RuntimeError(
                    f"Kimodo control API port {port} is already in use on {host}. "
                    f"Stop the existing robot_app.py process or run with --control-port {port + 1}."
                ) from exc
            raise
        thread = threading.Thread(
            target=self._control_server.serve_forever,
            name="kimodo-control-server",
            daemon=True,
        )
        thread.start()
        print(f"Kimodo control server listening on http://{host}:{port}")

    def _control_job_id(self, prompt: str, prefix: str = "generate_retarget") -> str:
        digest = hashlib.sha1(f"headless:{time.time()}:{prompt}".encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    def _set_control_job(self, job_id: str, **updates: object) -> None:
        with self._control_jobs_lock:
            current = dict(self._control_jobs.get(job_id) or {})
            current.update(updates)
            current["job_id"] = job_id
            current["updated_at"] = time.time()
            self._control_jobs[job_id] = current

    def _retarget_relative_stem(self, bvh_path: Path, output_root: Path) -> Path:
        try:
            return bvh_path.relative_to(output_root / "bvh").with_suffix("")
        except ValueError:
            return Path(bvh_path.stem)

    def _decode_remote_file(self, response: dict[str, object], *keys: str) -> bytes:
        for key in keys:
            value = response.get(key)
            if isinstance(value, str) and value:
                return base64.b64decode(value)
        raise KeyError(f"Retarget server response missing one of: {', '.join(keys)}")

    def _post_remote_retarget(self, target: str, bvh_path: Path, output_root: Path) -> dict[str, object]:
        if self.retarget_server_url is None:
            raise RuntimeError("No remote retarget server URL configured")
        bvh_bytes = bvh_path.read_bytes()
        relative_stem = self._retarget_relative_stem(bvh_path, output_root)
        payload = {
            "bvh_base64": base64.b64encode(bvh_bytes).decode("ascii"),
            "stem": str(relative_stem),
            "target": target,
        }
        request = urllib.request.Request(
            f"{self.retarget_server_url}/retarget/{target}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote retarget server failed with HTTP {exc.code}: {detail}") from exc

    def _run_remote_retarget_bvh_to_csv(self, bvh_path: Path, output_root: Path) -> tuple[RemoteT2RetargetJob, Path]:
        response = self._post_remote_retarget("t2", bvh_path, output_root)
        relative_stem = Path(str(response.get("stem") or self._retarget_relative_stem(bvh_path, output_root)))
        csv_path = output_root / "t2_csv" / relative_stem.with_suffix(".csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_bytes(self._decode_remote_file(response, "csv_base64", "t2_csv_base64"))

        log_path = output_root / "logs" / relative_stem.with_suffix(".retarget.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(response.get("log") or response.get("stdout") or ""), encoding="utf-8")
        return RemoteT2RetargetJob(output_root=output_root, relative_stem=relative_stem, csv_path=csv_path), log_path

    def _run_remote_retarget_bvh_to_t3_csv(self, bvh_path: Path, output_root: Path) -> tuple[RemoteT3RetargetJob, Path]:
        response = self._post_remote_retarget("t3", bvh_path, output_root)
        relative_stem = Path(str(response.get("stem") or self._retarget_relative_stem(bvh_path, output_root)))
        t3_csv_path = output_root / "t3_csv" / relative_stem.with_suffix(".csv")
        wheel_csv_path = output_root / "wheel_csv" / relative_stem.with_name(
            f"{relative_stem.name}_diff_drive.csv"
        )
        t3_csv_path.parent.mkdir(parents=True, exist_ok=True)
        wheel_csv_path.parent.mkdir(parents=True, exist_ok=True)
        t3_csv_path.write_bytes(self._decode_remote_file(response, "t3_csv_base64", "csv_base64"))
        wheel_csv_path.write_bytes(self._decode_remote_file(response, "wheel_csv_base64"))

        log_path = output_root / "logs" / relative_stem.with_suffix(".t3_retarget.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(response.get("log") or response.get("stdout") or ""), encoding="utf-8")
        job = RemoteT3RetargetJob(
            output_root=output_root,
            relative_stem=relative_stem,
            t3_csv_path=t3_csv_path,
            wheel_csv_path=wheel_csv_path,
        )
        return job, log_path

    def _run_headless_retarget_bvh_to_csv(self, bvh_path: Path, output_root: Path) -> tuple[SomaT2RetargetJob, Path]:
        if self.retarget_server_url:
            return self._run_remote_retarget_bvh_to_csv(bvh_path, output_root)
        job = SomaT2RetargetJob(
            retargeter_root=default_soma_retargeter_root(),
            bvh_path=bvh_path,
            output_root=output_root,
            conda_env=os.environ.get("KIMODO_RETARGET_CONDA_ENV", "soma-retargeter"),
        )
        result = job.run()
        log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".retarget.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout or "", encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"soma-retargeter failed with exit code {result.returncode}. Log: {log_path}")
        if not job.csv_path.is_file():
            raise FileNotFoundError(f"Expected retarget CSV was not created: {job.csv_path}")
        return job, log_path

    def _run_headless_retarget_bvh_to_t3_csv(self, bvh_path: Path, output_root: Path) -> tuple[SomaT3RetargetJob | RemoteT3RetargetJob, Path]:
        if self.retarget_server_url:
            return self._run_remote_retarget_bvh_to_t3_csv(bvh_path, output_root)
        job = SomaT3RetargetJob(
            retargeter_root=default_soma_retargeter_root(),
            bvh_path=bvh_path,
            output_root=output_root,
            conda_env=os.environ.get("KIMODO_RETARGET_CONDA_ENV", "soma-retargeter"),
        )
        result = job.run()
        log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".t3_retarget.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout or "", encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"T3 retargeter failed with exit code {result.returncode}. Log: {log_path}")
        if not job.t3_csv_path.is_file():
            raise FileNotFoundError(f"Expected T3 CSV was not created: {job.t3_csv_path}")
        if not job.wheel_csv_path.is_file():
            raise FileNotFoundError(f"Expected T3 wheel CSV was not created: {job.wheel_csv_path}")
        return job, log_path

    def _save_headless_bvh(
        self,
        *,
        stem: str,
        output_root: Path,
        model_name: str,
        model_fps: float,
        joints_pos: torch.Tensor,
        joints_rot: torch.Tensor,
        foot_contacts: torch.Tensor | None,
        standard_tpose: bool,
    ) -> Path:
        bundle = self.models[model_name]
        skeleton = bundle.skeleton
        relative_stem = Path(stem)
        bvh_path = output_root / "bvh" / relative_stem.with_suffix(".bvh")
        npz_path = output_root / "kimodo_npz" / relative_stem.with_suffix(".npz")
        bvh_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        local_rot_mats = global_rots_to_local_rots(joints_rot, skeleton)
        root_positions = joints_pos[:, skeleton.root_idx, :]
        save_motion_bvh(
            str(bvh_path),
            local_rot_mats,
            root_positions,
            skeleton=skeleton,
            fps=float(model_fps),
            standard_tpose=standard_tpose,
        )

        motion_data = {
            "posed_joints": joints_pos.detach().cpu().numpy(),
            "global_rot_mats": joints_rot.detach().cpu().numpy(),
            "local_rot_mats": local_rot_mats.detach().cpu().numpy(),
            "root_positions": root_positions.detach().cpu().numpy(),
        }
        if foot_contacts is not None:
            motion_data["foot_contacts"] = foot_contacts.detach().cpu().numpy()
        save_kimodo_npz(str(npz_path), motion_data)

        metadata = {
            "stem": str(relative_stem),
            "model_name": model_name,
            "fps": float(model_fps),
            "num_frames": int(joints_pos.shape[0]),
            "bvh_path": str(bvh_path),
            "npz_path": str(npz_path),
            "standard_tpose_bvh": bool(standard_tpose),
            "headless": True,
        }
        meta_path = output_root / "metadata" / relative_stem.with_suffix(".json")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return bvh_path

    def _generate_retarget_headless(self, payload: dict[str, object]) -> dict[str, object]:
        raw_prompts = payload.get("prompts")
        if isinstance(raw_prompts, list):
            prompts = [str(item).strip() for item in raw_prompts if str(item).strip()]
        else:
            prompts = [str(payload.get("prompt") or "").strip()]
        prompts = [prompt for prompt in prompts if prompt]
        if not prompts:
            raise ValueError("Missing prompt")
        prompt = prompts[0]

        model_name = resolve_model_name(str(payload.get("model") or self.default_model_name), "Kimodo")
        if "soma" not in model_name.lower():
            raise ValueError("Headless generate-retarget requires a SOMA model.")

        bundle = self.load_model(model_name)
        duration = float(payload.get("duration_seconds") or DEFAULT_CUR_DURATION)
        duration = max(0.1, _resolve_generate_duration_seconds(prompts, duration, float(bundle.model_fps)))
        seed = int(payload.get("seed") or 42)
        diffusion_steps = int(payload.get("diffusion_steps") or 100)
        stem = str(payload.get("stem") or "").strip() or _new_generated_stem()
        output_root = Path(str(payload.get("output_root") or _default_output_root())).expanduser().resolve()
        frames_per_prompt = max(1, int(round(duration * float(bundle.model_fps))))
        num_frames = [frames_per_prompt for _ in prompts]
        total_frames = sum(num_frames)
        standard_tpose = bool(payload.get("standard_tpose", True))
        job_id = self._control_job_id(" | ".join(prompts), prefix="headless_generate_retarget")

        self._set_control_job(
            job_id,
            status="queued",
            prompt=prompt,
            prompts=prompts,
            stem=stem,
            output_root=str(output_root),
            model_name=model_name,
            headless=True,
        )

        def worker() -> None:
            try:
                self._set_control_job(job_id, status="generating")
                if not self._cuda_healthy:
                    raise RuntimeError("CUDA is in a corrupted state. The space is restarting...")

                with self._generation_lock:
                    seed_everything(seed)
                    pred = bundle.model(
                        prompts,
                        num_frames,
                        diffusion_steps,
                        multi_prompt=True,
                        constraint_lst=[],
                        cfg_weight=[2.0, 2.0],
                        num_samples=1,
                        cfg_type="separated",
                        post_processing=True,
                        root_margin=0.04,
                        num_transition_frames=NB_TRANSITION_FRAMES,
                    )

                joints_pos = pred["posed_joints"][0].to(device=self.device, dtype=torch.float32)
                joints_rot = pred["global_rot_mats"][0].to(device=self.device, dtype=torch.float32)
                foot_contacts = pred.get("foot_contacts")
                if foot_contacts is not None:
                    foot_contacts = foot_contacts[0].to(device=self.device, dtype=torch.float32)

                self._set_control_job(job_id, status="saving", num_frames=total_frames)
                bvh_path = self._save_headless_bvh(
                    stem=stem,
                    output_root=output_root,
                    model_name=model_name,
                    model_fps=float(bundle.model_fps),
                    joints_pos=joints_pos,
                    joints_rot=joints_rot,
                    foot_contacts=foot_contacts,
                    standard_tpose=standard_tpose,
                )

                self._set_control_job(job_id, status="retargeting", bvh_path=str(bvh_path), retarget_target="t3")
                retarget_job, log_path = self._run_headless_retarget_bvh_to_t3_csv(bvh_path, output_root)
                self._set_control_job(
                    job_id,
                    status="done",
                    stem=str(retarget_job.relative_stem),
                    bvh_path=str(bvh_path),
                    csv_path=str(retarget_job.t3_csv_path),
                    t3_csv_path=str(retarget_job.t3_csv_path),
                    wheel_csv_path=str(retarget_job.wheel_csv_path),
                    log_path=str(log_path),
                    retarget_target="t3",
                )
            except Exception as exc:
                self._set_control_job(job_id, status="error", error=str(exc))

        threading.Thread(target=worker, name=f"headless-generate-retarget-{job_id}", daemon=True).start()
        return {"job_id": job_id, "status": "queued", "stem": stem}

    def _get_world_scene_data(self, world_scene_path: Path | None = None) -> dict[str, np.ndarray | str] | None:
        path = (world_scene_path or self.world_scene_path)
        if path is None:
            return None
        path = path.expanduser().resolve()
        if not path.is_file():
            return None
        with self._world_scene_lock:
            if self._world_scene_data is None or self._world_scene_data_path != path:
                self._world_scene_data = _load_world_scene(path)
                self._world_scene_data_path = path
            return self._world_scene_data

    def _add_world_scene_handle(
        self,
        client: viser.ClientHandle,
        world_scene_data: dict[str, np.ndarray | str],
        rotation_deg: np.ndarray,
        position: np.ndarray,
        scale: float,
    ) -> viser.SceneHandle:
        wxyz = _world_scene_wxyz(rotation_deg)
        scale = float(scale)
        if world_scene_data["kind"] == "gaussian_splats":
            return client.scene.add_gaussian_splats(
                "/world/scene",
                centers=world_scene_data["centers"] * scale,
                rgbs=world_scene_data["rgbs"],
                opacities=world_scene_data["opacities"],
                covariances=world_scene_data["covariances"] * (scale * scale),
                wxyz=wxyz,
                position=position,
            )
        return client.scene.add_point_cloud(
            "/world/scene",
            points=world_scene_data["points"] * scale,
            colors=world_scene_data["colors"],
            point_size=max(0.001, 0.02 * scale),
            point_shape="circle",
            precision="float32",
            wxyz=wxyz,
            position=position,
        )

    def _add_world_scene_to_client(self, client: viser.ClientHandle) -> None:
        try:
            world_scene_data = self._get_world_scene_data()
            if world_scene_data is None:
                return
            scale, rotation_deg, position = _world_scene_transform_for_path(self.world_scene_path)
            handle = self._add_world_scene_handle(
                client,
                world_scene_data,
                rotation_deg,
                position,
                scale,
            )
            self.world_scene_handles[client.client_id] = handle
            self.world_scene_client_transforms[client.client_id] = (
                self.world_scene_path,
                scale,
                rotation_deg.copy(),
                position.copy(),
            )
            client.add_notification(
                title="World loaded",
                body=str(self.world_scene_path),
                auto_close_seconds=4.0,
                color="green",
            )
        except Exception as exc:
            print(f"Failed to load world scene {self.world_scene_path}: {exc}")
            client.add_notification(
                title="World load failed",
                body=str(exc),
                auto_close_seconds=8.0,
                color="red",
            )

    def _create_world_scene_gui(self, client: viser.ClientHandle) -> None:
        world_handle = self.world_scene_handles.get(client.client_id)
        worlds_root = WORLD_SCENES_ROOT.expanduser().resolve()
        world_paths = _scan_world_scene_paths(worlds_root)
        world_labels = [_world_scene_label(path, worlds_root) for path in world_paths] if world_paths else ["<no worlds>"]
        selected_world_path = self.world_scene_path
        if selected_world_path is not None:
            selected_world_path = selected_world_path.expanduser().resolve()
        initial_world_label = (
            _world_scene_label(selected_world_path, worlds_root)
            if selected_world_path in world_paths
            else world_labels[0]
        )
        initial_world_path = selected_world_path if selected_world_path in world_paths else (world_paths[0] if world_paths else None)
        initial_scale, initial_rotation_deg, initial_position = _world_scene_transform_for_path(initial_world_path)
        initial_x_limits, initial_y_limits, initial_z_limits = _world_scene_move_limits_for_path(initial_world_path)

        with client.gui.add_folder("World Scene", expand_by_default=False):
            show_world_scene_checkbox = client.gui.add_checkbox("Load World", initial_value=False)
            show_floor_grid_checkbox = client.gui.add_checkbox("Show Floor Grid", initial_value=True)
            show_racks_checkbox = client.gui.add_checkbox("Show Physical Racks", initial_value=True)
            show_work_area_checkbox = client.gui.add_checkbox("Show 20-Square Boundary", initial_value=True)
            worlds_root_text = client.gui.add_text("Root", initial_value=str(worlds_root))
            world_dropdown = client.gui.add_dropdown(
                "World",
                options=world_labels,
                initial_value=initial_world_label,
            )
            refresh_worlds_button = client.gui.add_button("Refresh Worlds")
            world_scale = client.gui.add_slider(
                "World Scale",
                min=WORLD_SCENE_SCALE_LIMITS[0],
                max=WORLD_SCENE_SCALE_LIMITS[1],
                step=0.01,
                initial_value=initial_scale,
            )

            x_rotation = client.gui.add_slider(
                "Pitch X",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_rotation_deg[0]),
            )
            y_rotation = client.gui.add_slider(
                "Turn Y",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_rotation_deg[1]),
            )
            z_rotation = client.gui.add_slider(
                "Roll Z",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_rotation_deg[2]),
            )
            x_position = client.gui.add_slider(
                "Move X",
                min=initial_x_limits[0],
                max=initial_x_limits[1],
                step=0.01,
                initial_value=float(initial_position[0]),
            )
            y_position = client.gui.add_slider(
                "Height Y",
                min=initial_y_limits[0],
                max=initial_y_limits[1],
                step=0.01,
                initial_value=float(initial_position[1]),
            )
            z_position = client.gui.add_slider(
                "Move Z",
                min=initial_z_limits[0],
                max=initial_z_limits[1],
                step=0.01,
                initial_value=float(initial_position[2]),
            )

        applying_world_preset = False

        def current_worlds_root() -> Path:
            return Path(worlds_root_text.value).expanduser().resolve()

        def current_rotation_deg() -> np.ndarray:
            return np.array(
                [x_rotation.value, y_rotation.value, z_rotation.value],
                dtype=np.float64,
            )

        def current_position() -> np.ndarray:
            return np.array(
                [x_position.value, y_position.value, z_position.value],
                dtype=np.float64,
            )

        def selected_world() -> Path | None:
            value = str(world_dropdown.value)
            if value == "<no worlds>":
                return None
            return (current_worlds_root() / value).expanduser().resolve()

        def apply_move_limits(path: Path | None) -> None:
            x_limits, y_limits, z_limits = _world_scene_move_limits_for_path(path)
            x_position.min = x_limits[0]
            x_position.max = x_limits[1]
            y_position.min = y_limits[0]
            y_position.max = y_limits[1]
            z_position.min = z_limits[0]
            z_position.max = z_limits[1]

        def apply_world_preset(path: Path) -> None:
            scale, rotation_deg, position = _world_scene_transform_for_path(path)
            apply_move_limits(path)
            world_scale.value = scale
            x_rotation.value = float(rotation_deg[0])
            y_rotation.value = float(rotation_deg[1])
            z_rotation.value = float(rotation_deg[2])
            x_position.value = float(position[0])
            y_position.value = float(position[1])
            z_position.value = float(position[2])

        def reload_world(path: Path) -> None:
            nonlocal world_handle
            world_scene_data = self._get_world_scene_data(path)
            if world_scene_data is None:
                raise FileNotFoundError(path)
            if world_handle is not None:
                world_handle.remove()
            world_handle = self._add_world_scene_handle(
                client,
                world_scene_data,
                current_rotation_deg(),
                current_position(),
                float(world_scale.value),
            )
            self.world_scene_handles[client.client_id] = world_handle
            self.world_scene_path = path.expanduser().resolve()
            self.world_scene_client_transforms[client.client_id] = (
                self.world_scene_path,
                float(world_scale.value),
                current_rotation_deg(),
                current_position(),
            )
            world_handle.visible = bool(show_world_scene_checkbox.value)

        def sync_world_from_external_change(path: Path) -> None:
            nonlocal applying_world_preset
            root = current_worlds_root()
            label = _world_scene_label(path, root)
            try:
                applying_world_preset = True
                if label not in world_dropdown.options:
                    refresh_world_options(path)
                if label in world_dropdown.options:
                    world_dropdown.value = label
                apply_world_preset(path)
                if bool(show_world_scene_checkbox.value):
                    reload_world(path)
            finally:
                applying_world_preset = False

        self._world_scene_sync_callbacks[client.client_id] = sync_world_from_external_change

        def refresh_world_options(select_path: Path | None = None) -> None:
            root = current_worlds_root()
            paths = _scan_world_scene_paths(root)
            labels = [_world_scene_label(path, root) for path in paths] if paths else ["<no worlds>"]
            world_dropdown.options = labels
            if select_path is not None and select_path.resolve() in [path.resolve() for path in paths]:
                world_dropdown.value = _world_scene_label(select_path, root)
            elif str(world_dropdown.value) not in labels:
                world_dropdown.value = labels[0]

        def update_world_transform() -> None:
            if world_handle is None:
                return
            world_handle.wxyz = _world_scene_wxyz(current_rotation_deg())
            world_handle.position = current_position()
            path = selected_world()
            if path is not None:
                self.world_scene_client_transforms[client.client_id] = (
                    path.expanduser().resolve(),
                    float(world_scale.value),
                    current_rotation_deg(),
                    current_position(),
                )

        def set_floor_grid_visible(visible: bool) -> None:
            grid_handle = self.grid_handles.get(client.client_id)
            if grid_handle is not None:
                grid_handle.visible = visible

        def set_racks_visible(visible: bool) -> None:
            for rack_handle in self.rack_scene_handles.get(client.client_id, []):
                rack_handle.visible = visible

        def set_work_area_visible(visible: bool) -> None:
            boundary = self.work_area_boundary_handles.get(client.client_id)
            if boundary is not None:
                boundary.visible = visible

        set_floor_grid_visible(bool(show_floor_grid_checkbox.value))
        set_racks_visible(bool(show_racks_checkbox.value))
        set_work_area_visible(bool(show_work_area_checkbox.value))

        @show_world_scene_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            if bool(show_world_scene_checkbox.value):
                path = selected_world()
                if path is None:
                    show_world_scene_checkbox.value = False
                    return
                try:
                    reload_world(path)
                except Exception as exc:
                    show_world_scene_checkbox.value = False
                    event.client.add_notification(
                        title="World load failed",
                        body=str(exc),
                        auto_close_seconds=8.0,
                        color="red",
                    )
            elif world_handle is not None:
                world_handle.visible = False

        @show_floor_grid_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_floor_grid_visible(bool(show_floor_grid_checkbox.value))

        @show_racks_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_racks_visible(bool(show_racks_checkbox.value))

        @show_work_area_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_work_area_visible(bool(show_work_area_checkbox.value))

        @world_scale.on_update
        def _(event: viser.GuiEvent) -> None:
            if applying_world_preset:
                return
            path = selected_world()
            if path is None:
                return
            if not bool(show_world_scene_checkbox.value):
                return
            try:
                reload_world(path)
            except Exception as exc:
                event.client.add_notification(
                    title="World scale failed",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @world_dropdown.on_update
        def _(event: viser.GuiEvent) -> None:
            nonlocal applying_world_preset
            if applying_world_preset:
                return
            path = selected_world()
            if path is None:
                return
            try:
                applying_world_preset = True
                apply_world_preset(path)
                applying_world_preset = False
                if bool(show_world_scene_checkbox.value):
                    reload_world(path)
                    event.client.add_notification(
                        title="World loaded",
                        body=str(path),
                        auto_close_seconds=4.0,
                        color="green",
                    )
                for client_id, sync_world in list(self._world_scene_sync_callbacks.items()):
                    if client_id == client.client_id:
                        continue
                    try:
                        sync_world(path)
                    except Exception as exc:
                        print(f"Failed to sync world scene for client {client_id}: {exc}")
            except Exception as exc:
                if event.client is not None:
                    event.client.add_notification(
                        title="World load failed",
                        body=str(exc),
                        auto_close_seconds=8.0,
                        color="red",
                    )
            finally:
                applying_world_preset = False

        @refresh_worlds_button.on_click
        def _(event: viser.GuiEvent) -> None:
            refresh_world_options(selected_world())
            event.client.add_notification(
                title="Worlds refreshed",
                body=str(current_worlds_root()),
                auto_close_seconds=3.0,
                color="blue",
            )

        for control in (x_rotation, y_rotation, z_rotation, x_position, y_position, z_position):
            control.on_update(lambda _event: None if applying_world_preset else update_world_transform())

    @staticmethod
    def _hide_examples_folder(client: viser.ClientHandle) -> None:
        root = client.gui._container_handle_from_uuid.get("root")
        if root is None:
            return
        for child in list(root._children.values()):
            props = getattr(getattr(child, "_impl", None), "props", None)
            if getattr(props, "label", None) == "Examples":
                child.visible = False

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        workflow = self.robot_workflows.pop(client.client_id, None)
        if workflow is not None:
            workflow.tara_stop_event.set()
            workflow.connection.disconnect()
            workflow.clear_t2_preview()
            workflow.clear_placed_objects()
        self.world_scene_handles.pop(client.client_id, None)
        self.world_scene_client_transforms.pop(client.client_id, None)
        self.rack_scene_handles.pop(client.client_id, None)
        self.work_area_boundary_handles.pop(client.client_id, None)
        self._world_scene_sync_callbacks.pop(client.client_id, None)
        self._memory_sync_callbacks.pop(client.client_id, None)
        self._memory_list_callbacks.pop(client.client_id, None)
        self._base_memory_control_callbacks.pop(client.client_id, None)
        self._base_memory_list_callbacks.pop(client.client_id, None)
        self._generate_retarget_callbacks.pop(client.client_id, None)
        self._robot_control_callbacks.pop(client.client_id, None)
        super().on_client_disconnect(client)

    def set_frame(self, client_id: int, frame_idx: int, update_timeline: bool = True):
        super().set_frame(client_id, frame_idx, update_timeline=update_timeline)
        workflow = self.robot_workflows.get(client_id)
        if workflow is None:
            return
        held_objects = dict(workflow.picked_rack_objects_by_hand)
        if workflow.picked_rack_object is not None:
            held_objects.setdefault(workflow.pick_hand_side, workflow.picked_rack_object)
        if held_objects:
            session = self.client_sessions.get(client_id)
            for hand_side, object_key in held_objects.items():
                object_handle = workflow.rack_object_handles.get(object_key)
                home_position = workflow.rack_object_home_positions.get(object_key)
                if object_handle is None or home_position is None:
                    continue
                if (
                    workflow.place_release_frame is not None
                    and frame_idx >= workflow.place_release_frame
                ):
                    placed_position = workflow.place_object_positions_by_hand.get(hand_side)
                    if placed_position is None:
                        placed_position = workflow.place_object_position_world
                    if placed_position is not None:
                        object_handle.position = placed_position
                    else:
                        object_handle.position = home_position
                elif (
                    (
                        hold_start_frame := workflow.pick_hold_start_frames_by_side.get(
                            hand_side,
                            workflow.pick_hold_start_frame,
                        )
                    )
                    is not None
                    and frame_idx >= hold_start_frame
                    and session is not None
                    and session.motions
                ):
                    motion = next(iter(session.motions.values()))
                    active_hand_names = (
                        motion.skeleton.left_hand_joint_names
                        if hand_side == "left"
                        else motion.skeleton.right_hand_joint_names
                    )
                    hand_label = "Left" if hand_side == "left" else "Right"
                    hand_name = active_hand_names[0]
                    hand_index = motion.skeleton.bone_order_names.index(hand_name)
                    middle_name = (
                        f"{hand_label}HandMiddleEnd"
                        if f"{hand_label}HandMiddleEnd" in motion.skeleton.bone_order_names
                        else active_hand_names[-1]
                    )
                    middle_index = motion.skeleton.bone_order_names.index(middle_name)
                    motion_frame = min(int(frame_idx), motion.length - 1)
                    wrist_position = motion.joints_pos[motion_frame, hand_index].detach().cpu().numpy()
                    middle_position = motion.joints_pos[motion_frame, middle_index].detach().cpu().numpy()
                    palm_position = 0.35 * wrist_position + 0.65 * middle_position
                    hand_offset_local = workflow.pick_object_hand_offsets_by_side.get(hand_side)
                    hand_has_saved_offset = hand_offset_local is not None
                    if hand_offset_local is None:
                        hand_offset_local = (
                            workflow.pick_object_hand_offset
                            if workflow.pick_object_hand_offset is not None
                            else np.zeros(3, dtype=np.float64)
                        )
                    if workflow.pick_grasp_verified or hand_has_saved_offset:
                        hand_rotation = (
                            motion.joints_rot[motion_frame, hand_index].detach().cpu().numpy()
                        )
                        object_handle.position = palm_position + hand_rotation @ hand_offset_local
                    else:
                        object_handle.position = home_position
                else:
                    object_handle.position = home_position
        if workflow.t2_motion is not None:
            workflow.t2_motion.set_frame(frame_idx)
        if workflow.t3_motion is not None:
            workflow.t3_motion.apply_frame(frame_idx)
        if workflow.wheel_base is not None:
            workflow.wheel_base.apply_frame(0 if workflow.freeze_t3_base else frame_idx)

    def restore_robot_memory_state_from_loaded_motion(
        self,
        client: viser.ClientHandle,
        source_label: str,
    ) -> None:
        """Restore rack/pick workflow state after a BVH is loaded through the generic UI."""
        session = self.client_sessions.get(client.client_id)
        workflow = self.robot_workflows.get(client.client_id)
        if session is None or workflow is None or not session.motions:
            return

        labels = [source_label, Path(source_label).stem]
        outbound_rack = next(
            (rack for label in labels if (rack := _outbound_rack_from_memory_stem(label)) is not None),
            None,
        )
        pick_request = next(
            (request for label in labels if (request := _pick_request_from_memory_stem(label)) is not None),
            None,
        )
        motion = next(iter(session.motions.values()))
        if outbound_rack is not None:
            session.human_outbound_rack_poses[outbound_rack] = (
                motion.joints_pos[-1].detach().cpu().numpy().copy(),
                motion.joints_rot[-1].detach().cpu().numpy().copy(),
            )
            session.human_route_rack = outbound_rack
            session.human_route_kind = "outbound"
            return

        if pick_request is None:
            return
        object_key = (
            pick_request.rack_name,
            pick_request.shelf_number,
            pick_request.object_index,
        )
        object_position = workflow.rack_object_home_positions.get(object_key)
        if object_position is None:
            return
        restored_hand_side = "left" if pick_request.object_index == 3 else "right"
        restored_hand_names = (
            motion.skeleton.left_hand_joint_names
            if restored_hand_side == "left"
            else motion.skeleton.right_hand_joint_names
        )
        restored_hand_label = "Left" if restored_hand_side == "left" else "Right"
        hand_name = restored_hand_names[0]
        hand_index = motion.skeleton.bone_order_names.index(hand_name)
        middle_name = (
            f"{restored_hand_label}HandMiddleEnd"
            if f"{restored_hand_label}HandMiddleEnd" in motion.skeleton.bone_order_names
            else restored_hand_names[-1]
        )
        middle_index = motion.skeleton.bone_order_names.index(middle_name)
        attach_frame = min(
            motion.length - 1,
            max(0, int(round(motion.length * 0.52))),
        )
        wrist_attach = motion.joints_pos[attach_frame, hand_index].detach().cpu().numpy()
        middle_attach = motion.joints_pos[attach_frame, middle_index].detach().cpu().numpy()
        palm_attach = 0.35 * wrist_attach + 0.65 * middle_attach
        hand_rotation_attach = motion.joints_rot[attach_frame, hand_index].detach().cpu().numpy()

        workflow.picked_rack_object = object_key
        workflow.picked_rack_objects_by_hand.clear()
        workflow.pick_hand_side = restored_hand_side
        workflow.pick_hold_start_frame = attach_frame
        workflow.pick_hold_start_frames_by_side.clear()
        workflow.pick_object_hand_offset = (
            hand_rotation_attach.T @ (np.asarray(object_position, dtype=np.float64) - palm_attach)
        )
        workflow.pick_object_hand_offsets_by_side.clear()
        workflow.pick_reach_start_frame = 0
        workflow.pick_hand_target_world = None
        workflow.pick_target_path_world = None
        workflow.pick_grasp_target_world = None
        workflow.pick_grasp_verified = True
        workflow.pick_elbow_bend_hint_world = None
        workflow.place_release_frame = None
        workflow.place_object_position_world = None
        workflow.place_object_positions_by_hand.clear()
        workflow.place_reach_start_frame = None
        workflow.place_hand_target_world = None
        workflow.place_hand_targets_by_side.clear()
        workflow.place_target_path_world = None
        workflow.place_target_paths_by_side.clear()
        workflow.place_elbow_bend_hint_world = None
        workflow.place_elbow_bend_hints_by_side.clear()
        session.human_outbound_rack_poses[pick_request.rack_name] = (
            motion.joints_pos[0].detach().cpu().numpy().copy(),
            motion.joints_rot[0].detach().cpu().numpy().copy(),
        )
        session.human_route_rack = pick_request.rack_name
        session.human_route_kind = "pick_hold"
        self.set_frame(client.client_id, session.frame_idx)

    def generate(
        self,
        client: viser.ClientHandle,
        prompts: list[str],
        num_frames: list[int],
        num_samples: int,
        seed: int,
        diffusion_steps: int,
        cfg_weight: list[float] | None = None,
        cfg_type: str | None = None,
        postprocess_parameters: dict | None = None,
        transitions_parameters: dict | None = None,
        real_robot_rotations: bool = False,
    ) -> None:
        """Route exact Tara movement prompts without invoking the Kimodo model."""
        actions = [base_prompt_action(prompt) for prompt in prompts]
        requested_return_rack = requested_base_return_rack_name(prompts)
        requested_base_rack = None if requested_return_rack is not None else requested_base_rack_name(prompts)
        is_primitive_base_prompt = bool(prompts) and len(prompts) == len(num_frames) and all(
            action is not None for action in actions
        )
        workflow = self.robot_workflows.get(client.client_id)
        automatic_base_segments: list[tuple[str, int]] | None = None
        automatic_base_route_summary: str | None = None
        automatic_base_start_pose: tuple[float, float, float] | None = None
        automatic_base_route_rack: str | None = None
        automatic_base_route_kind: str | None = None
        model_pick_base_pose: tuple[np.ndarray, np.ndarray] | None = None
        model_pick_target_hand_rotations: np.ndarray | None = None
        model_pick_target_palm_normals: np.ndarray | None = None
        model_pick_palm_normal_local: np.ndarray | None = None
        model_pick_hand_side: str = "right"
        model_dual_pick_specs: dict[str, dict[str, object]] | None = None
        model_pick_existing_objects_by_hand: dict[str, tuple[str, int, int]] = {}
        model_pick_existing_offsets_by_hand: dict[str, np.ndarray] = {}
        model_return_hold_rotations: np.ndarray | None = None
        model_return_picked_object: tuple[str, int, int] | None = None
        model_return_hand_offset: np.ndarray | None = None
        model_return_hand_side: str = "right"
        model_return_picked_objects_by_hand: dict[str, tuple[str, int, int]] | None = None
        model_return_hand_offsets_by_side: dict[str, np.ndarray] | None = None
        model_place_base_pose: tuple[np.ndarray, np.ndarray] | None = None
        model_place_release_frame: int | None = None
        model_place_target_hand_rotations: np.ndarray | None = None
        model_place_hand_side: str = "right"
        model_place_specs: dict[str, dict[str, object]] | None = None
        rack_transfer_route: RackTransferRoute | None = None
        requested_rack_transfer: RackTransferRequest | None = None
        model_transfer_hold_rotations: np.ndarray | None = None
        model_transfer_picked_object: tuple[str, int, int] | None = None
        model_transfer_hand_offset: np.ndarray | None = None
        model_transfer_hand_side: str = "right"
        model_transfer_picked_objects_by_hand: dict[str, tuple[str, int, int]] | None = None
        model_transfer_hand_offsets_by_side: dict[str, np.ndarray] | None = None
        requested_rack_transfer_holding_pick = False
        requested_human_return_holding_pick = False

        def strengthen_neutral_walk_guidance() -> None:
            nonlocal cfg_weight
            weights = list(cfg_weight or [2.0, 2.0])
            if len(weights) == 1:
                weights[0] = max(float(weights[0]), 3.5)
            else:
                weights[0] = max(float(weights[0]), 3.5)
                weights[1] = max(float(weights[1]), 3.0)
            cfg_weight = weights

        def append_route_turn(
            segments: list[tuple[str, int]],
            labels: list[str],
            turn_delta: float,
            fps: float,
        ) -> None:
            turn_degrees = int(round(abs(np.rad2deg(turn_delta))))
            if turn_degrees not in (90, 180):
                raise ValueError(f"Unsupported TaraBase route turn: {turn_degrees} degrees")
            if turn_degrees == 90:
                user_side = "left" if turn_delta > 0.0 else "right"
            else:
                user_side = "left" if turn_delta < 0.0 else "right"
            action = base_prompt_action(f"turn {user_side} by {turn_degrees} degrees")
            if action is None:
                raise AssertionError("Generated TaraBase turn prompt was not recognized")
            frames = max(
                1,
                int(round(DEFAULT_TURN_SECONDS_PER_90_DEG * (turn_degrees / 90.0) * fps)),
            )
            segments.append((action, frames))
            labels.append(f"{user_side} {turn_degrees}deg")

        if requested_return_rack is not None:
            if workflow is None or workflow.wheel_base is None or workflow.wheel_base.length == 0:
                raise RuntimeError("Generate and preview the outbound base-to-rack motion before returning.")
            if workflow.base_route_rack != requested_return_rack or workflow.base_route_kind != "outbound":
                raise RuntimeError(
                    f"The loaded base motion is not the outbound route to {requested_return_rack.replace('_', ' ')}."
                )
            session = self.client_sessions[client.client_id]
            prompt_fps = float(session.model_fps)
            x_near = WORK_AREA_SIDE_SHIFT_M
            x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
            z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
            rack_target, rack_facing_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[requested_return_rack],
                RACK_MAP_YAWS_RAD.get(requested_return_rack, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            previous_motion = workflow.wheel_base.motion
            automatic_base_start_pose = (
                float(previous_motion["x"][-1]),
                float(previous_motion["z"][-1]),
                float(previous_motion["yaw"][-1]),
            )

            heading_vector = (np.sin(rack_facing_heading), np.cos(rack_facing_heading))
            backed_position = (
                rack_target[0] - 0.10 * heading_vector[0],
                0.0,
                rack_target[2] - 0.10 * heading_vector[1],
            )
            obstacles: list[tuple[float, float, float, float]] = []
            for rack_name, rack_center in RACK_MAP_POSITIONS.items():
                rack_yaw = RACK_MAP_YAWS_RAD.get(rack_name, 0.0)
                half_x = abs(np.cos(rack_yaw)) * RACK_WIDTH_M / 2.0 + abs(np.sin(rack_yaw)) * RACK_DEPTH_M / 2.0
                half_z = abs(np.sin(rack_yaw)) * RACK_WIDTH_M / 2.0 + abs(np.cos(rack_yaw)) * RACK_DEPTH_M / 2.0
                obstacles.append((float(rack_center[0]), float(rack_center[2]), half_x, half_z))
            safer_side = choose_safer_turn_side(
                backed_position,
                rack_facing_heading,
                (x_far, x_near),
                (z_far, 0.0),
                obstacles,
            )

            automatic_base_segments = []
            primitive_labels = ["backward 0.10m"]
            automatic_base_segments.append(
                ("backward", max(1, int(round(0.10 / DEFAULT_LINEAR_SPEED_M_S * prompt_fps))))
            )
            ninety_action = base_prompt_action(f"turn {safer_side} by 90 degrees")
            if ninety_action is None:
                raise AssertionError("Generated safe turnaround prompt was not recognized")
            ninety_frames = max(1, int(round(DEFAULT_TURN_SECONDS_PER_90_DEG * prompt_fps)))
            automatic_base_segments.extend([(ninety_action, ninety_frames), (ninety_action, ninety_frames)])
            primitive_labels.extend([f"{safer_side} 90deg", f"{safer_side} 90deg"])

            current_heading = (rack_facing_heading + 2.0 * np.pi) % (2.0 * np.pi) - np.pi
            current_x, current_z = backed_position[0], backed_position[2]
            if not np.isclose(current_x, 0.0):
                desired_heading = np.pi / 2.0 if current_x < 0.0 else -np.pi / 2.0
                turn_delta = (desired_heading - current_heading + np.pi) % (2.0 * np.pi) - np.pi
                if not np.isclose(turn_delta, 0.0):
                    append_route_turn(automatic_base_segments, primitive_labels, turn_delta, prompt_fps)
                distance = abs(current_x)
                automatic_base_segments.append(
                    ("forward", max(1, int(round(distance / DEFAULT_LINEAR_SPEED_M_S * prompt_fps))))
                )
                primitive_labels.append(f"forward {distance:.2f}m")
                current_heading = desired_heading
                current_x = 0.0
            if not np.isclose(current_z, 0.0):
                desired_heading = 0.0 if current_z < 0.0 else np.pi
                turn_delta = (desired_heading - current_heading + np.pi) % (2.0 * np.pi) - np.pi
                if not np.isclose(turn_delta, 0.0):
                    append_route_turn(automatic_base_segments, primitive_labels, turn_delta, prompt_fps)
                distance = abs(current_z)
                automatic_base_segments.append(
                    ("forward", max(1, int(round(distance / DEFAULT_LINEAR_SPEED_M_S * prompt_fps))))
                )
                primitive_labels.append(f"forward {distance:.2f}m")
            automatic_base_route_summary = ", ".join(primitive_labels)
            automatic_base_route_rack = requested_return_rack
            automatic_base_route_kind = "return"

        if requested_base_rack is not None:
            if requested_base_rack not in RACK_MAP_POSITIONS:
                raise ValueError(f"Unknown warehouse rack {requested_base_rack.replace('_', ' ')}")
            session = self.client_sessions[client.client_id]
            prompt_fps = float(session.model_fps)
            x_near = WORK_AREA_SIDE_SHIFT_M
            x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
            z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
            rack_target, rack_facing_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[requested_base_rack],
                RACK_MAP_YAWS_RAD.get(requested_base_rack, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            # TaraBase always enters the work area along Z before making any
            # cross-aisle move. Rack 3 therefore never travels along z=0.
            route = plan_cardinal_rack_route(
                approach_position=rack_target,
                final_heading=rack_facing_heading,
                total_frames=max(600, int(num_frames[0]) if num_frames else 600),
                fps=prompt_fps,
                first_axis="z",
            )
            for position in route.positions:
                if not (x_far <= position[0] <= x_near and z_far <= position[2] <= 0.0):
                    raise ValueError(f"Planned TaraBase route leaves the orange boundary at {position}")

            automatic_base_segments = []
            primitive_labels: list[str] = []
            for primitive in cardinal_route_primitives(route):
                if primitive.kind == "forward":
                    frames = max(1, int(round(primitive.value / DEFAULT_LINEAR_SPEED_M_S * prompt_fps)))
                    automatic_base_segments.append(("forward", frames))
                    primitive_labels.append(f"forward {primitive.value:.2f}m")
                    continue

                turn_degrees = int(round(abs(np.rad2deg(primitive.value))))
                if turn_degrees not in (90, 180):
                    raise ValueError(f"Unsupported TaraBase route turn: {turn_degrees} degrees")
                # Tara's calibrated prompt/visual convention swaps 90-degree
                # direction names; 180-degree turns use the hardware sign.
                if turn_degrees == 90:
                    user_side = "left" if primitive.value > 0.0 else "right"
                else:
                    user_side = "left" if primitive.value < 0.0 else "right"
                action = base_prompt_action(f"turn {user_side} by {turn_degrees} degrees")
                if action is None:
                    raise AssertionError("Generated TaraBase turn prompt was not recognized")
                frames = max(
                    1,
                    int(round(DEFAULT_TURN_SECONDS_PER_90_DEG * (turn_degrees / 90.0) * prompt_fps)),
                )
                automatic_base_segments.append((action, frames))
                primitive_labels.append(f"{user_side} {turn_degrees}deg")
            automatic_base_route_summary = ", ".join(primitive_labels)
            automatic_base_route_rack = requested_base_rack
            automatic_base_route_kind = "outbound"

        is_base_only_prompt = is_primitive_base_prompt or automatic_base_segments is not None

        if is_base_only_prompt:
            if workflow is None:
                raise RuntimeError("TaraBase workflow is unavailable for this client.")
            segments = automatic_base_segments or [
                (str(action), int(frames)) for action, frames in zip(actions, num_frames)
            ]
            if any(frames <= 0 for _, frames in segments):
                raise ValueError("Base movement prompts must contain at least one frame.")
            session = self.client_sessions[client.client_id]
            prompt_fps = float(session.model_fps)
            temp_csv = Path("/tmp") / f"kimodo_tara_prompt_{client.client_id}.csv"
            start_pose: tuple[float, float, float] | None = None
            continuation_source: str | None = None
            if automatic_base_start_pose is not None:
                start_pose = automatic_base_start_pose
                continuation_source = f"loaded outbound {automatic_base_route_rack} endpoint"
            elif workflow.continue_base_from_last_pose:
                requested_csv = workflow.continuation_base_csv_reference.strip()
                if requested_csv:
                    continuation_path = _resolve_base_csv_reference(
                        requested_csv,
                        workflow.output_root,
                    )
                    previous_motion = _read_diff_drive_csv(
                        continuation_path,
                        _csv_frame_rate(continuation_path, prompt_fps),
                    )
                    continuation_source = str(continuation_path)
                elif (
                    workflow.wheel_base is not None
                    and workflow.wheel_base.csv_path is not None
                    and workflow.wheel_base.length > 0
                ):
                    previous_motion = workflow.wheel_base.motion
                    continuation_source = str(workflow.wheel_base.csv_path)
                else:
                    raise ValueError(
                        "Continuation is checked, but no source CSV was specified or loaded."
                    )
                start_pose = (
                    float(previous_motion["x"][-1]),
                    float(previous_motion["z"][-1]),
                    float(previous_motion["yaw"][-1]),
                )
            start_pose_kwargs = (
                {
                    "initial_x_m": start_pose[0],
                    "initial_z_m": start_pose[1],
                    "initial_yaw_rad": start_pose[2],
                }
                if start_pose is not None
                else {}
            )
            write_prompt_base_csv(
                temp_csv,
                segments,
                fps=prompt_fps,
                **start_pose_kwargs,
            )

            position_offset = (
                workflow.wheel_base.position_offset.copy()
                if workflow.wheel_base is not None
                else np.zeros(3, dtype=np.float64)
            )
            if workflow.wheel_base is not None:
                workflow.wheel_base.clear()
            workflow.wheel_base = WheelBasePlayback(
                client,
                DEFAULT_WHEEL_URDF_PATH,
                temp_csv,
                prompt_fps,
                root_node_name=f"/tara_wheel_base_{client.client_id}",
                position_offset=position_offset,
                visual_yaw_offset_rad=float(np.pi),
            )
            workflow.wheel_base.set_visible(True)
            workflow.wheel_csv_path = temp_csv
            workflow.base_wheel_rpms = _load_base_wheel_rpms(temp_csv)
            workflow.arm_frames.clear()
            workflow.gripper_widths.clear()
            workflow.base_prompt_segments = segments
            workflow.base_prompt_fps = prompt_fps
            workflow.base_prompt_stem = f"tara_{uuid.uuid4().hex[:12]}"
            workflow.base_prompt_start_pose = start_pose
            workflow.base_route_rack = automatic_base_route_rack
            workflow.base_route_kind = automatic_base_route_kind
            session.cur_duration = sum(frames for _, frames in segments) / float(session.model_fps)
            session.max_frame_idx = sum(frames for _, frames in segments) - 1
            if workflow.status_markdown is not None:
                direction_labels = [action.replace("_", " ") for action, _ in segments]
                workflow.status_markdown.content = (
                    "Dedicated TaraBase motion generated (Kimodo model bypassed).\n\n"
                    f"Segments: `{', '.join(direction_labels)}`\n\n"
                    + (
                        f"Rack route: `{automatic_base_route_summary}`\n\n"
                        if automatic_base_route_summary is not None
                        else ""
                    )
                    + f"Frames: `{session.max_frame_idx + 1}` at `{prompt_fps:g} FPS`\n\n"
                    + (
                        f"Continues from `{continuation_source}`\n\n"
                        f"X `{start_pose[0]:.3f}`, Z `{start_pose[1]:.3f}`, "
                        f"yaw `{np.rad2deg(start_pose[2]):.1f}°`; new frame/time starts at `0`.\n\n"
                        if start_pose is not None
                        else "Starts from the origin; new frame/time starts at `0`.\n\n"
                    )
                    + "Click **Save Base CSV** to persist the frame-by-frame commands."
                )
            self.set_frame(client.client_id, 0)
            return

        requested_rack_transfer = _requested_rack_transfer(prompts)
        requested_dual_pick = (
            None if requested_rack_transfer is not None else _requested_dual_rack_pick(prompts)
        )
        requested_human_pick = (
            None
            if requested_rack_transfer is not None or requested_dual_pick is not None
            else requested_rack_pick(prompts)
        )
        requested_human_place = (
            False
            if requested_dual_pick is not None or requested_human_pick is not None
            else _requested_counter_place(prompts)
        )
        requested_human_return = (
            None
            if requested_rack_transfer is not None or requested_dual_pick is not None or requested_human_pick is not None or requested_human_place
            else requested_human_return_rack_name(prompts)
        )
        requested_rack = (
            None
            if (
                requested_rack_transfer is not None
                or requested_dual_pick is not None
                or requested_human_pick is not None
                or requested_human_place
                or requested_human_return is not None
            )
            else requested_rack_name(prompts)
        )
        if requested_dual_pick is not None or requested_human_pick is not None:
            session = self.client_sessions[client.client_id]
            is_dual_pick = requested_dual_pick is not None
            pick_rack = requested_dual_pick.rack_name if is_dual_pick else requested_human_pick.rack_name
            shelf_number = requested_dual_pick.shelf_number if is_dual_pick else requested_human_pick.shelf_number
            pick_object_indices = requested_dual_pick.object_indices if is_dual_pick else (requested_human_pick.object_index,)
            if pick_rack not in RACK_MAP_POSITIONS:
                raise ValueError(f"Unknown warehouse rack {pick_rack.replace('_', ' ')}")
            x_near = WORK_AREA_SIDE_SHIFT_M
            x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
            z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
            rack_target, rack_facing_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[pick_rack],
                RACK_MAP_YAWS_RAD.get(pick_rack, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            cached_outbound_pose = session.human_outbound_rack_poses.get(pick_rack)
            if (
                workflow is not None
                and workflow.pick_grasp_verified
                and session.human_route_rack == pick_rack
                and session.human_route_kind == "pick_hold"
                and session.motions
            ):
                current_motion = next(iter(session.motions.values()))
                current_root = (
                    current_motion.joints_pos[-1, current_motion.skeleton.root_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if float(np.linalg.norm(current_root[[0, 2]] - np.asarray(rack_target)[[0, 2]])) <= 0.35:
                    cached_outbound_pose = (
                        current_motion.joints_pos[-1].detach().cpu().numpy().copy(),
                        current_motion.joints_rot[-1].detach().cpu().numpy().copy(),
                    )
            if (
                cached_outbound_pose is None
                and session.human_route_rack == pick_rack
                and session.human_route_kind == "outbound"
                and session.motions
            ):
                outbound_motion = next(iter(session.motions.values()))
                cached_outbound_pose = (
                    outbound_motion.joints_pos[-1].detach().cpu().numpy().copy(),
                    outbound_motion.joints_rot[-1].detach().cpu().numpy().copy(),
                )
                session.human_outbound_rack_poses[pick_rack] = cached_outbound_pose
            if cached_outbound_pose is None and session.motions:
                current_motion = next(iter(session.motions.values()))
                current_root = (
                    current_motion.joints_pos[-1, current_motion.skeleton.root_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if float(np.linalg.norm(current_root[[0, 2]] - np.asarray(rack_target)[[0, 2]])) <= 0.35:
                    cached_outbound_pose = (
                        current_motion.joints_pos[-1].detach().cpu().numpy().copy(),
                        current_motion.joints_rot[-1].detach().cpu().numpy().copy(),
                    )
                    session.human_outbound_rack_poses[pick_rack] = cached_outbound_pose
                    session.human_route_rack = pick_rack
                    session.human_route_kind = "outbound"
            if cached_outbound_pose is None:
                raise RuntimeError(
                    f"Generate `move to {pick_rack.replace('_', ' ')}` before picking its object."
                )
            if len(num_frames) != 1 or num_frames[0] < int(round(4.0 * session.model_fps)):
                raise ValueError("A shelf-pick motion needs one segment of at least 4 seconds.")
            if not session.motions:
                raise RuntimeError("The outbound human motion is not available for pick initialization.")
            if workflow is None:
                raise RuntimeError("The robot workflow state is unavailable.")
            pick_object_positions: dict[int, np.ndarray] = {}
            for object_index in pick_object_indices:
                object_key = (pick_rack, shelf_number, object_index)
                object_world_position = workflow.rack_object_home_positions.get(object_key)
                if object_world_position is None:
                    raise RuntimeError(
                        f"Shelf {shelf_number} object {object_index} on {pick_rack} is unavailable."
                    )
                pick_object_positions[object_index] = np.asarray(object_world_position, dtype=np.float64)

            total_frames = int(num_frames[0])
            last_positions_world, last_rotations = cached_outbound_pose
            last_positions_world = np.asarray(last_positions_world, dtype=np.float64).copy()
            last_rotations = np.asarray(last_rotations, dtype=np.float64).copy()
            model_pick_base_pose = (last_positions_world, last_rotations)
            world_offset = np.asarray(rack_target, dtype=np.float64)

            root_constraint = session.constraints.get("2D Root")
            fullbody_constraint = session.constraints.get("Full-Body")
            end_effector_constraint = session.constraints.get("End-Effectors")
            if root_constraint is None or fullbody_constraint is None or end_effector_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide all pick constraints.")
            for constraint in (root_constraint, fullbody_constraint, end_effector_constraint):
                constraint.clear()

            stationary_root = np.repeat(
                np.asarray(rack_target, dtype=np.float64)[None, :],
                total_frames,
                axis=0,
            )
            root_constraint.set_smooth_path(False)
            root_constraint.add_interval(
                f"auto_{pick_rack}_pick_root",
                0,
                total_frames - 1,
                stationary_root,
            )
            root_constraint.set_dense_path(True)

            initial_pose_frames = min(5, total_frames)
            fullbody_constraint.add_interval(
                f"auto_{pick_rack}_pick_start_pose",
                0,
                initial_pose_frames - 1,
                np.repeat(last_positions_world[None, ...], initial_pose_frames, axis=0),
                np.repeat(last_rotations[None, ...], initial_pose_frames, axis=0),
            )

            reach_start_frame = 0
            hand_positions = np.repeat(last_positions_world[None, ...], total_frames, axis=0)
            hand_rotations = np.repeat(last_rotations[None, ...], total_frames, axis=0)
            last_frame = total_frames - 1
            pregrasp_frame = max(1, int(round(last_frame * 0.38)))
            grasp_frame = max(pregrasp_frame + 1, int(round(last_frame * 0.52)))
            approach_frame = max(
                pregrasp_frame + 1,
                int(round((pregrasp_frame + grasp_frame) * 0.5)),
            )
            approach_frame = min(approach_frame, grasp_frame - 1)
            lift_frame = max(grasp_frame + 1, int(round(last_frame * 0.64)))
            chest_frame = max(lift_frame + 1, int(round(last_frame * 0.84)))
            chest_frame = min(chest_frame, last_frame)
            rack_yaw = RACK_MAP_YAWS_RAD.get(pick_rack, 0.0)
            outward_normal = np.asarray(
                [np.cos(rack_yaw), 0.0, -np.sin(rack_yaw)],
                dtype=np.float64,
            )
            chest_index = session.skeleton.bone_order_names.index("Chest")
            right_shoulder_index = session.skeleton.bone_order_names.index("RightShoulder")
            left_shoulder_index = session.skeleton.bone_order_names.index("LeftShoulder")
            right_lateral = (
                last_positions_world[right_shoulder_index]
                - last_positions_world[left_shoulder_index]
            )
            right_lateral[1] = 0.0
            right_lateral /= max(float(np.linalg.norm(right_lateral)), 1e-9)
            # Direct pick path:
            #   current hand -> same-side diagonal pregrasp -> grasp -> 5 cm lift -> chest hold.
            # The pregrasp is offset 7 cm forward from the grasp point, 12 cm
            # toward the active hand side, and 5 cm higher for shelf/rack
            # clearance. This lets the hand approach like an open gripper
            # coming from the side, instead of pushing the item from the front.
            approach_clearance = 0.11
            pregrasp_clearance = 0.07
            lateral_clearance = 0.12
            pick_sides_and_objects = (
                (("right", 1), ("left", 3))
                if is_dual_pick
                else (("left" if pick_object_indices[0] == 3 else "right", pick_object_indices[0]),)
            )
            pick_specs: dict[str, dict[str, object]] = {}
            for pick_hand_side, object_index in pick_sides_and_objects:
                hand_label = "Left" if pick_hand_side == "left" else "Right"
                active_hand_names = (
                    session.skeleton.left_hand_joint_names
                    if pick_hand_side == "left"
                    else session.skeleton.right_hand_joint_names
                )
                active_hand_indices = [
                    session.skeleton.bone_order_names.index(name) for name in active_hand_names
                ]
                hand_root_index = active_hand_indices[0]
                middle_name = (
                    f"{hand_label}HandMiddleEnd"
                    if f"{hand_label}HandMiddleEnd" in session.skeleton.bone_order_names
                    else active_hand_names[-1]
                )
                thumb_name = (
                    f"{hand_label}HandThumbEnd"
                    if f"{hand_label}HandThumbEnd" in session.skeleton.bone_order_names
                    else None
                )
                middle_index = session.skeleton.bone_order_names.index(middle_name)
                thumb_index = (
                    session.skeleton.bone_order_names.index(thumb_name)
                    if thumb_name is not None
                    else None
                )
                active_lateral = right_lateral if pick_hand_side == "right" else -right_lateral
                object_position = pick_object_positions[object_index]
                grasp_target = object_position + approach_clearance * outward_normal + np.array([0.0, -0.025, 0.0])
                pregrasp_target = (
                    grasp_target
                    + pregrasp_clearance * outward_normal
                    + lateral_clearance * active_lateral
                    + np.array([0.0, 0.05, 0.0])
                )
                diagonal_approach_target = (
                    grasp_target
                    + 0.5 * pregrasp_clearance * outward_normal
                    + 0.5 * lateral_clearance * active_lateral
                    + np.array([0.0, 0.025, 0.0])
                )
                lift_target = grasp_target + np.array([0.0, 0.05, 0.0])
                chest_target = (
                    last_positions_world[chest_index]
                    - 0.24 * outward_normal
                    + 0.18 * active_lateral
                    + np.array([0.0, -0.10, 0.0])
                )
                hand_target_path = _smooth_waypoint_path(
                    total_frames,
                    [
                        (0, last_positions_world[hand_root_index]),
                        (pregrasp_frame, pregrasp_target),
                        (approach_frame, diagonal_approach_target),
                        (grasp_frame, grasp_target),
                        (lift_frame, lift_target),
                        (chest_frame, chest_target),
                        (total_frames - 1, chest_target),
                    ],
                )
                base_finger_direction = (
                    last_positions_world[middle_index] - last_positions_world[hand_root_index]
                )
                if thumb_index is not None:
                    base_thumb_direction = (
                        last_positions_world[thumb_index] - last_positions_world[hand_root_index]
                    )
                    base_palm_normal_world = (
                        np.cross(base_finger_direction, base_thumb_direction)
                        if pick_hand_side == "left"
                        else np.cross(base_thumb_direction, base_finger_direction)
                    )
                else:
                    base_palm_normal_world = last_rotations[hand_root_index][:, 1]
                if float(np.linalg.norm(base_palm_normal_world)) < 1e-6:
                    base_palm_normal_world = last_rotations[hand_root_index][:, 1]
                base_palm_normal_world /= max(float(np.linalg.norm(base_palm_normal_world)), 1e-9)
                if float(np.dot(base_palm_normal_world, np.array([0.0, 1.0, 0.0]))) < 0.0:
                    base_palm_normal_world = -base_palm_normal_world
                desired_palm_normal_world = np.array(
                    [0.0, -1.0 if pick_hand_side == "left" else 1.0, 0.0],
                    dtype=np.float64,
                )
                target_palm_normals = _smooth_waypoint_path(
                    total_frames,
                    [
                        (0, base_palm_normal_world),
                        (pregrasp_frame, desired_palm_normal_world),
                        (approach_frame, desired_palm_normal_world),
                        (grasp_frame, desired_palm_normal_world),
                        (lift_frame, desired_palm_normal_world),
                        (chest_frame, desired_palm_normal_world),
                        (total_frames - 1, desired_palm_normal_world),
                    ],
                )
                target_palm_norms = np.linalg.norm(target_palm_normals, axis=1, keepdims=True)
                target_palm_normals = target_palm_normals / np.maximum(target_palm_norms, 1e-9)
                rack_center = np.asarray(RACK_MAP_POSITIONS[pick_rack], dtype=np.float64)
                front_clearance = (hand_target_path - rack_center[None, :]) @ outward_normal
                minimum_front_clearance = float(RACK_WIDTH_M / 2.0 + 0.045)
                actual_front_clearance = float(front_clearance.min())
                if actual_front_clearance < minimum_front_clearance:
                    raise RuntimeError(
                        "The planned hand path does not clear the shelf front by 4.5 cm "
                        f"(needed {minimum_front_clearance:.3f} m from rack center, "
                        f"got {actual_front_clearance:.3f} m)."
                    )
                for frame in range(total_frames):
                    hand_positions[frame, active_hand_indices] += (
                        hand_target_path[frame] - last_positions_world[hand_root_index]
                    )
                pick_specs[pick_hand_side] = {
                    "object_index": object_index,
                    "object_key": (pick_rack, shelf_number, object_index),
                    "hand_target_path": hand_target_path,
                    "hand_target_world": chest_target,
                    "grasp_target_world": grasp_target,
                    "palm_normal_local": last_rotations[hand_root_index].T @ base_palm_normal_world,
                    "target_palm_normals": target_palm_normals,
                    "elbow_bend_hint": active_lateral + np.array([0.0, -0.35, 0.0]),
                    "hand_root_index": hand_root_index,
                    "hand_start_position": last_positions_world[hand_root_index],
                    "middle_index": middle_index,
                    "object_position": object_position,
                    "pregrasp_target": pregrasp_target,
                    "diagonal_approach_target": diagonal_approach_target,
                    "lift_target": lift_target,
                    "chest_target": chest_target,
                    "active_lateral": active_lateral,
                    "correction_start_frame": pregrasp_frame,
                }
            primary_side = "right" if "right" in pick_specs else next(iter(pick_specs))
            primary_spec = pick_specs[primary_side]
            object_index = int(primary_spec["object_index"])
            object_key = primary_spec["object_key"]
            object_position = primary_spec["object_position"]
            hand_target_path = primary_spec["hand_target_path"]
            grasp_target = primary_spec["grasp_target_world"]
            pregrasp_target = primary_spec["pregrasp_target"]
            lift_target = primary_spec["lift_target"]
            chest_target = primary_spec["chest_target"]
            model_pick_hand_side = primary_side
            model_pick_palm_normal_local = primary_spec["palm_normal_local"]
            model_pick_target_palm_normals = primary_spec["target_palm_normals"]
            model_pick_target_hand_rotations = None
            model_dual_pick_specs = pick_specs if is_dual_pick else None
            if not is_dual_pick and workflow.pick_grasp_verified:
                model_pick_existing_objects_by_hand = dict(workflow.picked_rack_objects_by_hand)
                model_pick_existing_offsets_by_hand = {
                    side: np.asarray(offset, dtype=np.float64).copy()
                    for side, offset in workflow.pick_object_hand_offsets_by_side.items()
                }
                if (
                    workflow.picked_rack_object is not None
                    and workflow.pick_object_hand_offset is not None
                ):
                    model_pick_existing_objects_by_hand.setdefault(
                        workflow.pick_hand_side,
                        workflow.picked_rack_object,
                    )
                    model_pick_existing_offsets_by_hand.setdefault(
                        workflow.pick_hand_side,
                        np.asarray(workflow.pick_object_hand_offset, dtype=np.float64).copy(),
                    )
                model_pick_existing_objects_by_hand.pop(primary_side, None)
                model_pick_existing_offsets_by_hand.pop(primary_side, None)
            rack_center = np.asarray(RACK_MAP_POSITIONS[pick_rack], dtype=np.float64)
            pick_constraint_debug_path = _save_pick_constraint_debug_json(
                rack_name=pick_rack,
                object_index=object_index,
                shelf_number=shelf_number,
                total_frames=total_frames,
                fps=float(session.model_fps),
                rack_target=np.asarray(rack_target, dtype=np.float64),
                rack_facing_heading=float(rack_facing_heading),
                object_position=object_position,
                outward_normal=outward_normal,
                pregrasp_frame=pregrasp_frame,
                grasp_frame=grasp_frame,
                lift_frame=lift_frame,
                chest_frame=chest_frame,
                pregrasp_target=pregrasp_target,
                grasp_target=grasp_target,
                lift_target=lift_target,
                chest_target=chest_target,
                root_path=stationary_root,
                right_hand_path=hand_target_path,
                right_hand_start_position=last_positions_world[int(primary_spec["hand_root_index"])],
                hand_specs=pick_specs,
            )
            end_effector_constraint.add_interval(
                f"auto_{pick_rack}_object_{object_index}_pick_limbs",
                0,
                total_frames - 1,
                hand_positions,
                hand_rotations,
                ["LeftHand", "LeftFoot", "RightFoot", "RightHand"],
                {"left-hand", "left-foot", "right-foot", "right-hand"},
            )

            session.first_heading_angle = rack_facing_heading
            session.constrained_root_heading_angle = None
            session.constrained_root_initial_turn_angle = None
            session.constrained_root_turn_end_frame = None
            session.constrained_root_headings = [rack_facing_heading] * total_frames
            session.generation_world_offset = tuple(float(value) for value in rack_target)
            session.hide_constraint_overlays = True
            session.human_route_rack = pick_rack
            session.human_route_kind = "pick_hold"
            for key, handle in workflow.rack_object_handles.items():
                home = workflow.rack_object_home_positions[key]
                handle.position = home
            if is_dual_pick:
                workflow.picked_rack_object = None
                workflow.picked_rack_objects_by_hand = {
                    side: spec["object_key"] for side, spec in pick_specs.items()
                }
                workflow.pick_hold_start_frames_by_side = {
                    side: grasp_frame for side in pick_specs
                }
            else:
                workflow.picked_rack_object = object_key
                workflow.picked_rack_objects_by_hand = dict(model_pick_existing_objects_by_hand)
                workflow.pick_hold_start_frames_by_side = {
                    side: 0 for side in model_pick_existing_objects_by_hand
                }
            workflow.pick_hand_side = primary_side
            workflow.pick_hold_start_frame = grasp_frame
            workflow.pick_object_hand_offset = None
            workflow.pick_object_hand_offsets_by_side = {
                side: offset.copy()
                for side, offset in model_pick_existing_offsets_by_hand.items()
            }
            workflow.pick_reach_start_frame = reach_start_frame
            workflow.pick_hand_target_world = chest_target
            workflow.pick_target_path_world = hand_target_path
            workflow.pick_grasp_target_world = grasp_target
            workflow.pick_grasp_verified = False
            workflow.pick_elbow_bend_hint_world = primary_spec["elbow_bend_hint"]
            workflow.place_release_frame = None
            workflow.place_object_position_world = None
            workflow.place_object_positions_by_hand.clear()
            workflow.place_reach_start_frame = None
            workflow.place_hand_target_world = None
            workflow.place_hand_targets_by_side.clear()
            workflow.place_target_path_world = None
            workflow.place_target_paths_by_side.clear()
            workflow.place_elbow_bend_hint_world = None
            workflow.place_elbow_bend_hints_by_side.clear()
            if is_dual_pick:
                pick_prompt_text = (
                    "An ordinary healthy person stands still, upright, and balanced. They clearly "
                    "first move both arms to close pregrasp poses for objects 1 and 3 at the same time, with the right hand "
                    "aligned to object 1 and the left hand aligned to object 3. From the pregrasp poses, both hands approach "
                    "the objects diagonally from the side, then grasp both objects, lift them vertically by 5 cm while keeping "
                    "the wrists straight, then bring both objects to mirrored holding poses in front of the chest. The hands "
                    "must not wander away from the objects or touch the "
                    "rack or shelf; each hand touches only its selected object. The body and feet remain still. If the "
                    "clip has extra time, finish the pick and stay frozen in the final holding pose without extra hand "
                    "or body movement."
                )
            else:
                pick_prompt_text = (
                    "An ordinary healthy person stands still, upright, and balanced. They clearly "
                    f"first move only the {primary_side} arm to a close pregrasp pose for the selected object. From the "
                    "pregrasp pose, the hand approaches the object diagonally from the side with the wrist straight in line "
                    f"with the forearm and the palm naturally aligned to support the object, grasps the object with the {primary_side} hand, "
                    "lifts it vertically by 5 cm while keeping the wrist straight, then brings it to a holding pose in front of the chest. The hand "
                    "must not wander away from the object or touch the rack or shelf; it touches only the selected "
                    f"object on shelf {shelf_number} of {pick_rack.replace('_', ' ')}. The body and feet remain still. "
                    "If the clip has extra time, finish the pick and stay frozen in the final holding pose without "
                    "extra hand or body movement."
                )
            prompts = [pick_prompt_text]
            strengthen_neutral_walk_guidance()
            cfg_weight = list(cfg_weight or [3.5, 3.0])
            if len(cfg_weight) > 1:
                cfg_weight[1] = max(float(cfg_weight[1]), 5.0)
            postprocess_parameters = dict(postprocess_parameters or {})
            postprocess_parameters["post_processing"] = True
            postprocess_parameters["root_margin"] = min(
                float(postprocess_parameters.get("root_margin", 0.04)),
                0.02,
            )
            client.add_notification(
                title=f"{pick_rack.replace('_', ' ').title()} Shelf {shelf_number} pick applied",
                body=(
                    (
                        f"Right hand grasps object 1 and left hand grasps object 3 at frame "
                        if is_dual_pick
                        else f"{primary_side.title()} hand approaches above the shelf, grasps object {object_index} at frame "
                    )
                    + f"{grasp_frame}, lifts it, and holds it at chest level.\n\n"
                    f"Saved constraint debug JSON:\n{pick_constraint_debug_path}"
                ),
                auto_close_seconds=7.0,
                color="green",
            )
            self._apply_constraint_overlay_visibility(session)
        elif requested_human_return is not None:
            session = self.client_sessions[client.client_id]
            x_near = WORK_AREA_SIDE_SHIFT_M
            x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
            z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
            rack_target, rack_facing_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[requested_human_return],
                RACK_MAP_YAWS_RAD.get(requested_human_return, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            if session.human_route_rack != requested_human_return or session.human_route_kind not in {
                "outbound",
                "pick_hold",
            }:
                if (
                    workflow is not None
                    and (
                        (
                            workflow.picked_rack_object is not None
                            and workflow.picked_rack_object[0] == requested_human_return
                            and workflow.pick_object_hand_offset is not None
                        )
                        or (
                            workflow.picked_rack_objects_by_hand
                            and all(
                                object_key[0] == requested_human_return
                                for object_key in workflow.picked_rack_objects_by_hand.values()
                            )
                            and workflow.pick_object_hand_offsets_by_side
                        )
                    )
                    and workflow.pick_grasp_verified
                ):
                    session.human_route_rack = requested_human_return
                    session.human_route_kind = "pick_hold"
                elif session.motions:
                    current_motion = next(iter(session.motions.values()))
                    current_root = (
                        current_motion.joints_pos[-1, current_motion.skeleton.root_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    if float(np.linalg.norm(current_root[[0, 2]] - np.asarray(rack_target)[[0, 2]])) <= 0.45:
                        session.human_route_rack = requested_human_return
                        session.human_route_kind = "outbound"
            if session.human_route_rack != requested_human_return or session.human_route_kind not in {
                "outbound",
                "pick_hold",
            }:
                raise RuntimeError(
                    f"Generate the outbound human motion to {requested_human_return.replace('_', ' ')} first, "
                    "or pick an item there before returning."
                )
            requested_human_return_holding_pick = session.human_route_kind == "pick_hold"
            if requested_human_return_holding_pick:
                if workflow is None:
                    raise RuntimeError("The robot workflow state is unavailable.")
                has_single_hold = (
                    workflow.picked_rack_object is not None
                    and workflow.pick_object_hand_offset is not None
                )
                has_dual_hold = (
                    bool(workflow.picked_rack_objects_by_hand)
                    and {"left", "right"} <= set(workflow.picked_rack_objects_by_hand)
                    and {"left", "right"} <= set(workflow.pick_object_hand_offsets_by_side)
                )
                if (not has_single_hold and not has_dual_hold) or not workflow.pick_grasp_verified:
                    raise RuntimeError("Pick an item successfully before generating a carry return.")
                if not session.motions:
                    raise RuntimeError("The pick-hold motion is not available for return initialization.")
                pick_motion = next(iter(session.motions.values()))
                model_return_hold_rotations = (
                    pick_motion.joints_rot[-1].detach().cpu().numpy().copy()
                )
                if has_dual_hold:
                    model_return_picked_object = None
                    model_return_hand_offset = None
                    model_return_picked_objects_by_hand = dict(workflow.picked_rack_objects_by_hand)
                    model_return_hand_offsets_by_side = {
                        side: np.asarray(offset, dtype=np.float64).copy()
                        for side, offset in workflow.pick_object_hand_offsets_by_side.items()
                    }
                    model_return_hand_side = "both"
                else:
                    model_return_picked_object = workflow.picked_rack_object
                    model_return_hand_offset = np.asarray(
                        workflow.pick_object_hand_offset,
                        dtype=np.float64,
                    ).copy()
                    model_return_picked_objects_by_hand = None
                    model_return_hand_offsets_by_side = None
                    model_return_hand_side = workflow.pick_hand_side
            if len(num_frames) != 1 or num_frames[0] < 2:
                raise ValueError("A human return prompt must contain one motion segment.")
            root_constraint = session.constraints.get("2D Root")
            fullbody_constraint = session.constraints.get("Full-Body")
            end_effector_constraint = session.constraints.get("End-Effectors")
            if root_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide a 2D Root constraint track.")
            if fullbody_constraint is None or end_effector_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide all return constraint tracks.")
            return_start_position = np.asarray(rack_target, dtype=np.float64)
            return_initial_positions: np.ndarray | None = None
            return_initial_rotations: np.ndarray | None = None
            if session.motions:
                source_motion = next(iter(session.motions.values()))
                source_root = (
                    source_motion.joints_pos[-1, session.skeleton.root_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if (
                    requested_human_return_holding_pick
                    or float(np.linalg.norm(source_root[[0, 2]] - np.asarray(rack_target)[[0, 2]])) <= 0.45
                ):
                    return_start_position = source_root
                    return_initial_positions = source_motion.joints_pos[-1].detach().cpu().numpy()
                    return_initial_rotations = source_motion.joints_rot[-1].detach().cpu().numpy()
            rack4_direct_counter_return = requested_human_return == "rack_4"
            human_reverse_distance = (
                0.12
                if requested_human_return_holding_pick
                else 0.0
            )
            human_reverse_seconds = (
                1.00
                if rack4_direct_counter_return and human_reverse_distance > 0.0
                else 0.50
            )
            heading_vector = (np.sin(rack_facing_heading), np.cos(rack_facing_heading))
            backed_position = (
                return_start_position[0] - human_reverse_distance * heading_vector[0],
                0.0,
                return_start_position[2] - human_reverse_distance * heading_vector[1],
            )
            obstacles = []
            for rack_name, rack_center in RACK_MAP_POSITIONS.items():
                rack_yaw = RACK_MAP_YAWS_RAD.get(rack_name, 0.0)
                half_x = abs(np.cos(rack_yaw)) * RACK_WIDTH_M / 2.0 + abs(np.sin(rack_yaw)) * RACK_DEPTH_M / 2.0
                half_z = abs(np.sin(rack_yaw)) * RACK_WIDTH_M / 2.0 + abs(np.cos(rack_yaw)) * RACK_DEPTH_M / 2.0
                obstacles.append((float(rack_center[0]), float(rack_center[2]), half_x, half_z))
            safer_side = choose_safer_turn_side(
                backed_position,
                rack_facing_heading,
                (x_far, x_near),
                (z_far, 0.0),
                obstacles,
            )
            turn_seconds_per_90 = 0.90 if requested_human_return_holding_pick else 1.10
            initial_hold_seconds = 0.50 if requested_human_return_holding_pick else 0.20
            final_hold_seconds = 0.50 if requested_human_return_holding_pick else 0.75
            return_walk_speed_m_s = (
                _rack_human_walk_speed_m_s(requested_human_return)
                if requested_human_return_holding_pick
                else 0.60
                if requested_human_return == "rack_1"
                else 0.90
            )
            return_first_axis = "z" if rack4_direct_counter_return else "x"
            return_target_position = (
                (float(return_start_position[0]), 0.0, 0.0)
                if rack4_direct_counter_return
                else (0.0, 0.0, 0.0)
            )
            turn_sign = 1.0 if safer_side == "left" else -1.0
            current_heading = (
                rack_facing_heading + turn_sign * np.pi + np.pi
            ) % (2.0 * np.pi) - np.pi
            extra_turn_seconds = 0.0
            current_x = float(backed_position[0])
            current_z = float(backed_position[2])
            target_x = float(return_target_position[0])
            target_z = float(return_target_position[2])
            for axis in ((0, 2) if return_first_axis == "x" else (2, 0)):
                if axis == 0:
                    if np.isclose(current_x, target_x):
                        continue
                    desired_heading = np.pi / 2.0 if target_x > current_x else -np.pi / 2.0
                    current_x = target_x
                else:
                    if np.isclose(current_z, target_z):
                        continue
                    desired_heading = 0.0 if target_z > current_z else np.pi
                    current_z = target_z
                turn_delta = (desired_heading - current_heading + np.pi) % (2.0 * np.pi) - np.pi
                extra_turn_seconds += turn_seconds_per_90 * abs(turn_delta) / (np.pi / 2.0)
                current_heading = desired_heading
            reverse_seconds = human_reverse_seconds if human_reverse_distance > 0.0 else 0.0
            minimum_duration_s = (
                initial_hold_seconds
                + reverse_seconds
                + 2.0 * turn_seconds_per_90
                + extra_turn_seconds
                + (
                    abs(backed_position[0] - target_x)
                    + abs(backed_position[2] - target_z)
                )
                / return_walk_speed_m_s
                + final_hold_seconds
            )
            requested_duration_s = int(num_frames[0]) / float(session.model_fps)
            if requested_duration_s < minimum_duration_s:
                num_frames = [int(np.ceil(minimum_duration_s * float(session.model_fps)))]
            route = plan_cardinal_return_route(
                start_position=return_start_position,
                start_heading=rack_facing_heading,
                turn_side=safer_side,
                total_frames=int(num_frames[0]),
                fps=float(session.model_fps),
                reverse_distance=human_reverse_distance,
                reverse_seconds=human_reverse_seconds,
                turn_seconds_per_90=turn_seconds_per_90,
                walk_speed_m_s=return_walk_speed_m_s,
                initial_hold_seconds=initial_hold_seconds,
                final_hold_seconds=final_hold_seconds,
                first_axis=return_first_axis,
                target_position=return_target_position,
            )
            if any(
                not (x_far <= position[0] <= x_near and z_far <= position[2] <= 0.0)
                for position in route.positions
            ):
                raise ValueError("The planned human return route leaves the orange boundary.")
            session.first_heading_angle = route.headings[0]
            session.constrained_root_heading_angle = None
            session.constrained_root_initial_turn_angle = None
            session.constrained_root_turn_end_frame = None
            session.constrained_root_headings = list(route.headings)
            session.generation_world_offset = None
            session.hide_constraint_overlays = True
            session.human_route_rack = requested_human_return
            session.human_route_kind = "carry_return" if requested_human_return_holding_pick else "return"
            for constraint in (root_constraint, fullbody_constraint, end_effector_constraint):
                constraint.clear()
            root_constraint.set_smooth_path(False)
            root_constraint.add_interval(
                f"auto_{requested_human_return}_human_return",
                0,
                int(num_frames[0]) - 1,
                np.asarray(route.positions, dtype=np.float64),
            )
            root_constraint.set_dense_path(True)
            if return_initial_positions is not None and return_initial_rotations is not None:
                initial_pose_frames = min(
                    max(5, int(round(initial_hold_seconds * float(session.model_fps)))),
                    int(num_frames[0]),
                )
                fullbody_constraint.add_interval(
                    f"auto_{requested_human_return}_return_start_pose",
                    0,
                    initial_pose_frames - 1,
                    np.repeat(return_initial_positions[None, ...], initial_pose_frames, axis=0),
                    np.repeat(return_initial_rotations[None, ...], initial_pose_frames, axis=0),
                )
            if requested_human_return_holding_pick:
                if model_return_hand_side == "both":
                    prompts = [
                        "An ordinary healthy person carefully carries two small objects, one in each hand, "
                        "held in mirrored positions in front of the chest. They stand upright and balanced, "
                        "keep both arms steady relative to the torso without swinging either object, make "
                        "controlled in-place turns with stable footwork, then walk at the same steady pace "
                        "back to the counter/origin. At the end they stand still at the counter/origin while "
                        "holding both objects in the same body-relative poses."
                    ]
                else:
                    held_hand_side = workflow.pick_hand_side if workflow is not None else "right"
                    other_hand_side = "left" if held_hand_side == "right" else "right"
                    prompts = [
                        f"An ordinary healthy person carefully carries a small object in the {held_hand_side} hand "
                        f"in front of the chest. They stand upright and balanced, keep the {held_hand_side} arm steady "
                        "relative to the torso without swinging it, make controlled in-place turns with stable "
                        f"footwork, then walk at the same steady pace back to the counter/origin. The {other_hand_side} arm "
                        "stays relaxed for balance, and at the end they stand still at the counter/origin."
                    ]
            else:
                prompts = [rack_return_model_prompt(requested_human_return)]
            strengthen_neutral_walk_guidance()
            postprocess_parameters = dict(postprocess_parameters or {})
            postprocess_parameters["post_processing"] = True
            postprocess_parameters["root_margin"] = min(
                float(postprocess_parameters.get("root_margin", 0.04)),
                0.02,
            )
            client.add_notification(
                title=f"{requested_human_return.replace('_', ' ').title()} human return applied",
                body=(
                    (
                        f"Reverse {human_reverse_distance:.2f} m over {human_reverse_seconds:.1f}s, "
                        if human_reverse_distance > 0.0
                        else ""
                    )
                    + f"turn {safer_side} twice by 90 degrees, then walk "
                    f"along straight aisles to the counter at {return_walk_speed_m_s:.2f} m/s."
                    + (
                        f" Duration extended to {int(num_frames[0]) / float(session.model_fps):.1f}s for stable turns."
                        if requested_duration_s < minimum_duration_s
                        else ""
                    )
                    + (
                        (
                            " Both picked items stay attached to both hands in the same body-relative hold poses."
                            if model_return_hand_side == "both"
                            else f" The picked item stays attached to the {model_return_hand_side} hand in the same body-relative hold pose."
                        )
                        if requested_human_return_holding_pick
                        else ""
                    )
                ),
                auto_close_seconds=7.0,
                color="green",
            )
            self._apply_constraint_overlay_visibility(session)
        elif requested_human_place:
            session = self.client_sessions[client.client_id]
            if workflow is None:
                raise RuntimeError("The robot workflow state is unavailable.")
            has_single_place_hold = (
                workflow.picked_rack_object is not None
                and workflow.pick_object_hand_offset is not None
            )
            has_dual_place_hold = (
                bool(workflow.picked_rack_objects_by_hand)
                and {"left", "right"} <= set(workflow.picked_rack_objects_by_hand)
                and {"left", "right"} <= set(workflow.pick_object_hand_offsets_by_side)
            )
            if (not has_single_place_hold and not has_dual_place_hold) or not workflow.pick_grasp_verified:
                raise RuntimeError("Pick an item and carry it back to the counter before placing it.")
            if session.human_route_kind != "carry_return":
                raise RuntimeError("Generate the carry return to the counter before placing the object.")
            if not session.motions:
                raise RuntimeError("The carry-return motion is not available for place initialization.")
            if len(num_frames) != 1:
                raise ValueError("A counter-place motion needs one segment.")
            total_frames = int(num_frames[0])
            if total_frames < 2:
                raise ValueError("A counter-place motion needs at least two frames.")
            source_motion = next(iter(session.motions.values()))
            last_positions_world = source_motion.joints_pos[-1].detach().cpu().numpy().copy()
            last_rotations = source_motion.joints_rot[-1].detach().cpu().numpy().copy()
            model_place_base_pose = (last_positions_world, last_rotations)

            root_constraint = session.constraints.get("2D Root")
            fullbody_constraint = session.constraints.get("Full-Body")
            end_effector_constraint = session.constraints.get("End-Effectors")
            if root_constraint is None or fullbody_constraint is None or end_effector_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide all place constraints.")
            for constraint in (root_constraint, fullbody_constraint, end_effector_constraint):
                constraint.clear()

            root_index = session.skeleton.root_idx
            stationary_root = np.repeat(
                last_positions_world[root_index][None, :],
                total_frames,
                axis=0,
            )
            root_constraint.set_smooth_path(False)
            root_constraint.add_interval(
                "auto_counter_place_root",
                0,
                total_frames - 1,
                stationary_root,
            )
            root_constraint.set_dense_path(True)

            initial_pose_frames = min(5, total_frames)
            fullbody_constraint.add_interval(
                "auto_counter_place_start_pose",
                0,
                initial_pose_frames - 1,
                np.repeat(last_positions_world[None, ...], initial_pose_frames, axis=0),
                np.repeat(last_rotations[None, ...], initial_pose_frames, axis=0),
            )

            right_shoulder_index = session.skeleton.bone_order_names.index("RightShoulder")
            left_shoulder_index = session.skeleton.bone_order_names.index("LeftShoulder")
            right_lateral = last_positions_world[right_shoulder_index] - last_positions_world[left_shoulder_index]
            right_lateral[1] = 0.0
            right_lateral /= max(float(np.linalg.norm(right_lateral)), 1e-9)
            tabletop_margin = RACK_PICK_OBJECT_SIZE_M / 2.0 + 0.02
            table_x_min = -COUNTER_TABLE_LENGTH_M / 2.0 + tabletop_margin
            table_x_max = COUNTER_TABLE_LENGTH_M / 2.0 - tabletop_margin
            table_z_min = COUNTER_TABLE_DISTANCE_FROM_ORIGIN_M + tabletop_margin
            table_z_max = (
                COUNTER_TABLE_DISTANCE_FROM_ORIGIN_M
                + COUNTER_TABLE_WIDTH_M
                - tabletop_margin
            )
            reach_start_frame = 0
            last_frame = total_frames - 1
            pre_release_frame = min(max(1, int(round(last_frame * 0.24))), last_frame)
            release_frame = min(
                max(pre_release_frame + 1, int(round(last_frame * 0.38))),
                last_frame,
            )
            frames_after_release = max(0, last_frame - release_frame)
            reserve_after_release_hold = 3 if total_frames >= 6 else 2
            release_hold_frames = min(
                max(1, int(round(float(session.model_fps)))),
                max(0, frames_after_release - reserve_after_release_hold),
            )
            release_hold_end_frame = min(
                last_frame,
                release_frame + release_hold_frames,
            )
            final_hold_start_frame = last_frame
            retreat_limit_frame = last_frame
            retreat_frame = min(
                max(release_hold_end_frame + 1, int(round(last_frame * 0.76))),
                retreat_limit_frame,
            )
            normal_frame = retreat_frame
            model_place_release_frame = release_frame

            if has_dual_place_hold:
                place_hands = {
                    side: (
                        workflow.picked_rack_objects_by_hand[side],
                        np.asarray(workflow.pick_object_hand_offsets_by_side[side], dtype=np.float64),
                    )
                    for side in ("right", "left")
                }
                model_place_hand_side = "both"
            else:
                place_hands = {
                    workflow.pick_hand_side: (
                        workflow.picked_rack_object,
                        np.asarray(workflow.pick_object_hand_offset, dtype=np.float64),
                    )
                }
                model_place_hand_side = workflow.pick_hand_side

            hand_positions = np.repeat(last_positions_world[None, ...], total_frames, axis=0)
            hand_rotations = np.repeat(last_rotations[None, ...], total_frames, axis=0)
            place_specs: dict[str, dict[str, object]] = {}
            backward_from_counter = np.array([0.0, 0.0, -1.0], dtype=np.float64)
            for place_side, (place_object_key, hand_offset_local) in place_hands.items():
                hand_label = "Left" if place_side == "left" else "Right"
                active_hand_names = (
                    session.skeleton.left_hand_joint_names
                    if place_side == "left"
                    else session.skeleton.right_hand_joint_names
                )
                active_hand_indices = [
                    session.skeleton.bone_order_names.index(name) for name in active_hand_names
                ]
                hand_root_index = active_hand_indices[0]
                middle_name = (
                    f"{hand_label}HandMiddleEnd"
                    if f"{hand_label}HandMiddleEnd" in session.skeleton.bone_order_names
                    else active_hand_names[-1]
                )
                middle_index = session.skeleton.bone_order_names.index(middle_name)
                active_lateral = right_lateral if place_side == "right" else -right_lateral
                wrist_start = last_positions_world[hand_root_index]
                middle_start = last_positions_world[middle_index]
                palm_start = 0.35 * wrist_start + 0.65 * middle_start
                palm_from_wrist = palm_start - wrist_start
                carried_object_start = (
                    palm_start
                    + last_rotations[hand_root_index] @ hand_offset_local
                )
                object_release_position = np.array(
                    [
                        float(np.clip(carried_object_start[0], table_x_min, table_x_max)),
                        COUNTER_TABLE_HEIGHT_M + RACK_PICK_OBJECT_SIZE_M / 2.0 + 0.005,
                        float(np.clip(carried_object_start[2], table_z_min, table_z_max)),
                    ],
                    dtype=np.float64,
                )
                if has_dual_place_hold:
                    object_release_position[0] = float(
                        np.clip(
                            object_release_position[0] + (0.07 if place_side == "right" else -0.07),
                            table_x_min,
                            table_x_max,
                        )
                    )
                release_target = (
                    object_release_position
                    - last_rotations[hand_root_index] @ hand_offset_local
                    - palm_from_wrist
                )
                pre_release_target = release_target + np.array([0.0, 0.02, 0.0]) + 0.02 * backward_from_counter
                retreat_target = (
                    release_target
                    + 0.14 * backward_from_counter
                    + 0.10 * active_lateral
                    + np.array([0.0, 0.06, 0.0])
                )
                retreat_target[2] = min(float(retreat_target[2]), table_z_min - 0.12)
                normal_target = retreat_target.copy()
                hand_target_path = _smooth_waypoint_path(
                    total_frames,
                    [
                        (0, last_positions_world[hand_root_index]),
                        (pre_release_frame, pre_release_target),
                        (release_frame, release_target),
                        (release_hold_end_frame, release_target),
                        (retreat_frame, retreat_target),
                        (total_frames - 1, normal_target),
                    ],
                )
                target_hand_rotations = np.full(
                    (total_frames, 3, 3),
                    np.nan,
                    dtype=np.float64,
                )
                target_hand_rotations[:] = last_rotations[hand_root_index]
                for frame in range(total_frames):
                    hand_positions[frame, active_hand_indices] += (
                        hand_target_path[frame] - last_positions_world[hand_root_index]
                    )
                place_specs[place_side] = {
                    "object_key": place_object_key,
                    "object_release_position": object_release_position,
                    "pre_release_target": pre_release_target,
                    "release_target": release_target,
                    "retreat_target": retreat_target,
                    "normal_target": normal_target,
                    "hand_target_path": hand_target_path,
                    "hand_target_world": normal_target,
                    "target_hand_rotations": target_hand_rotations,
                    "elbow_bend_hint": active_lateral + np.array([0.0, -0.35, -0.18]),
                    "hand_root_index": hand_root_index,
                    "hand_start_position": last_positions_world[hand_root_index],
                }
            primary_place_side = "right" if "right" in place_specs else next(iter(place_specs))
            primary_place_spec = place_specs[primary_place_side]
            model_place_specs = place_specs if has_dual_place_hold else None
            model_place_target_hand_rotations = primary_place_spec["target_hand_rotations"]
            place_constraint_debug_path = _save_counter_place_constraint_debug_json(
                object_key=primary_place_spec["object_key"],
                total_frames=total_frames,
                fps=float(session.model_fps),
                root_position=last_positions_world[root_index],
                pre_release_frame=pre_release_frame,
                release_frame=release_frame,
                release_hold_end_frame=release_hold_end_frame,
                retreat_frame=retreat_frame,
                normal_frame=normal_frame,
                object_release_position=primary_place_spec["object_release_position"],
                pre_release_target=primary_place_spec["pre_release_target"],
                release_target=primary_place_spec["release_target"],
                retreat_target=primary_place_spec["retreat_target"],
                normal_target=primary_place_spec["normal_target"],
                root_path=stationary_root,
                right_hand_path=primary_place_spec["hand_target_path"],
                right_hand_start_position=primary_place_spec["hand_start_position"],
            )
            end_effector_constraint.add_interval(
                "auto_counter_place_limbs",
                0,
                total_frames - 1,
                hand_positions,
                hand_rotations,
                ["LeftHand", "LeftFoot", "RightFoot", "RightHand"],
                {"left-hand", "left-foot", "right-foot", "right-hand"},
            )

            workflow.place_release_frame = release_frame
            workflow.place_object_position_world = primary_place_spec["object_release_position"]
            workflow.place_object_positions_by_hand = {
                side: np.asarray(spec["object_release_position"], dtype=np.float64).copy()
                for side, spec in place_specs.items()
            }
            workflow.place_reach_start_frame = reach_start_frame
            workflow.place_hand_target_world = primary_place_spec["hand_target_world"]
            workflow.place_hand_targets_by_side = {
                side: np.asarray(spec["hand_target_world"], dtype=np.float64).copy()
                for side, spec in place_specs.items()
            }
            workflow.place_target_path_world = primary_place_spec["hand_target_path"]
            workflow.place_target_paths_by_side = {
                side: np.asarray(spec["hand_target_path"], dtype=np.float64).copy()
                for side, spec in place_specs.items()
            }
            workflow.place_elbow_bend_hint_world = primary_place_spec["elbow_bend_hint"]
            workflow.place_elbow_bend_hints_by_side = {
                side: np.asarray(spec["elbow_bend_hint"], dtype=np.float64).copy()
                for side, spec in place_specs.items()
            }
            session.first_heading_angle = 0.0
            session.constrained_root_heading_angle = None
            session.constrained_root_initial_turn_angle = None
            session.constrained_root_turn_end_frame = None
            session.constrained_root_headings = [0.0] * total_frames
            session.generation_world_offset = None
            session.hide_constraint_overlays = True
            session.human_route_kind = "counter_place"
            if has_dual_place_hold:
                prompts = [
                    "An ordinary healthy person stands still at the counter while holding two small objects, "
                    "one in each hand. They move both arms to mirrored safe pre-release positions above the "
                    "counter without changing either hand orientation, lower both hands only 2 cm to the "
                    "release positions, hold still for one second while releasing both objects onto the "
                    "counter, move both empty hands diagonally backward/outward and slightly upward until they are outside the "
                    "counter edge, then stop and hold those cleared poses until the end while keeping the same hand "
                    "orientations. Do not add a later normal-pose travel segment, extra hand motion, or extra wrist "
                    "rotation. The empty hands must not touch the table or collide with the body. The body and feet remain still."
                ]
            else:
                other_hand_side = "left" if model_place_hand_side == "right" else "right"
                prompts = [
                    "An ordinary healthy person stands still at the counter while holding a small object "
                    f"in the {model_place_hand_side} hand. They move only the {model_place_hand_side} arm to a safe pre-release position above "
                    "the counter without changing the hand orientation, lower the hand only 2 cm to "
                    "the release position, hold still for one second while releasing the object onto the "
                    "counter, move the empty hand diagonally backward/outward and slightly upward until it is outside the "
                    "counter edge, then stop and hold that cleared pose until the end while keeping the "
                    "same hand orientation. Do not add a later normal-pose travel segment, extra hand "
                    "motion, or extra wrist rotation. The empty hand must not touch the table or collide "
                    f"with the body. The body, feet, and {other_hand_side} arm remain still."
                ]
            strengthen_neutral_walk_guidance()
            cfg_weight = list(cfg_weight or [3.5, 3.0])
            if len(cfg_weight) > 1:
                cfg_weight[1] = max(float(cfg_weight[1]), 5.0)
            postprocess_parameters = dict(postprocess_parameters or {})
            postprocess_parameters["post_processing"] = True
            postprocess_parameters["root_margin"] = min(
                float(postprocess_parameters.get("root_margin", 0.04)),
                0.02,
            )
            client.add_notification(
                title="Counter place constraints applied",
                body=(
                    f"Pre-release at frame {pre_release_frame}, release at frame {release_frame}, "
                    f"waits until frame {release_hold_end_frame}, then retreats and returns to normal.\n\n"
                    + ("Both hands place their held objects.\n\n" if has_dual_place_hold else "")
                    + f"Saved constraint debug JSON:\n{place_constraint_debug_path}"
                ),
                auto_close_seconds=7.0,
                color="green",
            )
            self._apply_constraint_overlay_visibility(session)
        elif requested_rack_transfer is not None:
            session = self.client_sessions[client.client_id]
            source_rack = requested_rack_transfer.source_rack
            target_rack = requested_rack_transfer.target_rack
            for rack_name in (source_rack, target_rack):
                if rack_name not in RACK_MAP_POSITIONS:
                    available_racks = ", ".join(name.replace("_", " ") for name in RACK_MAP_POSITIONS)
                    raise ValueError(
                        f"Unknown warehouse rack {rack_name.replace('_', ' ')}. "
                        f"Available racks: {available_racks}."
                    )
            if len(num_frames) != 1 or num_frames[0] < 2:
                raise ValueError("A rack-to-rack prompt must contain one motion segment of at least two frames.")

            root_constraint = session.constraints.get("2D Root")
            fullbody_constraint = session.constraints.get("Full-Body")
            end_effector_constraint = session.constraints.get("End-Effectors")
            if root_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide a 2D Root constraint track.")
            if fullbody_constraint is None or end_effector_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide all rack-to-rack constraint tracks.")

            x_near = WORK_AREA_SIDE_SHIFT_M
            x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
            z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
            source_position, source_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[source_rack],
                RACK_MAP_YAWS_RAD.get(source_rack, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            target_position, target_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[target_rack],
                RACK_MAP_YAWS_RAD.get(target_rack, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            start_position = tuple(float(value) for value in source_position)
            start_heading = float(source_heading)
            transfer_initial_positions: np.ndarray | None = None
            transfer_initial_rotations: np.ndarray | None = None
            if session.motions:
                current_motion = next(iter(session.motions.values()))
                current_root = (
                    current_motion.joints_pos[-1, current_motion.skeleton.root_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if float(np.linalg.norm(current_root[[0, 2]] - np.asarray(source_position)[[0, 2]])) <= 0.45:
                    start_position = tuple(float(value) for value in current_root)
                    if (
                        session.constrained_root_headings is not None
                        and len(session.constrained_root_headings) > 0
                    ):
                        start_heading = float(session.constrained_root_headings[-1])
                    transfer_initial_positions = current_motion.joints_pos[-1].detach().cpu().numpy()
                    transfer_initial_rotations = current_motion.joints_rot[-1].detach().cpu().numpy()
                    if (
                        workflow is not None
                        and workflow.pick_grasp_verified
                    ):
                        if (
                            workflow.picked_rack_objects_by_hand
                            and workflow.pick_object_hand_offsets_by_side
                        ):
                            requested_rack_transfer_holding_pick = True
                            model_transfer_hold_rotations = transfer_initial_rotations.copy()
                            model_transfer_picked_object = None
                            model_transfer_hand_offset = None
                            model_transfer_picked_objects_by_hand = dict(
                                workflow.picked_rack_objects_by_hand
                            )
                            model_transfer_hand_offsets_by_side = {
                                side: np.asarray(offset, dtype=np.float64).copy()
                                for side, offset in workflow.pick_object_hand_offsets_by_side.items()
                            }
                            model_transfer_hand_side = "both"
                        elif (
                            workflow.picked_rack_object is not None
                            and workflow.pick_object_hand_offset is not None
                        ):
                            requested_rack_transfer_holding_pick = True
                            model_transfer_hold_rotations = transfer_initial_rotations.copy()
                            model_transfer_picked_object = workflow.picked_rack_object
                            model_transfer_hand_offset = np.asarray(
                                workflow.pick_object_hand_offset,
                                dtype=np.float64,
                            ).copy()
                            model_transfer_hand_side = workflow.pick_hand_side

            rack_transfer_route = _plan_cardinal_rack_transfer_route(
                start_position=start_position,
                start_heading=start_heading,
                target_position=tuple(float(value) for value in target_position),
                final_heading=float(target_heading),
                total_frames=int(num_frames[0]),
                fps=float(session.model_fps),
                walk_speed_m_s=_rack_human_walk_speed_m_s(target_rack),
                first_axis="x" if source_rack in {"rack_1", "rack_2"} else "z",
                reverse_distance=0.12 if requested_rack_transfer_holding_pick else 0.08,
                reverse_seconds=0.60 if requested_rack_transfer_holding_pick else 0.40,
            )
            if any(
                not (x_far <= position[0] <= x_near and z_far <= position[2] <= 0.0)
                for position in rack_transfer_route.positions
            ):
                raise ValueError("The planned rack-to-rack route leaves the orange boundary.")

            session.first_heading_angle = rack_transfer_route.headings[0]
            session.constrained_root_heading_angle = None
            session.constrained_root_initial_turn_angle = None
            session.constrained_root_turn_end_frame = None
            session.constrained_root_headings = list(rack_transfer_route.headings)
            session.generation_world_offset = None
            session.hide_constraint_overlays = True
            session.human_route_rack = target_rack
            session.human_route_kind = "outbound"
            for constraint in (root_constraint, fullbody_constraint, end_effector_constraint):
                constraint.clear()
            root_constraint.set_smooth_path(False)
            root_constraint.add_interval(
                f"auto_{source_rack}_to_{target_rack}_cardinal_route",
                0,
                int(num_frames[0]) - 1,
                np.asarray(rack_transfer_route.positions, dtype=np.float64),
            )
            root_constraint.set_dense_path(True)
            if transfer_initial_positions is not None and transfer_initial_rotations is not None:
                initial_pose_frames = min(max(5, int(round(0.25 * float(session.model_fps)))), int(num_frames[0]))
                fullbody_constraint.add_interval(
                    f"auto_{source_rack}_to_{target_rack}_start_pose",
                    0,
                    initial_pose_frames - 1,
                    np.repeat(transfer_initial_positions[None, ...], initial_pose_frames, axis=0),
                    np.repeat(transfer_initial_rotations[None, ...], initial_pose_frames, axis=0),
                )
                if rack_transfer_route.reverse_end_frame > 0:
                    reverse_pose_frames = min(rack_transfer_route.reverse_end_frame + 1, int(num_frames[0]))
                    fullbody_constraint.add_interval(
                        f"auto_{source_rack}_to_{target_rack}_reverse_slide_pose",
                        0,
                        reverse_pose_frames - 1,
                        np.repeat(transfer_initial_positions[None, ...], reverse_pose_frames, axis=0),
                        np.repeat(transfer_initial_rotations[None, ...], reverse_pose_frames, axis=0),
                    )
            prompts = [
                (
                    "An ordinary healthy person carefully carries the held object or objects "
                    "in front of the chest. They keep the holding arm poses steady relative to the torso, "
                    "slide slightly backward from the rack while staying upright, make controlled in-place "
                    "turns, walk forward along the aisle at a steady pace, turn to face the next rack, "
                    "then walk forward into the same final rack approach position, "
                    f"and finish standing upright in front of {target_rack.replace('_', ' ')}. "
                    "No sitting, crouching, skating, or squatting."
                )
                if requested_rack_transfer_holding_pick
                else (
                    "An ordinary healthy person walks at a steady normal pace from one rack to another. "
                    f"They start in front of {source_rack.replace('_', ' ')}, slide slightly backward "
                    "from the rack while staying upright, make controlled in-place turns at the aisle corners, walk forward "
                    f"normally, and finish standing naturally in front of {target_rack.replace('_', ' ')} "
                    "facing the rack. No sitting, crouching, skating, or squatting."
                )
            ]
            strengthen_neutral_walk_guidance()
            postprocess_parameters = dict(postprocess_parameters or {})
            postprocess_parameters["post_processing"] = True
            postprocess_parameters["root_margin"] = min(
                float(postprocess_parameters.get("root_margin", 0.04)),
                0.02,
            )
            client.add_notification(
                title=(
                    f"{source_rack.replace('_', ' ').title()} to "
                    f"{target_rack.replace('_', ' ').title()} route applied"
                ),
                body=(
                    f"Starts at X {start_position[0]:.2f} m, Z {start_position[2]:.2f} m; "
                    f"ends at the same approach pose as `move to {target_rack.replace('_', ' ')}`: "
                    f"X {target_position[0]:.2f} m, Z {target_position[2]:.2f} m. "
                    f"Final heading: {np.rad2deg(target_heading):.1f} degrees. "
                    f"Stationary turns: {rack_transfer_route.turn_degrees}."
                ),
                auto_close_seconds=7.0,
                color="green",
            )
            self._apply_constraint_overlay_visibility(session)
        elif requested_rack is not None:
            if requested_rack not in RACK_MAP_POSITIONS:
                available_racks = ", ".join(name.replace("_", " ") for name in RACK_MAP_POSITIONS)
                raise ValueError(
                    f"Unknown warehouse rack {requested_rack.replace('_', ' ')}. "
                    f"Available racks: {available_racks}."
                )
            if len(num_frames) != 1 or num_frames[0] < 2:
                raise ValueError("A rack-walk prompt must contain one motion segment of at least two frames.")

            if workflow is not None:
                for key, handle in workflow.rack_object_handles.items():
                    handle.position = workflow.rack_object_home_positions[key]
                workflow.picked_rack_object = None
                workflow.picked_rack_objects_by_hand.clear()
                workflow.pick_hold_start_frame = None
                workflow.pick_hold_start_frames_by_side.clear()
                workflow.pick_object_hand_offset = None
                workflow.pick_object_hand_offsets_by_side.clear()
                workflow.pick_reach_start_frame = None
                workflow.pick_hand_target_world = None
                workflow.pick_target_path_world = None
                workflow.pick_grasp_target_world = None
                workflow.pick_grasp_verified = False
                workflow.pick_elbow_bend_hint_world = None

            session = self.client_sessions[client.client_id]
            root_constraint = session.constraints.get("2D Root")
            if root_constraint is None:
                raise RuntimeError("The active Kimodo model does not provide a 2D Root constraint track.")

            x_near = WORK_AREA_SIDE_SHIFT_M
            x_far = x_near - WORK_AREA_GRID_SHAPE[0] * WORK_AREA_GRID_SECTION_M
            z_far = -WORK_AREA_GRID_SHAPE[1] * WORK_AREA_GRID_SECTION_M
            rack_target, rack_facing_heading = rack_width_side_approach_pose(
                RACK_MAP_POSITIONS[requested_rack],
                RACK_MAP_YAWS_RAD.get(requested_rack, 0.0),
                RACK_WIDTH_M,
                RACK_HUMAN_APPROACH_CLEARANCE_M,
                (x_far, x_near),
                (z_far, 0.0),
            )
            rack_walk_speed_m_s = _rack_human_walk_speed_m_s(requested_rack)
            route = plan_cardinal_rack_route(
                approach_position=rack_target,
                final_heading=rack_facing_heading,
                total_frames=int(num_frames[0]),
                fps=float(session.model_fps),
                walk_speed_m_s=rack_walk_speed_m_s,
                first_axis="z" if requested_rack in {"rack_1", "rack_2", "rack_3"} else "x",
            )
            session.first_heading_angle = route.headings[0]
            session.constrained_root_heading_angle = None
            session.constrained_root_initial_turn_angle = None
            session.constrained_root_turn_end_frame = None
            session.constrained_root_headings = list(route.headings)
            session.generation_world_offset = None
            session.hide_constraint_overlays = False
            session.human_route_rack = requested_rack
            session.human_route_kind = "outbound"
            root_constraint.clear()
            root_constraint.set_smooth_path(False)
            root_constraint.add_interval(
                f"auto_{requested_rack}_cardinal_route",
                0,
                int(num_frames[0]) - 1,
                np.asarray(route.positions, dtype=np.float64),
            )
            root_constraint.set_dense_path(True)
            prompts = [rack_walk_model_prompt(prompts[0], requested_rack)]
            strengthen_neutral_walk_guidance()
            postprocess_parameters = dict(postprocess_parameters or {})
            postprocess_parameters["post_processing"] = True
            postprocess_parameters["root_margin"] = min(
                float(postprocess_parameters.get("root_margin", 0.04)),
                0.02,
            )
            client.add_notification(
                title=f"{requested_rack.replace('_', ' ').title()} route applied",
                body=(
                    "Human root path: origin to "
                    f"X {rack_target[0]:.2f} m, Z {rack_target[2]:.2f} m "
                    f"({RACK_HUMAN_APPROACH_CLEARANCE_M:.2f} m clear of the rack face). "
                    + "Locked to the inward-facing broad shelf face. "
                    + f"Walking speed: {rack_walk_speed_m_s:.2f} m/s. "
                    + f"Straight segments only; stationary turns: {route.turn_degrees}. "
                    + f"Final rack-facing heading: {np.rad2deg(route.final_heading):.1f} degrees."
                ),
                auto_close_seconds=7.0,
                color="green",
            )
        else:
            session = self.client_sessions[client.client_id]
            session.first_heading_angle = None
            session.constrained_root_heading_angle = None
            session.constrained_root_initial_turn_angle = None
            session.constrained_root_turn_end_frame = None
            session.constrained_root_headings = None
            session.generation_world_offset = None
            session.hide_constraint_overlays = False
            session.human_route_rack = None
            session.human_route_kind = None

        # Every other prompt remains a normal Kimodo generation. Do not carry
        # a previous dedicated base command into an unrelated human motion.
        if workflow is not None:
            workflow.base_prompt_segments.clear()
            workflow.base_prompt_stem = None
            workflow.base_prompt_start_pose = None
            workflow.wheel_csv_path = None
            if workflow.wheel_base is not None:
                workflow.wheel_base.visual_yaw_offset_rad = float(np.pi / 2.0)
                workflow.wheel_base.set_stationary()

        super().generate(
            client,
            prompts,
            num_frames,
            num_samples,
            seed,
            diffusion_steps,
            cfg_weight=cfg_weight,
            cfg_type=cfg_type,
            postprocess_parameters=postprocess_parameters,
            transitions_parameters=transitions_parameters,
            real_robot_rotations=real_robot_rotations,
        )
        if requested_rack is not None:
            session = self.client_sessions[client.client_id]
            if session.motions:
                outbound_motion = next(iter(session.motions.values()))
                session.human_outbound_rack_poses[requested_rack] = (
                    outbound_motion.joints_pos[-1].detach().cpu().numpy().copy(),
                    outbound_motion.joints_rot[-1].detach().cpu().numpy().copy(),
                )
        if requested_rack_transfer is not None:
            session = self.client_sessions[client.client_id]
            if session.motions:
                transfer_motion = next(iter(session.motions.values()))
                if rack_transfer_route is not None:
                    route_positions = np.asarray(rack_transfer_route.positions, dtype=np.float64)
                    frame_count = min(transfer_motion.length, len(route_positions))
                    root_index = transfer_motion.skeleton.root_idx
                    target_root_positions = torch.as_tensor(
                        route_positions[:frame_count],
                        device=transfer_motion.joints_pos.device,
                        dtype=transfer_motion.joints_pos.dtype,
                    )
                    root_delta = target_root_positions - transfer_motion.joints_pos[:frame_count, root_index]
                    root_delta[:, 1] = 0.0
                    transfer_motion.joints_pos[:frame_count] = (
                        transfer_motion.joints_pos[:frame_count] + root_delta[:, None, :]
                    )
                    foot_indices = list(getattr(transfer_motion.skeleton, "foot_joint_idx", []) or [])
                    if foot_indices:
                        foot_y = transfer_motion.joints_pos[:frame_count, foot_indices, 1]
                        floor_delta_y = -torch.min(foot_y, dim=1).values
                        transfer_motion.joints_pos[:frame_count, :, 1] += floor_delta_y[:, None]
                    transfer_motion.precompute_mesh_info()
                    self.set_frame(client.client_id, 0)
                if (
                    requested_rack_transfer_holding_pick
                    and workflow is not None
                    and model_transfer_hold_rotations is not None
                ):
                    if model_transfer_hand_side == "both":
                        _hold_right_arm_local_pose(
                            transfer_motion,
                            model_transfer_hold_rotations,
                            side="right",
                        )
                        _hold_right_arm_local_pose(
                            transfer_motion,
                            model_transfer_hold_rotations,
                            side="left",
                        )
                        workflow.picked_rack_object = None
                        workflow.pick_object_hand_offset = None
                        workflow.picked_rack_objects_by_hand = dict(
                            model_transfer_picked_objects_by_hand or {}
                        )
                        workflow.pick_object_hand_offsets_by_side = {
                            side: np.asarray(offset, dtype=np.float64).copy()
                            for side, offset in (model_transfer_hand_offsets_by_side or {}).items()
                        }
                        workflow.pick_hold_start_frames_by_side = {
                            side: 0 for side in workflow.picked_rack_objects_by_hand
                        }
                        workflow.pick_hand_side = "right"
                    else:
                        _hold_right_arm_local_pose(
                            transfer_motion,
                            model_transfer_hold_rotations,
                            side=model_transfer_hand_side,
                        )
                        workflow.picked_rack_object = model_transfer_picked_object
                        workflow.pick_object_hand_offset = model_transfer_hand_offset
                        workflow.picked_rack_objects_by_hand.clear()
                        workflow.pick_object_hand_offsets_by_side.clear()
                        workflow.pick_hold_start_frames_by_side.clear()
                        workflow.pick_hand_side = model_transfer_hand_side
                    workflow.pick_hold_start_frame = 0
                    workflow.pick_reach_start_frame = 0
                    workflow.pick_hand_target_world = None
                    workflow.pick_target_path_world = None
                    workflow.pick_grasp_target_world = None
                    workflow.pick_elbow_bend_hint_world = None
                    workflow.place_release_frame = None
                    workflow.place_object_position_world = None
                    workflow.place_object_positions_by_hand.clear()
                    workflow.place_reach_start_frame = None
                    workflow.place_hand_target_world = None
                    workflow.place_hand_targets_by_side.clear()
                    workflow.place_target_path_world = None
                    workflow.place_target_paths_by_side.clear()
                    workflow.place_elbow_bend_hint_world = None
                    workflow.place_elbow_bend_hints_by_side.clear()
                    workflow.pick_grasp_verified = True
                    self.set_frame(client.client_id, 0)
                session.human_outbound_rack_poses[requested_rack_transfer.target_rack] = (
                    transfer_motion.joints_pos[-1].detach().cpu().numpy().copy(),
                    transfer_motion.joints_rot[-1].detach().cpu().numpy().copy(),
                )
                session.human_route_rack = requested_rack_transfer.target_rack
                session.human_route_kind = "outbound"
        if requested_human_return is not None:
            session = self.client_sessions[client.client_id]
            if session.motions:
                motion = next(iter(session.motions.values()))
                route_positions = np.asarray(route.positions, dtype=np.float64)
                frame_count = min(motion.length, len(route_positions))
                root_index = motion.skeleton.root_idx
                target_root_positions = torch.as_tensor(
                    route_positions[:frame_count],
                    device=motion.joints_pos.device,
                    dtype=motion.joints_pos.dtype,
                )
                root_delta = target_root_positions - motion.joints_pos[:frame_count, root_index]
                root_delta[:, 1] = 0.0
                motion.joints_pos[:frame_count] = motion.joints_pos[:frame_count] + root_delta[:, None, :]
                foot_indices = list(getattr(motion.skeleton, "foot_joint_idx", []) or [])
                if foot_indices:
                    foot_y = motion.joints_pos[:frame_count, foot_indices, 1]
                    floor_delta_y = -torch.min(foot_y, dim=1).values
                    motion.joints_pos[:frame_count, :, 1] += floor_delta_y[:, None]
                motion.precompute_mesh_info()
                self.set_frame(client.client_id, 0)
        if (requested_dual_pick is not None or requested_human_pick is not None) and workflow is not None:
            session = self.client_sessions[client.client_id]
            if (
                session.motions
                and workflow.pick_reach_start_frame is not None
                and workflow.pick_hold_start_frame is not None
                and workflow.pick_hand_target_world is not None
                and workflow.pick_target_path_world is not None
                and workflow.pick_grasp_target_world is not None
                and model_pick_base_pose is not None
            ):
                motion = next(iter(session.motions.values()))
                base_positions, base_rotations = model_pick_base_pose
                active_specs = model_dual_pick_specs
                if active_specs is None:
                    active_specs = {
                        model_pick_hand_side: {
                            "object_key": workflow.picked_rack_object,
                            "hand_target_path": workflow.pick_target_path_world,
                            "hand_target_world": workflow.pick_hand_target_world,
                            "palm_normal_local": model_pick_palm_normal_local,
                            "target_palm_normals": model_pick_target_palm_normals,
                            "elbow_bend_hint": workflow.pick_elbow_bend_hint_world,
                        }
                    }
                    _freeze_body_except_right_arm(
                        motion,
                        base_rotations,
                        base_positions[motion.skeleton.root_idx],
                        side=model_pick_hand_side,
                    )
                else:
                    _freeze_body_except_arm_sides(
                        motion,
                        base_rotations,
                        base_positions[motion.skeleton.root_idx],
                        set(active_specs.keys()),
                    )

                grasp_frame = workflow.pick_hold_start_frame
                chest_index = motion.skeleton.bone_order_names.index("Chest")
                chest_final = motion.joints_pos[-1, chest_index].detach().cpu().numpy()
                max_final_error = 0.0
                max_palm_error = 0.0
                min_body_clearance = float("inf")
                per_hand_errors: dict[str, float] = {}
                workflow.pick_object_hand_offsets_by_side.clear()
                workflow.pick_object_hand_offsets_by_side.update(
                    {
                        side: offset.copy()
                        for side, offset in model_pick_existing_offsets_by_hand.items()
                    }
                )
                side_runtime: dict[str, dict[str, object]] = {}
                for side, spec in active_specs.items():
                    target_path = np.asarray(spec["hand_target_path"], dtype=np.float64)
                    target_world = np.asarray(spec["hand_target_world"], dtype=np.float64)
                    palm_normal_local = spec["palm_normal_local"]
                    target_palm_normals = spec["target_palm_normals"]
                    elbow_bend_hint = spec["elbow_bend_hint"]
                    _enforce_right_arm_reach(
                        motion,
                        workflow.pick_reach_start_frame,
                        grasp_frame,
                        target_world,
                        target_path_world=target_path,
                        target_hand_rotations_world=model_pick_target_hand_rotations,
                        target_palm_normals_world=target_palm_normals,
                        palm_normal_local=palm_normal_local,
                        elbow_bend_hint_world=elbow_bend_hint,
                        refresh_cache=False,
                        side=side,
                    )
                    active_hand_names = (
                        motion.skeleton.left_hand_joint_names
                        if side == "left"
                        else motion.skeleton.right_hand_joint_names
                    )
                    hand_label = "Left" if side == "left" else "Right"
                    hand_index = motion.skeleton.bone_order_names.index(active_hand_names[0])
                    middle_name = (
                        f"{hand_label}HandMiddleEnd"
                        if f"{hand_label}HandMiddleEnd" in motion.skeleton.bone_order_names
                        else active_hand_names[-1]
                    )
                    middle_index = motion.skeleton.bone_order_names.index(middle_name)
                    object_key = spec["object_key"]
                    object_position = np.asarray(
                        workflow.rack_object_home_positions[object_key],
                        dtype=np.float64,
                    )
                    correction_start = int(
                        spec.get(
                            "correction_start_frame",
                            max(0, int(round(motion.length * 0.20))),
                        )
                    )
                    correction_start = min(correction_start, max(0, grasp_frame - 1))
                    side_runtime[side] = {
                        "spec": spec,
                        "target_path": target_path.copy(),
                        "palm_normal_local": palm_normal_local,
                        "target_palm_normals": target_palm_normals,
                        "elbow_bend_hint": elbow_bend_hint,
                        "hand_index": hand_index,
                        "middle_index": middle_index,
                        "object_position": object_position,
                        "correction_start": correction_start,
                    }

                correction_passes = 4 if model_dual_pick_specs is not None else 2
                for _ in range(correction_passes):
                    all_close = True
                    for side, runtime in side_runtime.items():
                        spec = runtime["spec"]
                        corrected_path = runtime["target_path"]
                        hand_index = int(runtime["hand_index"])
                        middle_index = int(runtime["middle_index"])
                        object_position = np.asarray(runtime["object_position"], dtype=np.float64)
                        correction_start = int(runtime["correction_start"])
                        wrist_grasp = motion.joints_pos[grasp_frame, hand_index].detach().cpu().numpy()
                        middle_grasp = motion.joints_pos[grasp_frame, middle_index].detach().cpu().numpy()
                        palm_grasp = 0.35 * wrist_grasp + 0.65 * middle_grasp
                        palm_correction = object_position - palm_grasp
                        for frame in range(correction_start, motion.length):
                            progress = min(
                                1.0,
                                (frame - correction_start) / max(1, grasp_frame - correction_start),
                            )
                            progress = progress * progress * (3.0 - 2.0 * progress)
                            corrected_path[frame] += progress * palm_correction
                        spec["hand_target_path"] = corrected_path
                        spec["hand_target_world"] = corrected_path[-1]
                        final_error = _enforce_right_arm_reach(
                            motion,
                            workflow.pick_reach_start_frame,
                            grasp_frame,
                            corrected_path[-1],
                            target_path_world=corrected_path,
                            target_hand_rotations_world=model_pick_target_hand_rotations,
                            target_palm_normals_world=runtime["target_palm_normals"],
                            palm_normal_local=runtime["palm_normal_local"],
                            elbow_bend_hint_world=runtime["elbow_bend_hint"],
                            refresh_cache=False,
                            side=side,
                        )
                        wrist_grasp = motion.joints_pos[grasp_frame, hand_index].detach().cpu().numpy()
                        middle_grasp = motion.joints_pos[grasp_frame, middle_index].detach().cpu().numpy()
                        palm_grasp = 0.35 * wrist_grasp + 0.65 * middle_grasp
                        palm_error = float(np.linalg.norm(palm_grasp - object_position))
                        runtime["target_path"] = corrected_path
                        runtime["final_error"] = final_error
                        runtime["palm_error"] = palm_error
                        if palm_error > 0.025:
                            all_close = False
                    if all_close:
                        break

                for side, runtime in side_runtime.items():
                    hand_index = int(runtime["hand_index"])
                    middle_index = int(runtime["middle_index"])
                    object_position = np.asarray(runtime["object_position"], dtype=np.float64)
                    wrist_grasp = motion.joints_pos[grasp_frame, hand_index].detach().cpu().numpy()
                    middle_grasp = motion.joints_pos[grasp_frame, middle_index].detach().cpu().numpy()
                    palm_grasp = 0.35 * wrist_grasp + 0.65 * middle_grasp
                    palm_error = float(np.linalg.norm(palm_grasp - object_position))
                    final_error = float(runtime.get("final_error", 0.0))
                    hand_rotation_grasp = motion.joints_rot[grasp_frame, hand_index].detach().cpu().numpy()
                    workflow.pick_object_hand_offsets_by_side[side] = (
                        hand_rotation_grasp.T @ (object_position - palm_grasp)
                    )
                    palm_final = (
                        0.35 * motion.joints_pos[-1, hand_index]
                        + 0.65 * motion.joints_pos[-1, middle_index]
                    ).detach().cpu().numpy()
                    body_clearance = float(np.linalg.norm(palm_final - chest_final))
                    max_final_error = max(max_final_error, final_error)
                    max_palm_error = max(max_palm_error, palm_error)
                    min_body_clearance = min(min_body_clearance, body_clearance)
                    per_hand_errors[side] = palm_error
                if model_dual_pick_specs is None:
                    workflow.pick_object_hand_offset = workflow.pick_object_hand_offsets_by_side.get(model_pick_hand_side)
                    workflow.pick_target_path_world = active_specs[model_pick_hand_side]["hand_target_path"]
                    workflow.pick_hand_target_world = active_specs[model_pick_hand_side]["hand_target_world"]
                    if model_pick_existing_objects_by_hand:
                        workflow.picked_rack_objects_by_hand = dict(model_pick_existing_objects_by_hand)
                        workflow.picked_rack_objects_by_hand[model_pick_hand_side] = workflow.picked_rack_object
                        workflow.picked_rack_object = None
                        workflow.pick_object_hand_offset = None
                        workflow.pick_hold_start_frames_by_side.update(
                            {
                                side: 0
                                for side in model_pick_existing_objects_by_hand
                            }
                        )
                        workflow.pick_hold_start_frames_by_side[model_pick_hand_side] = grasp_frame
                motion.precompute_mesh_info()
                palm_error = max_palm_error
                final_error = max_final_error
                body_clearance = min_body_clearance
                # The selectable rack objects are 6 cm cubes. A 6.5 cm palm-center
                # tolerance still keeps the gripper on/near the object volume while
                # avoiding false failures from tiny wrist/retarget offsets.
                palm_grasp_tolerance = 0.065
                workflow.pick_grasp_verified = (
                    palm_error <= palm_grasp_tolerance and body_clearance >= 0.16
                )
                if not workflow.pick_grasp_verified or final_error > 0.08:
                    per_hand_text = ", ".join(
                        f"{side} {error:.3f} m" for side, error in sorted(per_hand_errors.items())
                    )
                    raise RuntimeError(
                        f"Collision-safe pick did not converge (palm {palm_error:.3f} m, "
                        f"allowed {palm_grasp_tolerance:.3f} m, chest hold {final_error:.3f} m, "
                        f"body clearance {body_clearance:.3f} m"
                        + (f", per-hand {per_hand_text}" if per_hand_text else "")
                        + "); "
                        "the object was not attached."
                    )
                self.set_frame(client.client_id, 0)
                client.add_notification(
                    title="Collision-safe model pick enforced",
                    body=(
                        f"Palm grasp error {palm_error * 100.0:.1f} cm; body clearance "
                        f"{body_clearance * 100.0:.1f} cm."
                        + (
                            "\n"
                            + ", ".join(
                                f"{side}: {error * 100.0:.1f} cm"
                                for side, error in sorted(per_hand_errors.items())
                            )
                            if len(per_hand_errors) > 1
                            else ""
                        )
                    ),
                    auto_close_seconds=5.0,
                    color="green",
                )
        if requested_human_place and workflow is not None:
            session = self.client_sessions[client.client_id]
            if (
                session.motions
                and workflow.place_reach_start_frame is not None
                and model_place_release_frame is not None
                and workflow.place_hand_target_world is not None
                and workflow.place_target_path_world is not None
                and model_place_base_pose is not None
            ):
                motion = next(iter(session.motions.values()))
                base_positions, base_rotations = model_place_base_pose
                release_frame = min(model_place_release_frame, motion.length - 1)
                if model_place_specs is not None:
                    _freeze_body_except_arm_sides(
                        motion,
                        base_rotations,
                        base_positions[motion.skeleton.root_idx],
                        set(model_place_specs.keys()),
                    )
                    final_error = 0.0
                    release_error = 0.0
                    per_hand_errors: dict[str, tuple[float, float]] = {}
                    for side, spec in model_place_specs.items():
                        hand_target_world = np.asarray(spec["hand_target_world"], dtype=np.float64)
                        target_path_world = np.asarray(spec["hand_target_path"], dtype=np.float64)
                        side_final_error = _enforce_right_arm_reach(
                            motion,
                            workflow.place_reach_start_frame,
                            model_place_release_frame,
                            hand_target_world,
                            target_path_world=target_path_world,
                            target_hand_rotations_world=spec["target_hand_rotations"],
                            elbow_bend_hint_world=spec["elbow_bend_hint"],
                            refresh_cache=False,
                            side=side,
                        )
                        active_hand_names = (
                            motion.skeleton.left_hand_joint_names
                            if side == "left"
                            else motion.skeleton.right_hand_joint_names
                        )
                        hand_index = motion.skeleton.bone_order_names.index(active_hand_names[0])
                        side_release_error = float(
                            np.linalg.norm(
                                motion.joints_pos[release_frame, hand_index].detach().cpu().numpy()
                                - target_path_world[release_frame]
                            )
                        )
                        final_error = max(final_error, side_final_error)
                        release_error = max(release_error, side_release_error)
                        per_hand_errors[side] = (side_release_error, side_final_error)
                    motion.precompute_mesh_info()
                else:
                    _freeze_body_except_right_arm(
                        motion,
                        base_rotations,
                        base_positions[motion.skeleton.root_idx],
                        side=model_place_hand_side,
                    )
                    final_error = _enforce_right_arm_reach(
                        motion,
                        workflow.place_reach_start_frame,
                        model_place_release_frame,
                        workflow.place_hand_target_world,
                        target_path_world=workflow.place_target_path_world,
                        target_hand_rotations_world=model_place_target_hand_rotations,
                        elbow_bend_hint_world=workflow.place_elbow_bend_hint_world,
                        refresh_cache=True,
                        side=model_place_hand_side,
                    )
                    active_hand_names = (
                        motion.skeleton.left_hand_joint_names
                        if model_place_hand_side == "left"
                        else motion.skeleton.right_hand_joint_names
                    )
                    hand_index = motion.skeleton.bone_order_names.index(
                        active_hand_names[0]
                    )
                    release_error = float(
                        np.linalg.norm(
                            motion.joints_pos[release_frame, hand_index].detach().cpu().numpy()
                            - workflow.place_target_path_world[release_frame]
                        )
                    )
                    per_hand_errors = {}
                if release_error > 0.08:
                    per_hand_text = ", ".join(
                        f"{side} release {errors[0]:.3f} m, final {errors[1]:.3f} m"
                        for side, errors in sorted(per_hand_errors.items())
                    )
                    raise RuntimeError(
                        f"Counter place did not converge (release {release_error:.3f} m, "
                        f"normal {final_error:.3f} m"
                        + (f", per-hand {per_hand_text}" if per_hand_text else "")
                        + ")."
                    )
                workflow.pick_hold_start_frame = 0
                workflow.pick_grasp_verified = True
                self.set_frame(client.client_id, 0)
                client.add_notification(
                    title="Counter place enforced",
                    body=(
                        f"Release hand error {release_error * 100.0:.1f} cm; "
                        f"normal-pose error {final_error * 100.0:.1f} cm."
                    ),
                    auto_close_seconds=5.0,
                    color="green",
                )
        if (
            requested_human_return_holding_pick
            and workflow is not None
            and model_return_hold_rotations is not None
        ):
            session = self.client_sessions[client.client_id]
            if session.motions:
                motion = next(iter(session.motions.values()))
                if model_return_hand_side == "both":
                    _hold_right_arm_local_pose(motion, model_return_hold_rotations, side="right")
                    _hold_right_arm_local_pose(motion, model_return_hold_rotations, side="left")
                    workflow.picked_rack_object = None
                    workflow.pick_object_hand_offset = None
                    workflow.picked_rack_objects_by_hand = dict(model_return_picked_objects_by_hand or {})
                    workflow.pick_object_hand_offsets_by_side = {
                        side: np.asarray(offset, dtype=np.float64).copy()
                        for side, offset in (model_return_hand_offsets_by_side or {}).items()
                    }
                    workflow.pick_hold_start_frames_by_side = {
                        side: 0 for side in workflow.picked_rack_objects_by_hand
                    }
                    workflow.pick_hand_side = "right"
                else:
                    _hold_right_arm_local_pose(motion, model_return_hold_rotations, side=model_return_hand_side)
                    workflow.picked_rack_object = model_return_picked_object
                    workflow.pick_object_hand_offset = model_return_hand_offset
                    workflow.picked_rack_objects_by_hand.clear()
                    workflow.pick_object_hand_offsets_by_side.clear()
                    workflow.pick_hold_start_frames_by_side.clear()
                    workflow.pick_hand_side = model_return_hand_side
                workflow.pick_hold_start_frame = 0
                workflow.pick_reach_start_frame = 0
                workflow.pick_hand_target_world = None
                workflow.pick_target_path_world = None
                workflow.pick_grasp_target_world = None
                workflow.pick_elbow_bend_hint_world = None
                workflow.place_release_frame = None
                workflow.place_object_position_world = None
                workflow.place_object_positions_by_hand.clear()
                workflow.place_reach_start_frame = None
                workflow.place_hand_target_world = None
                workflow.place_hand_targets_by_side.clear()
                workflow.place_target_path_world = None
                workflow.place_target_paths_by_side.clear()
                workflow.place_elbow_bend_hint_world = None
                workflow.place_elbow_bend_hints_by_side.clear()
                workflow.pick_grasp_verified = True
                self.set_frame(client.client_id, 0)
            client.add_notification(
                title="Carry return enforced",
                body=(
                    (
                        "Both arm local poses were held from the pick final frame, "
                        "so both items stay in the hands relative to the body during the return."
                        if model_return_hand_side == "both"
                        else (
                            f"{model_return_hand_side.title()} arm local pose was held from the pick final frame, "
                            "so the item stays in the hand relative to the body during the return."
                        )
                    )
                ),
                    auto_close_seconds=5.0,
                    color="green",
                )

    def _create_robot_pipeline_gui(self, client: viser.ClientHandle) -> None:
        session = self.client_sessions.get(client.client_id)
        if session is None:
            return
        workflow = self.robot_workflows.get(client.client_id)
        if workflow is None:
            return
        memories_root_default = workflow.output_root
        memory_stems = _scan_memory_stems(memories_root_default)
        memory_labels = (
            [_memory_label(memories_root_default, stem) for stem in memory_stems]
            if memory_stems
            else ["<no memories>"]
        )
        base_memory_stems = _scan_base_memory_stems(memories_root_default)
        base_memory_labels = (
            [_base_memory_label(memories_root_default, stem) for stem in base_memory_stems]
            if base_memory_stems
            else ["<no base memories>"]
        )

        with client.gui.add_folder("Memories", expand_by_default=True):
            memories_root_text = client.gui.add_text("Root", initial_value=str(memories_root_default))
            memory_dropdown = client.gui.add_dropdown(
                "Memory",
                options=memory_labels,
                initial_value=memory_labels[0],
            )
            memory_status = client.gui.add_markdown("Select a memory.")
            base_memory_dropdown = client.gui.add_dropdown(
                "Base Memory",
                options=base_memory_labels,
                initial_value=base_memory_labels[0],
            )
            base_memory_status = client.gui.add_markdown("Select a base memory.")
            continue_base_pose_checkbox = client.gui.add_checkbox(
                "Continue From Last Base Pose",
                initial_value=workflow.continue_base_from_last_pose,
                hint="Checked: continue from the previous CSV end pose. Unchecked: start at the origin.",
            )
            continuation_base_csv_text = client.gui.add_text(
                "Continuation Base CSV",
                initial_value=workflow.continuation_base_csv_reference,
                hint="Optional filename, memory stem, or absolute CSV path. Empty uses the loaded base preview.",
            )
            chain_base_memories_checkbox = client.gui.add_checkbox(
                "Chain Loaded Base Memories",
                initial_value=workflow.chain_loaded_base_memories,
                hint="Align the next loaded base memory's first frame to the previous memory's final pose.",
            )
            loop_base_memory_checkbox = client.gui.add_checkbox(
                "Loop Base Memory in Simulation",
                initial_value=workflow.loop_base_memory_preview,
                hint="Repeat the Viser base-memory preview until playback is paused.",
            )
            refresh_memories_button = client.gui.add_button("Refresh Memories")
            load_memory_button = client.gui.add_button("Load Memory")
            load_base_memory_button = client.gui.add_button("Load Base Memory")
            play_base_memory_preview_button = client.gui.add_button(
                "Play Base Memory in Simulation",
                color="green",
                hint="Preview once or loop in Viser according to the simulation loop checkbox.",
            )
            retarget_memory_button = client.gui.add_button("Retarget Memory to T2", color="green")
            retarget_memory_t3_button = client.gui.add_button("Retarget Memory to T3", color="green")
            save_generated_memory_button = client.gui.add_button(
                "Save Memories",
                color="blue",
                hint="Save the current SOMA motion into the BVH memories folder.",
            )
            base_csv_name_text = client.gui.add_text(
                "Base CSV Name",
                initial_value=workflow.base_prompt_stem or "tara_motion",
                hint="Saved as <name>_diff_drive.csv under wheel_csv.",
            )
            save_base_csv_button = client.gui.add_button(
                "Save Base CSV",
                color="blue",
                hint="Save the current forward/backward Tara motion frame by frame.",
            )

        with client.gui.add_folder("Preview Visibility", expand_by_default=True):
            show_human_checkbox = client.gui.add_checkbox(
                "Show Human",
                initial_value=session.gui_elements.gui_viz_skinned_mesh_checkbox.value,
            )
            show_t2_checkbox = client.gui.add_checkbox("Show T2 Robot", initial_value=True)
            show_t3_checkbox = client.gui.add_checkbox("Show T3 Robot", initial_value=True)
            show_wheel_base_checkbox = client.gui.add_checkbox("Show Tara Wheel Base", initial_value=False)
            show_floor_grid_checkbox = client.gui.add_checkbox("Show Floor Grid", initial_value=True)

        with client.gui.add_folder("Sync to Real Robot", expand_by_default=True):
            workflow.robot_markdown = client.gui.add_markdown("Disconnected.")
            dry_run_checkbox = client.gui.add_checkbox(
                "Dry run",
                initial_value=True,
                hint="Print streamed frames instead of publishing ROS commands.",
            )
            connect_button = client.gui.add_button("Connect")
            disconnect_button = client.gui.add_button("Disconnect")
            send_frame_button = client.gui.add_button("Send Current Frame")
            play_robot_button = client.gui.add_button("Play on Robot", color="green")
            approve_robot_play_button = client.gui.add_button("Approve Real Robot Play", color="orange")
            stop_robot_button = client.gui.add_button("Stop Robot", color="red")

        with client.gui.add_folder("TaraBase (ROS 2)", expand_by_default=False):
            tara_status = client.gui.add_markdown(
                f"Ready to publish `/base/cmd_wheel_rpm`; clamp +/-`{self.tara_max_rpm:g}` RPM."
            )
            tara_play_button = client.gui.add_button("Play / Resume TaraBase via ROS", color="green")
            tara_pause_button = client.gui.add_button("Stop TaraBase ROS Playback", color="orange")
            tara_restart_button = client.gui.add_button("Restart TaraBase via ROS")
            tara_reverse_button = client.gui.add_button("Reverse TaraBase via ROS")

        with client.gui.add_folder("TaraBase (WebSocket)", expand_by_default=False):
            tara_remote_url_text = client.gui.add_text(
                "TaraBase WebSocket URL",
                initial_value=self.tara_remote_url or "",
                hint="WebSocket endpoint exposed by tara_remote_server.py, for example ws://PI_IP:8094.",
            )
            tara_websocket_status = client.gui.add_markdown(
                f"Ready to stream the loaded TaraBase CSV; clamp +/-`{self.tara_max_rpm:g}` RPM."
            )
            tara_websocket_play_button = client.gui.add_button(
                "Play / Resume TaraBase via WebSocket", color="green"
            )
            tara_websocket_pause_button = client.gui.add_button(
                "Stop TaraBase WebSocket Playback", color="orange"
            )
            tara_websocket_restart_button = client.gui.add_button(
                "Restart TaraBase via WebSocket"
            )
            tara_websocket_reverse_button = client.gui.add_button(
                "Reverse TaraBase via WebSocket"
            )

        with client.gui.add_folder("Sync T3 to Real Robot", expand_by_default=True):
            combined_csv_path_text = client.gui.add_text(
                "Base + Arms CSV",
                initial_value="/home/jony/Downloads/soma-retargeter/assets/basearms.csv",
                hint="One CSV containing T2 arm joints, time_s, and TaraBase motor RPM columns.",
            )
            combined_status = client.gui.add_markdown("Load a combined CSV to preview the complete T3 motion.")
            load_combined_csv_button = client.gui.add_button("Load Combined T3 Motion", color="blue")
            enable_t3_hardware_checkbox = client.gui.add_checkbox(
                "Enable Real T3 Hardware",
                initial_value=False,
                hint="Safety gate. Leave off to preview base + arms together in Viser only.",
            )
            freeze_t3_base_checkbox = client.gui.add_checkbox(
                "Keep T3 Base Stationary",
                initial_value=False,
                hint="Animate only arms/grippers. No TaraBase commands are sent during synchronized hardware play.",
            )
            sync_t3_button = client.gui.add_button("Play Base + Arms Synchronized", color="green")
            stop_t3_sync_button = client.gui.add_button("Stop Synchronized Motion", color="red")

        with client.gui.add_folder("Prompt-to-Robot Job", expand_by_default=True):
            prompt_job_markdown = client.gui.add_markdown("No prompt job running.")
        prompt_job_status_targets: list[viser.GuiMarkdownHandle] = [prompt_job_markdown]

        with client.gui.add_folder("Robot Pipeline", expand_by_default=False):
            workflow.status_markdown = client.gui.add_markdown("No robot artifact generated yet.")
            clip_name_text = client.gui.add_text("Clip Name", initial_value="kimodo_motion")
            output_root_text = client.gui.add_text("Output Root", initial_value=str(workflow.output_root))
            retargeter_root_text = client.gui.add_text(
                "soma-retargeter",
                initial_value=str(default_soma_retargeter_root()),
            )
            conda_env_text = client.gui.add_text("Conda Env", initial_value="soma-retargeter")
            standard_tpose_checkbox = client.gui.add_checkbox(
                "Standard T-pose BVH",
                initial_value=False,
                hint="Use Kimodo's standard SOMA T-pose when exporting the BVH for retargeting.",
            )
            arms_only_checkbox = client.gui.add_checkbox(
                "Freeze T2 root/body",
                initial_value=False,
                hint="When enabled, preview only arms/grippers and keep the T2 body at frame 0.",
            )

            save_bvh_button = client.gui.add_button("Save SOMA BVH", color="blue")
            retarget_button = client.gui.add_button("Retarget to T2", color="green")
            retarget_t3_button = client.gui.add_button("Retarget to T3", color="green")
            csv_path_text = client.gui.add_text("T2 CSV", initial_value="")
            load_t2_button = client.gui.add_button("Load T2 Preview")

        def update_status(message: str) -> None:
            if workflow.status_markdown is not None:
                workflow.status_markdown.content = message
            client.flush()

        def update_prompt_job(message: str) -> None:
            for target in list(prompt_job_status_targets):
                try:
                    target.content = message
                except Exception:
                    prompt_job_status_targets.remove(target)
            client.flush()

        def update_robot_status(message: str) -> None:
            if workflow.robot_markdown is not None:
                workflow.robot_markdown.content = message

        def current_memories_root() -> Path:
            return Path(memories_root_text.value).expanduser().resolve()

        def selected_memory_stem() -> str | None:
            value = str(memory_dropdown.value)
            if value == "<no memories>":
                return None
            return _stem_from_memory_label(value)

        def selected_base_memory_stem() -> str | None:
            value = str(base_memory_dropdown.value)
            if value == "<no base memories>":
                return None
            return _stem_from_memory_label(value)

        def refresh_memory_options(select_stem: str | None = None) -> list[str]:
            root = current_memories_root()
            stems = _scan_memory_stems(root)
            labels = [_memory_label(root, stem) for stem in stems] if stems else ["<no memories>"]
            memory_dropdown.options = labels
            if select_stem is not None and select_stem in stems:
                memory_dropdown.value = _memory_label(root, select_stem)
            elif str(memory_dropdown.value) not in labels:
                memory_dropdown.value = labels[0]
            update_memory_status()
            refresh_base_memory_options()
            return stems

        def refresh_base_memory_options(select_stem: str | None = None) -> list[str]:
            root = current_memories_root()
            stems = _scan_base_memory_stems(root)
            labels = [_base_memory_label(root, stem) for stem in stems] if stems else ["<no base memories>"]
            base_memory_dropdown.options = labels
            if select_stem is not None and select_stem in stems:
                base_memory_dropdown.value = _base_memory_label(root, select_stem)
            elif str(base_memory_dropdown.value) not in labels:
                base_memory_dropdown.value = labels[0]
            update_base_memory_status()
            return stems

        def set_human_mesh_visible(visible: bool) -> None:
            session = self.client_sessions[client.client_id]
            session.gui_elements.gui_viz_skinned_mesh_checkbox.value = visible
            for motion in session.motions.values():
                motion.character.set_skinned_mesh_visibility(visible)

        def set_t2_visible(visible: bool) -> None:
            if workflow.t2_motion is not None:
                workflow.t2_motion.set_mesh_visibility(visible)

        def set_wheel_base_visible(visible: bool) -> None:
            if workflow.wheel_base is not None:
                workflow.wheel_base.set_visible(visible)

        def set_t3_visible(visible: bool) -> None:
            if workflow.t3_motion is not None:
                workflow.t3_motion.set_visible(visible)

        def set_floor_grid_visible(visible: bool) -> None:
            grid_handle = self.grid_handles.get(client.client_id)
            if grid_handle is not None:
                grid_handle.visible = visible

        set_floor_grid_visible(bool(show_floor_grid_checkbox.value))

        def current_human_root_position() -> np.ndarray | None:
            session = self.client_sessions[client.client_id]
            if not session.motions:
                return None
            motion = list(session.motions.values())[0]
            frame_idx = min(session.frame_idx, motion.joints_pos.shape[0] - 1)
            return (
                motion.joints_pos[frame_idx, session.skeleton.root_idx, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        def t2_to_human_offset(csv_path: Path) -> np.ndarray:
            human_root = current_human_root_position()
            if human_root is None:
                return np.zeros(3, dtype=np.float64)
            session = self.client_sessions[client.client_id]
            t2_motion = load_tara_motion_csv(csv_path, x_offset=0.0)
            t2_frame_idx = min(session.frame_idx, t2_motion.root_positions.shape[0] - 1)
            offset = np.zeros(3, dtype=np.float64)
            offset[0] = float(human_root[0] - t2_motion.root_positions[t2_frame_idx, 0])
            offset[2] = float(human_root[2] - t2_motion.root_positions[t2_frame_idx, 2])
            return offset

        def wheel_to_human_offset(csv_path: Path, fps: float, *, mirror_z_axis: bool = False) -> np.ndarray:
            human_root = current_human_root_position()
            if human_root is None:
                return np.zeros(3, dtype=np.float64)
            session = self.client_sessions[client.client_id]
            wheel_motion = _read_diff_drive_csv(csv_path, fps)
            wheel_z = -wheel_motion["z"] if mirror_z_axis else wheel_motion["z"]
            wheel_frame_idx = min(session.frame_idx, len(wheel_motion["x"]) - 1)
            offset = np.zeros(3, dtype=np.float64)
            offset[0] = float(human_root[0] - wheel_motion["x"][wheel_frame_idx])
            offset[2] = float(human_root[2] - wheel_z[wheel_frame_idx])
            return offset

        def update_memory_status() -> None:
            root = current_memories_root()
            stem = selected_memory_stem()
            if stem is None:
                memory_status.content = f"No BVH memories found in `{root / 'bvh'}`."
                return
            bvh_path = _memory_bvh_path(root, stem)
            t3_csv_path = _memory_t3_csv_path(root, stem)
            t2_csv_path = _memory_csv_path(root, stem)
            t3_csv_state = "available" if t3_csv_path.is_file() else "missing"
            t2_csv_state = "available" if t2_csv_path.is_file() else "missing"
            memory_status.content = (
                f"BVH: `{bvh_path}`\n\n"
                f"T3 CSV: `{t3_csv_state}`\n\n"
                f"`{t3_csv_path}`\n\n"
                f"T2 CSV fallback: `{t2_csv_state}`\n\n"
                f"`{t2_csv_path}`"
            )

        def update_base_memory_status() -> None:
            root = current_memories_root()
            stem = selected_base_memory_stem()
            if stem is None:
                base_memory_status.content = f"No base CSV memories found in `{root / 'wheel_csv'}`."
                return
            csv_path = _base_memory_csv_path(root, stem)
            csv_state = "available" if csv_path.is_file() else "missing"
            base_memory_status.content = (
                f"Base CSV: `{csv_state}`\n\n"
                f"`{csv_path}`"
            )

        def save_current_bvh(
            stem: str | None = None,
            output_root: Path | None = None,
            fps: float | None = None,
        ) -> Path:
            session = self.client_sessions[client.client_id]
            if "soma" not in session.model_name.lower():
                raise ValueError("Retargeting requires a SOMA model. Select a Kimodo-SOMA model first.")
            if not session.motions:
                raise ValueError("Generate or load a SOMA motion first.")

            motion = list(session.motions.values())[0]
            output_root = output_root or Path(output_root_text.value).expanduser().resolve()
            relative_stem = Path(stem) if stem is not None else Path(_safe_clip_name(clip_name_text.value))
            bvh_path = output_root / "bvh" / relative_stem.with_suffix(".bvh")
            npz_path = output_root / "kimodo_npz" / relative_stem.with_suffix(".npz")
            bvh_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path.parent.mkdir(parents=True, exist_ok=True)

            save_motion_bvh(
                str(bvh_path),
                motion.joints_local_rot,
                motion.joints_pos[:, session.skeleton.root_idx, :],
                skeleton=session.skeleton,
                fps=float(fps if fps is not None else _model_native_fps(self, session.model_name, session.model_fps)),
                standard_tpose=bool(standard_tpose_checkbox.value),
            )
            motion_data = {
                "posed_joints": motion.joints_pos.detach().cpu().numpy(),
                "global_rot_mats": motion.joints_rot.detach().cpu().numpy(),
                "local_rot_mats": motion.joints_local_rot.detach().cpu().numpy(),
                "root_positions": motion.joints_pos[:, session.skeleton.root_idx, :].detach().cpu().numpy(),
            }
            if motion.foot_contacts is not None:
                motion_data["foot_contacts"] = motion.foot_contacts.detach().cpu().numpy()
            save_kimodo_npz(str(npz_path), motion_data)

            metadata = {
                "stem": str(relative_stem),
                "model_name": session.model_name,
                "fps": float(fps if fps is not None else _model_native_fps(self, session.model_name, session.model_fps)),
                "num_frames": int(motion.joints_pos.shape[0]),
                "bvh_path": str(bvh_path),
                "npz_path": str(npz_path),
                "standard_tpose_bvh": bool(standard_tpose_checkbox.value),
            }
            meta_path = output_root / "metadata" / relative_stem.with_suffix(".json")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            workflow.output_root = output_root
            workflow.memory_stem = str(relative_stem)
            workflow.bvh_path = bvh_path
            workflow.npz_path = npz_path
            update_status(f"Saved SOMA BVH:\n\n`{bvh_path}`")
            return bvh_path

        def load_bvh_memory(bvh_path: Path) -> None:
            session = self.client_sessions[client.client_id]
            joints_pos, joints_rot, foot_contacts, skeleton = load_motion_file(bvh_path, self.device)
            self.clear_motions(client.client_id)
            session.skeleton = skeleton
            session.max_frame_idx = int(joints_pos.shape[0]) - 1
            try:
                session.model_fps = 1.0 / read_bvh_frame_time_seconds(bvh_path)
            except Exception:
                session.model_fps = _model_native_fps(self, session.model_name, session.model_fps or 30.0)
            client.timeline.set_fps(session.model_fps)
            session.cur_duration = (session.max_frame_idx + 1) / float(session.model_fps)
            self.add_character_motion(client, skeleton, joints_pos, joints_rot, foot_contacts)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            self.set_frame(client.client_id, 0)

        def load_memory(stem: str, *, preserve_base_pose: bool = False) -> None:
            root = current_memories_root()
            stem = _resolve_memory_stem(stem, root)
            bvh_path = _memory_bvh_path(root, stem)
            t3_csv_path = _memory_t3_csv_path(root, stem)
            t2_csv_path = _memory_csv_path(root, stem)
            if not bvh_path.is_file() and not t3_csv_path.is_file() and not t2_csv_path.is_file():
                raise FileNotFoundError(f"No BVH, T3 CSV, or T2 CSV found for memory `{stem}`.")
            if bvh_path.is_file():
                load_bvh_memory(bvh_path)
            workflow.output_root = root
            workflow.memory_stem = stem
            workflow.real_robot_previewed_memory_stem = stem
            workflow.bvh_path = bvh_path if bvh_path.is_file() else None
            workflow.npz_path = None
            if t3_csv_path.is_file():
                combined_csv_path_text.value = str(t3_csv_path)
                load_combined_t3_motion(t3_csv_path, preserve_base_pose=preserve_base_pose)
                if bvh_path.is_file():
                    update_status(f"Loaded memory with T3 CSV:\n\n`{stem}`")
                else:
                    update_status(f"Loaded T3 CSV-only robot memory:\n\n`{stem}`")
            elif t2_csv_path.is_file():
                load_t2_preview(t2_csv_path, preserve_base_pose=preserve_base_pose)
                if bvh_path.is_file():
                    update_status(f"Loaded memory with fallback T2 CSV:\n\n`{stem}`")
                else:
                    update_status(f"Loaded T2 CSV-only robot memory:\n\n`{stem}`")
            else:
                workflow.clear_t2_preview(clear_wheel_base=False)
                update_status(f"Loaded BVH memory without robot CSV:\n\n`{stem}`")
            csv_path_text.value = str(t2_csv_path)
            output_root_text.value = str(root)
            clip_name_text.value = Path(stem).name
            session = self.client_sessions.get(client.client_id)
            if session is not None:
                outbound_rack = _outbound_rack_from_memory_stem(stem)
                pick_request = _pick_request_from_memory_stem(stem)
                if outbound_rack is not None and session.motions:
                    outbound_motion = next(iter(session.motions.values()))
                    session.human_outbound_rack_poses[outbound_rack] = (
                        outbound_motion.joints_pos[-1].detach().cpu().numpy().copy(),
                        outbound_motion.joints_rot[-1].detach().cpu().numpy().copy(),
                    )
                    session.human_route_rack = outbound_rack
                    session.human_route_kind = "outbound"
                elif pick_request is not None and session.motions:
                    object_key = (
                        pick_request.rack_name,
                        pick_request.shelf_number,
                        pick_request.object_index,
                    )
                    object_position = workflow.rack_object_home_positions.get(object_key)
                    if object_position is not None:
                        pick_motion = next(iter(session.motions.values()))
                        restored_hand_side = "left" if pick_request.object_index == 3 else "right"
                        restored_hand_label = "Left" if restored_hand_side == "left" else "Right"
                        restored_hand_names = (
                            pick_motion.skeleton.left_hand_joint_names
                            if restored_hand_side == "left"
                            else pick_motion.skeleton.right_hand_joint_names
                        )
                        hand_name = restored_hand_names[0]
                        hand_index = pick_motion.skeleton.bone_order_names.index(hand_name)
                        middle_name = (
                            f"{restored_hand_label}HandMiddleEnd"
                            if f"{restored_hand_label}HandMiddleEnd" in pick_motion.skeleton.bone_order_names
                            else restored_hand_names[-1]
                        )
                        middle_index = pick_motion.skeleton.bone_order_names.index(middle_name)
                        attach_frame = min(
                            pick_motion.length - 1,
                            max(0, int(round(pick_motion.length * 0.52))),
                        )
                        wrist_attach = pick_motion.joints_pos[attach_frame, hand_index].detach().cpu().numpy()
                        middle_attach = pick_motion.joints_pos[attach_frame, middle_index].detach().cpu().numpy()
                        palm_attach = 0.35 * wrist_attach + 0.65 * middle_attach
                        hand_rotation_attach = pick_motion.joints_rot[attach_frame, hand_index].detach().cpu().numpy()
                        workflow.picked_rack_object = object_key
                        workflow.picked_rack_objects_by_hand.clear()
                        workflow.pick_hand_side = restored_hand_side
                        workflow.pick_hold_start_frame = attach_frame
                        workflow.pick_hold_start_frames_by_side.clear()
                        workflow.pick_object_hand_offset = (
                            hand_rotation_attach.T @ (np.asarray(object_position, dtype=np.float64) - palm_attach)
                        )
                        workflow.pick_object_hand_offsets_by_side.clear()
                        workflow.pick_reach_start_frame = 0
                        workflow.pick_hand_target_world = None
                        workflow.pick_target_path_world = None
                        workflow.pick_grasp_target_world = None
                        workflow.pick_grasp_verified = True
                        workflow.pick_elbow_bend_hint_world = None
                        workflow.place_release_frame = None
                        workflow.place_object_position_world = None
                        workflow.place_object_positions_by_hand.clear()
                        workflow.place_reach_start_frame = None
                        workflow.place_hand_target_world = None
                        workflow.place_hand_targets_by_side.clear()
                        workflow.place_target_path_world = None
                        workflow.place_target_paths_by_side.clear()
                        workflow.place_elbow_bend_hint_world = None
                        workflow.place_elbow_bend_hints_by_side.clear()
                        session.human_outbound_rack_poses[pick_request.rack_name] = (
                            pick_motion.joints_pos[0].detach().cpu().numpy().copy(),
                            pick_motion.joints_rot[0].detach().cpu().numpy().copy(),
                        )
                        session.human_route_rack = pick_request.rack_name
                        session.human_route_kind = "pick_hold"
                self.set_frame(client.client_id, 0)
                session.play_once = True
                session.playing = True

        def sync_memory_from_external_change(stem: str, preserve_base_pose: bool = False) -> str:
            root = current_memories_root()
            stem = _resolve_memory_stem(stem, root)
            refresh_memory_options(stem)
            load_memory(stem, preserve_base_pose=preserve_base_pose)
            return stem

        def list_memories_for_control_api() -> dict[str, object]:
            root = current_memories_root()
            stems = _scan_memory_stems(root)
            return {
                "memories_root": str(root),
                "stems": stems,
                "labels": [_memory_label(root, stem) for stem in stems],
            }

        self._memory_sync_callbacks[client.client_id] = sync_memory_from_external_change
        self._memory_list_callbacks[client.client_id] = list_memories_for_control_api

        def set_retarget_buttons_disabled(disabled: bool) -> None:
            retarget_memory_button.disabled = disabled
            retarget_memory_t3_button.disabled = disabled
            retarget_button.disabled = disabled
            retarget_t3_button.disabled = disabled

        def retarget_bvh_to_memory_csv(bvh_path: Path, output_root: Path, notify_client: viser.ClientHandle) -> None:
            if workflow.retarget_running:
                notify_client.add_notification(
                    title="Retarget already running",
                    body="Wait for the current soma-retargeter job to finish.",
                    auto_close_seconds=4.0,
                    color="orange",
                )
                return

            if self.retarget_server_url:
                relative_stem = self._retarget_relative_stem(bvh_path, output_root)
                expected_csv_path = output_root / "t2_csv" / relative_stem.with_suffix(".csv")
                job = None
            else:
                job = SomaT2RetargetJob(
                    retargeter_root=Path(retargeter_root_text.value).expanduser().resolve(),
                    bvh_path=bvh_path,
                    output_root=output_root,
                    conda_env=str(conda_env_text.value).strip() or "soma-retargeter",
                )
                relative_stem = job.relative_stem
                expected_csv_path = job.csv_path
            workflow.retarget_running = True
            set_retarget_buttons_disabled(True)
            update_status(f"Retargeting `{relative_stem}` to `{expected_csv_path}`...")
            retarget_notif = notify_client.add_notification(
                title="Retargeting started",
                body="Remote soma-retargeter is running." if self.retarget_server_url else "soma-retargeter is running headless.",
                loading=True,
                with_close_button=False,
            )

            def run_job() -> None:
                try:
                    if self.retarget_server_url:
                        retarget_job, log_path = self._run_remote_retarget_bvh_to_csv(bvh_path, output_root)
                    else:
                        assert job is not None
                        result = job.run()
                        log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".retarget.log")
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_text(result.stdout or "", encoding="utf-8")
                        if result.returncode != 0:
                            raise RuntimeError(f"soma-retargeter failed with exit code {result.returncode}. Log: {log_path}")
                        if not job.csv_path.is_file():
                            raise FileNotFoundError(f"Expected retarget CSV was not created: {job.csv_path}")
                        retarget_job = job
                    workflow.output_root = output_root
                    workflow.memory_stem = str(retarget_job.relative_stem)
                    workflow.bvh_path = bvh_path
                    if bvh_path.is_file():
                        load_bvh_memory(bvh_path)
                    load_t2_preview(retarget_job.csv_path)
                    refresh_memory_options(str(retarget_job.relative_stem))
                    retarget_notif.title = "Retargeting finished"
                    retarget_notif.body = str(retarget_job.csv_path)
                    retarget_notif.loading = False
                    retarget_notif.with_close_button = True
                    retarget_notif.auto_close_seconds = 5.0
                    retarget_notif.color = "green"
                except Exception as exc:
                    update_status(f"Retarget failed:\n\n`{exc}`")
                    retarget_notif.title = "Retargeting failed"
                    retarget_notif.body = str(exc)
                    retarget_notif.loading = False
                    retarget_notif.with_close_button = True
                    retarget_notif.auto_close_seconds = 9.0
                    retarget_notif.color = "red"
                finally:
                    workflow.retarget_running = False
                    set_retarget_buttons_disabled(False)

            threading.Thread(target=run_job, daemon=True).start()

        def retarget_bvh_to_t3_memory_csv(
            bvh_path: Path,
            output_root: Path,
            notify_client: viser.ClientHandle,
        ) -> None:
            if workflow.retarget_running:
                notify_client.add_notification(
                    title="Retarget already running",
                    body="Wait for the current soma-retargeter job to finish.",
                    auto_close_seconds=4.0,
                    color="orange",
                )
                return

            if self.retarget_server_url:
                relative_stem = self._retarget_relative_stem(bvh_path, output_root)
                expected_t3_csv_path = output_root / "t3_csv" / relative_stem.with_suffix(".csv")
                job = None
            else:
                job = SomaT3RetargetJob(
                    retargeter_root=Path(retargeter_root_text.value).expanduser().resolve(),
                    bvh_path=bvh_path,
                    output_root=output_root,
                    conda_env=str(conda_env_text.value).strip() or "soma-retargeter",
                )
                relative_stem = job.relative_stem
                expected_t3_csv_path = job.t3_csv_path
            workflow.retarget_running = True
            set_retarget_buttons_disabled(True)
            update_status(f"Retargeting `{relative_stem}` to T3 CSV `{expected_t3_csv_path}`...")
            retarget_notif = notify_client.add_notification(
                title="T3 retargeting started",
                body="Remote soma-retargeter bvh_to_t3 is running."
                if self.retarget_server_url
                else "soma-retargeter bvh_to_t3 is running headless.",
                loading=True,
                with_close_button=False,
            )

            def run_job() -> None:
                try:
                    if self.retarget_server_url:
                        retarget_job, log_path = self._run_remote_retarget_bvh_to_t3_csv(bvh_path, output_root)
                    else:
                        assert job is not None
                        result = job.run()
                        log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".t3_retarget.log")
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log_path.write_text(result.stdout or "", encoding="utf-8")
                        if result.returncode != 0:
                            raise RuntimeError(f"T3 retargeter failed with exit code {result.returncode}. Log: {log_path}")
                        if not job.t3_csv_path.is_file():
                            raise FileNotFoundError(f"Expected T3 CSV was not created: {job.t3_csv_path}")
                        if not job.wheel_csv_path.is_file():
                            raise FileNotFoundError(f"Expected T3 wheel CSV was not created: {job.wheel_csv_path}")
                        retarget_job = job
                    workflow.output_root = output_root
                    workflow.memory_stem = str(retarget_job.relative_stem)
                    workflow.bvh_path = bvh_path
                    load_bvh_memory(bvh_path)
                    combined_csv_path_text.value = str(retarget_job.t3_csv_path)
                    load_combined_t3_motion(retarget_job.t3_csv_path)
                    refresh_memory_options(str(retarget_job.relative_stem))
                    retarget_notif.title = "T3 retargeting finished"
                    retarget_notif.body = str(retarget_job.t3_csv_path)
                    retarget_notif.loading = False
                    retarget_notif.with_close_button = True
                    retarget_notif.auto_close_seconds = 5.0
                    retarget_notif.color = "green"
                except Exception as exc:
                    update_status(f"T3 retarget failed:\n\n`{exc}`")
                    retarget_notif.title = "T3 retargeting failed"
                    retarget_notif.body = str(exc)
                    retarget_notif.loading = False
                    retarget_notif.with_close_button = True
                    retarget_notif.auto_close_seconds = 9.0
                    retarget_notif.color = "red"
                finally:
                    workflow.retarget_running = False
                    set_retarget_buttons_disabled(False)

            threading.Thread(target=run_job, daemon=True).start()

        def load_t2_preview(csv_path: Path, *, preserve_base_pose: bool = False) -> None:
            csv_path = csv_path.expanduser().resolve()
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)

            preserved_position: np.ndarray | None = None
            if (
                preserve_base_pose
                and workflow.wheel_base is not None
                and workflow.wheel_base.length > 0
            ):
                prior = workflow.wheel_base
                prior_index = min(prior.frame_idx, prior.length - 1)
                preserved_position = prior.position_offset.copy()
                preserved_position[0] += float(prior.motion["x"][prior_index])
                preserved_position[2] += float(prior.motion["z"][prior_index])

            workflow.clear_t2_preview()
            retargeter_root = Path(retargeter_root_text.value).expanduser().resolve()
            urdf_path = retargeter_root / "antt_t2" / "T2_serial_nero_arms.urdf"
            preview_offset = (
                preserved_position
                if preserved_position is not None
                else t2_to_human_offset(csv_path)
            )
            workflow.t2_motion = T2ViewerMotion(
                name=f"t2_preview_{client.client_id}",
                server=client,
                csv_path=csv_path,
                urdf_path=urdf_path if urdf_path.is_file() else None,
                x_offset=0.0,
                position_offset=preview_offset,
                color=(145, 145, 145),
                arms_only=bool(arms_only_checkbox.value),
            )
            workflow.t2_motion.set_frame(0)
            workflow.t2_motion.set_mesh_visibility(bool(show_t2_checkbox.value))
            csv_fps = _csv_frame_rate(csv_path, self.tara_fps)
            show_t2_checkbox.value = True
            show_t3_checkbox.value = False
            show_wheel_base_checkbox.value = False
            workflow.wheel_csv_path = None
            workflow.base_prompt_fps = csv_fps
            workflow.base_prompt_segments.clear()
            workflow.base_prompt_stem = None
            workflow.arm_frames = load_arm_frames(csv_path, start_frame=0, max_frames=None, frame_stride=1)
            workflow.gripper_widths = _load_right_gripper_widths(csv_path)
            workflow.base_wheel_rpms.clear()
            workflow.csv_path = csv_path
            csv_path_text.value = str(csv_path)
            session = self.client_sessions[client.client_id]
            session.model_fps = csv_fps
            session.max_frame_idx = workflow.t2_motion.length - 1
            session.cur_duration = workflow.t2_motion.length / csv_fps
            client.timeline.set_fps(csv_fps)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            update_status(
                "Loaded T2 preview and robot arm frames:\n\n"
                f"`{csv_path}`\n\n"
                f"{_csv_motion_summary(csv_path)}"
            )

        def load_combined_t3_motion(csv_path: Path, *, preserve_base_pose: bool = False) -> None:
            csv_path = csv_path.expanduser().resolve()
            required = {
                "left_motor_rpm",
                "right_motor_rpm",
                "time_s",
                "right_joint1_dof",
                "left_joint1_dof",
            }
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            if not _csv_has_columns(csv_path, required):
                raise ValueError("Combined T3 CSV must contain arm joints, time_s, and both motor RPM columns.")

            preserved_position: np.ndarray | None = None
            preserved_yaw: float | None = None
            if preserve_base_pose and workflow.wheel_base is not None and workflow.wheel_base.length > 0:
                prior = workflow.wheel_base
                prior_index = min(prior.frame_idx, prior.length - 1)
                preserved_position = prior.position_offset.copy()
                preserved_position[0] += float(prior.motion["x"][prior_index])
                preserved_position[2] += float(prior.motion["z"][prior_index])
                preserved_yaw = float(prior.motion["yaw"][prior_index])

            # Combined T3 playback owns the whole robot by itself. Clear
            # standalone T2 and wheel-base actors so the timeline drives only
            # the T3 URDF for this retarget target.
            if workflow.t2_motion is not None:
                workflow.t2_motion.clear()
                workflow.t2_motion = None
            if workflow.wheel_base is not None:
                workflow.wheel_base.clear()
                workflow.wheel_base = None

            fps = _csv_frame_rate(csv_path, self.tara_fps)
            position_offset = (
                preserved_position
                if preserved_position is not None
                else wheel_to_human_offset(csv_path, fps, mirror_z_axis=True)
            )
            start_pose = (
                (0.0, 0.0, preserved_yaw)
                if preserved_yaw is not None
                else None
            )
            if not DEFAULT_T3_URDF_PATH.is_file():
                raise FileNotFoundError(f"T3 URDF not found: {DEFAULT_T3_URDF_PATH}")
            if workflow.t3_motion is None:
                workflow.t3_motion = T3Playback(
                    client,
                    DEFAULT_T3_URDF_PATH,
                    csv_path,
                    csv_path,
                    fps,
                    stiff_posture=True,
                    root_node_name=f"/t3_robot_{client.client_id}",
                    position_offset=position_offset,
                    visual_yaw_offset_rad=0.0,
                    initial_start_pose=start_pose,
                    mirror_wheel_z_axis=True,
                )
            else:
                workflow.t3_motion.visual_yaw_offset_rad = 0.0
                workflow.t3_motion.load_csvs(
                    csv_path,
                    csv_path,
                    fps,
                    start_pose=start_pose,
                    position_offset=position_offset,
                    mirror_wheel_z_axis=True,
                )
            workflow.freeze_t3_base = False
            workflow.t3_motion.set_base_frozen(False)
            workflow.t3_motion.apply_frame(0)

            show_t2_checkbox.value = False
            show_wheel_base_checkbox.value = False
            show_t3_checkbox.value = True
            set_t3_visible(True)
            workflow.wheel_csv_path = csv_path
            workflow.base_prompt_fps = fps
            workflow.base_prompt_segments.clear()
            workflow.base_prompt_stem = None
            workflow.arm_frames = load_arm_frames(csv_path, start_frame=0, max_frames=None, frame_stride=1)
            workflow.gripper_widths = _load_right_gripper_widths(csv_path)
            workflow.base_wheel_rpms = _load_base_wheel_rpms(csv_path)
            workflow.csv_path = csv_path
            csv_path_text.value = str(csv_path)
            session = self.client_sessions[client.client_id]
            session.model_fps = fps
            session.max_frame_idx = min(len(workflow.arm_frames), workflow.t3_motion.length) - 1
            session.cur_duration = (session.max_frame_idx + 1) / fps
            client.timeline.set_fps(fps)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            self.set_frame(client.client_id, 0)
            combined_csv_path_text.value = str(csv_path)
            combined_status.content = (
                f"Loaded `{csv_path.name}`: `{session.max_frame_idx + 1}` synchronized rows at `{fps:.2f} Hz`.\n\n"
                "T3 preview, TaraBase RPM, and both arms now share the same frame clock."
            )

        def play_loaded_motion_once() -> None:
            session = self.client_sessions[client.client_id]
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = False
            workflow.tara_stop_event.set()
            self.set_frame(client.client_id, 0)
            session.play_once = True
            session.playing = True

        def load_base_memory(stem: str, *, chain: bool | None = None) -> None:
            root = current_memories_root()
            csv_path = _base_memory_csv_path(root, stem).expanduser().resolve()
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)
            if not DEFAULT_WHEEL_URDF_PATH.is_file():
                raise FileNotFoundError(f"Tara wheel-base URDF not found: {DEFAULT_WHEEL_URDF_PATH}")

            # Loading only prepares the preview. Never inherit a running or
            # looping timeline from the previously loaded memory.
            session = self.client_sessions[client.client_id]
            session.playing = False
            session.play_once = False
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = False
            if workflow.connection.is_connected():
                workflow.connection.disconnect()

            should_chain = workflow.chain_loaded_base_memories if chain is None else chain
            previous_final_pose: tuple[float, float, float] | None = None
            if (
                should_chain
                and workflow.wheel_base is not None
                and workflow.wheel_base.csv_path is not None
                and workflow.wheel_base.length > 0
            ):
                previous_motion = workflow.wheel_base.motion
                previous_final_pose = (
                    float(previous_motion["x"][-1]),
                    float(previous_motion["z"][-1]),
                    float(previous_motion["yaw"][-1]),
                )
            position_offset = (
                workflow.wheel_base.position_offset.copy()
                if should_chain and workflow.wheel_base is not None
                else np.zeros(3, dtype=np.float64)
            )
            if workflow.t2_motion is not None:
                workflow.t2_motion.clear()
                workflow.t2_motion = None
            if workflow.t3_motion is not None:
                workflow.t3_motion.clear()
                workflow.t3_motion = None
            if workflow.wheel_base is not None:
                workflow.wheel_base.clear()
            base_fps = _csv_frame_rate(csv_path, workflow.base_prompt_fps)
            workflow.wheel_base = WheelBasePlayback(
                client,
                DEFAULT_WHEEL_URDF_PATH,
                csv_path,
                base_fps,
                root_node_name=f"/tara_wheel_base_{client.client_id}",
                position_offset=position_offset,
                visual_yaw_offset_rad=float(np.pi),
            )
            if previous_final_pose is not None:
                workflow.wheel_base.align_start_pose(*previous_final_pose)
            workflow.wheel_base.apply_frame(0)
            show_wheel_base_checkbox.value = True
            workflow.wheel_base.set_visible(True)
            show_t3_checkbox.value = False
            workflow.wheel_csv_path = csv_path
            workflow.base_wheel_rpms = _load_base_wheel_rpms(csv_path)
            workflow.arm_frames.clear()
            workflow.gripper_widths.clear()
            workflow.csv_path = None
            workflow.base_prompt_fps = base_fps
            workflow.base_prompt_segments.clear()
            workflow.base_prompt_stem = stem
            base_csv_name_text.value = Path(stem).name
            session.model_fps = workflow.base_prompt_fps
            session.max_frame_idx = workflow.wheel_base.length - 1
            session.cur_duration = workflow.wheel_base.length / workflow.base_prompt_fps
            client.timeline.set_fps(workflow.base_prompt_fps)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            update_status(
                "Loaded TaraBase CSV memory:\n\n"
                f"`{csv_path}`\n\n"
                + (
                    "Chained to the previous base memory's final position and heading.\n\n"
                    if previous_final_pose is not None
                    else ""
                )
                + "Use **Play / Resume TaraBase** to send this CSV to the base."
            )

        def base_memory_status() -> dict[str, object]:
            session = self.client_sessions[client.client_id]
            csv_path = workflow.wheel_base.csv_path if workflow.wheel_base is not None else None
            return {
                "playing": bool(session.playing),
                "frame": int(session.frame_idx),
                "max_frame": int(session.max_frame_idx),
                "memory_stem": workflow.base_prompt_stem,
                "csv_path": str(csv_path) if csv_path is not None else None,
                "chain_enabled": bool(workflow.chain_loaded_base_memories),
                "preview_only": not bool(workflow.stream_real_robot_playback),
            }

        def list_base_memories_for_control_api() -> dict[str, object]:
            root = current_memories_root()
            stems = _scan_base_memory_stems(root)
            return {"memories_root": str(root), "stems": stems}

        def control_base_memory_from_external_change(
            payload: dict[str, object],
        ) -> dict[str, object]:
            action = str(payload.get("action") or "status").strip().lower().replace("_", "-")
            session = self.client_sessions[client.client_id]
            if action == "status":
                return {"action": action, "status": base_memory_status()}
            if action == "stop":
                session.playing = False
                session.play_once = False
                workflow.stream_real_robot_playback = False
                return {"action": action, "status": base_memory_status()}
            if action in {"t3-load", "t3-preview", "t3-play"}:
                requested = str(payload.get("motion") or payload.get("memory") or "").strip().lower()
                aliases = {
                    "pick": "pick_item",
                    "pick-item": "pick_item",
                    "pick_item": "pick_item",
                    "drop": "drop_item",
                    "drop-item": "drop_item",
                    "drop_item": "drop_item",
                    "place": "drop_item",
                }
                motion_name = aliases.get(requested)
                if motion_name is None:
                    raise ValueError(f"Unknown combined T3 motion: {requested!r}")
                csv_path = T3_COMBINED_MOTION_PATHS[motion_name]
                if not csv_path.is_file():
                    raise FileNotFoundError(csv_path)
                workflow.stream_real_robot_playback = False
                workflow.real_robot_approval_pending = False
                load_combined_t3_motion(csv_path)
                should_play = action != "t3-load"
                session.play_once = should_play
                session.playing = should_play
                return {
                    "action": action,
                    "motion": motion_name,
                    "csv_path": str(csv_path),
                    "status": base_memory_status(),
                }
            if action not in {"load", "play", "preview"}:
                raise ValueError(f"Unknown base action: {action}")

            stem = str(payload.get("memory") or payload.get("stem") or "").strip()
            if not stem:
                raise ValueError("Missing base memory stem")
            root = current_memories_root()
            stem = _resolve_base_memory_stem(stem, root)
            chain_value = payload.get("chain")
            chain = bool(chain_value) if chain_value is not None else workflow.chain_loaded_base_memories
            workflow.chain_loaded_base_memories = chain

            # External planning is preview-only. A separate, explicit hardware
            # approval path remains responsible for sending wheel commands.
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = False
            load_base_memory(stem, chain=chain)
            self.set_frame(client.client_id, 0)
            session.play_once = action in {"play", "preview"}
            session.playing = action in {"play", "preview"}
            return {"action": action, "stem": stem, "status": base_memory_status()}

        self._base_memory_list_callbacks[client.client_id] = list_base_memories_for_control_api
        self._base_memory_control_callbacks[client.client_id] = control_base_memory_from_external_change

        def set_tara_status(message: str) -> None:
            tara_websocket_status.content = message

        def current_tara_remote_url() -> str | None:
            value = str(tara_remote_url_text.value).strip().rstrip("/")
            return value or None

        def websocket_url_is_valid(event: viser.GuiEvent) -> bool:
            remote_url = current_tara_remote_url()
            if remote_url and remote_url.startswith(("ws://", "wss://")):
                return True
            message = "Enter a TaraBase URL beginning with `ws://` or `wss://` first."
            set_tara_status(message)
            event.client.add_notification(
                title="TaraBase WebSocket URL required",
                body="Example: ws://PI_IP:8094",
                color="orange",
            )
            return False

        def tara_remote_worker(csv_path: Path, start_index: int, reverse_playback: bool) -> None:
            try:
                options = {
                    "port": self.tara_port,
                    "slave_id": self.tara_slave_id,
                    "baudrate": self.tara_baudrate,
                    "fps": workflow.base_prompt_fps,
                    "speed_scale": self.tara_rpm_scale,
                    "max_abs_rpm": self.tara_max_rpm,
                    "reverse_playback": reverse_playback,
                    "debug": self.tara_debug,
                    "start_index": start_index,
                }
                remote_url = current_tara_remote_url()
                if remote_url:
                    payload = {"action": "stream", "csv_text": csv_path.read_text(encoding="utf-8"), "options": options}
                    if remote_url.startswith(("ws://", "wss://")):
                        from websockets.sync.client import connect

                        with connect(remote_url, open_timeout=10, close_timeout=2) as websocket:
                            websocket.send(json.dumps(payload))
                            stream_started = False
                            while not workflow.tara_stop_event.is_set():
                                remote_status = json.loads(websocket.recv())
                                if remote_status.get("type") == "error":
                                    raise RuntimeError(str(remote_status.get("error", "Unknown TaraBase error")))
                                if remote_status.get("type") == "accepted":
                                    stream_started = True
                                    continue
                                if remote_status.get("type") != "status":
                                    continue
                                running = bool(remote_status.get("running", False))
                                stream_started = stream_started or running
                                workflow.tara_next_index = int(remote_status.get("index", workflow.tara_next_index))
                                set_tara_status(str(remote_status.get("message", "TaraBase stream running.")))
                                if stream_started and not running:
                                    break
                        return

                    request = urllib.request.Request(
                        f"{remote_url}/stream",
                        data=json.dumps({"csv_text": payload["csv_text"], "options": options}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    urllib.request.urlopen(request, timeout=10).close()
                    while not workflow.tara_stop_event.wait(0.2):
                        with urllib.request.urlopen(f"{remote_url}/status", timeout=5) as response:
                            remote_status = json.load(response)
                        workflow.tara_next_index = int(remote_status.get("index", workflow.tara_next_index))
                        set_tara_status(str(remote_status.get("message", "TaraBase stream running.")))
                        if not remote_status.get("running", False):
                            break
                    return

                stream_wheel_commands(
                    csv_path=csv_path,
                    port=self.tara_port,
                    slave_id=self.tara_slave_id,
                    baudrate=self.tara_baudrate,
                    fps=options["fps"],
                    speed_scale=self.tara_rpm_scale,
                    max_abs_rpm=self.tara_max_rpm,
                    left_motor_sign=1,
                    right_motor_sign=-1,
                    invert_turn_direction=True,
                    fit_to_rpm_limit=True,
                    reverse_playback=reverse_playback,
                    debug=self.tara_debug,
                    start_index=start_index,
                    stop_event=workflow.tara_stop_event,
                    status_callback=set_tara_status,
                )
            except Exception as exc:
                set_tara_status(f"TaraBase send failed: `{exc}`")
            finally:
                workflow.tara_thread = None

        def start_tara_stream(*, restart: bool, reverse_playback: bool | None = None) -> None:
            if workflow.tara_thread is not None and workflow.tara_thread.is_alive():
                set_tara_status("TaraBase is already playing.")
                return
            if workflow.wheel_csv_path is None or not workflow.wheel_csv_path.is_file():
                set_tara_status("Load a T2 preview first; its TaraBase CSV will be created automatically.")
                return
            direction = workflow.tara_active_direction if reverse_playback is None else ("reverse" if reverse_playback else "forward")
            if restart or direction != workflow.tara_active_direction:
                workflow.tara_next_index = 0
            workflow.tara_active_direction = direction
            workflow.tara_stop_event.clear()
            remote_url = current_tara_remote_url()
            set_tara_status(
                f"Starting `{direction}` from row `{workflow.tara_next_index + 1}` via "
                f"`{remote_url or self.tara_port}`..."
            )
            workflow.tara_thread = threading.Thread(
                target=tara_remote_worker,
                args=(workflow.wheel_csv_path, workflow.tara_next_index, direction == "reverse"),
                name=f"tara-stream-{client.client_id}",
                daemon=True,
            )
            workflow.tara_thread.start()

        def pause_tara_stream() -> None:
            workflow.tara_stop_event.set()
            remote_url = current_tara_remote_url()
            if remote_url:
                try:
                    if remote_url.startswith(("ws://", "wss://")):
                        from websockets.sync.client import connect

                        with connect(remote_url, open_timeout=5, close_timeout=2) as websocket:
                            websocket.send(json.dumps({"action": "stop"}))
                    else:
                        request = urllib.request.Request(f"{remote_url}/stop", data=b"{}", method="POST")
                        urllib.request.urlopen(request, timeout=5).close()
                except Exception as exc:
                    set_tara_status(f"Remote pause failed: `{exc}`")
                    return
            set_tara_status(f"Pause requested. Resume near row `{workflow.tara_next_index + 1}`.")

        def control_job_id(prompt: str) -> str:
            digest = hashlib.sha1(f"{client.client_id}:{time.time()}:{prompt}".encode("utf-8")).hexdigest()[:12]
            return f"generate_retarget_{digest}"

        def set_control_job(job_id: str, **updates: object) -> None:
            with self._control_jobs_lock:
                current = dict(self._control_jobs.get(job_id) or {})
                current.update(updates)
                current["job_id"] = job_id
                current["updated_at"] = time.time()
                self._control_jobs[job_id] = current

        def run_retarget_bvh_to_t3_memory_csv_sync(
            bvh_path: Path,
            output_root: Path,
        ) -> tuple[SomaT3RetargetJob | RemoteT3RetargetJob, Path]:
            if self.retarget_server_url:
                return self._run_remote_retarget_bvh_to_t3_csv(bvh_path, output_root)
            job = SomaT3RetargetJob(
                retargeter_root=Path(retargeter_root_text.value).expanduser().resolve(),
                bvh_path=bvh_path,
                output_root=output_root,
                conda_env=str(conda_env_text.value).strip() or "soma-retargeter",
            )
            result = job.run()
            log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".t3_retarget.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout or "", encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"T3 retargeter failed with exit code {result.returncode}. Log: {log_path}")
            if not job.t3_csv_path.is_file():
                raise FileNotFoundError(f"Expected T3 CSV was not created: {job.t3_csv_path}")
            if not job.wheel_csv_path.is_file():
                raise FileNotFoundError(f"Expected T3 wheel CSV was not created: {job.wheel_csv_path}")
            return job, log_path

        def generate_save_retarget_from_control_api(payload: dict[str, object]) -> dict[str, object]:
            raw_prompts = payload.get("prompts")
            if isinstance(raw_prompts, list):
                prompts = [str(item).strip() for item in raw_prompts if str(item).strip()]
            else:
                prompts = [str(payload.get("prompt") or "").strip()]
            prompts = [item for item in prompts if item]
            if not prompts:
                raise ValueError("Missing prompt")
            prompt = prompts[0]
            prompt_label = " | ".join(prompts)
            session = self.client_sessions[client.client_id]
            if "soma" not in session.model_name.lower():
                raise ValueError("Generate-retarget requires a SOMA model. Select a Kimodo-SOMA model first.")
            duration = float(payload.get("duration_seconds") or session.cur_duration or DEFAULT_CUR_DURATION)
            duration = max(0.1, _resolve_generate_duration_seconds(prompts, duration, float(session.model_fps)))
            seed = int(payload.get("seed") or 42)
            diffusion_steps = int(payload.get("diffusion_steps") or 100)
            stem = str(payload.get("stem") or "").strip() or _new_generated_stem()
            output_root = Path(str(payload.get("output_root") or current_memories_root())).expanduser().resolve()
            frames_per_prompt = max(1, int(round(duration * float(session.model_fps))))
            num_frames = [frames_per_prompt for _ in prompts]
            total_frames = sum(num_frames)
            job_id = control_job_id(prompt_label)
            session.cur_duration = total_frames / float(session.model_fps)
            session.max_frame_idx = total_frames - 1
            client.timeline.clear_prompts()
            start_frame = 0
            for index, item in enumerate(prompts):
                end_frame = start_frame + num_frames[index] - 1
                client.timeline.add_prompt(item, start_frame, end_frame, color=(64, 124, 186))
                start_frame = end_frame + 1
            client.timeline.set_current_frame(0)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            self.set_frame(client.client_id, 0)

            client.flush()

            set_control_job(
                job_id,
                status="queued",
                prompt=prompt_label,
                prompts=prompts,
                stem=stem,
                output_root=str(output_root),
                client_id=client.client_id,
            )
            update_prompt_job(
                    "**Prompt-to-Robot Job**\n\n"
                    "Status: `queued`\n\n"
                    f"Prompt: `{prompt_label}`\n\n"
                    f"Job: `{job_id}`\n\n"
                    f"Target stem: `{stem}`\n\n"
                "Progress:\n\n"
                "- [x] Queued\n"
                "- [ ] Generate SOMA motion\n"
                "- [ ] Save memory\n"
                "- [ ] Retarget to T3\n"
                "- [ ] Load result"
            )
            progress_notif = client.add_notification(
                title="Prompt-to-robot queued",
                body=f"`{prompt_label}`",
                loading=True,
                with_close_button=False,
            )
            client.flush()

            def show_control_stage(
                title: str,
                body: str,
                status: str,
                progress_lines: list[str],
                **job_updates: object,
            ) -> None:
                set_control_job(job_id, status=status, **job_updates)
                update_prompt_job(
                    "**Prompt-to-Robot Job**\n\n"
                    f"Status: `{status}`\n\n"
                    f"Prompt: `{prompt_label}`\n\n"
                    f"Job: `{job_id}`\n\n"
                    f"Stage: **{title}**\n\n"
                    f"{body}\n\n"
                    "Progress:\n\n"
                    + "\n".join(progress_lines)
                )
                update_status(
                    f"{title}\n\n"
                    f"Prompt: `{prompt_label}`\n\n"
                    f"Job: `{job_id}`\n\n"
                    f"{body}"
                )
                progress_notif.title = title
                progress_notif.body = body.replace("\n\n", "\n")
                progress_notif.loading = status not in {"done", "error"}
                client.flush()

            def worker() -> None:
                try:
                    show_control_stage(
                        "Generating SOMA motion",
                        f"Frames: `{total_frames}`\n\nSeed: `{seed}`\n\nDenoising steps: `{diffusion_steps}`",
                        "generating",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [ ] Save memory",
                            "- [ ] Retarget to T3",
                            "- [ ] Load result",
                        ],
                    )
                    self.generate(
                        client,
                        prompts,
                        num_frames,
                        1,
                        seed,
                        diffusion_steps,
                        cfg_weight=[2.0, 2.0],
                        cfg_type="separated",
                        postprocess_parameters={
                            "post_processing": "g1" not in session.model_name.lower(),
                            "root_margin": 0.04,
                        },
                        transitions_parameters={"num_transition_frames": NB_TRANSITION_FRAMES},
                        real_robot_rotations=False,
                    )
                    session.cur_duration = total_frames / float(session.model_fps)
                    session.max_frame_idx = total_frames - 1
                    self.set_frame(client.client_id, 0)

                    show_control_stage(
                        "Saving generated memory",
                        f"Writing BVH/NPZ under:\n\n`{output_root}`\n\nStem: `{stem}`",
                        "saving",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [x] Save memory",
                            "- [ ] Retarget to T3",
                            "- [ ] Load result",
                        ],
                    )
                    bvh_path = save_current_bvh(
                        stem=stem,
                        output_root=output_root,
                        fps=_model_native_fps(self, session.model_name, session.model_fps),
                    )
                    refresh_memory_options(stem)

                    show_control_stage(
                        "Retargeting to T3 robot",
                        f"BVH: `{bvh_path}`",
                        "retargeting",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [x] Save memory",
                            "- [x] Retarget to T3",
                            "- [ ] Load result",
                        ],
                        bvh_path=str(bvh_path),
                        retarget_target="t3",
                    )
                    retarget_job, log_path = run_retarget_bvh_to_t3_memory_csv_sync(bvh_path, output_root)
                    workflow.output_root = output_root
                    workflow.memory_stem = str(retarget_job.relative_stem)
                    workflow.bvh_path = bvh_path
                    load_bvh_memory(bvh_path)
                    combined_csv_path_text.value = str(retarget_job.t3_csv_path)
                    load_combined_t3_motion(retarget_job.t3_csv_path)
                    play_loaded_motion_once()
                    refresh_memory_options(str(retarget_job.relative_stem))
                    show_control_stage(
                        "Prompt-to-robot complete",
                        "Generated T3 motion is loaded in Viser and playing exactly once.\n\n"
                        f"Stem: `{retarget_job.relative_stem}`\n\n"
                        f"BVH: `{bvh_path}`\n\n"
                        f"T3 CSV: `{retarget_job.t3_csv_path}`\n\n"
                        f"Wheel CSV: `{retarget_job.wheel_csv_path}`\n\n"
                        f"Log: `{log_path}`",
                        "done",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [x] Save memory",
                            "- [x] Retarget to T3",
                            "- [x] Load result",
                        ],
                        stem=str(retarget_job.relative_stem),
                        bvh_path=str(bvh_path),
                        csv_path=str(retarget_job.t3_csv_path),
                        t3_csv_path=str(retarget_job.t3_csv_path),
                        wheel_csv_path=str(retarget_job.wheel_csv_path),
                        log_path=str(log_path),
                        retarget_target="t3",
                        playback_started=True,
                        playback_mode="viser_once",
                    )
                    progress_notif.loading = False
                    progress_notif.with_close_button = True
                    progress_notif.auto_close_seconds = 8.0
                    progress_notif.color = "green"
                except Exception as exc:
                    show_control_stage(
                        "Prompt-to-robot failed",
                        f"Error: `{exc}`",
                        "error",
                        [
                            "- [x] Queued",
                            "- [ ] Generate SOMA motion",
                            "- [ ] Save memory",
                            "- [ ] Retarget to T3",
                            "- [ ] Load result",
                        ],
                        error=str(exc),
                    )
                    progress_notif.loading = False
                    progress_notif.with_close_button = True
                    progress_notif.auto_close_seconds = 10.0
                    progress_notif.color = "red"

            threading.Thread(target=worker, name=f"generate-retarget-{job_id}", daemon=True).start()
            return {"job_id": job_id, "status": "queued", "stem": stem}

        self._generate_retarget_callbacks[client.client_id] = generate_save_retarget_from_control_api

        @show_human_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_human_mesh_visible(bool(show_human_checkbox.value))

        @show_t2_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_t2_visible(bool(show_t2_checkbox.value))

        @show_t3_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_t3_visible(bool(show_t3_checkbox.value))

        @freeze_t3_base_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            workflow.freeze_t3_base = bool(freeze_t3_base_checkbox.value)
            if workflow.t3_motion is not None:
                workflow.t3_motion.set_base_frozen(workflow.freeze_t3_base)
            if workflow.wheel_base is not None:
                frame_idx = self.client_sessions[client.client_id].frame_idx
                workflow.wheel_base.apply_frame(0 if workflow.freeze_t3_base else frame_idx)
            if workflow.freeze_t3_base and workflow.connection.is_connected():
                frame_idx = self.client_sessions[client.client_id].frame_idx
                workflow.connection.send_base_frame(frame_idx, (0.0, 0.0))
            if (
                workflow.freeze_t3_base
                and workflow.tara_thread is not None
                and workflow.tara_thread.is_alive()
            ):
                pause_tara_stream()

        @show_wheel_base_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_wheel_base_visible(bool(show_wheel_base_checkbox.value))

        @show_floor_grid_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_floor_grid_visible(bool(show_floor_grid_checkbox.value))

        def start_ros_base_playback(
            event: viser.GuiEvent,
            *,
            restart: bool,
            reverse_playback: bool = False,
        ) -> None:
            if workflow.wheel_csv_path is None or not workflow.wheel_csv_path.is_file():
                event.client.add_notification(
                    title="No TaraBase RPM rows loaded",
                    body="Load a combined T3 CSV or a base CSV memory first.",
                    color="red",
                )
                return
            session = self.client_sessions[client.client_id]
            if restart:
                self.set_frame(client.client_id, 0)
            if bool(enable_t3_hardware_checkbox.value):
                dry_run_checkbox.value = False
                if workflow.connection.is_connected() and workflow.connection.dry_run:
                    workflow.connection.disconnect()
                connect_robot(event)
                if not workflow.connection.is_connected():
                    tara_status.content = "ROS robot stream did not connect."
                    return
                # Match WebSocket playback progress instead of coupling the
                # hardware start row to whichever Viser frame is selected.
                start_index = (
                    0
                    if restart or reverse_playback
                    else workflow.tara_next_index
                )
                workflow.connection.send_base_csv(
                    workflow.wheel_csv_path.read_text(encoding="utf-8"),
                    {
                        "fps": workflow.base_prompt_fps,
                        "speed_scale": self.tara_rpm_scale,
                        "max_abs_rpm": self.tara_max_rpm,
                        "reverse_playback": reverse_playback,
                        "start_index": start_index,
                        "fit_to_rpm_limit": True,
                    },
                )
                # The ROS Tara node now owns the base frame clock. Keep the
                # Viser loop from also publishing per-frame wheel commands.
                workflow.stream_real_robot_playback = False
                tara_status.content = (
                    f"ROS TaraBase CSV playback accepted on `/base/play_csv` from row "
                    f"`{start_index + 1}` ({'reverse' if reverse_playback else 'forward'})."
                )
            else:
                workflow.stream_real_robot_playback = False
                tara_status.content = (
                    "Previewing TaraBase only; enable real T3 hardware to publish ROS commands."
                )
            session.play_once = True
            session.playing = True

        def stop_ros_base_playback() -> None:
            session = self.client_sessions[client.client_id]
            workflow.tara_next_index = session.frame_idx
            session.play_once = False
            session.playing = False
            workflow.stream_real_robot_playback = False
            if workflow.connection.is_connected():
                try:
                    workflow.connection.stop_base_csv()
                except RuntimeError:
                    pass
            workflow.connection.disconnect()
            tara_status.content = "Stopped ROS CSV playback and disabled TaraBase motors."

        @tara_play_button.on_click
        def _(event: viser.GuiEvent) -> None:
            start_ros_base_playback(event, restart=False)

        @tara_pause_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            stop_ros_base_playback()

        @tara_restart_button.on_click
        def _(event: viser.GuiEvent) -> None:
            start_ros_base_playback(event, restart=True)

        @tara_reverse_button.on_click
        def _(event: viser.GuiEvent) -> None:
            start_ros_base_playback(event, restart=True, reverse_playback=True)

        @tara_websocket_play_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if websocket_url_is_valid(event):
                start_tara_stream(restart=False)

        @tara_websocket_pause_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if websocket_url_is_valid(event):
                pause_tara_stream()

        @tara_websocket_restart_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if websocket_url_is_valid(event):
                start_tara_stream(restart=True, reverse_playback=False)

        @tara_websocket_reverse_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if websocket_url_is_valid(event):
                start_tara_stream(restart=True, reverse_playback=True)

        @load_combined_csv_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                load_combined_t3_motion(Path(combined_csv_path_text.value))
                event.client.add_notification(
                    title="Combined T3 motion loaded",
                    body=str(workflow.csv_path),
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                combined_status.content = f"Load failed: `{exc}`"
                event.client.add_notification(title="Failed to load combined CSV", body=str(exc), color="red")

        @arms_only_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            if workflow.csv_path is None:
                return
            try:
                load_t2_preview(workflow.csv_path)
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to reload T2 preview",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )

        @memory_dropdown.on_update
        def _(_event: viser.GuiEvent) -> None:
            update_memory_status()

        @base_memory_dropdown.on_update
        def _(_event: viser.GuiEvent) -> None:
            update_base_memory_status()

        @continue_base_pose_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            workflow.continue_base_from_last_pose = bool(continue_base_pose_checkbox.value)

        @continuation_base_csv_text.on_update
        def _(_event: viser.GuiEvent) -> None:
            workflow.continuation_base_csv_reference = str(continuation_base_csv_text.value).strip()

        @chain_base_memories_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            workflow.chain_loaded_base_memories = bool(chain_base_memories_checkbox.value)

        @loop_base_memory_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            workflow.loop_base_memory_preview = bool(loop_base_memory_checkbox.value)
            if session.playing and workflow.wheel_base is not None:
                session.play_once = not workflow.loop_base_memory_preview

        @memories_root_text.on_update
        def _(_event: viser.GuiEvent) -> None:
            workflow.output_root = current_memories_root()

        @refresh_memories_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stems = refresh_memory_options()
            base_stems = refresh_base_memory_options()
            event.client.add_notification(
                title="Memories refreshed",
                body=f"Found {len(stems)} BVH/T2 memories and {len(base_stems)} base memories.",
                auto_close_seconds=3.0,
                color="blue",
            )

        @load_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No memory selected",
                    body="Refresh memories or choose a BVH memory.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            try:
                load_memory(stem)
                for client_id, sync_memory in list(self._memory_sync_callbacks.items()):
                    if client_id == client.client_id:
                        continue
                    try:
                        sync_memory(stem)
                    except Exception as exc:
                        print(f"Failed to sync memory for client {client_id}: {exc}")
                event.client.add_notification(
                    title="Memory loaded",
                    body=stem,
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to load memory",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @load_base_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_base_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No base memory selected",
                    body="Refresh memories or choose a base CSV memory.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            try:
                load_base_memory(stem)
                self.set_frame(client.client_id, 0)
                session.play_once = True
                session.playing = True
                event.client.add_notification(
                    title="Base memory playing once",
                    body=f"{stem} — simulation starts at frame 0 and stops at the final frame.",
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to load base memory",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @play_base_memory_preview_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_base_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No base memory selected",
                    body="Refresh memories or choose a base CSV memory.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            try:
                # This button is deliberately preview-only. Disabling the
                # stream flag before changing frames guarantees that the
                # generic playback loop cannot publish these rows to hardware.
                workflow.stream_real_robot_playback = False
                workflow.real_robot_approval_pending = False
                load_base_memory(stem)
                self.set_frame(client.client_id, 0)
                session.play_once = not workflow.loop_base_memory_preview
                session.playing = True
                playback_mode = "looping" if workflow.loop_base_memory_preview else "playing once"
                event.client.add_notification(
                    title=f"Base memory {playback_mode}",
                    body=(
                        f"{stem} — simulation only; "
                        + (
                            "repeats until playback is paused."
                            if workflow.loop_base_memory_preview
                            else "stops at the final frame."
                        )
                    ),
                    auto_close_seconds=5.0,
                    color="green",
                )
            except Exception as exc:
                session.play_once = False
                session.playing = False
                event.client.add_notification(
                    title="Failed to play base memory",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @retarget_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No memory selected",
                    body="Choose a BVH memory before retargeting.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            root = current_memories_root()
            bvh_path = _memory_bvh_path(root, stem)
            if not bvh_path.is_file():
                event.client.add_notification(
                    title="Memory BVH missing",
                    body=str(bvh_path),
                    auto_close_seconds=5.0,
                    color="red",
                )
                return
            retarget_bvh_to_memory_csv(bvh_path, root, event.client)

        @retarget_memory_t3_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No memory selected",
                    body="Choose a BVH memory before retargeting to T3.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            root = current_memories_root()
            bvh_path = _memory_bvh_path(root, stem)
            if not bvh_path.is_file():
                event.client.add_notification(
                    title="Memory BVH missing",
                    body=str(bvh_path),
                    auto_close_seconds=5.0,
                    color="red",
                )
                return
            retarget_bvh_to_t3_memory_csv(bvh_path, root, event.client)

        @save_base_csv_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if not workflow.base_prompt_segments:
                event.client.add_notification(
                    title="No dedicated base motion",
                    body='Generate using only a TaraBase prompt such as "move forward" or "turn left by 90 degrees" first.',
                    auto_close_seconds=6.0,
                    color="orange",
                )
                return
            requested_name = str(base_csv_name_text.value).strip()
            stem = requested_name or workflow.base_prompt_stem or f"tara_{uuid.uuid4().hex[:12]}"
            stem = stem.removesuffix(".csv").removesuffix("_diff_drive")
            output_path = current_memories_root() / "wheel_csv" / f"{_safe_clip_name(stem)}_diff_drive.csv"
            try:
                start_pose = workflow.base_prompt_start_pose
                write_prompt_base_csv(
                    output_path,
                    workflow.base_prompt_segments,
                    fps=workflow.base_prompt_fps,
                    **(
                        {
                            "initial_x_m": start_pose[0],
                            "initial_z_m": start_pose[1],
                            "initial_yaw_rad": start_pose[2],
                        }
                        if start_pose is not None
                        else {}
                    ),
                )
                workflow.wheel_csv_path = output_path
                workflow.base_prompt_stem = output_path.stem.removesuffix("_diff_drive")
                base_csv_name_text.value = workflow.base_prompt_stem
                refresh_base_memory_options(workflow.base_prompt_stem)
                event.client.add_notification(
                    title="Base CSV saved",
                    body=str(output_path),
                    auto_close_seconds=7.0,
                    color="green",
                )
                update_status(
                    "Saved frame-by-frame TaraBase motion:\n\n"
                    f"`{output_path}`\n\n"
                    f"Frames: `{sum(frames for _, frames in workflow.base_prompt_segments)}` at "
                    f"`{workflow.base_prompt_fps:g} FPS`."
                    + (
                        f" Start pose: X `{start_pose[0]:.3f}`, Z `{start_pose[1]:.3f}`, "
                        f"yaw `{np.rad2deg(start_pose[2]):.1f}°`; frame/time starts at `0`."
                        if start_pose is not None
                        else " Start pose: origin; frame/time starts at `0`."
                    )
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to save Base CSV",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @save_generated_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = _new_generated_stem()
            root = current_memories_root()
            session = self.client_sessions[client.client_id]
            generated_fps = _model_native_fps(self, session.model_name, session.model_fps)
            try:
                bvh_path = save_current_bvh(stem=stem, output_root=root, fps=generated_fps)
                refresh_memory_options(stem)
                csv_path_text.value = str(_memory_csv_path(root, stem))
                output_root_text.value = str(root)
                clip_name_text.value = Path(stem).name
                event.client.add_notification(
                    title="Memory saved",
                    body=str(bvh_path),
                    auto_close_seconds=5.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to save generated memory",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @save_bvh_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                bvh_path = save_current_bvh()
                event.client.add_notification(
                    title="SOMA BVH saved",
                    body=str(bvh_path),
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to save BVH",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )

        @retarget_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                bvh_path = workflow.bvh_path if workflow.bvh_path is not None else save_current_bvh()
            except Exception as exc:
                event.client.add_notification(
                    title="Cannot start retarget",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )
                return

            retarget_bvh_to_memory_csv(bvh_path, Path(output_root_text.value).expanduser().resolve(), event.client)

        @retarget_t3_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                bvh_path = workflow.bvh_path if workflow.bvh_path is not None else save_current_bvh()
            except Exception as exc:
                event.client.add_notification(
                    title="Cannot start T3 retarget",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )
                return

            retarget_bvh_to_t3_memory_csv(
                bvh_path,
                Path(output_root_text.value).expanduser().resolve(),
                event.client,
            )

        @load_t2_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                load_t2_preview(Path(csv_path_text.value))
                event.client.add_notification(
                    title="T2 preview loaded",
                    body=str(workflow.csv_path),
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to load T2 preview",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )

        update_memory_status()
        update_base_memory_status()

        def connect_robot(event: viser.GuiEvent) -> None:
            if not workflow.arm_frames and not workflow.base_wheel_rpms:
                event.client.add_notification(
                    title="No robot CSV loaded",
                    body="Load a T2/T3 CSV or TaraBase CSV before connecting.",
                    auto_close_seconds=5.0,
                    color="red",
                )
                return

            workflow.connection.dry_run = bool(dry_run_checkbox.value)
            workflow.connection.require_base = bool(
                workflow.base_wheel_rpms and not workflow.freeze_t3_base
            )
            workflow.connection.stream_script = default_t2_stream_script()
            if workflow.connection.is_connected():
                update_robot_status("Already connected.")
                return

            def on_status(title: str, body: str) -> None:
                update_robot_status(f"{title}.\n\n`{body[-900:]}`")

            try:
                workflow.stream_real_robot_playback = False
                workflow.connection.connect(on_status=on_status)
                update_robot_status("Connecting...")
                if workflow.arm_frames:
                    current_frame = self.client_sessions[client.client_id].frame_idx
                    self._send_robot_frame(workflow, current_frame)
                event.client.add_notification(
                    title="T2 Nero connecting",
                    body="Current visualizer frames will stream to the robot.",
                    auto_close_seconds=4.0,
                    color="blue",
                )
            except Exception as exc:
                update_robot_status(f"Connection failed.\n\n`{exc}`")
                event.client.add_notification(
                    title="Robot connection failed",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        def robot_control_status() -> dict[str, object]:
            session = self.client_sessions[client.client_id]
            real_robot_playing = (
                bool(session.playing)
                and workflow.connection.is_connected()
                and workflow.stream_real_robot_playback
            )
            previewed_for_current_memory = (
                workflow.memory_stem is not None
                and workflow.real_robot_previewed_memory_stem == workflow.memory_stem
            )
            current_hardware_path = (
                workflow.csv_path
                if workflow.arm_frames and workflow.csv_path is not None
                else workflow.wheel_csv_path
            )
            return {
                "connected": workflow.connection.is_connected(),
                "dry_run": bool(workflow.connection.dry_run),
                "playing": real_robot_playing,
                "visualizer_playing": bool(session.playing),
                "frame": int(session.frame_idx),
                "max_frame": int(session.max_frame_idx),
                "has_robot_frames": bool(workflow.arm_frames),
                "robot_frame_count": len(workflow.arm_frames),
                "csv_path": str(workflow.csv_path) if workflow.csv_path is not None else None,
                "memory_stem": workflow.memory_stem,
                "approval_pending": bool(workflow.real_robot_approval_pending),
                "streaming_enabled": bool(workflow.stream_real_robot_playback),
                "previewed_for_current_memory": bool(previewed_for_current_memory),
                "previewed_for_hardware": bool(
                    current_hardware_path is not None
                    and current_hardware_path.expanduser().resolve() in workflow.hardware_previewed_csv_paths
                ),
                "hardware_csv_path": str(current_hardware_path) if current_hardware_path is not None else None,
                "action_status": (
                    "playing"
                    if real_robot_playing
                    else "approval_pending"
                    if workflow.real_robot_approval_pending
                    else "connected"
                    if workflow.connection.is_connected()
                    else "disconnected"
                ),
                "last_output": workflow.connection.last_output[-6:],
            }

        def connect_robot_for_control_api(payload: dict[str, object]) -> None:
            if not workflow.arm_frames and not workflow.base_wheel_rpms:
                raise RuntimeError("No T3 arm or TaraBase frames are loaded.")

            dry_run_value = payload.get("dry_run")
            if dry_run_value is None:
                dry_run = bool(dry_run_checkbox.value)
            else:
                dry_run = bool(dry_run_value)
                dry_run_checkbox.value = dry_run
            workflow.connection.dry_run = dry_run
            workflow.connection.require_base = bool(
                workflow.base_wheel_rpms and not workflow.freeze_t3_base
            )
            workflow.connection.stream_script = default_t2_stream_script()
            if workflow.connection.is_connected() and bool(workflow.connection.dry_run) == dry_run:
                update_robot_status("Already connected.")
                return
            workflow.stream_real_robot_playback = False

            def on_status(title: str, body: str) -> None:
                update_robot_status(f"{title}.\n\n`{body[-900:]}`")

            workflow.connection.connect(on_status=on_status)
            update_robot_status("Connecting...")
            workflow.connection.wait_until_ready(timeout=10.0)
            update_robot_status("Connected and subscriber-ready.")
            current_frame = self.client_sessions[client.client_id].frame_idx
            self._send_robot_frame(workflow, current_frame)

        def preview_once_before_real_robot_play() -> None:
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = True
            session = self.client_sessions[client.client_id]
            self.set_frame(client.client_id, 0)
            session.play_once = True
            session.playing = True
            update_robot_status(
                "Previewing once in Viser only. Approve before playing on the real robot."
            )

        def request_real_robot_approval_without_preview() -> None:
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = True
            update_robot_status(
                "Preview already checked for this memory. Approve before playing on the real robot."
            )

        def play_once_on_real_robot() -> None:
            workflow.real_robot_approval_pending = False
            workflow.stream_real_robot_playback = True
            session = self.client_sessions[client.client_id]
            self.set_frame(client.client_id, 0)
            session.play_once = True
            session.playing = True
            update_robot_status("Playing once on the real robot.")

        def robot_control_from_external_change(payload: dict[str, object]) -> dict[str, object]:
            action = str(payload.get("action") or "").strip().lower().replace("_", "-")
            session = self.client_sessions[client.client_id]
            if action == "status":
                return {"action": action, "status": robot_control_status()}
            if action == "connect":
                connect_robot_for_control_api(payload)
                return {"action": action, "status": robot_control_status()}
            if action == "disconnect":
                workflow.stream_real_robot_playback = False
                workflow.real_robot_approval_pending = False
                workflow.connection.disconnect()
                update_robot_status("Disconnected.")
                return {"action": action, "status": robot_control_status()}
            if action == "play":
                current_hardware_path = (
                    workflow.csv_path
                    if workflow.arm_frames and workflow.csv_path is not None
                    else workflow.wheel_csv_path
                )
                if bool(payload.get("require_preview")):
                    if current_hardware_path is None:
                        raise RuntimeError("No hardware motion CSV is loaded.")
                    if current_hardware_path.expanduser().resolve() not in workflow.hardware_previewed_csv_paths:
                        raise RuntimeError(
                            "This exact CSV has not completed a Viser preview in the current session."
                        )
                connect_robot_for_control_api(payload)
                if not workflow.connection.dry_run and not bool(payload.get("approved")):
                    if (
                        workflow.memory_stem is not None
                        and workflow.real_robot_previewed_memory_stem == workflow.memory_stem
                    ):
                        request_real_robot_approval_without_preview()
                        return {"action": "approval-required", "status": robot_control_status()}
                    preview_once_before_real_robot_play()
                    return {"action": "preview-before-real-play", "status": robot_control_status()}
                play_once_on_real_robot()
                return {"action": action, "status": robot_control_status()}
            if action in {"approve-play", "approve-real-play", "replay", "rerun"}:
                approved_payload = dict(payload)
                approved_payload["dry_run"] = False
                connect_robot_for_control_api(approved_payload)
                play_once_on_real_robot()
                return {"action": action, "status": robot_control_status()}
            if action in {"stop", "emergency-stop"}:
                session.play_once = False
                session.playing = False
                workflow.stream_real_robot_playback = False
                workflow.real_robot_approval_pending = False
                workflow.connection.disconnect()
                update_robot_status(
                    "EMERGENCY STOP: zero base RPM sent; arm streaming stopped and disconnected."
                    if action == "emergency-stop"
                    else "Stopped and disconnected."
                )
                return {"action": action, "status": robot_control_status()}
            raise ValueError(f"Unknown robot action: {action}")

        self._robot_control_callbacks[client.client_id] = robot_control_from_external_change

        @connect_button.on_click
        def _(event: viser.GuiEvent) -> None:
            connect_robot(event)

        @disconnect_button.on_click
        def _(event: viser.GuiEvent) -> None:
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = False
            workflow.connection.disconnect()
            update_robot_status("Disconnected.")
            event.client.add_notification(
                title="T2 Nero disconnected",
                body="Robot streaming is off.",
                auto_close_seconds=3.0,
                color="blue",
            )

        @send_frame_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                self._send_robot_frame(workflow, self.client_sessions[client.client_id].frame_idx)
                update_robot_status(f"Sent frame `{self.client_sessions[client.client_id].frame_idx}`.")
            except Exception as exc:
                update_robot_status(f"Send failed.\n\n`{exc}`")
                event.client.add_notification(
                    title="Send current frame failed",
                    body=str(exc),
                    auto_close_seconds=6.0,
                    color="red",
                )

        @play_robot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            connect_robot(event)
            if bool(dry_run_checkbox.value):
                workflow.real_robot_approval_pending = False
                workflow.stream_real_robot_playback = True
                session = self.client_sessions[client.client_id]
                self.set_frame(client.client_id, 0)
                session.play_once = True
                session.playing = True
                update_robot_status("Playing once in dry-run mode.")
                return
            preview_once_before_real_robot_play()

        @approve_robot_play_button.on_click
        def _(event: viser.GuiEvent) -> None:
            dry_run_checkbox.value = False
            try:
                connect_robot(event)
                play_once_on_real_robot()
            except Exception as exc:
                update_robot_status(f"Approval play failed.\n\n`{exc}`")
                event.client.add_notification(
                    title="Real robot play failed",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @stop_robot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            session = self.client_sessions[client.client_id]
            session.play_once = False
            session.playing = False
            workflow.stream_real_robot_playback = False
            workflow.real_robot_approval_pending = False
            workflow.connection.disconnect()
            update_robot_status("Stopped and disconnected.")

        @sync_t3_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                requested_csv = Path(combined_csv_path_text.value).expanduser().resolve()
                if workflow.csv_path != requested_csv or workflow.t3_motion is None:
                    load_combined_t3_motion(requested_csv)

                session = self.client_sessions[client.client_id]
                session.play_once = False
                session.playing = False
                workflow.stream_real_robot_playback = False
                workflow.tara_stop_event.set()
                self.set_frame(client.client_id, 0)

                if bool(enable_t3_hardware_checkbox.value):
                    dry_run_checkbox.value = False
                    if workflow.connection.is_connected() and workflow.connection.dry_run:
                        workflow.connection.disconnect()
                    connect_robot(event)
                    if not workflow.connection.is_connected():
                        raise RuntimeError("T2 arm stream did not connect; synchronized hardware playback was not started.")
                    workflow.stream_real_robot_playback = True
                    self._send_robot_frame(workflow, 0)
                    if workflow.freeze_t3_base:
                        mode = "real T3 arms with TaraBase stationary (no base commands sent)"
                    else:
                        mode = "real T3 hardware over ROS 2 (arms + /base/cmd_wheel_rpm)"
                else:
                    mode = (
                        "Viser arms-only preview with base stationary"
                        if workflow.freeze_t3_base
                        else "Viser preview only (hardware safety gate is off)"
                    )

                session.play_once = True
                session.playing = True
                combined_status.content = (
                    f"Playing `{workflow.csv_path.name}` from row 1 on {mode}.\n\n"
                    f"Clock: `{workflow.base_prompt_fps:.2f} Hz`; rows: `{session.max_frame_idx + 1}`."
                )
            except Exception as exc:
                workflow.stream_real_robot_playback = False
                workflow.tara_stop_event.set()
                combined_status.content = f"Synchronized play failed: `{exc}`"
                event.client.add_notification(title="T3 synchronized play failed", body=str(exc), color="red")

        @stop_t3_sync_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            session = self.client_sessions[client.client_id]
            session.play_once = False
            session.playing = False
            workflow.stream_real_robot_playback = False
            workflow.tara_stop_event.set()
            workflow.connection.disconnect()
            combined_status.content = f"Stopped near row `{session.frame_idx + 1}`; base and arms are disconnected."

    @staticmethod
    def _arm_frame_for_index(workflow: RobotWorkflowState, frame_idx: int) -> ArmFrame:
        if not workflow.arm_frames:
            raise RuntimeError("No T2 arm frames are loaded.")
        return workflow.arm_frames[max(0, min(int(frame_idx), len(workflow.arm_frames) - 1))]

    @staticmethod
    def _gripper_width_for_index(workflow: RobotWorkflowState, frame_idx: int) -> float | None:
        if not workflow.gripper_widths:
            return None
        return workflow.gripper_widths[max(0, min(int(frame_idx), len(workflow.gripper_widths) - 1))]

    def _send_robot_frame(self, workflow: RobotWorkflowState, frame_idx: int) -> None:
        base_wheel_rpm: tuple[float, float] | None = None
        if (
            workflow.stream_real_robot_playback
            and workflow.base_wheel_rpms
            and not workflow.freeze_t3_base
        ):
            left_rpm, right_rpm = workflow.base_wheel_rpms[
                max(0, min(int(frame_idx), len(workflow.base_wheel_rpms) - 1))
            ]
            # Match the standalone TaraBase sender:
            # invert_turn_direction=True maps (L, R) -> (R, L), then the real
            # right motor wiring uses right_motor_sign=-1.
            base_wheel_rpm = (
                float(np.clip(right_rpm * self.tara_rpm_scale, -self.tara_max_rpm, self.tara_max_rpm)),
                float(np.clip(-left_rpm * self.tara_rpm_scale, -self.tara_max_rpm, self.tara_max_rpm)),
            )
        if workflow.arm_frames:
            workflow.connection.send_frame(
                self._arm_frame_for_index(workflow, frame_idx),
                gripper=self._gripper_width_for_index(workflow, frame_idx),
                base_wheel_rpm=base_wheel_rpm,
            )
        elif base_wheel_rpm is not None:
            workflow.connection.send_base_frame(frame_idx, base_wheel_rpm)

    def run(self) -> None:
        try:
            update_counter = 0
            cuda_check_interval = 300
            while True:
                last_update_time = time.time()
                fps_candidates = [bundle.model_fps for bundle in self.models.values()]
                fps_candidates.extend(
                    session.model_fps
                    for session in self.client_sessions.values()
                    if session.model_fps and session.model_fps > 0.0
                )
                playback_fps = max(fps_candidates, default=60.0) * 2.0

                for client_id, session in list(self.client_sessions.items()):
                    if not session.model_fps or session.model_fps <= 0.0:
                        continue
                    playback_speed = max(float(session.playback_speed), 1e-6)
                    update_interval = max(1, int(playback_fps / (playback_speed * session.model_fps)))
                    new_frame_idx = session.frame_idx
                    if session.playing and update_counter % update_interval == 0:
                        if session.frame_idx >= session.max_frame_idx:
                            if session.play_once:
                                session.playing = False
                                session.play_once = False
                                workflow = self.robot_workflows.get(client_id)
                                if workflow is not None:
                                    was_hardware_playback = bool(workflow.stream_real_robot_playback)
                                    if not was_hardware_playback:
                                        previewed_path = (
                                            workflow.csv_path
                                            if workflow.arm_frames and workflow.csv_path is not None
                                            else workflow.wheel_csv_path
                                        )
                                        if previewed_path is not None:
                                            workflow.hardware_previewed_csv_paths.add(
                                                previewed_path.expanduser().resolve()
                                            )
                                    workflow.stream_real_robot_playback = False
                                    if workflow.base_wheel_rpms:
                                        workflow.connection.disconnect()
                                    if workflow.real_robot_approval_pending and workflow.robot_markdown is not None:
                                        workflow.real_robot_previewed_memory_stem = workflow.memory_stem
                                        workflow.robot_markdown.content = (
                                            "Viser preview finished. Approve before playing on the real robot."
                                        )
                                new_frame_idx = session.max_frame_idx
                            else:
                                new_frame_idx = 0
                        else:
                            new_frame_idx = session.frame_idx + 1

                        if self.client_active(client_id):
                            self.set_frame(client_id, new_frame_idx)

                if update_counter % cuda_check_interval == 0:
                    self.check_cuda_health()

                time_remaining = max(0, 1.0 / playback_fps - (time.time() - last_update_time))
                time.sleep(time_remaining)
                update_counter += 1
                update_counter %= max(1, int(playback_fps))
        finally:
            for workflow in self.robot_workflows.values():
                workflow.tara_stop_event.set()
                workflow.connection.disconnect()
                workflow.clear_t2_preview()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kimodo robot workflow demo UI.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Default model to load. A SOMA model is required for the retarget workflow.",
    )
    parser.add_argument(
        "--text-encoder-mode",
        choices=("api", "local", "auto", "dummy"),
        default=None,
        help="Text encoder mode. The robot demo defaults to api to avoid loading the 8B fallback in-process.",
    )
    parser.add_argument(
        "--text-encoder-url",
        default=None,
        help=f"Text encoder server URL. Defaults to TEXT_ENCODER_URL or {DEFAULT_TEXT_ENCODER_URL}.",
    )
    parser.add_argument(
        "--model-server-url",
        default=os.environ.get("KIMODO_MODEL_SERVER_URL"),
        help="Optional FastAPI Kimodo model server URL. If set, the Viser UI sends generation requests remotely.",
    )
    parser.add_argument(
        "--retarget-server-url",
        default=os.environ.get("KIMODO_RETARGET_SERVER_URL"),
        help="Optional FastAPI soma-retargeter server URL. If set, BVH retargeting runs remotely.",
    )
    parser.add_argument(
        "--world-scene-path",
        default=os.environ.get("KIMODO_WORLD_SCENE_PATH", str(DEFAULT_WORLD_SCENE_PATH)),
        help="Optional world PLY to load into the Viser scene without changing the human or robot meshes.",
    )
    parser.add_argument(
        "--control-host",
        default=os.environ.get("KIMODO_CONTROL_HOST", "127.0.0.1"),
        help="Host for the lightweight Kimodo control API. Use 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=int(os.environ.get("KIMODO_CONTROL_PORT", "8787")),
        help="Port for the lightweight Kimodo control API.",
    )
    parser.add_argument(
        "--tara-remote-url",
        default=os.environ.get("TARA_REMOTE_URL", "ws://192.168.31.145:8094"),
        help="Raspberry Pi Tara server URL, preferably ws://PI_IP:8094 (HTTP is also supported).",
    )
    parser.add_argument("--tara-port", default="/dev/ttyUSB0", help="Tara serial port on the Raspberry Pi.")
    parser.add_argument("--tara-slave-id", type=int, default=1, help="Tara MODBUS slave id.")
    parser.add_argument("--tara-baudrate", type=int, default=115200, help="Tara serial baudrate.")
    parser.add_argument("--tara-fps", type=float, default=30.0, help="T2/wheel command CSV frame rate.")
    parser.add_argument("--tara-max-rpm", type=float, default=30.0, help="Safety RPM clamp.")
    parser.add_argument("--tara-rpm-scale", type=float, default=1.0, help="Tara RPM calibration scale.")
    parser.add_argument("--tara-debug", action="store_true", help="Enable Tara MODBUS debug output.")
    args = parser.parse_args()

    if args.tara_fps <= 0:
        parser.error("--tara-fps must be positive")
    if args.tara_max_rpm <= 0:
        parser.error("--tara-max-rpm must be positive")
    if args.tara_rpm_scale <= 0:
        parser.error("--tara-rpm-scale must be positive")

    text_encoder_mode = _configure_text_encoder_runtime(args.text_encoder_mode, args.text_encoder_url)
    resolved = resolve_model_name(args.model, "Kimodo")
    world_scene_path = Path(args.world_scene_path).expanduser() if args.world_scene_path else None
    try:
        demo = RobotDemo(
            default_model_name=resolved,
            world_scene_path=world_scene_path,
            tara_remote_url=args.tara_remote_url,
            tara_port=args.tara_port,
            tara_slave_id=args.tara_slave_id,
            tara_baudrate=args.tara_baudrate,
            tara_fps=args.tara_fps,
            tara_max_rpm=args.tara_max_rpm,
            tara_rpm_scale=args.tara_rpm_scale,
            tara_debug=args.tara_debug,
            model_server_url=args.model_server_url,
            retarget_server_url=args.retarget_server_url,
        )
    except Exception:
        raise SystemExit(_text_encoder_startup_error(text_encoder_mode)) from None
    try:
        demo.start_control_server(args.control_host, args.control_port)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    demo.run()


if __name__ == "__main__":
    main()
