# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ANTT T2 URDF and CSV motion playback helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import viser
import viser.transforms as tf

try:
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError:  # pragma: no cover - optional at import time, required for URDF-backed T2 playback.
    yourdfpy = None
    ViserUrdf = None


# MuJoCo (z-up, x-forward) -> kimodo (y-up, z-forward)
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

# soma-retargeter T2 CSVs are yawed 90 degrees relative to the BVH forward direction.
CSV_TO_KIMODO_YAW = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)
URDF_TO_SCENE_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)
SCENE_YAW_CORRECTION = Rotation.from_euler("y", -90.0, degrees=True).as_matrix()

TARA_MODEL_SCALE = 1.0

_TARA_CSV_JOINT_NAME_MAP = {
    "waist_yaw_joint_dof": "waist_yaw_joint",
    "waist_roll_joint_dof": "waist_roll_joint",
    "waist_pitch_joint_dof": "waist_pitch_joint",
    "head_pitch_joint_dof": "head_pitch_joint",
    "head_yaw_joint_dof": "head_yaw_joint",
    "right_joint1_dof": "right_joint1",
    "right_joint2_dof": "right_joint2",
    "right_joint3_dof": "right_joint3",
    "right_joint4_dof": "right_joint4",
    "right_joint5_dof": "right_joint5",
    "right_joint6_dof": "right_joint6",
    "right_joint7_dof": "right_joint7",
    "right_gripper_joint1_dof": "right_gripper_joint1",
    "right_gripper_joint2_dof": "right_gripper_joint2",
    "left_joint1_dof": "left_joint1",
    "left_joint2_dof": "left_joint2",
    "left_joint3_dof": "left_joint3",
    "left_joint4_dof": "left_joint4",
    "left_joint5_dof": "left_joint5",
    "left_joint6_dof": "left_joint6",
    "left_joint7_dof": "left_joint7",
    "left_gripper_joint1_dof": "left_gripper_joint1",
    "left_gripper_joint2_dof": "left_gripper_joint2",
    "left_hip_pitch_joint_dof": "left_hip_pitch_joint",
    "left_hip_roll_joint_dof": "left_hip_roll_joint",
    "left_hip_yaw_joint_dof": "left_hip_yaw_joint",
    "left_knee_joint_dof": "left_knee_joint",
    "left_ankle_roll_joint_dof": "left_ankle_roll_joint",
    "left_ankle_pitch_joint_dof": "left_ankle_pitch_joint",
    "right_hip_pitch_joint_dof": "right_hip_pitch_joint",
    "right_hip_roll_joint_dof": "right_hip_roll_joint",
    "right_hip_yaw_joint_dof": "right_hip_yaw_joint",
    "right_knee_joint_dof": "right_knee_joint",
    "right_ankle_roll_joint_dof": "right_ankle_roll_joint",
    "right_ankle_pitch_joint_dof": "right_ankle_pitch_joint",
}

_TARA_CSV_TO_URDF_JOINT_NAME_MAP = {
    "waist_yaw_joint": "waist_yaw_joint",
    "waist_roll_joint": "waist_roll_joint",
    "waist_pitch_joint": "waist_pitch_joint",
    "head_pitch_joint": "head_pitch_joint",
    "head_yaw_joint": "head_yaw_joint",
    "right_joint1": "right_joint1",
    "right_joint2": "right_joint2",
    "right_joint3": "right_joint3",
    "right_joint4": "right_joint4",
    "right_joint5": "right_joint5",
    "right_joint6": "right_joint6",
    "right_joint7": "right_joint7",
    "right_gripper_joint1": "right_gripper_joint1",
    "right_gripper_joint2": "right_gripper_joint2",
    "left_joint1": "left_joint1",
    "left_joint2": "left_joint2",
    "left_joint3": "left_joint3",
    "left_joint4": "left_joint4",
    "left_joint5": "left_joint5",
    "left_joint6": "left_joint6",
    "left_joint7": "left_joint7",
    "left_gripper_joint1": "left_gripper_joint1",
    "left_gripper_joint2": "left_gripper_joint2",
    "left_hip_pitch_joint": "left_hip_pitch_joint",
    "left_hip_roll_joint": "left_hip_roll_joint",
    "left_hip_yaw_joint": "left_hip_yaw_joint",
    "left_knee_joint": "left_knee_joint",
    "left_ankle_roll_joint": "left_ankle_roll_joint",
    "left_ankle_pitch_joint": "left_ankle_pitch_joint",
    "right_hip_pitch_joint": "right_hip_pitch_joint",
    "right_hip_roll_joint": "right_hip_roll_joint",
    "right_hip_yaw_joint": "right_hip_yaw_joint",
    "right_knee_joint": "right_knee_joint",
    "right_ankle_roll_joint": "right_ankle_roll_joint",
    "right_ankle_pitch_joint": "right_ankle_pitch_joint",
}

_TARA_URDF_NEUTRAL_JOINTS = {
    "left_joint2": -0.5,
    "left_joint4": 1.2,
    "left_joint6": 0.8,
    "right_joint2": -0.5,
    "right_joint4": 1.2,
    "right_joint6": 0.8,
}

_TARA_ARM_JOINT_NAMES = {
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
    "right_gripper_joint1",
    "right_gripper_joint2",
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "left_gripper_joint1",
    "left_gripper_joint2",
}


@dataclass
class TaraMotionData:
    root_positions: np.ndarray
    root_rotations: np.ndarray
    joint_angles: dict[str, np.ndarray]

    @property
    def length(self) -> int:
        return int(self.root_positions.shape[0])


def load_tara_motion_csv(path: str | Path, x_offset: float = 0.0) -> TaraMotionData:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    expected_prefix = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    if header[: len(expected_prefix)] != expected_prefix:
        raise ValueError(f"{path}: unsupported Tara CSV header.")

    unknown_columns = [name for name in header[7:] if name not in _TARA_CSV_JOINT_NAME_MAP]
    if unknown_columns:
        raise ValueError(f"{path}: unsupported Tara CSV joints: {unknown_columns}")

    csv_data = np.loadtxt(path, delimiter=",", skiprows=1)
    if csv_data.ndim == 1:
        csv_data = csv_data[None, :]

    root_positions_mujoco = csv_data[:, 1:4] * 0.01
    root_rot_mujoco = Rotation.from_euler("xyz", np.deg2rad(csv_data[:, 4:7])).as_matrix()

    root_positions = (MUJOCO_TO_KIMODO @ root_positions_mujoco[..., None]).squeeze(-1)
    root_rotations = MUJOCO_TO_KIMODO[None, ...] @ root_rot_mujoco @ MUJOCO_TO_KIMODO.T

    root_positions = (CSV_TO_KIMODO_YAW @ root_positions[..., None]).squeeze(-1)
    root_rotations = CSV_TO_KIMODO_YAW[None, ...] @ root_rotations
    root_positions *= TARA_MODEL_SCALE
    root_positions[:, 0] += x_offset

    joint_angles = {
        _TARA_CSV_JOINT_NAME_MAP[column_name]: np.deg2rad(csv_data[:, column_idx]).astype(np.float64)
        for column_idx, column_name in enumerate(header[7:], start=7)
    }
    return TaraMotionData(root_positions=root_positions, root_rotations=root_rotations, joint_angles=joint_angles)


class TaraUrdfRig:
    """URDF-backed ANTT T2 rig for playback."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        urdf_path: str | Path,
        color: tuple[int, int, int],
    ):
        if ViserUrdf is None or yourdfpy is None:
            raise ImportError("ANTT T2 URDF playback requires yourdfpy and viser.extras.ViserUrdf")

        self.server = server
        self.color = color
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.root_handle = self.server.scene.add_frame(f"/{name}/tara_urdf", show_axes=False)
        self.urdf = ViserUrdf(
            self.server,
            self.urdf_path,
            scale=1.0,
            root_node_name=self.root_handle.name,
            load_meshes=True,
            load_collision_meshes=False,
        )
        self._joint_names = self.urdf.get_actuated_joint_names()
        self._joint_name_to_idx = {joint_name: idx for idx, joint_name in enumerate(self._joint_names)}
        self._neutral_cfg = np.zeros(len(self._joint_names), dtype=np.float64)
        for joint_name, joint_value in _TARA_URDF_NEUTRAL_JOINTS.items():
            joint_idx = self._joint_name_to_idx.get(joint_name)
            if joint_idx is not None:
                self._neutral_cfg[joint_idx] = joint_value
        self._cfg = self._neutral_cfg.copy()
        self._default_rot = SCENE_YAW_CORRECTION @ URDF_TO_SCENE_ROT
        self._rest_scene_y_min = self._compute_rest_scene_y_min()
        self._ground_y_offset = 0.0
        self.root_handle.wxyz = tf.SO3.from_matrix(self._default_rot).wxyz
        self.root_handle.position = np.array([0.0, self._ground_y_offset, 0.0], dtype=np.float64)
        self.urdf.update_cfg(self._cfg)

    def _compute_rest_scene_y_min(self) -> float:
        scene = getattr(self.urdf, "_urdf", None)
        if scene is None or scene.scene is None or scene.scene.bounds is None:
            return 0.0
        bounds = np.asarray(scene.scene.bounds, dtype=np.float64)
        corners = np.array(
            [
                [x, y, z]
                for x in (bounds[0, 0], bounds[1, 0])
                for y in (bounds[0, 1], bounds[1, 1])
                for z in (bounds[0, 2], bounds[1, 2])
            ],
            dtype=np.float64,
        )
        transformed = (self._default_rot @ corners.T).T
        return float(transformed[:, 1].min())

    def ground_from_frame0(self, root_pos: np.ndarray) -> None:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        self._ground_y_offset = -(float(root_pos[1]) + self._rest_scene_y_min)

    def set_visibility(self, visible: bool) -> None:
        self.root_handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        for mesh in getattr(self.urdf, "_meshes", []):
            if hasattr(mesh, "opacity"):
                mesh.opacity = opacity

    def clear(self) -> None:
        self.urdf.remove()
        self.root_handle.remove()

    def set_pose(self, root_pos: np.ndarray, root_rot: np.ndarray, joint_angles: dict[str, float]) -> None:
        self.root_handle.position = np.asarray(root_pos, dtype=np.float64) + np.array(
            [0.0, self._ground_y_offset, 0.0], dtype=np.float64
        )
        self.root_handle.wxyz = tf.SO3.from_matrix(root_rot @ self._default_rot).wxyz
        self._cfg[:] = self._neutral_cfg
        for csv_joint_name, joint_angle in joint_angles.items():
            urdf_joint_name = _TARA_CSV_TO_URDF_JOINT_NAME_MAP.get(csv_joint_name)
            if urdf_joint_name is None:
                continue
            joint_idx = self._joint_name_to_idx.get(urdf_joint_name)
            if joint_idx is not None:
                self._cfg[joint_idx] = float(joint_angle)
        self.urdf.update_cfg(self._cfg)


class T2ViewerMotion:
    """Playback adapter for ANTT T2 URDF motion clips."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        csv_path: str | Path,
        urdf_path: str | Path | None = None,
        x_offset: float = 0.0,
        color: tuple[int, int, int] = (160, 160, 160),
        arms_only: bool = False,
    ):
        repo_root = Path(__file__).resolve().parents[3]
        preferred_urdf_path = (
            Path(urdf_path).expanduser().resolve()
            if urdf_path is not None
            else repo_root / "soma-retargeter" / "antt_t2" / "T2_serial_nero_arms.urdf"
        )

        if ViserUrdf is None or yourdfpy is None:
            raise ImportError("ANTT T2 URDF playback requires yourdfpy and viser.extras.ViserUrdf")
        if not preferred_urdf_path.exists():
            raise FileNotFoundError(f"ANTT T2 robot URDF not found: {preferred_urdf_path}")
        self.rig = TaraUrdfRig(
            name=name,
            server=server,
            urdf_path=preferred_urdf_path,
            color=color,
        )
        self.motion = load_tara_motion_csv(csv_path, x_offset=x_offset)
        self.arms_only = arms_only
        self.rig.ground_from_frame0(self.motion.root_positions[0])
        self.length = self.motion.length
        self.cur_frame_idx = 0

    def set_frame(self, idx: int) -> None:
        idx = min(int(idx), self.length - 1)
        joint_angles = {joint_name: values[idx] for joint_name, values in self.motion.joint_angles.items()}
        root_idx = 0 if self.arms_only else idx
        if self.arms_only:
            joint_angles = {name: value for name, value in joint_angles.items() if name in _TARA_ARM_JOINT_NAMES}
        self.rig.set_pose(
            root_pos=self.motion.root_positions[root_idx],
            root_rot=self.motion.root_rotations[root_idx],
            joint_angles=joint_angles,
        )
        self.cur_frame_idx = idx

    def clear(self) -> None:
        self.rig.clear()

    def set_mesh_visibility(self, visible: bool) -> None:
        self.rig.set_visibility(visible)

    def set_skeleton_visibility(self, visible: bool) -> None:
        del visible

    def set_show_foot_contacts(self, show: bool) -> None:
        del show

    def set_mesh_opacity(self, opacity: float) -> None:
        self.rig.set_opacity(opacity)
