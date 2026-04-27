# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tara/T1 mesh rig and CSV motion playback helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import viser
import viser.transforms as tf
try:
    import yourdfpy
    from viser.extras import ViserUrdf
except ImportError:  # pragma: no cover - optional at import time, required for URDF-backed Tara playback.
    yourdfpy = None
    ViserUrdf = None

from .coords import rotation_matrix_from_two_vec


# MuJoCo (z-up, x-forward) -> kimodo (y-up, z-forward)
MUJOCO_TO_KIMODO = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float64)

# soma-retargeter CSVs are yawed 90 degrees relative to the BVH forward direction.
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
    "aa_head_yaw_dof": "AAHead_yaw",
    "head_pitch_dof": "Head_pitch",
    "left_nero_joint1_dof": "Left_Nero_Joint1",
    "left_nero_joint2_dof": "Left_Nero_Joint2",
    "left_nero_joint3_dof": "Left_Nero_Joint3",
    "left_nero_joint4_dof": "Left_Nero_Joint4",
    "left_nero_joint5_dof": "Left_Nero_Joint5",
    "left_nero_joint6_dof": "Left_Nero_Joint6",
    "left_nero_joint7_dof": "Left_Nero_Joint7",
    "right_nero_joint1_dof": "Right_Nero_Joint1",
    "right_nero_joint2_dof": "Right_Nero_Joint2",
    "right_nero_joint3_dof": "Right_Nero_Joint3",
    "right_nero_joint4_dof": "Right_Nero_Joint4",
    "right_nero_joint5_dof": "Right_Nero_Joint5",
    "right_nero_joint6_dof": "Right_Nero_Joint6",
    "right_nero_joint7_dof": "Right_Nero_Joint7",
    "waist_dof": "Waist",
    "left_hip_pitch_dof": "Left_Hip_Pitch",
    "left_hip_roll_dof": "Left_Hip_Roll",
    "left_hip_yaw_dof": "Left_Hip_Yaw",
    "left_knee_pitch_dof": "Left_Knee_Pitch",
    "left_ankle_pitch_dof": "Left_Ankle_Pitch",
    "left_ankle_roll_dof": "Left_Ankle_Roll",
    "right_hip_pitch_dof": "Right_Hip_Pitch",
    "right_hip_roll_dof": "Right_Hip_Roll",
    "right_hip_yaw_dof": "Right_Hip_Yaw",
    "right_knee_pitch_dof": "Right_Knee_Pitch",
    "right_ankle_pitch_dof": "Right_Ankle_Pitch",
    "right_ankle_roll_dof": "Right_Ankle_Roll",
    # New T2 CSV schema already uses URDF-style joint names.
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
    "AAHead_yaw": "AAHead_yaw",
    "Head_pitch": "Head_pitch",
    "Left_Nero_Joint1": "left_joint1",
    "Left_Nero_Joint2": "left_joint2",
    "Left_Nero_Joint3": "left_joint3",
    "Left_Nero_Joint4": "left_joint4",
    "Left_Nero_Joint5": "left_joint5",
    "Left_Nero_Joint6": "left_joint6",
    "Left_Nero_Joint7": "left_joint7",
    "Right_Nero_Joint1": "right_joint1",
    "Right_Nero_Joint2": "right_joint2",
    "Right_Nero_Joint3": "right_joint3",
    "Right_Nero_Joint4": "right_joint4",
    "Right_Nero_Joint5": "right_joint5",
    "Right_Nero_Joint6": "right_joint6",
    "Right_Nero_Joint7": "right_joint7",
    "Waist": "Waist",
    "Left_Hip_Pitch": "Left_Hip_Pitch",
    "Left_Hip_Roll": "Left_Hip_Roll",
    "Left_Hip_Yaw": "Left_Hip_Yaw",
    "Left_Knee_Pitch": "Left_Knee_Pitch",
    "Left_Ankle_Pitch": "Left_Ankle_Pitch",
    "Left_Ankle_Roll": "Left_Ankle_Roll",
    "Right_Hip_Pitch": "Right_Hip_Pitch",
    "Right_Hip_Roll": "Right_Hip_Roll",
    "Right_Hip_Yaw": "Right_Hip_Yaw",
    "Right_Knee_Pitch": "Right_Knee_Pitch",
    "Right_Ankle_Pitch": "Right_Ankle_Pitch",
    "Right_Ankle_Roll": "Right_Ankle_Roll",
    # Identity mapping for the newer T2 CSV schema.
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


def _parse_vec3(value: str | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float64)
    return np.array([float(x) for x in value.split()], dtype=np.float64)


def _parse_rot(elem: ET.Element) -> np.ndarray:
    quat = elem.get("quat")
    if quat is not None:
        wxyz = np.array([float(x) for x in quat.split()], dtype=np.float64)
        return tf.SO3(wxyz=wxyz).as_matrix()

    euler = elem.get("euler")
    if euler is not None:
        return Rotation.from_euler("xyz", [float(x) for x in euler.split()]).as_matrix()

    return np.eye(3, dtype=np.float64)


def _convert_pos_mujoco_to_kimodo(pos: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ pos


def _convert_rot_mujoco_to_kimodo(rot: np.ndarray) -> np.ndarray:
    return MUJOCO_TO_KIMODO @ rot @ MUJOCO_TO_KIMODO.T


@dataclass
class TaraBody:
    name: str
    parent_idx: int | None
    local_pos: np.ndarray
    local_rot: np.ndarray
    joint_name: str | None
    joint_axis: np.ndarray | None
    has_freejoint: bool


@dataclass
class TaraMeshSpec:
    mesh_name: str
    mesh_path: Path
    body_idx: int
    body_name: str
    geom_pos: np.ndarray
    geom_rot: np.ndarray


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


def _load_mesh_geometry(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(mesh_path, process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    vertices = (mesh.vertices @ MUJOCO_TO_KIMODO.T) * TARA_MODEL_SCALE
    return vertices, mesh.faces


def load_tara_model(xml_path: str | Path) -> tuple[list[TaraBody], list[TaraMeshSpec]]:
    xml_path = Path(xml_path).expanduser().resolve()
    root = ET.parse(xml_path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{xml_path}: missing worldbody")

    mesh_name_to_path = {}
    for mesh in root.findall(".//asset/mesh"):
        mesh_name = mesh.get("name")
        mesh_file = mesh.get("file")
        if mesh_name and mesh_file:
            mesh_name_to_path[mesh_name] = (xml_path.parent / mesh_file).resolve()

    bodies: list[TaraBody] = []
    meshes: list[TaraMeshSpec] = []

    def _visit(body_elem: ET.Element, parent_idx: int | None) -> None:
        body_idx = len(bodies)
        joint_elem = body_elem.find("joint")
        joint_axis = None
        joint_name = None
        if joint_elem is not None:
            joint_name = joint_elem.get("name")
            axis = _parse_vec3(joint_elem.get("axis"))
            if np.linalg.norm(axis) > 0.0:
                joint_axis = _convert_pos_mujoco_to_kimodo(axis)
                joint_axis = joint_axis / np.linalg.norm(joint_axis)

        bodies.append(
            TaraBody(
                name=body_elem.get("name", f"body_{body_idx}"),
                parent_idx=parent_idx,
                local_pos=_convert_pos_mujoco_to_kimodo(_parse_vec3(body_elem.get("pos"))) * TARA_MODEL_SCALE,
                local_rot=_convert_rot_mujoco_to_kimodo(_parse_rot(body_elem)),
                joint_name=joint_name,
                joint_axis=joint_axis,
                has_freejoint=body_elem.find("freejoint") is not None,
            )
        )

        for geom_elem in body_elem.findall("geom"):
            if geom_elem.get("class") != "visual":
                continue
            mesh_name = geom_elem.get("mesh")
            if not mesh_name or mesh_name not in mesh_name_to_path:
                continue
            meshes.append(
                TaraMeshSpec(
                    mesh_name=mesh_name,
                    mesh_path=mesh_name_to_path[mesh_name],
                    body_idx=body_idx,
                    body_name=body_elem.get("name", f"body_{body_idx}"),
                    geom_pos=_convert_pos_mujoco_to_kimodo(_parse_vec3(geom_elem.get("pos"))) * TARA_MODEL_SCALE,
                    geom_rot=_convert_rot_mujoco_to_kimodo(_parse_rot(geom_elem)),
                )
            )

        for child_body in body_elem.findall("body"):
            _visit(child_body, body_idx)

    for body_elem in worldbody.findall("body"):
        _visit(body_elem, parent_idx=None)

    return bodies, meshes


class TaraMeshRig:
    """Simple mesh rig that evaluates Tara/T1 MJCF body transforms directly."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        xml_path: str | Path,
        color: tuple[int, int, int],
    ):
        self.server = server
        self.color = color
        self.bodies, mesh_specs = load_tara_model(xml_path)
        self.mesh_handles: list[viser.SceneHandle] = []
        self.mesh_items: list[dict[str, object]] = []

        for mesh_spec in mesh_specs:
            vertices, faces = _load_mesh_geometry(mesh_spec.mesh_path)
            handle = self.server.scene.add_mesh_simple(
                f"/{name}/tara_mesh/{mesh_spec.body_idx:02d}_{mesh_spec.body_name}_{mesh_spec.mesh_name}",
                vertices=vertices,
                faces=faces,
                opacity=None,
                color=self.color,
                wireframe=False,
                visible=True,
            )
            self.mesh_handles.append(handle)
            self.mesh_items.append(
                {
                    "handle": handle,
                    "body_idx": mesh_spec.body_idx,
                    "geom_pos": mesh_spec.geom_pos,
                    "geom_rot": mesh_spec.geom_rot,
                }
            )

    def set_visibility(self, visible: bool) -> None:
        for handle in self.mesh_handles:
            handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        for handle in self.mesh_handles:
            handle.opacity = opacity

    def clear(self) -> None:
        for handle in self.mesh_handles:
            self.server.scene.remove_by_name(handle.name)
        self.mesh_handles = []
        self.mesh_items = []

    def set_pose(
        self, root_pos: np.ndarray, root_rot: np.ndarray, joint_angles: dict[str, float]
    ) -> tuple[np.ndarray, np.ndarray]:
        body_positions: list[np.ndarray] = [np.zeros(3, dtype=np.float64) for _ in self.bodies]
        body_rotations: list[np.ndarray] = [np.eye(3, dtype=np.float64) for _ in self.bodies]

        for body_idx, body in enumerate(self.bodies):
            joint_rot = np.eye(3, dtype=np.float64)
            if body.joint_name is not None and body.joint_axis is not None:
                joint_angle = float(joint_angles.get(body.joint_name, 0.0))
                joint_rot = Rotation.from_rotvec(body.joint_axis * joint_angle).as_matrix()

            if body.parent_idx is None:
                body_rotations[body_idx] = root_rot @ body.local_rot @ joint_rot
                if body.has_freejoint:
                    body_positions[body_idx] = root_pos
                else:
                    body_positions[body_idx] = root_pos + root_rot @ body.local_pos
            else:
                parent_pos = body_positions[body.parent_idx]
                parent_rot = body_rotations[body.parent_idx]
                body_positions[body_idx] = parent_pos + parent_rot @ body.local_pos
                body_rotations[body_idx] = parent_rot @ body.local_rot @ joint_rot

        for mesh_item in self.mesh_items:
            body_idx = mesh_item["body_idx"]
            body_pos = body_positions[body_idx]
            body_rot = body_rotations[body_idx]
            geom_pos = mesh_item["geom_pos"]
            geom_rot = mesh_item["geom_rot"]
            handle = mesh_item["handle"]
            handle.position = body_pos + body_rot @ geom_pos
            handle.wxyz = tf.SO3.from_matrix(body_rot @ geom_rot).wxyz

        return np.stack(body_positions, axis=0), np.stack(body_rotations, axis=0)


class TaraUrdfRig:
    """URDF-backed Tara/T1 rig for playback when a suitable URDF is available."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        urdf_path: str | Path,
        color: tuple[int, int, int],
    ):
        if ViserUrdf is None or yourdfpy is None:
            raise ImportError("URDF Tara playback requires yourdfpy and viser.extras.ViserUrdf")

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


class TaraSkeletonRig:
    """Lightweight skeleton renderer for Tara using the MJCF body hierarchy."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer | viser.ClientHandle,
        bodies: list[TaraBody],
        joint_color: tuple[int, int, int] = (255, 235, 0),
        bone_color: tuple[int, int, int] = (27, 106, 0),
    ):
        self.server = server
        self.bodies = bodies
        self.num_joints = len(bodies)
        self.joint_colors = np.full((self.num_joints, 3), joint_color, dtype=np.float64)
        self.bone_indices = [idx for idx, body in enumerate(bodies) if body.parent_idx is not None]
        self.label_handles: list[viser.SceneHandle] = []
        self.label_joint_indices: list[int] = []

        joint_mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.02)
        bone_mesh = trimesh.creation.cylinder(radius=0.01, height=1.0)

        init_joint_wxyzs = np.concatenate([np.ones((self.num_joints, 1)), np.zeros((self.num_joints, 3))], axis=1)
        self.joints_batched_mesh = server.scene.add_batched_meshes_simple(
            f"/{name}/tara_skeleton/joints",
            vertices=joint_mesh.vertices,
            faces=joint_mesh.faces,
            batched_wxyzs=init_joint_wxyzs,
            batched_positions=np.zeros((self.num_joints, 3)),
            batched_scales=np.ones((self.num_joints, 3)),
            batched_colors=self.joint_colors,
        )
        init_bone_wxyzs = np.concatenate([np.ones((len(self.bone_indices), 1)), np.zeros((len(self.bone_indices), 3))], axis=1)
        self.bones_batched_mesh = server.scene.add_batched_meshes_simple(
            f"/{name}/tara_skeleton/bones",
            vertices=bone_mesh.vertices,
            faces=bone_mesh.faces,
            batched_wxyzs=init_bone_wxyzs,
            batched_positions=np.zeros((len(self.bone_indices), 3)),
            batched_scales=np.ones((len(self.bone_indices), 3)),
            batched_colors=np.full((len(self.bone_indices), 3), bone_color, dtype=np.float64),
        )

        for joint_idx, body in enumerate(bodies):
            joint_name = body.joint_name
            if joint_name is None or "Nero_Joint" not in joint_name:
                continue
            label = server.scene.add_label(
                name=f"/{name}/tara_skeleton/labels/{joint_name}",
                text=joint_name,
                position=np.zeros(3, dtype=np.float64),
                font_size_mode="screen",
                font_screen_scale=0.55,
                anchor="bottom-center",
            )
            self.label_handles.append(label)
            self.label_joint_indices.append(joint_idx)

    def set_visibility(self, visible: bool) -> None:
        self.joints_batched_mesh.visible = visible
        self.bones_batched_mesh.visible = visible
        for handle in self.label_handles:
            handle.visible = visible

    def clear(self) -> None:
        self.server.scene.remove_by_name(self.joints_batched_mesh.name)
        self.server.scene.remove_by_name(self.bones_batched_mesh.name)
        for handle in self.label_handles:
            self.server.scene.remove_by_name(handle.name)

    def set_pose(self, joint_positions: np.ndarray, joint_rotations: np.ndarray) -> None:
        display_joint_positions = joint_positions.copy()

        # Some Tara arm joints are authored at zero translation in MJCF, so consecutive
        # joints collapse to the same point. Apply a tiny display-only separation along
        # the current joint axis so the 7-DoF chain remains readable in the skeleton view.
        for child_idx, body in enumerate(self.bodies):
            parent_idx = body.parent_idx
            if parent_idx is None or body.joint_axis is None:
                continue
            parent_pos = display_joint_positions[parent_idx]
            child_pos = display_joint_positions[child_idx]
            if np.linalg.norm(child_pos - parent_pos) >= 1e-8:
                continue
            axis_world = joint_rotations[child_idx] @ body.joint_axis
            axis_norm = np.linalg.norm(axis_world)
            if axis_norm < 1e-8:
                continue
            axis_world = axis_world / axis_norm
            display_joint_positions[child_idx] = child_pos + axis_world * 0.035

        self.joints_batched_mesh.batched_positions = display_joint_positions
        for label_idx, handle in enumerate(self.label_handles):
            joint_idx = self.label_joint_indices[label_idx]
            handle.position = display_joint_positions[joint_idx] + np.array([0.0, 0.035, 0.0], dtype=np.float64)

        bone_positions = np.zeros((len(self.bone_indices), 3), dtype=np.float64)
        bone_wxyzs = np.zeros((len(self.bone_indices), 4), dtype=np.float64)
        bone_scales = np.ones((len(self.bone_indices), 3), dtype=np.float64)
        for bone_list_idx, child_idx in enumerate(self.bone_indices):
            parent_idx = self.bodies[child_idx].parent_idx
            assert parent_idx is not None
            child_pos = display_joint_positions[child_idx]
            parent_pos = display_joint_positions[parent_idx]
            bone_pos = (child_pos + parent_pos) / 2.0
            bone_length = np.linalg.norm(child_pos - parent_pos)
            if bone_length < 1e-8:
                bone_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            else:
                bone_dir = (child_pos - parent_pos) / bone_length
                bone_rot = rotation_matrix_from_two_vec(np.array([0.0, 0.0, 1.0], dtype=np.float64), bone_dir)
                bone_wxyz = tf.SO3.from_matrix(bone_rot).wxyz
            bone_positions[bone_list_idx] = bone_pos
            bone_wxyzs[bone_list_idx] = bone_wxyz
            bone_scales[bone_list_idx] = np.array([1.0, 1.0, bone_length], dtype=np.float64)

        self.bones_batched_mesh.batched_positions = bone_positions
        self.bones_batched_mesh.batched_wxyzs = bone_wxyzs
        self.bones_batched_mesh.batched_scales = bone_scales


class TaraViewerMotion:
    """Playback adapter for Tara/T1 mesh-only motion clips."""

    def __init__(
        self,
        name: str,
        server: viser.ViserServer,
        tara_root: str | Path | None,
        csv_path: str | Path,
        urdf_path: str | Path | None = None,
        x_offset: float = 0.0,
        color: tuple[int, int, int] = (160, 160, 160),
    ):
        tara_root_path = Path(tara_root).expanduser().resolve() if tara_root is not None else None
        repo_root = Path(__file__).resolve().parents[3]
        preferred_urdf_path = (
            Path(urdf_path).expanduser().resolve()
            if urdf_path is not None
            else repo_root / "soma-retargeter" / "antt_t2" / "T2_serial_nero_arms.urdf"
        )
        secondary_urdf_path = repo_root / "unitree_ros" / "robots" / "h2_description" / "H2.urdf"
        tertiary_urdf_path = repo_root / "unitree_ros" / "robots" / "g1_description" / "g1_29dof_rev_1_0.urdf"
        quaternary_urdf_path = repo_root / "unitree_ros" / "robots" / "r1_description" / "R1.urdf"

        self.skeleton_rig: TaraSkeletonRig | None
        if preferred_urdf_path.exists() and ViserUrdf is not None and yourdfpy is not None:
            self.rig = TaraUrdfRig(
                name=name,
                server=server,
                urdf_path=preferred_urdf_path,
                color=color,
            )
            self.skeleton_rig = None
        elif urdf_path is None and secondary_urdf_path.exists() and ViserUrdf is not None and yourdfpy is not None:
            self.rig = TaraUrdfRig(name=name, server=server, urdf_path=secondary_urdf_path, color=color)
            self.skeleton_rig = None
        elif urdf_path is None and tertiary_urdf_path.exists() and ViserUrdf is not None and yourdfpy is not None:
            self.rig = TaraUrdfRig(name=name, server=server, urdf_path=tertiary_urdf_path, color=color)
            self.skeleton_rig = None
        elif urdf_path is None and quaternary_urdf_path.exists() and ViserUrdf is not None and yourdfpy is not None:
            self.rig = TaraUrdfRig(name=name, server=server, urdf_path=quaternary_urdf_path, color=color)
            self.skeleton_rig = None
        else:
            if urdf_path is not None:
                raise FileNotFoundError(f"Tara robot URDF not found or unavailable: {preferred_urdf_path}")
            if tara_root_path is None:
                raise FileNotFoundError("Tara root is required only for MJCF fallback. Pass --tara-root or --robot-path.")
            xml_path = tara_root_path / "T1_serial.xml"
            self.rig = TaraMeshRig(name=name, server=server, xml_path=xml_path, color=color)
            self.skeleton_rig = TaraSkeletonRig(name=name, server=server, bodies=self.rig.bodies)
        self.motion = load_tara_motion_csv(csv_path, x_offset=x_offset)
        if isinstance(self.rig, TaraUrdfRig):
            self.rig.ground_from_frame0(self.motion.root_positions[0])
        self.length = self.motion.length
        self.cur_frame_idx = 0

    def set_frame(self, idx: int) -> None:
        idx = min(int(idx), self.length - 1)
        joint_angles = {joint_name: values[idx] for joint_name, values in self.motion.joint_angles.items()}
        rig_result = self.rig.set_pose(
            root_pos=self.motion.root_positions[idx],
            root_rot=self.motion.root_rotations[idx],
            joint_angles=joint_angles,
        )
        if self.skeleton_rig is not None:
            joint_positions, joint_rotations = rig_result
            self.skeleton_rig.set_pose(joint_positions, joint_rotations)
        self.cur_frame_idx = idx

    def clear(self) -> None:
        self.rig.clear()
        if self.skeleton_rig is not None:
            self.skeleton_rig.clear()

    def set_mesh_visibility(self, visible: bool) -> None:
        self.rig.set_visibility(visible)

    def set_skeleton_visibility(self, visible: bool) -> None:
        if self.skeleton_rig is not None:
            self.skeleton_rig.set_visibility(visible)

    def set_show_foot_contacts(self, show: bool) -> None:
        del show

    def set_mesh_opacity(self, opacity: float) -> None:
        self.rig.set_opacity(opacity)
