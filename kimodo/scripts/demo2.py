# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arm-only Kimodo editor for ANTT T1 using PyRoki."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as tf
import yourdfpy
from viser.extras import ViserUrdf

from kimodo.demo import ui
from kimodo.demo.config import DEFAULT_PLAYBACK_SPEED
from kimodo.demo.state import ClientSession
from kimodo.skeleton import SkeletonBase
from kimodo.viz.playback import CharacterMotion
from kimodo.viz.scene import Character
from kimodo.viz.viser_utils import ConstraintSet
from kimodo.tools import to_numpy, to_torch

from .demo_lite import DemoLiteCharacterMotion, _load_motion_npz, _model_name_for_skeleton

REPO_ROOT = Path(__file__).resolve().parents[2]
PYROKI_ROOT = REPO_ROOT / "pyroki"
if str(PYROKI_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PYROKI_ROOT / "src"))

DEFAULT_PROXY_MOTION = (
    REPO_ROOT / "kimodo" / "assets" / "demo" / "examples" / "kimodo-soma-rp" / "01_single_text_prompt" / "motion.npz"
)
DEFAULT_EXAMPLES_DIR = REPO_ROOT / "kimodo" / "assets" / "demo" / "examples" / "antt-t1-demo2"
ANTT_T1_URDF = REPO_ROOT / "pyroki" / "examples" / "nero_both_arms" / "antt_t1.urdf"
ARM_HOME_CFG = np.array(
    [
        0.0,
        1.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
URDF_TO_SCENE_ROT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)
ROBOT_ROOT_NAME = "/antt_t1"
LEFT_UI_LINK_NAME = "left_gripper_base"
RIGHT_UI_LINK_NAME = "right_gripper_base"
SELF_COLLISION_MARGIN = 0.06
SELF_COLLISION_WEIGHT = 10.0
LEFT_ARM_JOINT_MASK = np.array([1.0] * 7 + [0.0] * 11, dtype=np.float32)
RIGHT_ARM_JOINT_MASK = np.array([0.0] * 9 + [1.0] * 7 + [0.0] * 2, dtype=np.float32)
DUAL_ARM_JOINT_MASK = np.array([1.0] * 7 + [0.0] * 2 + [1.0] * 7 + [0.0] * 2, dtype=np.float32)


class ArmTargetKeyframeSet(ConstraintSet):
    """Arm-only EE constraint visualization with simple target markers."""

    _COLOR_MAP = {
        "LeftHand": (0, 200, 255),
        "RightHand": (255, 140, 0),
    }
    _EE_TYPE_MAP = {
        "LeftHand": "left-hand",
        "RightHand": "right-hand",
    }

    def __init__(self, name: str, server: viser.ViserServer, skeleton: SkeletonBase, display_name: str | None = None):
        super().__init__(name, server, skeleton, display_name=display_name)
        self.frame2keyid = {}

    def _iter_targets(self, joint_names: list[str]) -> list[tuple[str, int]]:
        targets: list[tuple[str, int]] = []
        for joint_name in joint_names:
            if joint_name == "LeftHand":
                proxy_joint_name = self.skeleton.left_hand_joint_names[0]
            elif joint_name == "RightHand":
                proxy_joint_name = self.skeleton.right_hand_joint_names[0]
            elif joint_name == "Hips":
                continue
            else:
                raise ValueError(f"Invalid joint name: {joint_name}")
            targets.append((joint_name, self.skeleton.bone_order_names_index[proxy_joint_name]))
        return list(dict.fromkeys(targets))

    def _label_position(self, joints_pos: np.ndarray, joint_names: list[str]) -> np.ndarray:
        targets = self._iter_targets(joint_names)
        if not targets:
            return joints_pos[self.skeleton.root_idx] + np.array([0.0, 0.06, 0.0], dtype=np.float32)
        points = np.array([joints_pos[idx] for _, idx in targets], dtype=np.float32)
        return points.mean(axis=0) + np.array([0.0, 0.06, 0.0], dtype=np.float32)

    def create_scene_elements(
        self,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray | None,
        joint_names: list[str],
        viz_label: bool = True,
    ) -> None:
        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot) if joints_rot is not None else None
        targets = self._iter_targets(joint_names)

        marker_handles = {}
        axes_positions = []
        axes_wxyzs = []
        for joint_name, joint_idx in targets:
            marker_handles[joint_name] = self.server.scene.add_icosphere(
                f"/{self.name}/target_{frame_idx}_{joint_name}",
                radius=0.035,
                color=self._COLOR_MAP[joint_name],
                position=tuple(joints_pos_np[joint_idx]),
            )
            if joints_rot_np is not None:
                axes_positions.append(joints_pos_np[joint_idx])
                axes_wxyzs.append(tf.SO3.from_matrix(joints_rot_np[joint_idx]).wxyz)

        scene_data = {"markers": marker_handles}
        if axes_positions:
            scene_data["ee_axes"] = self.server.scene.add_batched_axes(
                f"/{self.name}/ee_axes_{frame_idx}",
                batched_wxyzs=np.asarray(axes_wxyzs, dtype=np.float32),
                batched_positions=np.asarray(axes_positions, dtype=np.float32),
                axes_length=0.09,
                axes_radius=0.008,
            )
        if viz_label:
            label = self.server.scene.add_label(
                name=f"/{self.name}/label_{frame_idx}",
                text=f"{self.display_name} @ {frame_idx}",
                position=self._label_position(joints_pos_np, joint_names),
                font_size_mode="screen",
                font_screen_scale=0.7,
                anchor="bottom-center",
            )
            label.visible = self.labels_visible
            scene_data["label"] = label

        self.scene_elements[frame_idx] = scene_data

    def add_keyframe(
        self,
        keyframe_id: str,
        frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        joint_names: list[str],
        end_effector_type: str,
        viz_label: bool = True,
        exists_ok: bool = False,
    ) -> None:
        need_create_viz = True
        joint_names_input = joint_names

        if not isinstance(end_effector_type, set):
            end_effector_type = {end_effector_type}

        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot)

        if frame_idx in self.keyframes:
            if joint_names != self.keyframes[frame_idx]["joint_names"]:
                merged_joint_names = set(joint_names)
                merged_joint_names.update(set(self.keyframes[frame_idx]["joint_names"]))
                joint_names = list(merged_joint_names)
                end_effector_type.update(self.keyframes[frame_idx]["end_effector_type"])
                self.clear(frame_idx)
            else:
                need_create_viz = False
                scene_data = self.scene_elements[frame_idx]
                for joint_name, joint_idx in self._iter_targets(joint_names):
                    scene_data["markers"][joint_name].position = joints_pos_np[joint_idx]
                if "ee_axes" in scene_data:
                    axes_positions = [joints_pos_np[joint_idx] for _, joint_idx in self._iter_targets(joint_names)]
                    axes_wxyzs = [tf.SO3.from_matrix(joints_rot_np[joint_idx]).wxyz for _, joint_idx in self._iter_targets(joint_names)]
                    scene_data["ee_axes"].batched_positions = np.asarray(axes_positions, dtype=np.float32)
                    scene_data["ee_axes"].batched_wxyzs = np.asarray(axes_wxyzs, dtype=np.float32)
                if viz_label and "label" in scene_data:
                    scene_data["label"].position = self._label_position(joints_pos_np, joint_names)
                    scene_data["label"].visible = self.labels_visible

        if need_create_viz:
            self.create_scene_elements(frame_idx, joints_pos_np, joints_rot_np, joint_names, viz_label=viz_label)

        self.keyframes[frame_idx] = {
            "joints_pos": joints_pos_np,
            "joints_rot": joints_rot_np,
            "joint_names": joint_names,
            "end_effector_type": end_effector_type,
        }

        if frame_idx not in self.frame2keyid:
            self.frame2keyid[frame_idx] = []

        known_keyframe_ids = {k: idx for idx, (k, _) in enumerate(self.frame2keyid[frame_idx])}
        if keyframe_id in known_keyframe_ids:
            if not exists_ok:
                raise AssertionError("keyframe_id already exists in this frame!")
            idx = known_keyframe_ids[keyframe_id]
            self.frame2keyid[frame_idx][idx] = (keyframe_id, joint_names_input)
        else:
            self.frame2keyid[frame_idx].append((keyframe_id, joint_names_input))

    def add_interval(
        self,
        interval_id: str,
        start_frame_idx: int,
        end_frame_idx: int,
        joints_pos: torch.Tensor | np.ndarray,
        joints_rot: torch.Tensor | np.ndarray,
        joint_names: list[str],
        end_effector_type: str,
    ) -> None:
        num_frames = end_frame_idx - start_frame_idx + 1
        joints_pos_np = to_numpy(joints_pos)
        joints_rot_np = to_numpy(joints_rot)
        assert joints_pos_np.shape[0] == num_frames
        assert joints_rot_np.shape[0] == num_frames

        for frame_idx in range(start_frame_idx, end_frame_idx + 1):
            rel_idx = frame_idx - start_frame_idx
            self.add_keyframe(
                interval_id,
                frame_idx,
                joints_pos_np[rel_idx],
                joints_rot_np[rel_idx],
                joint_names,
                end_effector_type,
                viz_label=False,
            )
        self._add_interval_label(start_frame_idx, end_frame_idx)

    def remove_keyframe(self, keyframe_id: str, frame_idx: int) -> None:
        if frame_idx not in self.keyframes:
            return

        remaining_joint_names = set()
        delete_idx = None
        for i, (keyid, joint_names) in enumerate(self.frame2keyid[frame_idx]):
            if keyid == keyframe_id:
                delete_idx = i
            else:
                remaining_joint_names.update(joint_names)
        if delete_idx is None:
            return

        self.frame2keyid[frame_idx].pop(delete_idx)
        if len(remaining_joint_names) == 0:
            del self.frame2keyid[frame_idx]
            self.clear(frame_idx)
            return

        new_joint_names = list(remaining_joint_names)
        self.clear(frame_idx, scene_elements_only=True)
        joints_pos = self.keyframes[frame_idx]["joints_pos"]
        joints_rot = self.keyframes[frame_idx]["joints_rot"]
        self.create_scene_elements(frame_idx, joints_pos, joints_rot, new_joint_names)
        self.keyframes[frame_idx]["joint_names"] = new_joint_names
        self.keyframes[frame_idx]["end_effector_type"] = {
            self._EE_TYPE_MAP[name] for name in new_joint_names if name in self._EE_TYPE_MAP
        }

    def update_target_position(self, frame_idx: int, joint_name: str, position: np.ndarray) -> None:
        if frame_idx not in self.keyframes:
            return
        if joint_name not in ("LeftHand", "RightHand"):
            raise ValueError(f"Unsupported target joint for update: {joint_name}")

        joint_idx = self._iter_targets([joint_name])[0][1]
        position_np = np.asarray(position, dtype=np.float32)
        self.keyframes[frame_idx]["joints_pos"][joint_idx] = position_np

        scene_data = self.scene_elements.get(frame_idx)
        if scene_data is None:
            return
        marker = scene_data.get("markers", {}).get(joint_name)
        if marker is not None:
            marker.position = position_np

        if "ee_axes" in scene_data:
            axes_positions = [
                self.keyframes[frame_idx]["joints_pos"][idx]
                for _, idx in self._iter_targets(self.keyframes[frame_idx]["joint_names"])
            ]
            scene_data["ee_axes"].batched_positions = np.asarray(axes_positions, dtype=np.float32)

        label = scene_data.get("label")
        if label is not None:
            label.position = self._label_position(
                self.keyframes[frame_idx]["joints_pos"],
                self.keyframes[frame_idx]["joint_names"],
            )

    def update_target_pose(self, frame_idx: int, joint_name: str, position: np.ndarray, wxyz: np.ndarray) -> None:
        if frame_idx not in self.keyframes:
            return
        if joint_name not in ("LeftHand", "RightHand"):
            raise ValueError(f"Unsupported target joint for update: {joint_name}")

        joint_idx = self._iter_targets([joint_name])[0][1]
        position_np = np.asarray(position, dtype=np.float32)
        rot_np = tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix().astype(np.float32)
        self.keyframes[frame_idx]["joints_pos"][joint_idx] = position_np
        self.keyframes[frame_idx]["joints_rot"][joint_idx] = rot_np

        scene_data = self.scene_elements.get(frame_idx)
        if scene_data is None:
            return
        marker = scene_data.get("markers", {}).get(joint_name)
        if marker is not None:
            marker.position = position_np

        if "ee_axes" in scene_data:
            targets = self._iter_targets(self.keyframes[frame_idx]["joint_names"])
            scene_data["ee_axes"].batched_positions = np.asarray(
                [self.keyframes[frame_idx]["joints_pos"][idx] for _, idx in targets],
                dtype=np.float32,
            )
            scene_data["ee_axes"].batched_wxyzs = np.asarray(
                [tf.SO3.from_matrix(self.keyframes[frame_idx]["joints_rot"][idx]).wxyz for _, idx in targets],
                dtype=np.float32,
            )

        label = scene_data.get("label")
        if label is not None:
            label.position = self._label_position(
                self.keyframes[frame_idx]["joints_pos"],
                self.keyframes[frame_idx]["joint_names"],
            )

    def _get_label_pos(self, frame_idx: int):
        data = self.keyframes[frame_idx]
        return self._label_position(data["joints_pos"], data["joint_names"])

    def remove_interval(self, interval_id: str, start_frame_idx: int, end_frame_idx: int) -> None:
        self._remove_interval_and_update_label(interval_id, start_frame_idx, end_frame_idx)

    def get_constraint_info(self, device: str | None = None):
        all_joints_pos = []
        all_joints_rot = []
        all_joints_names = []
        all_end_effector_type = []
        for value in self.keyframes.values():
            joints_pos = to_torch(value["joints_pos"], device=device)
            joints_rot = to_torch(value["joints_rot"], device=device)
            if len(joints_pos.shape) == 2:
                all_joints_pos.append(joints_pos[None])
            else:
                all_joints_pos.append(joints_pos)
            if len(joints_rot.shape) == 3:
                all_joints_rot.append(joints_rot[None])
            else:
                all_joints_rot.append(joints_rot)
            all_joints_names.append(value["joint_names"])
            all_end_effector_type.append(value["end_effector_type"])

        all_joints_pos = torch.cat(all_joints_pos, dim=0) if all_joints_pos else None
        all_joints_rot = torch.cat(all_joints_rot, dim=0) if all_joints_rot else None

        return {
            "frame_idx": self.get_frame_idx(),
            "joints_pos": all_joints_pos,
            "joints_rot": all_joints_rot,
            "joint_names": all_joints_names,
            "end_effector_type": all_end_effector_type,
        }

    def clear(self, frame_idx: int | None = None, scene_elements_only: bool = False) -> None:
        frame_idx_list = list(self.keyframes.keys()) if frame_idx is None else [frame_idx]
        for fidx in frame_idx_list:
            scene_data = self.scene_elements.get(fidx)
            if scene_data is None:
                continue
            for marker in scene_data.get("markers", {}).values():
                self.server.scene.remove_by_name(marker.name)
            if "ee_axes" in scene_data:
                self.server.scene.remove_by_name(scene_data["ee_axes"].name)
            if "label" in scene_data:
                self.server.scene.remove_by_name(scene_data["label"].name)
            self.scene_elements.pop(fidx, None)
            if not scene_elements_only:
                self.keyframes.pop(fidx, None)

        if frame_idx is None:
            for interval_label in list(self.interval_labels.values()):
                self.server.scene.remove_by_name(interval_label.name)
            self.interval_labels.clear()

    def set_overlay_visibility(self, only_frame: int | None = None) -> None:
        show_all = only_frame is None
        for fidx, scene_data in self.scene_elements.items():
            visible = show_all or fidx == only_frame
            for marker in scene_data.get("markers", {}).values():
                marker.visible = visible
            if "ee_axes" in scene_data:
                scene_data["ee_axes"].visible = visible
            label = scene_data.get("label")
            if label is not None:
                label.visible = visible and self.labels_visible
        for interval_label in self.interval_labels.values():
            interval_label.visible = show_all and self.labels_visible


class Demo2:
    def __init__(self, motion_path: Path, host: str = "127.0.0.1", port: int = 7860, fps: float = 30.0):
        self.device = "cpu"
        self.motion_path = motion_path
        self.playback_fps = fps
        self.playback_only = True
        self.client_sessions: dict[int, ClientSession] = {}
        self.start_direction_markers = {}
        self.grid_handles = {}
        self.floor_len = 20.0

        self.joints_pos, self.joints_rot, self.foot_contacts, self.skeleton = _load_motion_npz(motion_path, self.device)
        self.model_name = _model_name_for_skeleton(self.skeleton)
        self.model_fps = fps

        self.left_proxy_joint_idx = self.skeleton.bone_order_names_index[self.skeleton.left_hand_joint_names[0]]
        self.right_proxy_joint_idx = self.skeleton.bone_order_names_index[self.skeleton.right_hand_joint_names[0]]

        self.antt_urdf = yourdfpy.URDF.load(str(ANTT_T1_URDF))
        _, _, _, _, _, pk = self._import_pyroki()
        self.robot_coll = pk.collision.RobotCollision.from_urdf(self.antt_urdf)
        self.robot_home_targets = self._compute_robot_home_targets()
        self.proxy_joints_pos, self.proxy_joints_rot = self._build_static_proxy_motion()
        self._solver_kernels = None
        self.ori_weight = 10.0

        self.server = viser.ViserServer(
            host=host,
            port=port,
            label="Kimodo Demo2",
            enable_camera_keyboard_controls=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)

    def _urdf_point_to_scene(self, point: np.ndarray) -> np.ndarray:
        return (URDF_TO_SCENE_ROT @ np.asarray(point, dtype=np.float32)).astype(np.float32)

    def _scene_point_to_urdf(self, point: np.ndarray) -> np.ndarray:
        return (URDF_TO_SCENE_ROT.T @ np.asarray(point, dtype=np.float32)).astype(np.float32)

    def _scene_wxyz_to_urdf(self, wxyz: np.ndarray) -> np.ndarray:
        rot = URDF_TO_SCENE_ROT.T @ tf.SO3(np.asarray(wxyz, dtype=np.float32)).as_matrix()
        return tf.SO3.from_matrix(rot.astype(np.float64)).wxyz.astype(np.float32)

    def _compute_robot_home_targets(self) -> dict[str, np.ndarray]:
        home_urdf = yourdfpy.URDF.load(str(ANTT_T1_URDF))
        home_urdf.update_cfg(ARM_HOME_CFG.copy())
        return {
            "left_hand": self._urdf_point_to_scene(
                home_urdf.get_transform(LEFT_UI_LINK_NAME, home_urdf.base_link)[:3, 3]
            ),
            "right_hand": self._urdf_point_to_scene(
                home_urdf.get_transform(RIGHT_UI_LINK_NAME, home_urdf.base_link)[:3, 3]
            ),
            "root": self._urdf_point_to_scene(home_urdf.get_transform("tbar_link", home_urdf.base_link)[:3, 3]),
        }

    def _build_static_proxy_motion(self) -> tuple[torch.Tensor, torch.Tensor]:
        total_frames = self.joints_pos.shape[0]
        base_pos = self.joints_pos[0].clone()
        base_rot = self.joints_rot[0].clone()

        source_hand_center = (base_pos[self.left_proxy_joint_idx] + base_pos[self.right_proxy_joint_idx]) / 2.0
        target_hand_center = torch.from_numpy(
            (self.robot_home_targets["left_hand"] + self.robot_home_targets["right_hand"]) / 2.0
        ).to(device=base_pos.device, dtype=base_pos.dtype)
        base_pos = base_pos + (target_hand_center - source_hand_center)

        base_pos[self.skeleton.root_idx] = torch.from_numpy(self.robot_home_targets["root"]).to(
            device=base_pos.device,
            dtype=base_pos.dtype,
        )
        base_pos[self.left_proxy_joint_idx] = torch.from_numpy(self.robot_home_targets["left_hand"]).to(
            device=base_pos.device,
            dtype=base_pos.dtype,
        )
        base_pos[self.right_proxy_joint_idx] = torch.from_numpy(self.robot_home_targets["right_hand"]).to(
            device=base_pos.device,
            dtype=base_pos.dtype,
        )

        proxy_joints_pos = base_pos.unsqueeze(0).repeat(total_frames, 1, 1)
        proxy_joints_rot = base_rot.unsqueeze(0).repeat(total_frames, 1, 1, 1)
        return proxy_joints_pos, proxy_joints_rot

    def client_active(self, client_id: int) -> bool:
        return client_id in self.client_sessions

    def get_examples_base_dir(self, model_name: str, absolute: bool = True) -> str:
        del model_name, absolute
        DEFAULT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        return str(DEFAULT_EXAMPLES_DIR)

    def set_timeline_defaults(self, timeline, model_fps: float) -> None:
        timeline.set_defaults(
            default_text="ANTT T1 Arm Plan",
            default_duration=max(1, self.joints_pos.shape[0] - 1),
            min_duration=1,
            max_duration=max(1, self.joints_pos.shape[0] - 1),
            default_num_frames_zoom=max(60, self.joints_pos.shape[0]),
            max_frames_zoom=max(60, self.joints_pos.shape[0] * 2),
            fps=model_fps,
        )

    def build_constraint_tracks(self, client: viser.ClientHandle, skeleton: SkeletonBase):
        return {
            "End-Effectors": ArmTargetKeyframeSet(
                name="End-Effectors",
                server=client,
                skeleton=skeleton,
            ),
        }

    def _apply_constraint_overlay_visibility(self, session: ClientSession) -> None:
        only_frame = session.frame_idx if session.show_only_current_constraint else None
        for constraint in session.constraints.values():
            constraint.set_overlay_visibility(only_frame=only_frame)

    def configure_theme(self, client: viser.ClientHandle, dark_mode: bool = False, titlebar_dark_mode_checkbox_uuid=None):
        from kimodo.demo.app import Demo

        return Demo.configure_theme(self, client, dark_mode, titlebar_dark_mode_checkbox_uuid)

    def setup_scene(self, client: viser.ClientHandle) -> None:
        from kimodo.demo.app import Demo

        Demo.setup_scene(self, client)

    def set_start_direction_visible(self, client_id: int, visible: bool) -> None:
        from kimodo.demo.app import Demo

        return Demo.set_start_direction_visible(self, client_id, visible)

    def set_constraint_tracks_visible(self, session: ClientSession, visible: bool) -> None:
        from kimodo.demo.app import Demo

        return Demo.set_constraint_tracks_visible(self, session, visible)

    def add_character_motion(
        self,
        client: viser.ClientHandle,
        skeleton: SkeletonBase,
        joints_pos: torch.Tensor | None = None,
        joints_rot: torch.Tensor | None = None,
        foot_contacts: torch.Tensor | None = None,
    ) -> None:
        client_id = client.client_id
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        character_name = f"proxy_character{len(session.motions)}"

        character = Character(
            character_name,
            client,
            skeleton,
            create_skeleton_mesh=False,
            create_skinned_mesh=False,
            visible_skeleton=False,
            visible_skinned_mesh=False,
            skinned_mesh_opacity=0.0,
            show_foot_contacts=False,
            dark_mode=session.gui_elements.gui_dark_mode_checkbox.value,
            mesh_mode="soma_skin",
            gui_use_soma_layer_checkbox=session.gui_elements.gui_use_soma_layer_checkbox,
        )
        if joints_pos is None:
            joints_pos = self.proxy_joints_pos
        if joints_rot is None:
            joints_rot = self.proxy_joints_rot
        motion = DemoLiteCharacterMotion(character, joints_pos, joints_rot, foot_contacts)
        session.motions[character_name] = motion
        motion.character.set_skeleton_visibility(False)
        motion.set_frame(session.frame_idx)

    def clear_motions(self, client_id: int) -> None:
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        for motion in list(session.motions.values()):
            motion.clear()
        session.motions.clear()

    def set_frame(self, client_id: int, frame_idx: int, update_timeline: bool = True):
        if not self.client_active(client_id):
            return
        session = self.client_sessions[client_id]
        session.frame_idx = frame_idx
        if update_timeline:
            session.client.timeline.set_current_frame(frame_idx)
        for motion in list(session.motions.values()):
            motion.set_frame(frame_idx)
        self._apply_constraint_overlay_visibility(session)
        traj = getattr(session, "robot_joint_trajectory", None)
        urdf_vis = getattr(session, "robot_urdf_vis", None)
        if traj is not None and urdf_vis is not None and len(traj) > 0:
            clamped_idx = min(frame_idx, len(traj) - 1)
            urdf_vis.update_cfg(np.asarray(traj[clamped_idx], dtype=np.float32))
        if getattr(session, "edit_mode", False):
            self.refresh_custom_edit_gizmos(session)

    def enter_custom_edit_mode(self, session: ClientSession) -> bool:
        session.custom_edit_gizmos = {}
        self.refresh_custom_edit_gizmos(session)
        return True

    def exit_custom_edit_mode(self, session: ClientSession) -> bool:
        self.clear_custom_edit_gizmos(session)
        return True

    def clear_custom_edit_gizmos(self, session: ClientSession) -> None:
        for handle in getattr(session, "custom_edit_gizmos", {}).values():
            session.client.scene.remove_by_name(handle.name)
        session.custom_edit_gizmos = {}

    def _proxy_joint_idx_for_name(self, joint_name: str) -> int:
        if joint_name == "LeftHand":
            return self.left_proxy_joint_idx
        if joint_name == "RightHand":
            return self.right_proxy_joint_idx
        raise ValueError(f"Unsupported joint name: {joint_name}")

    def refresh_custom_edit_gizmos(self, session: ClientSession) -> None:
        self.clear_custom_edit_gizmos(session)
        ee_constraint = session.constraints.get("End-Effectors")
        if ee_constraint is None or session.frame_idx not in ee_constraint.keyframes:
            return

        scene_data = ee_constraint.scene_elements.get(session.frame_idx, {})
        markers = scene_data.get("markers", {})
        for joint_name in ("LeftHand", "RightHand"):
            marker = markers.get(joint_name)
            if marker is None:
                continue
            handle = session.client.scene.add_transform_controls(
                f"/demo2/edit_target_{joint_name}",
                scale=0.18,
                line_width=3.0,
                active_axes=(True, True, True),
                disable_axes=False,
                disable_sliders=False,
                disable_rotations=False,
                depth_test=False,
                position=np.asarray(marker.position, dtype=np.float32),
                wxyz=tf.SO3.from_matrix(
                    ee_constraint.keyframes[session.frame_idx]["joints_rot"][self._proxy_joint_idx_for_name(joint_name)]
                ).wxyz,
            )
            session.custom_edit_gizmos[joint_name] = handle

            def set_update_callback(name: str, gizmo_handle):
                @gizmo_handle.on_update
                def _(_) -> None:
                    if session.frame_idx not in ee_constraint.keyframes:
                        return
                    ee_constraint.update_target_pose(
                        session.frame_idx,
                        name,
                        np.asarray(gizmo_handle.position, dtype=np.float32),
                        np.asarray(gizmo_handle.wxyz, dtype=np.float32),
                    )

            set_update_callback(joint_name, handle)

    def compute_model_constraints_lst(self, session: ClientSession, _model_bundle, num_frames: int):
        del session, _model_bundle, num_frames
        return []

    def _import_pyroki(self):
        import jax
        import jax.numpy as jnp
        import jax_dataclasses as jdc
        import jaxlie
        import jaxls
        import pyroki as pk

        return jax, jnp, jdc, jaxlie, jaxls, pk

    def _get_solver_kernels(self):
        if self._solver_kernels is not None:
            return self._solver_kernels

        jax, jnp, jdc, jaxlie, jaxls, pk = self._import_pyroki()

        @jdc.jit
        def solve_one_target(
            robot_model,
            robot_coll,
            target_wxyz,
            target_position,
            target_joint_index,
            prev_cfg_jax,
            ori_weight,
            joint_mask,
        ):
            joint_var = robot_model.joint_var_cls(0)
            target_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz),
                target_position,
            )
            costs = [
                pk.costs.pose_cost_analytic_jac(
                    robot_model,
                    joint_var,
                    target_pose,
                    target_joint_index,
                    pos_weight=50.0,
                    ori_weight=ori_weight,
                    joint_mask=joint_mask,
                ),
                pk.costs.rest_cost(
                    joint_var,
                    rest_pose=prev_cfg_jax,
                    weight=0.35,
                ),
                pk.costs.self_collision_cost(
                    robot_model,
                    robot_coll=robot_coll,
                    joint_var=joint_var,
                    margin=SELF_COLLISION_MARGIN,
                    weight=SELF_COLLISION_WEIGHT,
                ),
                pk.costs.limit_constraint(
                    robot_model,
                    joint_var,
                ),
            ]
            solution = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg_jax)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=5.0),
                )
            )
            return solution[joint_var]

        @jdc.jit
        def solve_two_targets(
            robot_model,
            robot_coll,
            target_wxyzs,
            target_positions,
            target_joint_indices,
            prev_cfg_jax,
            ori_weight,
            joint_mask,
        ):
            joint_var = robot_model.joint_var_cls(0)
            target_pose = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyzs),
                target_positions,
            )
            batched_robot = jax.tree.map(lambda x: x[None], robot_model)
            batch_axes = target_pose.get_batch_axes()
            costs = [
                pk.costs.pose_cost_analytic_jac(
                    batched_robot,
                    robot_model.joint_var_cls(jnp.full(batch_axes, 0)),
                    target_pose,
                    target_joint_indices,
                    pos_weight=50.0,
                    ori_weight=ori_weight,
                    joint_mask=joint_mask,
                ),
                pk.costs.rest_cost(
                    joint_var,
                    rest_pose=prev_cfg_jax,
                    weight=0.35,
                ),
                pk.costs.self_collision_cost(
                    robot_model,
                    robot_coll=robot_coll,
                    joint_var=joint_var,
                    margin=SELF_COLLISION_MARGIN,
                    weight=SELF_COLLISION_WEIGHT,
                ),
                pk.costs.limit_constraint(
                    robot_model,
                    joint_var,
                ),
            ]
            solution = (
                jaxls.LeastSquaresProblem(costs=costs, variables=[joint_var])
                .analyze()
                .solve(
                    initial_vals=jaxls.VarValues.make([joint_var.with_value(prev_cfg_jax)]),
                    verbose=False,
                    linear_solver="dense_cholesky",
                    trust_region=jaxls.TrustRegionConfig(lambda_initial=5.0),
                )
            )
            return solution[joint_var]

        self._solver_kernels = (solve_one_target, solve_two_targets)
        return self._solver_kernels

    def _solve_ik_frame(
        self,
        robot,
        target_link_names: list[str],
        target_positions: np.ndarray,
        target_wxyzs: np.ndarray,
        joint_mask: np.ndarray,
        prev_cfg: np.ndarray,
    ) -> np.ndarray:
        _, jnp, _, _, _, _ = self._import_pyroki()
        solve_one_target, solve_two_targets = self._get_solver_kernels()
        target_link_indices = np.array([robot.links.names.index(name) for name in target_link_names], dtype=np.int32)
        if len(target_link_names) == 1:
            solved = solve_one_target(
                robot,
                self.robot_coll,
                jnp.array(target_wxyzs[0], dtype=jnp.float32),
                jnp.array(target_positions[0], dtype=jnp.float32),
                jnp.array(target_link_indices[0], dtype=jnp.int32),
                jnp.array(prev_cfg, dtype=jnp.float32),
                jnp.array(self.ori_weight, dtype=jnp.float32),
                jnp.array(joint_mask, dtype=jnp.float32),
            )
        elif len(target_link_names) == 2:
            solved = solve_two_targets(
                robot,
                self.robot_coll,
                jnp.array(target_wxyzs, dtype=jnp.float32),
                jnp.array(target_positions, dtype=jnp.float32),
                jnp.array(target_link_indices, dtype=jnp.int32),
                jnp.array(prev_cfg, dtype=jnp.float32),
                jnp.array(self.ori_weight, dtype=jnp.float32),
                jnp.array(joint_mask, dtype=jnp.float32),
            )
        else:
            raise ValueError(f"Unsupported number of IK targets: {len(target_link_names)}")
        return np.asarray(solved, dtype=np.float32)

    def _build_target_path(self, frame_to_position: dict[int, np.ndarray], total_frames: int) -> np.ndarray | None:
        if not frame_to_position:
            return None
        frames = np.array(sorted(frame_to_position.keys()), dtype=np.int32)
        positions = np.array([frame_to_position[frame] for frame in frames], dtype=np.float32)
        target_path = np.empty((total_frames, 3), dtype=np.float32)
        frame_domain = np.arange(total_frames, dtype=np.float32)
        for coord in range(3):
            target_path[:, coord] = np.interp(
                frame_domain,
                frames.astype(np.float32),
                positions[:, coord],
                left=positions[0, coord],
                right=positions[-1, coord],
            )
        return target_path

    def _extract_hand_targets(
        self, session: ClientSession
    ) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, tuple[np.ndarray, np.ndarray]], int]:
        ee_constraint = session.constraints.get("End-Effectors")
        if ee_constraint is None:
            return {}, {}, session.max_frame_idx + 1
        info = ee_constraint.get_constraint_info(device=self.device)
        frame_idx = info["frame_idx"]
        joints_pos = info["joints_pos"]
        joint_names = info["joint_names"]
        if frame_idx is None or joints_pos is None:
            return {}, {}, session.max_frame_idx + 1

        left_targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        right_targets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        frame_indices = [int(frame) for frame in frame_idx]
        for idx, frame in enumerate(frame_indices):
            pose = joints_pos[idx].detach().cpu().numpy()
            rots = info["joints_rot"][idx].detach().cpu().numpy()
            names = set(joint_names[idx])
            if "LeftHand" in names:
                left_targets[frame] = (
                    pose[self.left_proxy_joint_idx].astype(np.float32),
                    tf.SO3.from_matrix(rots[self.left_proxy_joint_idx]).wxyz.astype(np.float32),
                )
            if "RightHand" in names:
                right_targets[frame] = (
                    pose[self.right_proxy_joint_idx].astype(np.float32),
                    tf.SO3.from_matrix(rots[self.right_proxy_joint_idx]).wxyz.astype(np.float32),
                )

        return left_targets, right_targets, session.max_frame_idx + 1

    def _interpolate_joint_trajectory(
        self,
        solved_cfg_by_frame: dict[int, np.ndarray],
        total_frames: int,
        dof: int,
        initial_cfg: np.ndarray,
    ) -> np.ndarray:
        trajectory = np.zeros((total_frames, dof), dtype=np.float32)
        solved_frames = sorted(solved_cfg_by_frame.keys())
        if not solved_frames:
            return trajectory

        first_frame = solved_frames[0]
        if first_frame > 0:
            start_cfg = np.asarray(initial_cfg, dtype=np.float32)
            end_cfg = solved_cfg_by_frame[first_frame]
            span = first_frame
            for rel_idx in range(span + 1):
                s = rel_idx / max(span, 1)
                alpha = 10 * s**3 - 15 * s**4 + 6 * s**5
                trajectory[rel_idx] = (1.0 - alpha) * start_cfg + alpha * end_cfg
        else:
            trajectory[0] = solved_cfg_by_frame[first_frame]

        for start_frame, end_frame in zip(solved_frames[:-1], solved_frames[1:]):
            start_cfg = solved_cfg_by_frame[start_frame]
            end_cfg = solved_cfg_by_frame[end_frame]
            span = max(end_frame - start_frame, 1)
            for rel_idx in range(span + 1):
                s = rel_idx / span
                alpha = 10 * s**3 - 15 * s**4 + 6 * s**5
                trajectory[start_frame + rel_idx] = (1.0 - alpha) * start_cfg + alpha * end_cfg

        last_frame = solved_frames[-1]
        trajectory[last_frame:] = solved_cfg_by_frame[last_frame]
        return trajectory

    def _solve_robot_trajectory(self, session: ClientSession) -> tuple[np.ndarray, int]:
        left_targets, right_targets, total_frames = self._extract_hand_targets(session)
        if not left_targets and not right_targets:
            raise ValueError("Add at least one Left Hand or Right Hand constraint before solving.")

        try:
            _, _, _, _, _, pk = self._import_pyroki()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PyRoki dependencies are missing. Install them in the kimodo env with "
                "`python -m pip install -e ./pyroki`."
            ) from exc

        robot = pk.Robot.from_urdf(self.antt_urdf, default_joint_cfg=ARM_HOME_CFG.copy())
        prev_cfg = ARM_HOME_CFG.copy()
        solved_cfg_by_frame: dict[int, np.ndarray] = {}
        constrained_frames = sorted(set(left_targets.keys()) | set(right_targets.keys()))

        for frame_idx in constrained_frames:
            target_link_names: list[str] = []
            target_positions: list[np.ndarray] = []
            target_wxyzs: list[np.ndarray] = []
            left_active = frame_idx in left_targets
            right_active = frame_idx in right_targets
            if frame_idx in left_targets:
                target_link_names.append(LEFT_UI_LINK_NAME)
                target_positions.append(self._scene_point_to_urdf(left_targets[frame_idx][0]))
                target_wxyzs.append(self._scene_wxyz_to_urdf(left_targets[frame_idx][1]))
            if frame_idx in right_targets:
                target_link_names.append(RIGHT_UI_LINK_NAME)
                target_positions.append(self._scene_point_to_urdf(right_targets[frame_idx][0]))
                target_wxyzs.append(self._scene_wxyz_to_urdf(right_targets[frame_idx][1]))
            if target_link_names:
                if left_active and right_active:
                    joint_mask = DUAL_ARM_JOINT_MASK
                elif left_active:
                    joint_mask = LEFT_ARM_JOINT_MASK
                else:
                    joint_mask = RIGHT_ARM_JOINT_MASK
                prev_cfg = self._solve_ik_frame(
                    robot=robot,
                    target_link_names=target_link_names,
                    target_positions=np.asarray(target_positions, dtype=np.float32),
                    target_wxyzs=np.asarray(target_wxyzs, dtype=np.float32),
                    joint_mask=joint_mask,
                    prev_cfg=prev_cfg,
                )
                solved_cfg_by_frame[frame_idx] = prev_cfg.copy()

        trajectory = self._interpolate_joint_trajectory(
            solved_cfg_by_frame,
            total_frames=total_frames,
            dof=robot.joints.num_actuated_joints,
            initial_cfg=ARM_HOME_CFG.copy(),
        )
        return trajectory, robot.joints.num_actuated_joints

    def _add_robot_ui(self, client: viser.ClientHandle) -> None:
        client_id = client.client_id
        with client.gui.add_folder("ANTT T1", expand_by_default=True):
            solve_button = client.gui.add_button("Solve With PyRoki")
            status_text = client.gui.add_text("Status", initial_value="No solved trajectory yet")
            loop_checkbox = client.gui.add_checkbox("Loop Playback", initial_value=True)
            ori_weight_slider = client.gui.add_slider("Orientation Weight", min=0.0, max=50.0, step=0.5, initial_value=10.0)

        @solve_button.on_click
        def _(event: viser.GuiEvent) -> None:
            if not self.client_active(client_id):
                return
            session = self.client_sessions[client_id]
            status_text.value = "Solving..."
            try:
                trajectory, dof = self._solve_robot_trajectory(session)
            except Exception as exc:  # noqa: BLE001
                status_text.value = f"Error: {exc}"
                event.client.add_notification(
                    title="PyRoki solve failed",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )
                return

            session.robot_joint_trajectory = trajectory
            session.robot_joint_dof = dof
            status_text.value = f"Solved {len(trajectory)} frames ({dof} joints)"
            self.set_frame(client_id, session.frame_idx, update_timeline=False)
            event.client.add_notification(
                title="PyRoki solve complete",
                body=f"Solved {len(trajectory)} frames for ANTT T1.",
                auto_close_seconds=5.0,
                color="green",
            )

        if self.client_active(client_id):
            self.client_sessions[client_id].loop_playback = loop_checkbox.value

        @loop_checkbox.on_update
        def _(_) -> None:
            if not self.client_active(client_id):
                return
            self.client_sessions[client_id].loop_playback = loop_checkbox.value

        @ori_weight_slider.on_update
        def _(_) -> None:
            self.ori_weight = float(ori_weight_slider.value)

    def _add_robot_visualization(self, client: viser.ClientHandle, session: ClientSession) -> None:
        self.server.scene.remove_by_name(ROBOT_ROOT_NAME)
        robot_root = client.scene.add_frame(
            ROBOT_ROOT_NAME,
            show_axes=False,
            position=(0.0, 0.0, 0.0),
        )
        robot_root.wxyz = tf.SO3.from_matrix(URDF_TO_SCENE_ROT.astype(np.float64)).wxyz
        urdf_vis = ViserUrdf(
            self.server,
            self.antt_urdf,
            root_node_name=ROBOT_ROOT_NAME,
        )
        session.robot_urdf_vis = urdf_vis
        session.robot_joint_trajectory = None
        session.robot_joint_dof = len(ARM_HOME_CFG)
        urdf_vis.update_cfg(ARM_HOME_CFG.copy())

    def on_client_connect(self, client: viser.ClientHandle) -> None:
        self.server.scene.remove_by_name(ROBOT_ROOT_NAME)
        self.setup_scene(client)
        constraint_tracks = self.build_constraint_tracks(client, self.skeleton)
        (
            gui_elements,
            timeline_tracks,
            example_dict,
            gui_examples_dropdown,
            gui_save_example_path_text,
            gui_model_selector,
        ) = ui.create_gui(
            demo=self,
            client=client,
            model_name=self.model_name,
            model_fps=self.model_fps,
            playback_only=True,
            allowed_constraint_tracks=("Left Hand", "Right Hand"),
        )

        timeline_data = {
            "tracks": timeline_tracks,
            "tracks_ids": {val["name"]: key for key, val in timeline_tracks.items()},
            "keyframes": {},
            "intervals": {},
            "keyframe_update_lock": threading.Lock(),
            "keyframe_move_timers": {},
            "pending_keyframe_moves": {},
            "constraint_tracks_visible": True,
            "dense_path_after_release_timer": None,
        }

        max_frame_idx = self.joints_pos.shape[0] - 1
        session = ClientSession(
            client=client,
            gui_elements=gui_elements,
            motions={},
            constraints=constraint_tracks,
            timeline_data=timeline_data,
            frame_idx=0,
            playing=False,
            playback_speed=DEFAULT_PLAYBACK_SPEED,
            cur_duration=max_frame_idx / self.model_fps,
            max_frame_idx=max_frame_idx,
            updating_motions=False,
            edit_mode=False,
            model_name=self.model_name,
            model_fps=self.model_fps,
            skeleton=self.skeleton,
            motion_rep=None,
            examples_base_dir=self.get_examples_base_dir(self.model_name, absolute=True),
            example_dict=example_dict,
            gui_examples_dropdown=gui_examples_dropdown,
            gui_save_example_path_text=gui_save_example_path_text,
            gui_model_selector=gui_model_selector,
        )
        session.loop_playback = True
        self.client_sessions[client.client_id] = session

        self.add_character_motion(client, self.skeleton)
        self._add_robot_visualization(client, session)
        self._add_robot_ui(client)

        session.gui_elements.gui_viz_skeleton_checkbox.value = False
        session.gui_elements.gui_viz_skeleton_checkbox.visible = False
        session.gui_elements.gui_viz_foot_contacts_checkbox.visible = False
        session.gui_elements.gui_viz_skinned_mesh_checkbox.value = False
        session.gui_elements.gui_viz_skinned_mesh_checkbox.visible = False
        session.gui_elements.gui_viz_skinned_mesh_opacity_slider.visible = False

        client.timeline.add_prompt("ANTT T1 Arm Plan", 0, max_frame_idx, color=(100, 200, 150))
        client.timeline.set_current_frame(0)
        self.set_frame(client.client_id, 0, update_timeline=False)

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        self.server.scene.remove_by_name(ROBOT_ROOT_NAME)
        self.client_sessions.pop(client.client_id, None)
        self.start_direction_markers.pop(client.client_id, None)
        self.grid_handles.pop(client.client_id, None)

    def run(self) -> None:
        update_counter = 0
        while True:
            for client_id, session in list(self.client_sessions.items()):
                update_interval = max(1, int(self.playback_fps / max(session.playback_speed * session.model_fps, 1e-6)))
                if session.playing and update_counter % update_interval == 0:
                    if session.frame_idx >= session.max_frame_idx:
                        if getattr(session, "loop_playback", True):
                            new_frame_idx = 0
                        else:
                            session.playing = False
                            new_frame_idx = session.max_frame_idx
                    else:
                        new_frame_idx = session.frame_idx + 1
                    if self.client_active(client_id):
                        self.set_frame(client_id, new_frame_idx)
            update_counter = (update_counter + 1) % max(int(self.playback_fps), 1)
            threading.Event().wait(1.0 / max(self.playback_fps, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kimodo arm-only ANTT T1 demo.")
    parser.add_argument("motion", nargs="?", default=str(DEFAULT_PROXY_MOTION), help="Proxy .npz motion used for EE editing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    demo = Demo2(Path(args.motion).expanduser().resolve(), host=args.host, port=args.port, fps=args.fps)
    demo.run()


if __name__ == "__main__":
    main()
