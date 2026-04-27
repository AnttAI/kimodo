# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playback-only Kimodo demo UI with the real timeline/editor layout."""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import viser

from kimodo.constraints import FullBodyConstraintSet, Root2DConstraintSet, TYPE_TO_CLASS
from kimodo.demo import ui
from kimodo.demo.config import DEFAULT_PLAYBACK_SPEED, MODEL_EXAMPLES_DIRS
from kimodo.demo.state import ClientSession
from kimodo.motion_io import load_motion_file
from kimodo.skeleton import SkeletonBase
from kimodo.viz.playback import CharacterMotion
from kimodo.viz.scene import Character

WORLD_ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "worlds"
DEFAULT_WORLD_GLB = WORLD_ASSET_DIR / "simple_kitchen.glb"


class DemoLiteCharacterMotion(CharacterMotion):
    """Avoid expensive startup-wide precomputation in playback-only mode."""

    def precompute_mesh_info(self):
        if self.character.skeleton_mesh is not None:
            self.character.skeleton_mesh.mesh_info_cache = None
        if self.character.skinned_mesh is not None:
            self.character.skinned_verts_cache = None

def _model_name_for_skeleton(skeleton: SkeletonBase) -> str:
    if skeleton.nbjoints in (30, 77):
        return "kimodo-soma-rp"
    if skeleton.nbjoints == 34:
        return "kimodo-g1-rp"
    if skeleton.nbjoints == 22:
        return "kimodo-smplx-rp"
    raise ValueError(f"Unsupported skeleton with {skeleton.nbjoints} joints")


class DemoLite:
    def __init__(self, motion_path: Path, host: str = "127.0.0.1", port: int = 7860, fps: float = 30.0):
        self.device = "cpu"
        self.motion_path = motion_path
        self.playback_fps = fps
        self.models: dict[str, object] = {}
        self.playback_only = True
        self.client_sessions: dict[int, ClientSession] = {}
        self.start_direction_markers = {}
        self.grid_handles = {}
        self.floor_len = 20.0

        self.joints_pos, self.joints_rot, self.foot_contacts, self.skeleton = load_motion_file(
            motion_path, self.device
        )
        self.model_name = _model_name_for_skeleton(self.skeleton)
        self.model_fps = fps

        self.server = viser.ViserServer(
            host=host,
            port=port,
            label="Kimodo Demo Lite",
            enable_camera_keyboard_controls=False,
        )
        self.server.scene.world_axes.visible = False
        self.server.scene.set_up_direction("+y")
        self.server.on_client_connect(self.on_client_connect)
        self.server.on_client_disconnect(self.on_client_disconnect)

    def client_active(self, client_id: int) -> bool:
        return client_id in self.client_sessions

    def get_examples_base_dir(self, model_name: str, absolute: bool = True) -> str:
        return MODEL_EXAMPLES_DIRS[model_name]

    def set_timeline_defaults(self, timeline, model_fps: float) -> None:
        timeline.set_defaults(
            default_text="Loaded motion",
            default_duration=max(1, self.joints_pos.shape[0] - 1),
            min_duration=1,
            max_duration=max(1, self.joints_pos.shape[0] - 1),
            default_num_frames_zoom=max(60, self.joints_pos.shape[0]),
            max_frames_zoom=max(60, self.joints_pos.shape[0] * 2),
            fps=model_fps,
        )

    def build_constraint_tracks(self, client: viser.ClientHandle, skeleton: SkeletonBase):
        from kimodo.demo.app import Demo

        return Demo.build_constraint_tracks(self, client, skeleton)

    def _apply_constraint_overlay_visibility(self, session: ClientSession) -> None:
        from kimodo.demo.app import Demo

        return Demo._apply_constraint_overlay_visibility(self, session)

    def configure_theme(self, client: viser.ClientHandle, dark_mode: bool = False, titlebar_dark_mode_checkbox_uuid=None):
        from kimodo.demo.app import Demo

        return Demo.configure_theme(self, client, dark_mode, titlebar_dark_mode_checkbox_uuid)

    def setup_scene(self, client: viser.ClientHandle) -> None:
        from kimodo.demo.app import Demo

        Demo.setup_scene(self, client)
        if DEFAULT_WORLD_GLB.exists():
            client.scene.add_glb(
                "/environment/kitchen",
                DEFAULT_WORLD_GLB.read_bytes(),
                position=(0.0, 0.0, -2.5),
                cast_shadow=True,
                receive_shadow=True,
            )

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
        character_name = f"character{len(session.motions)}"

        character = Character(
            character_name,
            client,
            skeleton,
            create_skeleton_mesh=True,
            create_skinned_mesh=False,
            visible_skeleton=True,
            visible_skinned_mesh=False,
            skinned_mesh_opacity=1.0,
            show_foot_contacts=session.gui_elements.gui_viz_foot_contacts_checkbox.value,
            dark_mode=session.gui_elements.gui_dark_mode_checkbox.value,
            mesh_mode="soma_skin",
            gui_use_soma_layer_checkbox=session.gui_elements.gui_use_soma_layer_checkbox,
        )

        if joints_pos is None:
            joints_pos = self.joints_pos
        if joints_rot is None:
            joints_rot = self.joints_rot
        motion = DemoLiteCharacterMotion(character, joints_pos, joints_rot, foot_contacts)
        session.motions[character_name] = motion
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

    def compute_model_constraints_lst(self, session: ClientSession, _model_bundle, num_frames: int):
        if not session.constraints:
            return []

        dense_smooth_root_pos_2d = None
        if session.constraints["2D Root"].dense_path:
            dense_smooth_root_pos_2d = session.constraints["2D Root"].get_constraint_info(device=self.device)[
                "root_pos"
            ][:, [0, 2]]

        model_constraints = []
        for track_name, constraint in session.constraints.items():
            constraint_info = constraint.get_constraint_info(device=self.device)
            frame_idx = constraint_info["frame_idx"]
            valid_info = [(i, fi) for i, fi in enumerate(frame_idx) if fi < num_frames]
            valid_idx = [i for i, _ in valid_info]
            valid_frame_idx = [fi for _, fi in valid_info]
            if not valid_frame_idx:
                continue

            frame_indices = torch.tensor(valid_frame_idx)
            if track_name == "2D Root":
                smooth_root_pos_2d = constraint_info["root_pos"][valid_idx][:, [0, 2]].to(self.device)
                model_constraints.append(Root2DConstraintSet(self.skeleton, frame_indices, smooth_root_pos_2d))
            elif track_name == "Full-Body":
                constraint_joints_pos = constraint_info["joints_pos"][valid_idx].to(self.device)
                constraint_joints_rot = constraint_info["joints_rot"][valid_idx].to(self.device)
                smooth_root_pos_2d = dense_smooth_root_pos_2d[frame_indices] if dense_smooth_root_pos_2d is not None else None
                model_constraints.append(
                    FullBodyConstraintSet(
                        self.skeleton,
                        frame_indices,
                        constraint_joints_pos,
                        constraint_joints_rot,
                        smooth_root_2d=smooth_root_pos_2d,
                    )
                )
            elif track_name == "End-Effectors":
                constraint_joints_pos = constraint_info["joints_pos"][valid_idx].to(self.device)
                constraint_joints_rot = constraint_info["joints_rot"][valid_idx].to(self.device)
                end_effector_type_set_lst = [
                    end_effector_type_set
                    for i, end_effector_type_set in enumerate(constraint_info["end_effector_type"])
                    if i in valid_idx
                ]
                cls_idx = {}
                for idx, end_effector_type_set in enumerate(end_effector_type_set_lst):
                    for end_effector_type in end_effector_type_set:
                        cls_idx.setdefault(TYPE_TO_CLASS[end_effector_type], []).append(idx)
                for cls, lst_idx in cls_idx.items():
                    frame_indices_cls = frame_indices[lst_idx]
                    smooth_root_pos_2d = (
                        dense_smooth_root_pos_2d[frame_indices_cls] if dense_smooth_root_pos_2d is not None else None
                    )
                    model_constraints.append(
                        cls(
                            self.skeleton,
                            frame_indices_cls,
                            constraint_joints_pos[lst_idx],
                            constraint_joints_rot[lst_idx],
                            smooth_root_2d=smooth_root_pos_2d,
                        )
                    )

        return model_constraints

    def on_client_connect(self, client: viser.ClientHandle) -> None:
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
        self.client_sessions[client.client_id] = session
        session.gui_elements.gui_viz_skeleton_checkbox.value = True
        self.add_character_motion(client, self.skeleton, self.joints_pos, self.joints_rot, self.foot_contacts)
        client.timeline.add_prompt("Loaded motion", 0, max_frame_idx, color=(100, 200, 150))
        client.timeline.set_current_frame(0)

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        self.client_sessions.pop(client.client_id, None)
        self.start_direction_markers.pop(client.client_id, None)
        self.grid_handles.pop(client.client_id, None)

    def run(self) -> None:
        update_counter = 0
        while True:
            for client_id, session in list(self.client_sessions.items()):
                update_interval = max(1, int(self.playback_fps / max(session.playback_speed * session.model_fps, 1e-6)))
                if session.playing and update_counter % update_interval == 0:
                    new_frame_idx = 0 if session.frame_idx >= session.max_frame_idx else session.frame_idx + 1
                    if self.client_active(client_id):
                        self.set_frame(client_id, new_frame_idx)
            update_counter = (update_counter + 1) % max(int(self.playback_fps), 1)
            threading.Event().wait(1.0 / max(self.playback_fps, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kimodo demo UI in playback-only mode.")
    parser.add_argument("motion", help="Path to a motion file (.npz or compatible .bvh).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    demo = DemoLite(Path(args.motion).expanduser().resolve(), host=args.host, port=args.port, fps=args.fps)
    demo.run()


if __name__ == "__main__":
    main()
