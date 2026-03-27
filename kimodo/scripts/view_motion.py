# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playback-only viewer for existing Kimodo motion files."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import viser

from kimodo.skeleton import SkeletonBase
from kimodo.skeleton.registry import build_skeleton
from kimodo.viz.playback import CharacterMotion
from kimodo.viz.scene import Character


@dataclass
class LoadedMotion:
    path: Path
    motion: CharacterMotion


class ViewerCharacterMotion(CharacterMotion):
    """Lighter playback class that avoids precomputing all mesh frames at startup."""

    def precompute_mesh_info(self):
        if self.character.skeleton_mesh is not None:
            self.character.skeleton_mesh.mesh_info_cache = None
        if self.character.skinned_mesh is not None:
            self.character.skinned_verts_cache = None


def _infer_mesh_mode(skeleton: SkeletonBase) -> str:
    if skeleton.nbjoints == 34:
        return "g1_stl"
    if skeleton.nbjoints == 22:
        return "smplx_skin"
    if skeleton.nbjoints in (30, 77):
        return "soma_skin"
    raise ValueError(f"Unsupported skeleton with {skeleton.nbjoints} joints")


def _load_motion_npz(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SkeletonBase]:
    with np.load(path, allow_pickle=False) as data:
        if "joints_pos" in data:
            joints_pos = torch.from_numpy(data["joints_pos"]).to(device)
        elif "posed_joints" in data:
            joints_pos = torch.from_numpy(data["posed_joints"]).to(device)
        else:
            raise ValueError(f"{path}: missing 'joints_pos' or 'posed_joints'")

        if "joints_rot" in data:
            joints_rot = torch.from_numpy(data["joints_rot"]).to(device)
        elif "global_rot_mats" in data:
            joints_rot = torch.from_numpy(data["global_rot_mats"]).to(device)
        else:
            raise ValueError(f"{path}: missing 'joints_rot' or 'global_rot_mats'")

        foot_contacts = torch.from_numpy(data["foot_contacts"]).to(device) if "foot_contacts" in data else None

    if joints_pos.ndim == 4:
        joints_pos = joints_pos[0]
    if joints_rot.ndim == 5:
        joints_rot = joints_rot[0]
    if foot_contacts is not None and foot_contacts.ndim == 3:
        foot_contacts = foot_contacts[0]

    if joints_pos.ndim != 3 or joints_rot.ndim != 4:
        raise ValueError(f"{path}: unexpected tensor shapes for motion data")

    skeleton = build_skeleton(joints_pos.shape[1])
    return joints_pos, joints_rot, foot_contacts, skeleton


def _configure_client_scene(client: viser.ClientHandle) -> None:
    client.camera.position = np.array([2.75, 1.9, 7.7], dtype=np.float64)
    client.camera.look_at = np.array([0.0, 0.9, 0.0], dtype=np.float64)
    client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    client.camera.fov = np.deg2rad(45.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="View existing Kimodo motions in the forked Viser UI.")
    parser.add_argument("motions", nargs="+", help="One or more .npz motion files to load.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the viewer server to.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the viewer server.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS used by the viewer controls.")
    parser.add_argument("--device", default="cpu", help="Torch device for loading motion tensors.")
    args = parser.parse_args()

    motion_paths = [Path(p).expanduser().resolve() for p in args.motions]
    for path in motion_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    server = viser.ViserServer(host=args.host, port=args.port, label="Kimodo Viewer")
    print(f"Viewer server initialized on http://{args.host}:{args.port}", flush=True)
    server.scene.world_axes.visible = False
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/grid",
        width=20.0,
        height=20.0,
        wxyz=viser.transforms.SO3.from_x_radians(-np.pi / 2.0).wxyz,
        position=(0.0, 0.0001, 0.0),
        fade_distance=60.0,
        section_color=(180, 180, 180),
        infinite_grid=True,
    )

    loaded: list[LoadedMotion] = []
    max_frame = 0
    for idx, path in enumerate(motion_paths):
        print(f"Loading motion: {path}", flush=True)
        joints_pos, joints_rot, foot_contacts, skeleton = _load_motion_npz(path, args.device)
        character = Character(
            name=f"motion_{idx}",
            server=server,
            skeleton=skeleton,
            create_skeleton_mesh=True,
            create_skinned_mesh=False,
            visible_skeleton=True,
            visible_skinned_mesh=False,
            skinned_mesh_opacity=1.0,
            show_foot_contacts=True,
            dark_mode=False,
            mesh_mode=_infer_mesh_mode(skeleton),
        )
        motion = ViewerCharacterMotion(character, joints_pos, joints_rot, foot_contacts)
        motion.set_frame(0)
        loaded.append(LoadedMotion(path=path, motion=motion))
        max_frame = max(max_frame, motion.length - 1)
        print(f"Loaded {path.name}: {motion.length} frames", flush=True)

    server.on_client_connect(_configure_client_scene)

    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Play")
        frame_slider = server.gui.add_slider("Frame", min=0, max=max_frame, step=1, initial_value=0)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    with server.gui.add_folder("Display"):
        mesh_checkbox = server.gui.add_checkbox("Show Mesh", initial_value=True)
        skeleton_checkbox = server.gui.add_checkbox("Show Skeleton", initial_value=False)
        contacts_checkbox = server.gui.add_checkbox("Show Foot Contacts", initial_value=True)
        opacity_slider = server.gui.add_slider("Mesh Opacity", min=0.05, max=1.0, step=0.05, initial_value=1.0)

    with server.gui.add_folder("Loaded Motions"):
        server.gui.add_markdown("\n".join(f"- `{item.path}`" for item in loaded))

    state = {
        "playing": False,
        "frame": 0,
        "speed": 1.0,
        "updating_slider": False,
    }
    state_lock = threading.Lock()

    def _set_frame(frame_idx: int) -> None:
        frame_idx = max(0, min(max_frame, int(frame_idx)))
        for item in loaded:
            item.motion.set_frame(frame_idx)
        with state_lock:
            state["updating_slider"] = True
        frame_slider.value = frame_idx
        with state_lock:
            state["frame"] = frame_idx
            state["updating_slider"] = False

    @play_button.on_click
    def _(_event: viser.GuiEvent) -> None:
        with state_lock:
            state["playing"] = not state["playing"]
            playing = state["playing"]
        play_button.label = "Pause" if playing else "Play"

    @frame_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        with state_lock:
            if state["updating_slider"]:
                return
        _set_frame(frame_slider.value)

    @speed_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        with state_lock:
            state["speed"] = float(speed_slider.value)

    @mesh_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.character.set_skinned_mesh_visibility(mesh_checkbox.value)

    @skeleton_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.character.set_skeleton_visibility(skeleton_checkbox.value)

    @contacts_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.character.set_show_foot_contacts(contacts_checkbox.value)
            item.motion.set_frame(frame_slider.value)

    @opacity_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.character.set_skinned_mesh_opacity(opacity_slider.value)

    def _playback_loop() -> None:
        while True:
            time.sleep(max(1.0 / args.fps, 0.001))
            with state_lock:
                if not state["playing"]:
                    continue
                next_frame = state["frame"] + state["speed"]
            if next_frame > max_frame:
                next_frame = 0
            _set_frame(int(next_frame))

    threading.Thread(target=_playback_loop, daemon=True).start()

    print(f"Kimodo viewer running on http://{args.host}:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
