# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playback-only viewer for existing Kimodo motion files."""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser

from kimodo.motion_io import load_motion_file
from kimodo.skeleton import SkeletonBase
from kimodo.viz.playback import CharacterMotion
from kimodo.viz.scene import Character
from kimodo.viz.tara_rig import T2ViewerMotion


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

    def set_mesh_visibility(self, visible: bool) -> None:
        self.character.set_skinned_mesh_visibility(visible)

    def set_skeleton_visibility(self, visible: bool) -> None:
        self.character.set_skeleton_visibility(visible)

    def set_show_foot_contacts(self, show: bool) -> None:
        self.character.set_show_foot_contacts(show)

    def set_mesh_opacity(self, opacity: float) -> None:
        self.character.set_skinned_mesh_opacity(opacity)


def _infer_mesh_mode(skeleton: SkeletonBase) -> str:
    if skeleton.nbjoints == 34:
        return "g1_stl"
    if skeleton.nbjoints == 22:
        return "smplx_skin"
    if skeleton.nbjoints in (30, 77):
        return "soma_skin"
    raise ValueError(f"Unsupported skeleton with {skeleton.nbjoints} joints")


def _configure_client_scene(client: viser.ClientHandle) -> None:
    client.camera.position = np.array([2.75, 1.9, 7.7], dtype=np.float64)
    client.camera.look_at = np.array([0.0, 0.9, 0.0], dtype=np.float64)
    client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    client.camera.fov = np.deg2rad(45.0)


def _scan_motion_pairs(motions_root: Path) -> list[str]:
    bvh_dir = motions_root / "bvh"
    csv_dir = motions_root / "csv"
    t2_csv_dir = motions_root / "t2_csv"
    if not bvh_dir.is_dir() or not csv_dir.is_dir() or not t2_csv_dir.is_dir():
        return []

    bvh_stems = {path.stem for path in bvh_dir.glob("*.bvh")}
    csv_stems = {path.stem for path in csv_dir.glob("*.csv") if not path.name.endswith(".generated.csv")}
    t2_stems = {path.stem for path in t2_csv_dir.glob("*.csv")}
    return sorted(bvh_stems & csv_stems & t2_stems)


def _resolve_robot_path(robot_path: Path | None) -> Path | None:
    if robot_path is None:
        return None
    robot_path = robot_path.expanduser().resolve()
    if robot_path.is_dir():
        urdf_candidates = sorted(robot_path.glob("*.urdf"))
        if len(urdf_candidates) != 1:
            raise FileNotFoundError(
                f"{robot_path}: expected exactly one .urdf file for --robot-path, found {len(urdf_candidates)}"
            )
        return urdf_candidates[0]
    if robot_path.suffix.lower() != ".urdf":
        raise ValueError(f"{robot_path}: --robot-path must point to a .urdf file or a folder containing one.")
    if not robot_path.exists():
        raise FileNotFoundError(robot_path)
    return robot_path


def _motion_paths_for_stem(motions_root: Path, stem: str) -> list[Path]:
    return [
        motions_root / "bvh" / f"{stem}.bvh",
        motions_root / "t2_csv" / f"{stem}.csv",
        motions_root / "csv" / f"{stem}.csv",
    ]


def _default_browser_stem(motions_root: Path, options: list[str]) -> str:
    for stem in options:
        if (motions_root / "t2_csv" / f"{stem}.csv").exists():
            return stem
    return options[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="View existing Kimodo motions in the forked Viser UI.")
    parser.add_argument("motions", nargs="*", help="One or more motion files (.npz, compatible .bvh, or G1 .csv).")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the viewer server to.")
    parser.add_argument("--port", type=int, default=7860, help="Port for the viewer server.")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS used by the viewer controls.")
    parser.add_argument("--device", default="cpu", help="Torch device for loading motion tensors.")
    parser.add_argument("--spread", type=float, default=1.5, help="X offset in meters between loaded motions.")
    parser.add_argument(
        "--motions-root",
        default=None,
        help="Base folder containing matching bvh/, t2_csv/, and csv/ subfolders for UI selection.",
    )
    parser.add_argument(
        "--robot-path",
        default=None,
        help="Robot URDF file, or a folder containing exactly one robot URDF.",
    )
    args = parser.parse_args()

    motion_paths = [Path(p).expanduser().resolve() for p in args.motions]
    for path in motion_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    motions_root = Path(args.motions_root).expanduser().resolve() if args.motions_root else None
    robot_path = _resolve_robot_path(Path(args.robot_path)) if args.robot_path else None
    if not motion_paths and motions_root is None:
        raise ValueError("Provide motion paths or --motions-root.")

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

    server.on_client_connect(_configure_client_scene)

    with server.gui.add_folder("Playback"):
        play_button = server.gui.add_button("Play")
        frame_slider = server.gui.add_slider("Frame", min=0, max=0, step=1, initial_value=0)
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)

    with server.gui.add_folder("Display"):
        mesh_checkbox = server.gui.add_checkbox("Show Mesh", initial_value=True)
        skeleton_checkbox = server.gui.add_checkbox("Show Skeleton", initial_value=True)
        contacts_checkbox = server.gui.add_checkbox("Show Foot Contacts", initial_value=True)
        opacity_slider = server.gui.add_slider("Mesh Opacity", min=0.05, max=1.0, step=0.05, initial_value=1.0)

    with server.gui.add_folder("Loaded Motions"):
        loaded_markdown = server.gui.add_markdown("No motions loaded.")

    browser_dropdown = None
    browser_path_text = None
    if motions_root is not None:
        with server.gui.add_folder("Motion Browser"):
            browser_path_text = server.gui.add_text("Base Folder", initial_value=str(motions_root))
            pair_options = _scan_motion_pairs(motions_root)
            browser_dropdown = server.gui.add_dropdown(
                "Motion Clip",
                options=pair_options if pair_options else ["<none>"],
                initial_value=_default_browser_stem(motions_root, pair_options) if pair_options else "<none>",
            )
            refresh_button = server.gui.add_button("Refresh")
            load_pair_button = server.gui.add_button("Load Motion Set")

    state = {
        "playing": False,
        "frame": 0,
        "speed": 1.0,
        "updating_slider": False,
    }
    state_lock = threading.Lock()

    def _refresh_loaded_markdown() -> None:
        if loaded:
            loaded_markdown.content = "\n".join(f"- `{item.path}`" for item in loaded)
        else:
            loaded_markdown.content = "No motions loaded."

    def _clear_loaded() -> None:
        nonlocal loaded, max_frame
        for item in loaded:
            item.motion.clear()
        loaded.clear()
        max_frame = 0
        frame_slider.max = 0
        frame_slider.value = 0
        with state_lock:
            state["frame"] = 0
            state["playing"] = False
        play_button.label = "Play"
        _refresh_loaded_markdown()

    def _load_paths(paths: list[Path]) -> None:
        nonlocal loaded, max_frame
        _clear_loaded()
        num_motions = len(paths)
        for idx, path in enumerate(paths):
            print(f"Loading motion: {path}", flush=True)
            x_offset = (idx - (num_motions - 1) / 2.0) * args.spread if args.spread != 0.0 else 0.0
            if path.parent.name == "t2_csv":
                motion = T2ViewerMotion(
                    name=f"motion_{idx}",
                    server=server,
                    csv_path=path,
                    urdf_path=robot_path,
                    x_offset=x_offset,
                )
            else:
                joints_pos, joints_rot, foot_contacts, skeleton = load_motion_file(path, args.device)
                if x_offset != 0.0:
                    joints_pos = joints_pos.clone()
                    joints_pos[..., 0] += x_offset
                mesh_mode = _infer_mesh_mode(skeleton)
                character = Character(
                    name=f"motion_{idx}",
                    server=server,
                    skeleton=skeleton,
                    create_skeleton_mesh=True,
                    create_skinned_mesh=True,
                    visible_skeleton=skeleton_checkbox.value,
                    visible_skinned_mesh=mesh_checkbox.value,
                    skinned_mesh_opacity=opacity_slider.value,
                    show_foot_contacts=contacts_checkbox.value,
                    dark_mode=False,
                    mesh_mode=mesh_mode,
                )
                motion = ViewerCharacterMotion(character, joints_pos, joints_rot, foot_contacts)
            motion.set_frame(0)
            loaded.append(LoadedMotion(path=path, motion=motion))
            max_frame = max(max_frame, motion.length - 1)
            print(f"Loaded {path.name}: {motion.length} frames", flush=True)

        frame_slider.max = max_frame
        _set_frame(0)
        _refresh_loaded_markdown()

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
            item.motion.set_mesh_visibility(mesh_checkbox.value)

    @skeleton_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.set_skeleton_visibility(skeleton_checkbox.value)

    @contacts_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.set_show_foot_contacts(contacts_checkbox.value)
            item.motion.set_frame(frame_slider.value)

    @opacity_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        for item in loaded:
            item.motion.set_mesh_opacity(opacity_slider.value)

    if motions_root is not None and browser_dropdown is not None and browser_path_text is not None:

        def _refresh_browser_options() -> list[str]:
            root = Path(browser_path_text.value).expanduser().resolve()
            options = _scan_motion_pairs(root)
            browser_dropdown.options = options if options else ["<none>"]
            browser_dropdown.value = _default_browser_stem(root, options) if options else "<none>"
            return options

        @refresh_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            options = _refresh_browser_options()
            _event.client.add_notification(
                title="Motion browser refreshed",
                body=f"Found {len(options)} paired motions.",
                auto_close_seconds=3.0,
                color="blue",
            )

        @load_pair_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            root = Path(browser_path_text.value).expanduser().resolve()
            stem = str(browser_dropdown.value)
            if stem == "<none>":
                _event.client.add_notification(
                    title="No motion selected",
                    body="Refresh the browser or choose a valid clip.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            motion_set_paths = _motion_paths_for_stem(root, stem)
            if not motion_set_paths[0].exists() or not motion_set_paths[1].exists():
                _event.client.add_notification(
                    title="Motion set missing",
                    body=f"Expected both {stem}.bvh and {stem}.csv.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            _load_paths(motion_set_paths)
            _event.client.add_notification(
                title="Motion set loaded",
                body=f"{stem} ({len(motion_set_paths)} streams)",
                auto_close_seconds=3.0,
                color="green",
            )

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

    if motion_paths:
        _load_paths(motion_paths)
    elif motions_root is not None and browser_dropdown is not None:
        options = _scan_motion_pairs(motions_root)
        if options:
            _load_paths(_motion_paths_for_stem(motions_root, _default_browser_stem(motions_root, options)))

    print(f"Kimodo viewer running on http://{args.host}:{args.port}", flush=True)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
