# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Playback-only viewer for existing Kimodo motion files."""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import viser

from kimodo.motion_io import load_motion_file
from kimodo.scripts.t2_csv_arm_publisher import ArmFrame, load_arm_frames
from kimodo.skeleton import SkeletonBase
from kimodo.viz.playback import CharacterMotion
from kimodo.viz.scene import Character
from kimodo.viz.tara_rig import T2ViewerMotion


@dataclass
class LoadedMotion:
    path: Path
    motion: CharacterMotion


T2_ARM_COLUMNS = [f"{side}_joint{i}_dof" for side in ("right", "left") for i in range(1, 8)]


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
    t2_csv_dir = motions_root / "t2_csv"
    if not bvh_dir.is_dir() or not t2_csv_dir.is_dir():
        return []

    bvh_stems = {str(path.relative_to(bvh_dir).with_suffix("")) for path in bvh_dir.rglob("*.bvh")}
    t2_stems = {str(path.relative_to(t2_csv_dir).with_suffix("")) for path in t2_csv_dir.rglob("*.csv")}
    return sorted(bvh_stems & t2_stems)


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
    relative_stem = Path(stem)
    candidates = [
        motions_root / "bvh" / relative_stem.with_suffix(".bvh"),
        motions_root / "t2_csv" / relative_stem.with_suffix(".csv"),
        motions_root / "csv" / relative_stem.with_suffix(".csv"),
    ]
    return [path for path in candidates if path.exists()]


def _default_browser_stem(motions_root: Path, options: list[str]) -> str:
    for stem in options:
        if _motion_paths_for_stem(motions_root, stem)[1].exists():
            return stem
    return options[0]


def _is_t2_csv_path(path: Path) -> bool:
    return "t2_csv" in path.parts or _has_t2_arm_columns(path)


def _has_t2_arm_columns(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open(encoding="utf-8") as f:
            header = f.readline().strip().split(",")
    except OSError:
        return False
    return all(column in header for column in T2_ARM_COLUMNS)


def _resolve_robot_stream_script(script_path: str | None) -> Path:
    if script_path:
        resolved = Path(script_path).expanduser().resolve()
    else:
        kimodo_repo_root = Path(__file__).resolve().parents[2]
        resolved = kimodo_repo_root / "scripts" / "stream_t2_robot_sync.sh"
        if not resolved.is_file():
            workspace_root = Path(__file__).resolve().parents[3]
            resolved = workspace_root / "soma-retargeter" / "scripts" / "stream_t2_robot_sync.sh"
    if not resolved.is_file():
        raise FileNotFoundError(
            f"T2 robot stream script not found: {resolved}. "
            "Pass --robot-stream-script if the wrapper lives elsewhere."
        )
    return resolved


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
    parser.add_argument(
        "--t2-arms-only",
        action="store_true",
        help="For T2 CSV playback, apply only arm and gripper joints from CSV; keep root and other joints static.",
    )
    parser.add_argument(
        "--robot-stream-script",
        default=None,
        help="Path to scripts/stream_t2_robot_sync.sh used by the Connect Robot checkbox.",
    )
    args = parser.parse_args()

    motion_paths = [Path(p).expanduser().resolve() for p in args.motions]
    for path in motion_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    motions_root = Path(args.motions_root).expanduser().resolve() if args.motions_root else None
    robot_path = _resolve_robot_path(Path(args.robot_path)) if args.robot_path else None
    robot_stream_script = _resolve_robot_stream_script(args.robot_stream_script)
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
        speed_slider = server.gui.add_slider("Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0)
        connect_robot_checkbox = server.gui.add_checkbox("Connect Robot", initial_value=False)

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
        "frame_cursor": 0.0,
        "speed": 1.0,
        "updating_slider": False,
    }
    state_lock = threading.Lock()
    robot_state = {
        "process": None,
        "lock": threading.Lock(),
        "frames": [],
        "suppress_checkbox_callback": False,
    }

    def _refresh_loaded_markdown() -> None:
        if loaded:
            loaded_markdown.content = "\n".join(f"- `{item.path}`" for item in loaded)
        else:
            loaded_markdown.content = "No motions loaded."

    def _start_robot_stream(event_client: viser.ClientHandle) -> None:
        with robot_state["lock"]:
            active_process = robot_state["process"]
            has_frames = bool(robot_state["frames"])
        if not has_frames:
            _set_connect_robot_checkbox(False)
            event_client.add_notification(
                title="No T2 CSV loaded",
                body="Load a motion set with a T2 CSV before connecting the robot.",
                auto_close_seconds=5.0,
                color="red",
            )
            return
        if active_process is not None and active_process.poll() is None:
            _publish_robot_frame(frame_slider.value)
            return

        try:
            process = subprocess.Popen(
                [str(robot_stream_script)],
                cwd=str(robot_stream_script.parent.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            _set_connect_robot_checkbox(False)
            event_client.add_notification(
                title="Robot connection failed",
                body=str(exc),
                auto_close_seconds=8.0,
                color="red",
            )
            return

        with robot_state["lock"]:
            robot_state["process"] = process

        def _watch_robot_process() -> None:
            output_chunks: list[str] = []
            if process.stdout is not None:
                for line in process.stdout:
                    output_chunks.append(line.rstrip())
                    print(f"[ROBOT] {line.rstrip()}", flush=True)
            return_code = process.wait()
            with robot_state["lock"]:
                if robot_state["process"] is process:
                    robot_state["process"] = None
            if connect_robot_checkbox.value:
                _set_connect_robot_checkbox(False)
                output_text = "\n".join(output_chunks[-8:])
                event_client.add_notification(
                    title="Robot disconnected",
                    body=output_text[-900:] if output_text else f"Exit code {return_code}",
                    auto_close_seconds=8.0,
                    color="orange" if return_code < 0 else "red",
                )

        threading.Thread(target=_watch_robot_process, daemon=True).start()
        event_client.add_notification(
            title="Robot connecting",
            body="Visualizer frames will stream to the robot when ROS subscribers are ready.",
            auto_close_seconds=4.0,
            color="blue",
        )
        _publish_robot_frame(frame_slider.value)

    def _stop_robot_stream() -> None:
        with robot_state["lock"]:
            process = robot_state["process"]
            robot_state["process"] = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        process.terminate()

    def _set_connect_robot_checkbox(value: bool) -> None:
        with robot_state["lock"]:
            robot_state["suppress_checkbox_callback"] = True
        try:
            connect_robot_checkbox.value = value
        finally:
            with robot_state["lock"]:
                robot_state["suppress_checkbox_callback"] = False

    def _publish_robot_frame(frame_idx: int | float) -> None:
        with robot_state["lock"]:
            process = robot_state["process"]
            frames: list[ArmFrame] = robot_state["frames"]
        if process is None or process.poll() is not None or process.stdin is None or not frames:
            return
        frame = frames[max(0, min(int(frame_idx), len(frames) - 1))]
        payload = json.dumps(
            {
                "frame_index": frame.frame_index,
                "right": frame.right,
                "left": frame.left,
            },
            separators=(",", ":"),
        )
        try:
            process.stdin.write(payload + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            _set_connect_robot_checkbox(False)
            _stop_robot_stream()

    def _clear_loaded() -> None:
        nonlocal loaded, max_frame
        for item in loaded:
            item.motion.clear()
        loaded.clear()
        with robot_state["lock"]:
            robot_state["frames"] = []
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
            if _is_t2_csv_path(path):
                motion = T2ViewerMotion(
                    name=f"motion_{idx}",
                    server=server,
                    csv_path=path,
                    urdf_path=robot_path,
                    x_offset=x_offset,
                    arms_only=args.t2_arms_only,
                )
                if _has_t2_arm_columns(path):
                    with robot_state["lock"]:
                        robot_state["frames"] = load_arm_frames(path, start_frame=0, max_frames=None, frame_stride=1)
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

    def _set_frame(frame_idx: int, update_cursor: bool = True) -> None:
        frame_idx = max(0, min(max_frame, int(frame_idx)))
        for item in loaded:
            item.motion.set_frame(frame_idx)
        if connect_robot_checkbox.value:
            _publish_robot_frame(frame_idx)
        with state_lock:
            state["updating_slider"] = True
        frame_slider.value = frame_idx
        with state_lock:
            state["frame"] = frame_idx
            if update_cursor:
                state["frame_cursor"] = float(frame_idx)
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

    @connect_robot_checkbox.on_update
    def _(_event: viser.GuiEvent) -> None:
        with robot_state["lock"]:
            if robot_state["suppress_checkbox_callback"]:
                return
        if connect_robot_checkbox.value:
            if _event.client is None:
                return
            _start_robot_stream(_event.client)
        else:
            _stop_robot_stream()
            if _event.client is not None:
                _event.client.add_notification(
                    title="Robot disconnected",
                    body="Visualizer frame streaming is off.",
                    auto_close_seconds=3.0,
                    color="blue",
                )

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
            required_paths = [
                root / "bvh" / Path(stem).with_suffix(".bvh"),
                root / "t2_csv" / Path(stem).with_suffix(".csv"),
            ]
            if not required_paths[0].exists() or not required_paths[1].exists():
                _event.client.add_notification(
                    title="Motion set missing",
                    body=f"Expected both bvh/{stem}.bvh and t2_csv/{stem}.csv.",
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
                next_frame = state["frame_cursor"] + state["speed"]
            if next_frame > max_frame:
                next_frame = 0.0
            with state_lock:
                state["frame_cursor"] = next_frame
            _set_frame(int(next_frame), update_cursor=False)

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
