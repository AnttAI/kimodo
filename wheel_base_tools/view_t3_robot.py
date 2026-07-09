#!/usr/bin/env python3
"""View the T3 robot: T2 upper body mounted on the two-wheel base."""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wheel_base_tools.t2_csv_to_diff_drive import convert_t2_csv_to_diff_drive
from wheel_base_tools.view_two_wheel_base import _output_diff_drive_path, _scan_csvs
from wheel_base_tools.view_two_wheel_base_tara_send import _read_diff_drive_csv


DEFAULT_T3_URDF_PATH = REPO_ROOT / "robot_demo_outputs" / "t3_robot" / "T3.urdf"
DEFAULT_T2_CSV_ROOT = Path("/home/jony/Downloads/soma-retargeter/assets/motions/t2_csv")
DEFAULT_WHEEL_CSV_ROOT = REPO_ROOT / "robot_demo_outputs" / "wheel_base_robot"

WHEEL_Z_UP_TO_SCENE_Y_UP = Rotation.from_euler("x", -90.0, degrees=True)
T3_STIFF_POSTURE_JOINTS = {
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
}

_TARA_RIG_SPEC = importlib.util.spec_from_file_location("kimodo_tara_rig_light", REPO_ROOT / "kimodo" / "viz" / "tara_rig.py")
if _TARA_RIG_SPEC is None or _TARA_RIG_SPEC.loader is None:
    raise ImportError("Could not load kimodo/viz/tara_rig.py")
_tara_rig = importlib.util.module_from_spec(_TARA_RIG_SPEC)
sys.modules[_TARA_RIG_SPEC.name] = _tara_rig
_TARA_RIG_SPEC.loader.exec_module(_tara_rig)


def _default_label(options: dict[str, Path], preferred_name: str | None = None) -> str:
    if not options:
        return "<none>"
    if preferred_name is not None:
        for label, path in options.items():
            if path.name == preferred_name or label == preferred_name:
                return label
    return next(iter(options))


class T3Playback:
    def __init__(
        self,
        server: viser.ViserServer,
        urdf_path: Path,
        t2_csv_path: Path | None,
        wheel_csv_path: Path,
        fps: float,
        stiff_posture: bool,
        *,
        root_node_name: str = "/t3_robot",
        position_offset: np.ndarray | None = None,
        visual_yaw_offset_rad: float = 0.0,
        initial_start_pose: tuple[float, float, float] | None = None,
        mirror_wheel_z_axis: bool = False,
    ):
        self.server = server
        self.urdf_path = urdf_path
        self.fps = fps
        self.stiff_posture = stiff_posture
        self.position_offset = (
            np.zeros(3, dtype=np.float64)
            if position_offset is None
            else np.asarray(position_offset, dtype=np.float64)
        )
        self.visual_yaw_offset_rad = float(visual_yaw_offset_rad)
        self.mirror_wheel_z_axis = bool(mirror_wheel_z_axis)
        self.root_frame = server.scene.add_frame(root_node_name, show_axes=False)
        self.robot = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            root_node_name=self.root_frame.name,
            scale=1.0,
            load_meshes=True,
            load_collision_meshes=False,
        )
        self.joint_names = self.robot.get_actuated_joint_names()
        self.joint_index = {name: idx for idx, name in enumerate(self.joint_names)}
        self.cfg = np.zeros(len(self.joint_names), dtype=np.float64)
        self.t2_csv_path = t2_csv_path
        self.wheel_csv_path = wheel_csv_path
        self.t2_motion = (
            _tara_rig.load_tara_motion_csv(t2_csv_path)
            if t2_csv_path is not None
            else None
        )
        self.wheel_motion = _read_diff_drive_csv(wheel_csv_path, fps)
        self._apply_wheel_frame_transform()
        self.base_frozen = False
        self.frame_idx = 0
        if initial_start_pose is not None:
            self._align_wheel_motion(*initial_start_pose)
        self.apply_frame(0)

    @property
    def length(self) -> int:
        wheel_length = int(len(self.wheel_motion["x"]))
        if self.t2_motion is None:
            return wheel_length
        return min(int(self.t2_motion.length), wheel_length)

    def load_csvs(
        self,
        t2_csv_path: Path | None,
        wheel_csv_path: Path,
        fps: float,
        *,
        start_pose: tuple[float, float, float] | None = None,
        position_offset: np.ndarray | None = None,
        mirror_wheel_z_axis: bool | None = None,
    ) -> None:
        self.t2_csv_path = t2_csv_path
        self.wheel_csv_path = wheel_csv_path
        self.fps = fps
        if mirror_wheel_z_axis is not None:
            self.mirror_wheel_z_axis = bool(mirror_wheel_z_axis)
        if position_offset is not None:
            self.position_offset = np.asarray(position_offset, dtype=np.float64).copy()
        self.t2_motion = (
            _tara_rig.load_tara_motion_csv(t2_csv_path)
            if t2_csv_path is not None
            else None
        )
        self.wheel_motion = _read_diff_drive_csv(wheel_csv_path, fps)
        self._apply_wheel_frame_transform()
        if start_pose is not None:
            self._align_wheel_motion(*start_pose)
        self.apply_frame(0)

    def _apply_wheel_frame_transform(self) -> None:
        if not self.mirror_wheel_z_axis:
            return
        self.wheel_motion["z"] = -self.wheel_motion["z"]
        self.wheel_motion["yaw"] = -self.wheel_motion["yaw"]

    def _align_wheel_motion(self, x_m: float, z_m: float, yaw_rad: float) -> None:
        if self.length <= 0:
            return
        source_x = float(self.wheel_motion["x"][0])
        source_z = float(self.wheel_motion["z"][0])
        source_yaw = float(self.wheel_motion["yaw"][0])
        yaw_delta = float(yaw_rad) - source_yaw
        cos_yaw = math.cos(yaw_delta)
        sin_yaw = math.sin(yaw_delta)
        relative_x = self.wheel_motion["x"] - source_x
        relative_z = self.wheel_motion["z"] - source_z
        self.wheel_motion["x"] = float(x_m) + cos_yaw * relative_x - sin_yaw * relative_z
        self.wheel_motion["z"] = float(z_m) + sin_yaw * relative_x + cos_yaw * relative_z
        self.wheel_motion["yaw"] = self.wheel_motion["yaw"] + yaw_delta

    def align_start_pose(self, x_m: float, z_m: float, yaw_rad: float) -> None:
        """Rigidly align this T3 base motion to a preceding final pose."""
        self._align_wheel_motion(x_m, z_m, yaw_rad)
        self.apply_frame(0)

    def apply_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(int(frame_idx), self.length - 1))
        self.frame_idx = frame_idx
        base_frame_idx = 0 if self.base_frozen else frame_idx
        self.root_frame.position = np.array(
            [
                self.wheel_motion["x"][base_frame_idx] + self.position_offset[0],
                self.position_offset[1],
                self.wheel_motion["z"][base_frame_idx] + self.position_offset[2],
            ],
            dtype=np.float64,
        )
        display_yaw = self.wheel_motion["yaw"][base_frame_idx] + self.visual_yaw_offset_rad
        yaw_rot = Rotation.from_euler("y", -display_yaw)
        self.root_frame.wxyz = (yaw_rot * WHEEL_Z_UP_TO_SCENE_Y_UP).as_quat(scalar_first=True)

        if self.t2_motion is not None:
            t2_frame_idx = min(frame_idx, self.t2_motion.length - 1)
            for joint_name, values in self.t2_motion.joint_angles.items():
                if self.stiff_posture and joint_name in T3_STIFF_POSTURE_JOINTS:
                    continue
                joint_idx = self.joint_index.get(joint_name)
                if joint_idx is not None:
                    self.cfg[joint_idx] = float(values[t2_frame_idx])

        left_idx = self.joint_index.get("left_wheel_joint")
        right_idx = self.joint_index.get("right_wheel_joint")
        if left_idx is not None:
            self.cfg[left_idx] = math.remainder(self.wheel_motion["left_angle"][base_frame_idx], 2.0 * math.pi)
        if right_idx is not None:
            self.cfg[right_idx] = math.remainder(self.wheel_motion["right_angle"][base_frame_idx], 2.0 * math.pi)
        self.robot.update_cfg(self.cfg)

    def set_base_frozen(self, frozen: bool) -> None:
        self.base_frozen = bool(frozen)
        self.apply_frame(self.frame_idx)

    def set_visible(self, visible: bool) -> None:
        self.root_frame.visible = bool(visible)
        self.robot.show_visual = bool(visible)

    def clear(self) -> None:
        self.root_frame.remove()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch T3 playback from a T2 CSV.")
    parser.add_argument("--t3-urdf", type=Path, default=DEFAULT_T3_URDF_PATH, help="T3 URDF path.")
    parser.add_argument("--t2-csv-root", type=Path, default=DEFAULT_T2_CSV_ROOT, help="Folder containing T2 CSV files.")
    parser.add_argument("--wheel-csv-root", type=Path, default=DEFAULT_WHEEL_CSV_ROOT, help="Folder for generated wheel CSVs.")
    parser.add_argument("--fps", type=float, default=30.0, help="CSV frame rate in Hz. Default: 30.")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host. Default: 0.0.0.0.")
    parser.add_argument("--port", type=int, default=8092, help="Viser port. Default: 8092.")
    parser.add_argument(
        "--follow-t2-posture",
        action="store_true",
        help="Let T3 copy T2 waist/head joints from the CSV. By default T3 keeps these joints neutral and stiff.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t3_urdf_path = args.t3_urdf.expanduser().resolve()
    t2_csv_root = args.t2_csv_root.expanduser().resolve()
    wheel_csv_root = args.wheel_csv_root.expanduser().resolve()
    wheel_csv_root.mkdir(parents=True, exist_ok=True)

    if not t3_urdf_path.exists():
        raise FileNotFoundError(f"T3 URDF not found: {t3_urdf_path}. Run wheel_base_tools/create_t3_robot.py first.")

    t2_options = _scan_csvs(t2_csv_root)
    if not t2_options:
        raise FileNotFoundError(f"No T2 CSV files found under {t2_csv_root}")
    t2_label = _default_label(t2_options, "Neutral_walk_forward_002__A057.csv")
    t2_csv = t2_options[t2_label]
    wheel_csv = _output_diff_drive_path(t2_csv, wheel_csv_root)
    if not wheel_csv.exists():
        convert_t2_csv_to_diff_drive(
            t2_csv,
            wheel_csv,
            fps=args.fps,
            wheel_radius_m=0.10,
            wheel_separation_m=0.38,
            max_forward_speed=3.0,
            max_yaw_rate=12.0,
        )

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+y")
    server.scene.add_grid(
        "/ground",
        width=6.0,
        height=6.0,
        plane="xz",
        position=(0.0, 0.0, 0.0),
        cell_size=0.3,
        section_size=0.6,
    )

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = np.array([2.4, 1.7, 3.4], dtype=np.float64)
        client.camera.look_at = np.array([0.0, 0.85, 0.0], dtype=np.float64)
        client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        client.camera.fov = np.deg2rad(45.0)

    t3 = T3Playback(server, t3_urdf_path, t2_csv, wheel_csv, args.fps, stiff_posture=not args.follow_t2_posture)
    start_time = time.time()

    with server.gui.add_folder("T3 Robot"):
        t2_dropdown = server.gui.add_dropdown("T2 CSV", tuple(t2_options.keys()), initial_value=t2_label)
        play = server.gui.add_checkbox("Play T3", initial_value=False)
        loop = server.gui.add_checkbox("Loop T3", initial_value=True)
        freeze_base = server.gui.add_checkbox("Keep T3 Base Stationary", initial_value=False)
        speed = server.gui.add_slider("T3 speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        frame = server.gui.add_slider("T3 frame", min=0, max=t3.length - 1, step=1, initial_value=0)
        reset = server.gui.add_button("Reset T3")

    def reset_clock(frame_idx: int = 0) -> None:
        nonlocal start_time
        start_time = time.time() - (frame_idx / args.fps) / float(speed.value)

    @t2_dropdown.on_update
    def _(_) -> None:
        label = str(t2_dropdown.value)
        if label not in t2_options:
            return
        play.value = False
        t2_csv_now = t2_options[label]
        wheel_csv_now = _output_diff_drive_path(t2_csv_now, wheel_csv_root)
        if not wheel_csv_now.exists():
            convert_t2_csv_to_diff_drive(
                t2_csv_now,
                wheel_csv_now,
                fps=args.fps,
                wheel_radius_m=0.10,
                wheel_separation_m=0.38,
                max_forward_speed=3.0,
                max_yaw_rate=12.0,
            )
        t3.load_csvs(t2_csv_now, wheel_csv_now, args.fps)
        frame.max = t3.length - 1
        frame.value = 0
        reset_clock(0)

    @freeze_base.on_update
    def _(_) -> None:
        t3.set_base_frozen(bool(freeze_base.value))

    @frame.on_update
    def _(_) -> None:
        if not play.value:
            t3.apply_frame(int(frame.value))
            reset_clock(int(frame.value))

    @reset.on_click
    def _(_) -> None:
        play.value = False
        frame.value = 0
        t3.apply_frame(0)
        reset_clock(0)

    print(f"T3 viewer: http://{args.host}:{args.port}")
    print(f"T3 URDF: {t3_urdf_path}")
    while True:
        now = time.time()
        if play.value:
            frame_idx = int((now - start_time) * args.fps * float(speed.value))
            if frame_idx >= t3.length:
                if loop.value:
                    start_time = now
                    frame_idx = 0
                else:
                    frame_idx = t3.length - 1
                    play.value = False
            frame.value = frame_idx
            t3.apply_frame(frame_idx)
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
