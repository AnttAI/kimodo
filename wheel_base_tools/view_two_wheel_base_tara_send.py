#!/usr/bin/env python3
"""View T2 and the wheel base, with a button to stream CSV RPMs to TaraBase."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import viser
from viser.extras import ViserUrdf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wheel_base_tools.t2_csv_to_diff_drive import convert_t2_csv_to_diff_drive
from wheel_base_tools.send_diff_drive_csv_to_tara import (
    estimate_travel_distance_m,
    load_wheel_commands,
    stream_wheel_commands,
)

DEFAULT_WHEEL_URDF_PATH = REPO_ROOT / "robot_demo_outputs" / "wheel_base_robot" / "two_wheel_base.urdf"
DEFAULT_WHEEL_CSV_ROOT = REPO_ROOT / "robot_demo_outputs" / "wheel_base_robot"
DEFAULT_T2_CSV_ROOT = Path("/home/jony/Downloads/soma-retargeter/assets/motions/t2_csv")
DEFAULT_T2_URDF_PATH = Path("/home/jony/Downloads/soma-retargeter/antt_t2/T2_serial_nero_arms.urdf")
DEFAULT_WHEEL_RADIUS_M = 0.10
DEFAULT_WHEEL_SEPARATION_M = 0.38

WHEEL_SCENE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
T2_SCENE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WHEEL_Z_UP_TO_SCENE_Y_UP = Rotation.from_euler("x", -90.0, degrees=True)
T2_YAW_CORRECTION = Rotation.from_euler("y", 180.0, degrees=True).as_matrix()

_TARA_RIG_SPEC = importlib.util.spec_from_file_location("kimodo_tara_rig_light", REPO_ROOT / "kimodo" / "viz" / "tara_rig.py")
if _TARA_RIG_SPEC is None or _TARA_RIG_SPEC.loader is None:
    raise ImportError("Could not load kimodo/viz/tara_rig.py")
_tara_rig = importlib.util.module_from_spec(_TARA_RIG_SPEC)
sys.modules[_TARA_RIG_SPEC.name] = _tara_rig
_TARA_RIG_SPEC.loader.exec_module(_tara_rig)
T2ViewerMotion = _tara_rig.T2ViewerMotion


def _relative_label(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _scan_csvs(root: Path, suffix: str | tuple[str, ...] | None = None) -> dict[str, Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        return {}
    paths = sorted(root.rglob("*.csv"))
    if suffix is not None:
        paths = [path for path in paths if path.name.endswith(suffix)]
    return {_relative_label(path, root): path for path in paths}


def _default_label(options: dict[str, Path], preferred_name: str | None = None) -> str:
    if not options:
        return "<none>"
    if preferred_name is not None:
        for label, path in options.items():
            if path.name == preferred_name or label == preferred_name:
                return label
    return next(iter(options))


def _output_diff_drive_path(t2_csv: Path, wheel_csv_root: Path) -> Path:
    return wheel_csv_root / f"{t2_csv.stem}_diff_drive.csv"


def _read_diff_drive_csv(
    path: Path,
    fps: float,
    wheel_radius_m: float = DEFAULT_WHEEL_RADIUS_M,
    wheel_separation_m: float = DEFAULT_WHEEL_SEPARATION_M,
) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        header = reader.fieldnames
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} has no motion rows")

    if wheel_radius_m <= 0:
        raise ValueError(f"wheel radius must be positive, got {wheel_radius_m}")
    if wheel_separation_m <= 0:
        raise ValueError(f"wheel separation must be positive, got {wheel_separation_m}")

    def column(name: str) -> np.ndarray:
        try:
            return np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} has invalid numeric values in column {name!r}") from exc

    rpm_columns: tuple[str, str] | None = None
    if "left_motor_rpm" in header and "right_motor_rpm" in header:
        rpm_columns = ("left_motor_rpm", "right_motor_rpm")
    elif "left_wheel_rpm" in header and "right_wheel_rpm" in header:
        rpm_columns = ("left_wheel_rpm", "right_wheel_rpm")

    visual_velocity_columns = (
        "forward_velocity_m_s",
        "yaw_rate_rad_s",
        "left_wheel_rad_s",
        "right_wheel_rad_s",
    )
    if all(name in header for name in visual_velocity_columns):
        # Prompt-generated CSVs intentionally separate logical visualization
        # velocities from hardware-specific motor RPM polarity.
        forward_velocity_m_s = column("forward_velocity_m_s")
        yaw_rate_rad_s = column("yaw_rate_rad_s")
        left_rad_s = column("left_wheel_rad_s")
        right_rad_s = column("right_wheel_rad_s")
    elif rpm_columns is not None:
        rpm_to_rad_s = 2.0 * math.pi / 60.0
        left_rad_s = column(rpm_columns[0]) * rpm_to_rad_s
        right_rad_s = column(rpm_columns[1]) * rpm_to_rad_s
        left_linear_m_s = left_rad_s * wheel_radius_m
        right_linear_m_s = right_rad_s * wheel_radius_m
        forward_velocity_m_s = 0.5 * (left_linear_m_s + right_linear_m_s)
        yaw_rate_rad_s = (right_linear_m_s - left_linear_m_s) / wheel_separation_m
    else:
        required = visual_velocity_columns
        missing = [name for name in required if name not in header]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s): {', '.join(missing)}. "
                "Expected RPM columns left_motor_rpm/right_motor_rpm or the legacy velocity columns."
            )
        forward_velocity_m_s = column("forward_velocity_m_s")
        yaw_rate_rad_s = column("yaw_rate_rad_s")
        left_rad_s = column("left_wheel_rad_s")
        right_rad_s = column("right_wheel_rad_s")

    columns = {
        "forward_velocity_m_s": forward_velocity_m_s,
        "yaw_rate_rad_s": yaw_rate_rad_s,
        "left_wheel_rad_s": left_rad_s,
        "right_wheel_rad_s": right_rad_s,
    }
    root_yaw = column("root_yaw_rad") if "root_yaw_rad" in header else None
    dt = 1.0 / fps
    n = len(rows)

    x = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    if "root_x_m" in header and "root_y_m" in header:
        x = column("root_x_m")
        z = column("root_y_m")
    yaw = np.zeros(n, dtype=np.float64)
    if root_yaw is not None:
        yaw[0] = root_yaw[0]
    left_angle = np.zeros(n, dtype=np.float64)
    right_angle = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        v = forward_velocity_m_s[i - 1]
        omega = yaw_rate_rad_s[i - 1]
        yaw_mid = yaw[i - 1] + 0.5 * omega * dt
        if "root_x_m" not in header or "root_y_m" not in header:
            x[i] = x[i - 1] + v * math.cos(yaw_mid) * dt
            z[i] = z[i - 1] + v * math.sin(yaw_mid) * dt
        yaw[i] = root_yaw[i] if root_yaw is not None else yaw[i - 1] + omega * dt
        left_angle[i] = left_angle[i - 1] + left_rad_s[i - 1] * dt
        right_angle[i] = right_angle[i - 1] + right_rad_s[i - 1] * dt

    return {
        "x": x,
        "z": z,
        "yaw": yaw,
        "left_angle": left_angle,
        "right_angle": right_angle,
        "duration": np.arange(n, dtype=np.float64) * dt,
        **columns,
    }


class WheelBasePlayback:
    def __init__(
        self,
        server: viser.ViserServer | viser.ClientHandle,
        urdf_path: Path,
        csv_path: Path | None,
        fps: float,
        *,
        root_node_name: str = "/wheel_base",
        position_offset: np.ndarray | None = None,
        visual_yaw_offset_rad: float = 0.0,
    ):
        self.server = server
        self.urdf_path = urdf_path
        self.fps = fps
        self.position_offset = np.zeros(3, dtype=np.float64) if position_offset is None else np.asarray(position_offset)
        self.visual_yaw_offset_rad = float(visual_yaw_offset_rad)
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
        self.cfg = np.zeros(len(self.joint_names), dtype=np.float64)
        self.joint_index = {name: idx for idx, name in enumerate(self.joint_names)}
        self.csv_path = csv_path
        self.motion = self._stationary_motion() if csv_path is None else _read_diff_drive_csv(csv_path, fps)
        self.frame_idx = 0
        self.apply_frame(0)

    @property
    def length(self) -> int:
        return int(len(self.motion["x"]))

    def load_csv(self, csv_path: Path, fps: float) -> None:
        self.csv_path = csv_path
        self.fps = fps
        self.motion = _read_diff_drive_csv(csv_path, fps)
        self.apply_frame(0)

    def align_start_pose(self, x_m: float, z_m: float, yaw_rad: float) -> None:
        """Rigidly align this motion's first frame to a preceding final pose."""
        if self.length <= 0:
            return
        source_x = float(self.motion["x"][0])
        source_z = float(self.motion["z"][0])
        source_yaw = float(self.motion["yaw"][0])
        yaw_delta = float(yaw_rad) - source_yaw
        cos_yaw = math.cos(yaw_delta)
        sin_yaw = math.sin(yaw_delta)
        relative_x = self.motion["x"] - source_x
        relative_z = self.motion["z"] - source_z
        self.motion["x"] = float(x_m) + cos_yaw * relative_x - sin_yaw * relative_z
        self.motion["z"] = float(z_m) + sin_yaw * relative_x + cos_yaw * relative_z
        self.motion["yaw"] = self.motion["yaw"] + yaw_delta
        self.apply_frame(0)

    @staticmethod
    def _stationary_motion() -> dict[str, np.ndarray]:
        zero = np.zeros(1, dtype=np.float64)
        return {
            "x": zero.copy(),
            "z": zero.copy(),
            "yaw": zero.copy(),
            "left_angle": zero.copy(),
            "right_angle": zero.copy(),
            "duration": zero.copy(),
            "forward_velocity_m_s": zero.copy(),
            "yaw_rate_rad_s": zero.copy(),
            "left_wheel_rad_s": zero.copy(),
            "right_wheel_rad_s": zero.copy(),
        }

    def set_stationary(self) -> None:
        self.csv_path = None
        self.motion = self._stationary_motion()
        self.apply_frame(0)

    def apply_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(int(frame_idx), self.length - 1))
        self.frame_idx = frame_idx
        self.root_frame.position = np.array(
            [
                self.motion["x"][frame_idx] + WHEEL_SCENE_OFFSET[0] + self.position_offset[0],
                WHEEL_SCENE_OFFSET[1] + self.position_offset[1],
                self.motion["z"][frame_idx] + WHEEL_SCENE_OFFSET[2] + self.position_offset[2],
            ],
            dtype=np.float64,
        )
        # Stored diff-drive headings use math convention in the viewer ground
        # plane: atan2(+Z, +X). In a right-handed +Y-up scene, positive
        # Rotation.from_euler("y", theta) turns +X toward -Z, so the visual
        # root needs the opposite sign to face along the CSV path tangent.
        display_yaw = self.motion["yaw"][frame_idx] + self.visual_yaw_offset_rad
        yaw_rot = Rotation.from_euler("y", -display_yaw)
        self.root_frame.wxyz = (yaw_rot * WHEEL_Z_UP_TO_SCENE_Y_UP).as_quat(scalar_first=True)

        left_idx = self.joint_index.get("left_wheel_joint")
        right_idx = self.joint_index.get("right_wheel_joint")
        if left_idx is not None:
            self.cfg[left_idx] = math.remainder(self.motion["left_angle"][frame_idx], 2.0 * math.pi)
        if right_idx is not None:
            self.cfg[right_idx] = math.remainder(self.motion["right_angle"][frame_idx], 2.0 * math.pi)
        self.robot.update_cfg(self.cfg)

    def set_visible(self, visible: bool) -> None:
        self.root_frame.visible = bool(visible)
        self.robot.show_visual = bool(visible)

    def clear(self) -> None:
        self.root_frame.remove()


class T2Playback:
    def __init__(self, server: viser.ViserServer, urdf_path: Path, csv_path: Path):
        self.server = server
        self.urdf_path = urdf_path
        self.viewer = T2ViewerMotion(
            name="t2_robot",
            server=server,
            csv_path=csv_path,
            urdf_path=urdf_path,
            position_offset=T2_SCENE_OFFSET,
        )
        self.csv_path = csv_path
        self.frame_idx = 0
        self.viewer.set_frame(0)

    @property
    def length(self) -> int:
        return int(self.viewer.length)

    def load_csv(self, csv_path: Path) -> None:
        self.viewer.clear()
        self.viewer = T2ViewerMotion(
            name="t2_robot",
            server=self.server,
            csv_path=csv_path,
            urdf_path=self.urdf_path,
            position_offset=T2_SCENE_OFFSET,
        )
        self.csv_path = csv_path
        self.apply_frame(0)

    def apply_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(int(frame_idx), self.length - 1))
        self.frame_idx = frame_idx
        joint_angles = {joint_name: values[frame_idx] for joint_name, values in self.viewer.motion.joint_angles.items()}
        self.viewer.rig.set_pose(
            root_pos=T2_YAW_CORRECTION @ self.viewer.motion.root_positions[frame_idx],
            root_rot=T2_YAW_CORRECTION @ self.viewer.motion.root_rotations[frame_idx],
            joint_angles=joint_angles,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch combined T2 and wheel-base CSV playback UI.")
    parser.add_argument("--wheel-urdf", type=Path, default=DEFAULT_WHEEL_URDF_PATH, help="Wheel-base URDF path.")
    parser.add_argument("--t2-urdf", type=Path, default=DEFAULT_T2_URDF_PATH, help="T2 URDF path.")
    parser.add_argument("--t2-csv-root", type=Path, default=DEFAULT_T2_CSV_ROOT, help="Folder containing T2 CSV files.")
    parser.add_argument(
        "--wheel-csv-root",
        type=Path,
        default=DEFAULT_WHEEL_CSV_ROOT,
        help="Folder containing/generated diff-drive CSV files.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="CSV frame rate in Hz. Default: 30.")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host. Default: 0.0.0.0.")
    parser.add_argument("--port", type=int, default=8091, help="Viser port. Default: 8091.")
    parser.add_argument("--tara-port", default="/dev/ttyUSB0", help="TaraBase USB serial port. Default: /dev/ttyUSB0.")
    parser.add_argument("--tara-slave-id", type=int, default=1, help="TaraBase MODBUS slave id. Default: 1.")
    parser.add_argument("--tara-baudrate", type=int, default=115200, help="TaraBase serial baudrate. Default: 115200.")
    parser.add_argument("--tara-max-rpm", type=float, default=30.0, help="Safety clamp for sent commands. Default: +/-50 RPM.")
    parser.add_argument("--tara-rpm-scale", type=float, default=1.0, help="Calibration multiplier for sent RPM commands. Default: 1.")
    parser.add_argument("--tara-debug", action="store_true", help="Enable TaraBase minimalmodbus debug logs while sending.")
    parser.add_argument(
        "--tara-remote-url",
        help=(
            "Raspberry Pi Tara remote URL. Prefer ws://192.168.1.50:8094; "
            "HTTP URLs such as http://192.168.1.50:8093 remain supported."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    wheel_urdf_path = args.wheel_urdf.expanduser().resolve()
    t2_urdf_path = args.t2_urdf.expanduser().resolve()
    t2_csv_root = args.t2_csv_root.expanduser().resolve()
    wheel_csv_root = args.wheel_csv_root.expanduser().resolve()
    wheel_csv_root.mkdir(parents=True, exist_ok=True)

    if not wheel_urdf_path.exists():
        raise FileNotFoundError(f"Wheel URDF not found: {wheel_urdf_path}")
    if not t2_urdf_path.exists():
        raise FileNotFoundError(f"T2 URDF not found: {t2_urdf_path}")

    t2_options = _scan_csvs(t2_csv_root)
    wheel_options = _scan_csvs(wheel_csv_root, suffix=("_diff_drive.csv", "_motor_rpm.csv"))
    if not t2_options:
        raise FileNotFoundError(f"No T2 CSV files found in {t2_csv_root}")
    if not wheel_options:
        first_t2 = next(iter(t2_options.values()))
        output_csv = _output_diff_drive_path(first_t2, wheel_csv_root)
        convert_t2_csv_to_diff_drive(first_t2, output_csv, args.fps, 0.10, 0.38, 0.4, 1.0, yaw_source="t2")
        wheel_options = _scan_csvs(wheel_csv_root, suffix=("_diff_drive.csv", "_motor_rpm.csv"))

    t2_label = _default_label(t2_options, "Neutral_walk_forward_002__A057.csv")
    wheel_label = _default_label(wheel_options, "kimodo_picking_item_from_the_shelf_office_diff_drive.csv")

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
        client.camera.position = np.array([2.4, 1.6, 3.4], dtype=np.float64)
        client.camera.look_at = np.array([0.0, 0.65, 0.0], dtype=np.float64)
        client.camera.up_direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    t2 = T2Playback(server, t2_urdf_path, t2_options[t2_label])
    wheel = WheelBasePlayback(server, wheel_urdf_path, wheel_options[wheel_label], args.fps)

    t2_start_time = time.time()
    wheel_start_time = time.time()
    wheel_reverse_play = False
    wheel_reverse_start_time = time.time()
    wheel_reverse_start_frame = 0

    with server.gui.add_folder("T2 Robot"):
        t2_dropdown = server.gui.add_dropdown("T2 CSV", tuple(t2_options.keys()), initial_value=t2_label)
        t2_play = server.gui.add_checkbox("Play T2", initial_value=False)
        t2_loop = server.gui.add_checkbox("Loop T2", initial_value=True)
        t2_speed = server.gui.add_slider("T2 speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        t2_frame = server.gui.add_slider("T2 frame", min=0, max=t2.length - 1, step=1, initial_value=0)
        t2_reset = server.gui.add_button("Reset T2")

    with server.gui.add_folder("Wheel Base"):
        wheel_dropdown = server.gui.add_dropdown("Wheel CSV", tuple(wheel_options.keys()), initial_value=wheel_label)
        wheel_play = server.gui.add_checkbox("Play wheel base", initial_value=False)
        wheel_loop = server.gui.add_checkbox("Loop wheel base", initial_value=True)
        wheel_speed = server.gui.add_slider("Wheel speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        wheel_frame = server.gui.add_slider("Wheel frame", min=0, max=wheel.length - 1, step=1, initial_value=0)
        wheel_reset = server.gui.add_button("Reset wheel base")

    with server.gui.add_folder("Synchronized Playback"):
        both_speed = server.gui.add_slider("Both speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        both_frame = server.gui.add_slider(
            "Both frame",
            min=0,
            max=min(t2.length, wheel.length) - 1,
            step=1,
            initial_value=0,
        )
        play_both = server.gui.add_button("Play both")
        pause_both = server.gui.add_button("Pause both")
        reset_both = server.gui.add_button("Reset both")
        reverse_visual_both = server.gui.add_button("Reverse sim to start")

    with server.gui.add_folder("Convert T2 to Wheel CSV"):
        max_forward_speed = server.gui.add_slider("Max forward speed", min=0.05, max=4.0, step=0.05, initial_value=3.0)
        max_yaw_rate = server.gui.add_slider("Max yaw rate", min=0.1, max=15.0, step=0.1, initial_value=12.0)
        convert_button = server.gui.add_button("Create wheel CSV from selected T2")
        convert_status = server.gui.add_markdown("Ready.")

    with server.gui.add_folder("TaraBase Hardware"):
        tara_play_button = server.gui.add_button("Play TaraBase from pause")
        tara_pause_button = server.gui.add_button("Pause TaraBase")
        tara_restart_button = server.gui.add_button("Restart TaraBase CSV")
        tara_reverse_button = server.gui.add_button("Move hardware back to start")
        tara_status = server.gui.add_markdown(
            f"Ready. Port `{args.tara_port}`, clamp +/-{args.tara_max_rpm:g} RPM, "
            f"RPM scale `{args.tara_rpm_scale:g}`."
        )

    with server.gui.add_folder("Visibility"):
        wheel_visual = server.gui.add_checkbox("Show wheel visual", initial_value=wheel.robot.show_visual)
        wheel_collision = server.gui.add_checkbox("Show wheel collision", initial_value=wheel.robot.show_collision)

    def refresh_wheel_dropdown(select_path: Path | None = None) -> None:
        nonlocal wheel_options
        wheel_options = _scan_csvs(wheel_csv_root, suffix="_diff_drive.csv")
        labels = tuple(wheel_options.keys()) or ("<none>",)
        wheel_dropdown.options = labels
        if select_path is not None:
            label = _relative_label(select_path, wheel_csv_root)
            if label in wheel_options:
                wheel_dropdown.value = label
                return
        if str(wheel_dropdown.value) not in wheel_options and wheel_options:
            wheel_dropdown.value = next(iter(wheel_options))

    def reset_t2_clock(frame_idx: int = 0) -> None:
        nonlocal t2_start_time
        t2_start_time = time.time() - (frame_idx / args.fps) / float(t2_speed.value)

    def reset_wheel_clock(frame_idx: int = 0) -> None:
        nonlocal wheel_start_time
        wheel_start_time = time.time() - (frame_idx / args.fps) / float(wheel_speed.value)

    def start_wheel_reverse_visual() -> None:
        nonlocal wheel_reverse_play, wheel_reverse_start_time, wheel_reverse_start_frame
        wheel_play.value = False
        wheel_loop.value = False
        wheel_reverse_start_frame = int(wheel_frame.value)
        if wheel_reverse_start_frame <= 0:
            wheel_reverse_start_frame = wheel.length - 1
            wheel_frame.value = wheel_reverse_start_frame
            wheel.apply_frame(wheel_reverse_start_frame)
        wheel_speed.value = 1.0
        wheel_reverse_start_time = time.time()
        wheel_reverse_play = True

    def stop_wheel_reverse_visual() -> None:
        nonlocal wheel_reverse_play
        wheel_reverse_play = False

    tara_stop_event = threading.Event()
    tara_thread: threading.Thread | None = None
    tara_next_index = 0
    tara_active_label: str | None = None
    tara_active_direction = "forward"

    def set_tara_status(message: str) -> None:
        tara_status.content = message

    def selected_tara_distance_text() -> str:
        label = str(wheel_dropdown.value)
        if label not in wheel_options:
            return "No valid wheel CSV selected."
        commands = load_wheel_commands(wheel_options[label])
        distance_m = estimate_travel_distance_m(
            commands,
            fps=args.fps,
            speed_scale=args.tara_rpm_scale,
            playback_speed=1.0,
            max_abs_rpm=args.tara_max_rpm,
            fit_to_rpm_limit=True,
        )
        return f"Estimated full motion distance: `{distance_m:.3f} m`."

    def tara_progress(
        index: int,
        total: int,
        csv_left: float,
        csv_right: float,
        scaled_left: float,
        scaled_right: float,
        path_left: float,
        path_right: float,
        sdk_left: float,
        sdk_right: float,
        distance_m: float,
        total_distance_m: float,
    ) -> None:
        nonlocal tara_next_index
        tara_next_index = index
        tara_status.content = (
            f"Sending `{wheel_dropdown.value}`: frame `{index}/{total}`\n\n"
            f"CSV L/R `{csv_left:.1f}` / `{csv_right:.1f}` RPM\n\n"
            f"Scaled L/R `{scaled_left:.1f}` / `{scaled_right:.1f}` RPM\n\n"
            f"Path L/R after turn fix `{path_left:.1f}` / `{path_right:.1f}` RPM\n\n"
            f"SDK command L/R `{sdk_left:.1f}` / `{sdk_right:.1f}` RPM\n\n"
            f"Distance `{distance_m:.3f}/{total_distance_m:.3f} m`"
        )

    def tara_worker(csv_path: Path, start_index: int, reverse_playback: bool) -> None:
        nonlocal tara_next_index
        try:
            if args.tara_remote_url:
                remote_url = args.tara_remote_url.rstrip("/")
                stream_payload = {
                    "action": "stream",
                    "csv_text": csv_path.read_text(encoding="utf-8"),
                    "options": {
                        "port": args.tara_port,
                        "slave_id": args.tara_slave_id,
                        "baudrate": args.tara_baudrate,
                        "fps": args.fps,
                        "speed_scale": args.tara_rpm_scale,
                        "max_abs_rpm": args.tara_max_rpm,
                        "reverse_playback": reverse_playback,
                        "debug": args.tara_debug,
                        "start_index": start_index,
                    }
                }

                if remote_url.startswith(("ws://", "wss://")):
                    from websockets.sync.client import connect

                    with connect(remote_url, open_timeout=10, close_timeout=2) as websocket:
                        websocket.send(json.dumps(stream_payload))
                        stream_started = False
                        while True:
                            remote_status = json.loads(websocket.recv())
                            if remote_status.get("type") == "error":
                                raise RuntimeError(str(remote_status.get("error", "Unknown remote error")))
                            if remote_status.get("type") == "accepted":
                                stream_started = True
                                continue
                            if remote_status.get("type") != "status":
                                continue
                            running = bool(remote_status.get("running", False))
                            stream_started = stream_started or running
                            tara_next_index = int(remote_status.get("index", tara_next_index))
                            tara_status.content = str(
                                remote_status.get("message", "TaraBase remote stream running.")
                            )
                            if stream_started and not running:
                                break
                    return

                payload = json.dumps(
                    {key: value for key, value in stream_payload.items() if key != "action"}
                ).encode("utf-8")
                request = urllib.request.Request(
                    f"{remote_url}/stream",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    response.read()

                while True:
                    with urllib.request.urlopen(f"{remote_url}/status", timeout=5) as response:
                        remote_status = json.load(response)
                    tara_next_index = int(remote_status.get("index", tara_next_index))
                    tara_status.content = str(remote_status.get("message", "TaraBase remote stream running."))
                    if not remote_status.get("running", False):
                        break
                    time.sleep(0.2)
                return

            stream_wheel_commands(
                csv_path=csv_path,
                port=args.tara_port,
                slave_id=args.tara_slave_id,
                baudrate=args.tara_baudrate,
                fps=args.fps,
                speed_scale=args.tara_rpm_scale,
                playback_speed=1.0,
                max_abs_rpm=args.tara_max_rpm,
                left_motor_sign=1,
                right_motor_sign=-1,
                invert_turn_direction=True,
                fit_to_rpm_limit=True,
                reverse_playback=reverse_playback,
                debug=args.tara_debug,
                start_index=start_index,
                stop_event=tara_stop_event,
                progress_callback=tara_progress,
                status_callback=set_tara_status,
            )
        except Exception as exc:
            tara_status.content = f"TaraBase send failed: `{exc}`"
        finally:
            tara_status.content = f"{tara_status.content}\n\nStream ended."

    @t2_dropdown.on_update
    def _(_) -> None:
        nonlocal t2
        label = str(t2_dropdown.value)
        if label not in t2_options:
            return
        t2_play.value = False
        t2.load_csv(t2_options[label])
        t2_frame.max = t2.length - 1
        both_frame.max = min(t2.length, wheel.length) - 1
        t2_frame.value = 0
        both_frame.value = 0
        reset_t2_clock(0)

    @wheel_dropdown.on_update
    def _(_) -> None:
        label = str(wheel_dropdown.value)
        if label not in wheel_options:
            return
        wheel_play.value = False
        wheel.load_csv(wheel_options[label], args.fps)
        wheel_frame.max = wheel.length - 1
        both_frame.max = min(t2.length, wheel.length) - 1
        wheel_frame.value = 0
        both_frame.value = 0
        reset_wheel_clock(0)
        tara_status.content = selected_tara_distance_text()

    @t2_frame.on_update
    def _(_) -> None:
        if not t2_play.value:
            t2.apply_frame(int(t2_frame.value))
            reset_t2_clock(int(t2_frame.value))

    @wheel_frame.on_update
    def _(_) -> None:
        if not wheel_play.value:
            wheel.apply_frame(int(wheel_frame.value))
            reset_wheel_clock(int(wheel_frame.value))

    @t2_reset.on_click
    def _(_) -> None:
        t2_play.value = False
        t2_frame.value = 0
        t2.apply_frame(0)
        reset_t2_clock(0)

    @wheel_reset.on_click
    def _(_) -> None:
        wheel_play.value = False
        wheel_frame.value = 0
        wheel.apply_frame(0)
        reset_wheel_clock(0)

    @both_speed.on_update
    def _(_) -> None:
        t2_speed.value = both_speed.value
        wheel_speed.value = both_speed.value
        reset_t2_clock(int(t2_frame.value))
        reset_wheel_clock(int(wheel_frame.value))

    @both_frame.on_update
    def _(_) -> None:
        if t2_play.value or wheel_play.value:
            return
        frame_idx = int(both_frame.value)
        t2_frame.value = min(frame_idx, t2.length - 1)
        wheel_frame.value = min(frame_idx, wheel.length - 1)
        t2.apply_frame(int(t2_frame.value))
        wheel.apply_frame(int(wheel_frame.value))
        reset_t2_clock(int(t2_frame.value))
        reset_wheel_clock(int(wheel_frame.value))

    @play_both.on_click
    def _(_) -> None:
        t2_speed.value = both_speed.value
        wheel_speed.value = both_speed.value
        reset_t2_clock(int(t2_frame.value))
        reset_wheel_clock(int(wheel_frame.value))
        t2_play.value = True
        wheel_play.value = True

    @pause_both.on_click
    def _(_) -> None:
        t2_play.value = False
        wheel_play.value = False
        stop_wheel_reverse_visual()
        both_frame.value = min(int(t2_frame.value), int(wheel_frame.value), int(both_frame.max))
        reset_t2_clock(int(t2_frame.value))
        reset_wheel_clock(int(wheel_frame.value))

    @reset_both.on_click
    def _(_) -> None:
        t2_play.value = False
        wheel_play.value = False
        stop_wheel_reverse_visual()
        t2_frame.value = 0
        wheel_frame.value = 0
        both_frame.value = 0
        t2.apply_frame(0)
        wheel.apply_frame(0)
        reset_t2_clock(0)
        reset_wheel_clock(0)

    @reverse_visual_both.on_click
    def _(_) -> None:
        t2_play.value = False
        wheel_play.value = False
        start_wheel_reverse_visual()

    @convert_button.on_click
    def _(_) -> None:
        t2_label_now = str(t2_dropdown.value)
        if t2_label_now not in t2_options:
            convert_status.content = "No valid T2 CSV selected."
            return
        t2_csv = t2_options[t2_label_now]
        output_csv = _output_diff_drive_path(t2_csv, wheel_csv_root)
        try:
            convert_t2_csv_to_diff_drive(
                input_csv=t2_csv,
                output_csv=output_csv,
                fps=args.fps,
                wheel_radius_m=0.10,
                wheel_separation_m=0.38,
                max_forward_speed=float(max_forward_speed.value),
                max_yaw_rate=float(max_yaw_rate.value),
                yaw_source="t2",
            )
        except Exception as exc:
            convert_status.content = f"Conversion failed: `{exc}`"
            return
        refresh_wheel_dropdown(output_csv)
        wheel.load_csv(output_csv, args.fps)
        wheel_frame.max = wheel.length - 1
        both_frame.max = min(t2.length, wheel.length) - 1
        wheel_frame.value = 0
        both_frame.value = 0
        reset_wheel_clock(0)
        convert_status.content = f"Created `{_relative_label(output_csv, wheel_csv_root)}`."

    def start_tara_stream(*, restart: bool, reverse_playback: bool | None = None) -> None:
        nonlocal tara_thread, tara_next_index, tara_active_label, tara_active_direction
        if tara_thread is not None and tara_thread.is_alive():
            tara_status.content = "TaraBase is already playing."
            return
        label = str(wheel_dropdown.value)
        if label not in wheel_options:
            tara_status.content = "No valid wheel CSV selected."
            return
        direction = tara_active_direction if reverse_playback is None else ("reverse" if reverse_playback else "forward")
        if restart or tara_active_label != label or tara_active_direction != direction:
            tara_next_index = 0
            tara_active_label = label
            tara_active_direction = direction
        tara_stop_event.clear()
        csv_path = wheel_options[label]
        tara_status.content = (
            f"Starting TaraBase {tara_active_direction} stream from `{label}` row {tara_next_index + 1}..."
            f"\n\nClamp +/-`{args.tara_max_rpm:g}` RPM, RPM scale `{args.tara_rpm_scale:g}`."
        )
        tara_thread = threading.Thread(
            target=tara_worker,
            args=(csv_path, tara_next_index, tara_active_direction == "reverse"),
            daemon=True,
        )
        tara_thread.start()

    @tara_play_button.on_click
    def _(_) -> None:
        start_tara_stream(restart=False)

    @tara_restart_button.on_click
    def _(_) -> None:
        stop_wheel_reverse_visual()
        start_tara_stream(restart=True, reverse_playback=False)

    @tara_reverse_button.on_click
    def _(_) -> None:
        start_tara_stream(restart=True, reverse_playback=True)

    @tara_pause_button.on_click
    def _(_) -> None:
        tara_stop_event.set()
        if args.tara_remote_url:
            try:
                remote_url = args.tara_remote_url.rstrip("/")
                if remote_url.startswith(("ws://", "wss://")):
                    from websockets.sync.client import connect

                    with connect(remote_url, open_timeout=5, close_timeout=2) as websocket:
                        websocket.send(json.dumps({"action": "stop"}))
                        while json.loads(websocket.recv()).get("type") != "stopped":
                            pass
                else:
                    request = urllib.request.Request(f"{remote_url}/stop", data=b"{}", method="POST")
                    urllib.request.urlopen(request, timeout=5).close()
            except Exception as exc:
                tara_status.content = f"Remote pause failed: `{exc}`"
                return
        stop_wheel_reverse_visual()
        tara_status.content = f"Pause requested. Resume will continue near row {tara_next_index + 1}."

    @wheel_visual.on_update
    def _(_) -> None:
        wheel.robot.show_visual = wheel_visual.value

    @wheel_collision.on_update
    def _(_) -> None:
        wheel.robot.show_collision = wheel_collision.value

    print(f"Open http://localhost:{args.port}")
    print(f"T2 CSV root: {t2_csv_root}")
    print(f"Wheel CSV root: {wheel_csv_root}")
    print(f"TaraBase port: {args.tara_port}")
    print(f"TaraBase max RPM: {args.tara_max_rpm:g}")
    print(f"TaraBase RPM scale: {args.tara_rpm_scale:g}")
    if args.tara_remote_url:
        print(f"TaraBase remote server: {args.tara_remote_url}")

    while True:
        now = time.time()
        if t2_play.value:
            frame_idx = int((now - t2_start_time) * args.fps * float(t2_speed.value))
            if frame_idx >= t2.length:
                if t2_loop.value:
                    t2_start_time = now
                    frame_idx = 0
                else:
                    frame_idx = t2.length - 1
                    t2_play.value = False
            t2_frame.value = frame_idx
            t2.apply_frame(frame_idx)

        if wheel_play.value:
            stop_wheel_reverse_visual()
            frame_idx = int((now - wheel_start_time) * args.fps * float(wheel_speed.value))
            if frame_idx >= wheel.length:
                if wheel_loop.value:
                    wheel_start_time = now
                    frame_idx = 0
                else:
                    frame_idx = wheel.length - 1
                    wheel_play.value = False
            wheel_frame.value = frame_idx
            wheel.apply_frame(frame_idx)

        if wheel_reverse_play:
            elapsed_frames = int((now - wheel_reverse_start_time) * args.fps)
            frame_idx = wheel_reverse_start_frame - elapsed_frames
            if frame_idx <= 0:
                frame_idx = 0
                wheel_reverse_play = False
            wheel_frame.value = frame_idx
            wheel.apply_frame(frame_idx)

        if t2_play.value and (wheel_play.value or wheel_reverse_play):
            both_frame.value = min(int(t2_frame.value), int(wheel_frame.value), int(both_frame.max))

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
