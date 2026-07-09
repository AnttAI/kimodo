#!/usr/bin/env python3
"""View T2, G1, and the generated wheel-base URDF with independent CSV playback."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import pickle
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

DEFAULT_WHEEL_URDF_PATH = REPO_ROOT / "robot_demo_outputs" / "wheel_base_robot" / "two_wheel_base.urdf"
DEFAULT_WHEEL_CSV_ROOT = REPO_ROOT / "robot_demo_outputs" / "wheel_base_robot"
DEFAULT_T2_CSV_ROOT = Path("/home/jony/Downloads/soma-retargeter/assets/motions/t2_csv")
DEFAULT_T2_URDF_PATH = Path("/home/jony/Downloads/soma-retargeter/antt_t2/T2_serial_nero_arms.urdf")
DEFAULT_G1_URDF_PATH = Path("/home/jony/Downloads/viser-sandbox/unitree_g1/g1_custom_collision_29dof.urdf")
DEFAULT_G1_MOTION_ROOT = Path(
    "/home/jony/Documents/GR00T-WholeBodyControl/GR00T-WholeBodyControl/motionbricks/out"
)
DEFAULT_G1_MOTION_PATH = DEFAULT_G1_MOTION_ROOT / "my_motionbricks_motion1_g1.csv"
FALLBACK_G1_MOTION_PATH = Path("/home/jony/Downloads/pyroki/examples/retarget_helpers/humanoid/g1_motion_gmr.pkl")
DEFAULT_WHEEL_RADIUS_M = 0.10
DEFAULT_WHEEL_SEPARATION_M = 0.38

WHEEL_SCENE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
T2_SCENE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
G1_SCENE_OFFSET = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WHEEL_Z_UP_TO_SCENE_Y_UP = Rotation.from_euler("x", -90.0, degrees=True)
G1_Z_UP_TO_SCENE_Y_UP = Rotation.from_euler("x", -90.0, degrees=True)
G1_YAW_ALIGNMENT = Rotation.from_euler("y", 90.0, degrees=True)
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


def _scan_g1_motions(root: Path, default_motion_path: Path) -> dict[str, Path]:
    root = root.expanduser().resolve()
    options: dict[str, Path] = {}
    if root.is_dir():
        paths = sorted([path for path in root.rglob("*.csv") if "g1" in path.name.lower()] + list(root.rglob("*.pkl")))
        options.update({_relative_label(path, root): path for path in paths})
    for path in (default_motion_path.expanduser().resolve(), FALLBACK_G1_MOTION_PATH.expanduser().resolve()):
        if path.exists():
            options.setdefault(path.name, path)
    return options


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
        reader = csv.reader(f)
        header = next(reader)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]

    if wheel_radius_m <= 0:
        raise ValueError(f"wheel radius must be positive, got {wheel_radius_m}")
    if wheel_separation_m <= 0:
        raise ValueError(f"wheel separation must be positive, got {wheel_separation_m}")

    def column(name: str) -> np.ndarray:
        return data[:, header.index(name)].astype(np.float64)

    rpm_columns: tuple[str, str] | None = None
    if "left_motor_rpm" in header and "right_motor_rpm" in header:
        rpm_columns = ("left_motor_rpm", "right_motor_rpm")
    elif "left_wheel_rpm" in header and "right_wheel_rpm" in header:
        rpm_columns = ("left_wheel_rpm", "right_wheel_rpm")

    if rpm_columns is not None:
        rpm_to_rad_s = 2.0 * math.pi / 60.0
        left_rad_s = column(rpm_columns[0]) * rpm_to_rad_s
        right_rad_s = column(rpm_columns[1]) * rpm_to_rad_s
        left_linear_m_s = left_rad_s * wheel_radius_m
        right_linear_m_s = right_rad_s * wheel_radius_m
        forward_velocity_m_s = 0.5 * (left_linear_m_s + right_linear_m_s)
        yaw_rate_rad_s = (right_linear_m_s - left_linear_m_s) / wheel_separation_m
    else:
        required = (
            "forward_velocity_m_s",
            "yaw_rate_rad_s",
            "left_wheel_rad_s",
            "right_wheel_rad_s",
        )
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
    root_yaw = data[:, header.index("root_yaw_rad")].astype(np.float64) if "root_yaw_rad" in header else None
    dt = 1.0 / fps
    n = data.shape[0]

    x = np.zeros(n, dtype=np.float64)
    z = np.zeros(n, dtype=np.float64)
    if "root_x_m" in header and "root_y_m" in header:
        x = data[:, header.index("root_x_m")].astype(np.float64)
        z = data[:, header.index("root_y_m")].astype(np.float64)
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
    def __init__(self, server: viser.ViserServer, urdf_path: Path, csv_path: Path, fps: float):
        self.server = server
        self.urdf_path = urdf_path
        self.fps = fps
        self.root_frame = server.scene.add_frame("/wheel_base", show_axes=False)
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
        self.motion = _read_diff_drive_csv(csv_path, fps)
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

    def apply_frame(self, frame_idx: int) -> None:
        frame_idx = max(0, min(int(frame_idx), self.length - 1))
        self.frame_idx = frame_idx
        self.root_frame.position = np.array(
            [
                self.motion["x"][frame_idx] + WHEEL_SCENE_OFFSET[0],
                WHEEL_SCENE_OFFSET[1],
                self.motion["z"][frame_idx] + WHEEL_SCENE_OFFSET[2],
            ],
            dtype=np.float64,
        )
        # Stored diff-drive headings use math convention in the viewer ground
        # plane: atan2(+Z, +X). In a right-handed +Y-up scene, positive
        # Rotation.from_euler("y", theta) turns +X toward -Z, so the visual
        # root needs the opposite sign to face along the CSV path tangent.
        yaw_rot = Rotation.from_euler("y", -self.motion["yaw"][frame_idx])
        self.root_frame.wxyz = (yaw_rot * WHEEL_Z_UP_TO_SCENE_Y_UP).as_quat(scalar_first=True)

        left_idx = self.joint_index.get("left_wheel_joint")
        right_idx = self.joint_index.get("right_wheel_joint")
        if left_idx is not None:
            self.cfg[left_idx] = math.remainder(self.motion["left_angle"][frame_idx], 2.0 * math.pi)
        if right_idx is not None:
            self.cfg[right_idx] = math.remainder(self.motion["right_angle"][frame_idx], 2.0 * math.pi)
        self.robot.update_cfg(self.cfg)


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

    def set_visible(self, visible: bool) -> None:
        self.viewer.set_mesh_visibility(visible)


class G1RobotView:
    def __init__(self, server: viser.ViserServer, urdf_path: Path, motion_path: Path | None = None):
        self.server = server
        self.urdf_path = urdf_path
        self.root_frame = server.scene.add_frame("/g1_robot", show_axes=False)
        self.root_frame.position = G1_SCENE_OFFSET
        self.root_frame.wxyz = (G1_YAW_ALIGNMENT * G1_Z_UP_TO_SCENE_Y_UP).as_quat(scalar_first=True)
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
        self.motion_path: Path | None = None
        self.motion: dict[str, object] | None = None
        self.motion_joint_indices: list[int | None] = list(range(len(self.joint_names)))
        self.root_positions = np.zeros((1, 3), dtype=np.float64)
        self.root_origin = np.zeros(3, dtype=np.float64)
        self.frame_idx = 0
        if motion_path is not None:
            self.load_motion(motion_path)
        self.robot.update_cfg(self.cfg)

    @property
    def length(self) -> int:
        if self.motion is None:
            return 1
        return int(self.motion["num_frames"])

    @property
    def fps(self) -> float:
        if self.motion is None:
            return 30.0
        return float(self.motion["fps"])

    def load_motion(self, motion_path: Path) -> None:
        if motion_path.suffix.lower() == ".csv":
            self._load_csv_motion(motion_path)
        else:
            self._load_pickle_motion(motion_path)
        self.apply_frame(0)

    def _load_pickle_motion(self, motion_path: Path) -> None:
        with motion_path.open("rb") as f:
            motion = pickle.load(f)
        missing = [key for key in ("dof_pos", "fps", "num_frames") if key not in motion]
        if missing:
            raise ValueError(f"{motion_path} is missing required G1 motion key(s): {', '.join(missing)}")

        dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)
        if dof_pos.ndim != 2:
            raise ValueError(f"{motion_path} dof_pos must be 2D, got shape {dof_pos.shape}")
        motion["dof_pos"] = dof_pos
        motion["num_frames"] = int(min(int(motion["num_frames"]), dof_pos.shape[0]))
        root_pos = np.asarray(motion.get("root_pos", np.zeros((motion["num_frames"], 3))), dtype=np.float64)
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            root_pos = np.zeros((motion["num_frames"], 3), dtype=np.float64)
        self.root_positions = root_pos[: motion["num_frames"]]
        self.root_origin = self.root_positions[0].copy()

        all_joint_names = motion.get("all_joint_names")
        if all_joint_names is None:
            self.motion_joint_indices = [idx if idx < dof_pos.shape[1] else None for idx in range(len(self.joint_names))]
        else:
            urdf_motion_names = [name if str(name).endswith("_joint") else f"{name}_joint" for name in all_joint_names]
            self.motion_joint_indices = [
                urdf_motion_names.index(joint_name) if joint_name in urdf_motion_names else None
                for joint_name in self.joint_names
            ]

        self.motion_path = motion_path
        self.motion = motion

    def _load_csv_motion(self, motion_path: Path) -> None:
        with motion_path.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
        data = np.loadtxt(motion_path, delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data[None, :]

        csv_joint_to_column = {
            name.removesuffix("_dof"): idx
            for idx, name in enumerate(header)
            if name.endswith("_joint_dof")
        }
        self.motion_joint_indices = [csv_joint_to_column.get(joint_name) for joint_name in self.joint_names]
        if all(idx is None for idx in self.motion_joint_indices):
            raise ValueError(f"{motion_path} does not contain any G1 joint *_dof columns matching the URDF")

        motion = {
            "dof_pos": np.deg2rad(data.astype(np.float64)),
            "fps": 30.0,
            "num_frames": int(data.shape[0]),
        }
        if all(name in header for name in ("root_translateX", "root_translateY", "root_translateZ")):
            root_xyz_cm = data[
                :,
                [
                    header.index("root_translateX"),
                    header.index("root_translateY"),
                    header.index("root_translateZ"),
                ],
            ].astype(np.float64)
            self.root_positions = root_xyz_cm / 100.0
        else:
            self.root_positions = np.zeros((int(data.shape[0]), 3), dtype=np.float64)
        self.root_origin = self.root_positions[0].copy()
        self.motion_path = motion_path
        self.motion = motion

    def apply_frame(self, frame_idx: int) -> None:
        if self.motion is None:
            return
        frame_idx = max(0, min(int(frame_idx), self.length - 1))
        self.frame_idx = frame_idx
        dof_pos = self.motion["dof_pos"]
        root_delta = self.root_positions[frame_idx] - self.root_origin
        self.root_frame.position = np.array(
            [
                G1_SCENE_OFFSET[0] + root_delta[0],
                G1_SCENE_OFFSET[1] + root_delta[2],
                G1_SCENE_OFFSET[2] + root_delta[1],
            ],
            dtype=np.float64,
        )
        self.root_frame.wxyz = (G1_YAW_ALIGNMENT * G1_Z_UP_TO_SCENE_Y_UP).as_quat(scalar_first=True)
        for cfg_idx, motion_idx in enumerate(self.motion_joint_indices):
            if motion_idx is not None:
                self.cfg[cfg_idx] = float(dof_pos[frame_idx, motion_idx])
        self.robot.update_cfg(self.cfg)

    def set_visible(self, visible: bool) -> None:
        self.root_frame.visible = visible
        self.robot.show_visual = visible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch combined T2 and wheel-base CSV playback UI.")
    parser.add_argument("--wheel-urdf", type=Path, default=DEFAULT_WHEEL_URDF_PATH, help="Wheel-base URDF path.")
    parser.add_argument("--t2-urdf", type=Path, default=DEFAULT_T2_URDF_PATH, help="T2 URDF path.")
    parser.add_argument("--g1-urdf", type=Path, default=DEFAULT_G1_URDF_PATH, help="G1 URDF path.")
    parser.add_argument("--g1-motion-root", type=Path, default=DEFAULT_G1_MOTION_ROOT, help="Folder containing G1 .pkl motion files.")
    parser.add_argument("--g1-motion", type=Path, default=DEFAULT_G1_MOTION_PATH, help="Default G1 .pkl motion path.")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    wheel_urdf_path = args.wheel_urdf.expanduser().resolve()
    t2_urdf_path = args.t2_urdf.expanduser().resolve()
    g1_urdf_path = args.g1_urdf.expanduser().resolve()
    t2_csv_root = args.t2_csv_root.expanduser().resolve()
    g1_motion_root = args.g1_motion_root.expanduser().resolve()
    g1_motion_path = args.g1_motion.expanduser().resolve()
    wheel_csv_root = args.wheel_csv_root.expanduser().resolve()
    wheel_csv_root.mkdir(parents=True, exist_ok=True)

    if not wheel_urdf_path.exists():
        raise FileNotFoundError(f"Wheel URDF not found: {wheel_urdf_path}")
    if not t2_urdf_path.exists():
        raise FileNotFoundError(f"T2 URDF not found: {t2_urdf_path}")
    if not g1_urdf_path.exists():
        raise FileNotFoundError(f"G1 URDF not found: {g1_urdf_path}")

    t2_options = _scan_csvs(t2_csv_root)
    g1_options = _scan_g1_motions(g1_motion_root, g1_motion_path)
    wheel_options = _scan_csvs(wheel_csv_root, suffix=("_diff_drive.csv", "_motor_rpm.csv"))
    if not t2_options:
        raise FileNotFoundError(f"No T2 CSV files found in {t2_csv_root}")
    if not g1_options:
        raise FileNotFoundError(f"No G1 .pkl motion files found in {g1_motion_root} or at {g1_motion_path}")
    if not wheel_options:
        first_t2 = next(iter(t2_options.values()))
        output_csv = _output_diff_drive_path(first_t2, wheel_csv_root)
        convert_t2_csv_to_diff_drive(first_t2, output_csv, args.fps, 0.10, 0.38, 0.4, 1.0)
        wheel_options = _scan_csvs(wheel_csv_root, suffix=("_diff_drive.csv", "_motor_rpm.csv"))

    t2_label = _default_label(t2_options, "Neutral_walk_forward_002__A057.csv")
    g1_label = _default_label(g1_options, g1_motion_path.name)
    wheel_label = _default_label(wheel_options, "Neutral_walk_forward_002__A057_diff_drive.csv")

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
    g1 = G1RobotView(server, g1_urdf_path, g1_options[g1_label])
    wheel = WheelBasePlayback(server, wheel_urdf_path, wheel_options[wheel_label], args.fps)

    t2_start_time = time.time()
    g1_start_time = time.time()
    wheel_start_time = time.time()

    with server.gui.add_folder("T2 Robot"):
        t2_dropdown = server.gui.add_dropdown("T2 CSV", tuple(t2_options.keys()), initial_value=t2_label)
        t2_play = server.gui.add_checkbox("Play T2", initial_value=False)
        t2_loop = server.gui.add_checkbox("Loop T2", initial_value=True)
        t2_speed = server.gui.add_slider("T2 speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        t2_frame = server.gui.add_slider("T2 frame", min=0, max=t2.length - 1, step=1, initial_value=0)
        t2_reset = server.gui.add_button("Reset T2")

    with server.gui.add_folder("G1 Robot"):
        g1_dropdown = server.gui.add_dropdown("G1 motion", tuple(g1_options.keys()), initial_value=g1_label)
        g1_play = server.gui.add_checkbox("Play G1", initial_value=False)
        g1_loop = server.gui.add_checkbox("Loop G1", initial_value=True)
        g1_speed = server.gui.add_slider("G1 speed", min=0.1, max=3.0, step=0.1, initial_value=1.0)
        g1_frame = server.gui.add_slider("G1 frame", min=0, max=g1.length - 1, step=1, initial_value=0)
        g1_reset = server.gui.add_button("Reset G1")

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

    with server.gui.add_folder("Convert T2 to Wheel CSV"):
        max_forward_speed = server.gui.add_slider("Max forward speed", min=0.05, max=4.0, step=0.05, initial_value=3.0)
        max_yaw_rate = server.gui.add_slider("Max yaw rate", min=0.1, max=15.0, step=0.1, initial_value=12.0)
        convert_button = server.gui.add_button("Create wheel CSV from selected T2")
        convert_status = server.gui.add_markdown("Ready.")

    with server.gui.add_folder("Visibility"):
        t2_visible = server.gui.add_checkbox("Show T2 robot", initial_value=True)
        g1_visible = server.gui.add_checkbox("Show G1 robot", initial_value=False)
        tara_base_visible = server.gui.add_checkbox("Show Tara base robot", initial_value=True)
        wheel_visual = server.gui.add_checkbox("Show wheel visual", initial_value=wheel.robot.show_visual)
        wheel_collision = server.gui.add_checkbox("Show wheel collision", initial_value=wheel.robot.show_collision)
    g1.set_visible(False)

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

    def reset_g1_clock(frame_idx: int = 0) -> None:
        nonlocal g1_start_time
        g1_start_time = time.time() - (frame_idx / g1.fps) / float(g1_speed.value)

    def reset_wheel_clock(frame_idx: int = 0) -> None:
        nonlocal wheel_start_time
        wheel_start_time = time.time() - (frame_idx / args.fps) / float(wheel_speed.value)

    @t2_dropdown.on_update
    def _(_) -> None:
        nonlocal t2
        label = str(t2_dropdown.value)
        if label not in t2_options:
            return
        t2_play.value = False
        t2.load_csv(t2_options[label])
        t2.set_visible(bool(t2_visible.value))
        t2_frame.max = t2.length - 1
        both_frame.max = min(t2.length, wheel.length) - 1
        t2_frame.value = 0
        both_frame.value = 0
        reset_t2_clock(0)

    @g1_dropdown.on_update
    def _(_) -> None:
        label = str(g1_dropdown.value)
        if label not in g1_options:
            return
        g1_play.value = False
        try:
            g1.load_motion(g1_options[label])
        except Exception as exc:
            print(f"Failed to load G1 motion {g1_options[label]}: {exc}")
            return
        g1_frame.max = g1.length - 1
        g1_frame.value = 0
        reset_g1_clock(0)

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

    @g1_frame.on_update
    def _(_) -> None:
        if not g1_play.value:
            g1.apply_frame(int(g1_frame.value))
            reset_g1_clock(int(g1_frame.value))

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

    @g1_reset.on_click
    def _(_) -> None:
        g1_play.value = False
        g1_frame.value = 0
        g1.apply_frame(0)
        reset_g1_clock(0)

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
        both_frame.value = min(int(t2_frame.value), int(wheel_frame.value), int(both_frame.max))
        reset_t2_clock(int(t2_frame.value))
        reset_wheel_clock(int(wheel_frame.value))

    @reset_both.on_click
    def _(_) -> None:
        t2_play.value = False
        wheel_play.value = False
        t2_frame.value = 0
        wheel_frame.value = 0
        both_frame.value = 0
        t2.apply_frame(0)
        wheel.apply_frame(0)
        reset_t2_clock(0)
        reset_wheel_clock(0)

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

    @wheel_visual.on_update
    def _(_) -> None:
        wheel.robot.show_visual = bool(tara_base_visible.value) and bool(wheel_visual.value)

    @wheel_collision.on_update
    def _(_) -> None:
        wheel.robot.show_collision = bool(tara_base_visible.value) and bool(wheel_collision.value)

    @t2_visible.on_update
    def _(_) -> None:
        t2.set_visible(bool(t2_visible.value))

    @g1_visible.on_update
    def _(_) -> None:
        g1.set_visible(bool(g1_visible.value))

    @tara_base_visible.on_update
    def _(_) -> None:
        visible = bool(tara_base_visible.value)
        wheel.root_frame.visible = visible
        wheel.robot.show_visual = visible and bool(wheel_visual.value)
        wheel.robot.show_collision = visible and bool(wheel_collision.value)

    print(f"Open http://localhost:{args.port}")
    print(f"T2 CSV root: {t2_csv_root}")
    print(f"G1 motion root: {g1_motion_root}")
    print(f"Wheel CSV root: {wheel_csv_root}")

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

        if g1_play.value:
            frame_idx = int((now - g1_start_time) * g1.fps * float(g1_speed.value))
            if frame_idx >= g1.length:
                if g1_loop.value:
                    g1_start_time = now
                    frame_idx = 0
                else:
                    frame_idx = g1.length - 1
                    g1_play.value = False
            g1_frame.value = frame_idx
            g1.apply_frame(frame_idx)

        if wheel_play.value:
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

        if t2_play.value and wheel_play.value:
            both_frame.value = min(int(t2_frame.value), int(wheel_frame.value), int(both_frame.max))

        time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
