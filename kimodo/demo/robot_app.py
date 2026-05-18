# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-focused Kimodo demo.

This app keeps the regular Kimodo authoring UI intact and adds a separate
SOMA -> T2 -> real robot workflow panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.server
import json
import os
import re
import threading
import time
import urllib.parse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import viser
import viser.transforms as tf
from plyfile import PlyData

from kimodo.demo.app import Demo
from kimodo.demo.config import DEFAULT_CUR_DURATION, DEFAULT_MODEL, NB_TRANSITION_FRAMES
from kimodo.exports.bvh import read_bvh_frame_time_seconds, save_motion_bvh
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import DEFAULT_TEXT_ENCODER_URL, resolve_model_name
from kimodo.motion_io import load_motion_file
from kimodo.retarget.soma_t2 import SomaT2RetargetJob, default_soma_retargeter_root
from kimodo.robot.t2_nero import T2NeroConnection, default_t2_stream_script
from kimodo.scripts.t2_csv_arm_publisher import ArmFrame, load_arm_frames
from kimodo.viz.tara_rig import T2ViewerMotion, load_tara_motion_csv


MEMORIES_ROOT = Path("/home/jony/Downloads/soma-retargeter/assets/motions")
WORLD_SCENES_ROOT = Path("/home/jony/Downloads/worlds")
DEFAULT_WORLD_SCENE_PATH = WORLD_SCENES_ROOT / "scene.ply"
TEXT_ENCODER_SERVER_COMMAND = "python -m kimodo.scripts.run_text_encoder_server"
GAUSSIAN_SH_C0 = 0.28209479177387814
WORLD_SCENE_ROTATION_DEG = np.array([-90.0, 90.0, 0.0], dtype=np.float64)
WORLD_SCENE_POSITION = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WORLD_SCENE_SCALE = 1.0
WORLD_SCENE_SCALE_LIMITS = (0.01, 10.0)
WORLD_SCENE_MOVE_X_LIMITS = (-5.0, 5.0)
WORLD_SCENE_HEIGHT_LIMITS = (-5.0, 5.0)
WORLD_SCENE_MOVE_Z_LIMITS = (-5.0, 15.0)
OFFICE_WORLD_SCENE_MOVE_LIMITS = (-50.0, 50.0)
OFFICE_WORLD_SCENE_HEIGHT_LIMITS = (-50.0, 50.0)
WORLD_OBJECT_PICK_MAX_DISTANCE = 0.18
WORLD_SCENE_PRESETS = {
    "scene.ply": {
        "scale": 0.75,
        "rotation_deg": np.array([-178.0, 84.0, 0.0], dtype=np.float64),
        "position": np.array([-2.39, 1.17, 3.77], dtype=np.float64),
    },
    "office_world.ply": {
        "scale": 1.0,
        "rotation_deg": np.array([0.0, 70.0, -180.0], dtype=np.float64),
        "position": np.array([16.64, -10.91, 21.29], dtype=np.float64),
    }
}
WORLD_SCENE_ALIASES = {
    "office": "office_world.ply",
    "office_world": "office_world.ply",
    "office-world": "office_world.ply",
    "home": "home.ply",
    "scene": "scene.ply",
    "default": "scene.ply",
}


def _world_scene_move_limits_for_path(path: Path | None) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if path is not None and path.name == "office_world.ply":
        return (
            OFFICE_WORLD_SCENE_MOVE_LIMITS,
            OFFICE_WORLD_SCENE_HEIGHT_LIMITS,
            OFFICE_WORLD_SCENE_MOVE_LIMITS,
        )
    return WORLD_SCENE_MOVE_X_LIMITS, WORLD_SCENE_HEIGHT_LIMITS, WORLD_SCENE_MOVE_Z_LIMITS


def _world_scene_wxyz(rotation_deg: np.ndarray) -> np.ndarray:
    rotation_rad = np.deg2rad(rotation_deg)
    return (
        tf.SO3.from_y_radians(float(rotation_rad[1]))
        @ tf.SO3.from_x_radians(float(rotation_rad[0]))
        @ tf.SO3.from_z_radians(float(rotation_rad[2]))
    ).wxyz


def _world_scene_transform_for_path(path: Path | None) -> tuple[float, np.ndarray, np.ndarray]:
    preset = WORLD_SCENE_PRESETS.get(path.name if path is not None else "")
    if preset is None:
        return WORLD_SCENE_SCALE, WORLD_SCENE_ROTATION_DEG.copy(), WORLD_SCENE_POSITION.copy()
    return (
        float(preset["scale"]),
        np.asarray(preset["rotation_deg"], dtype=np.float64).copy(),
        np.asarray(preset["position"], dtype=np.float64).copy(),
    )


def _configure_text_encoder_runtime(text_encoder_mode: str | None, text_encoder_url: str | None) -> str:
    if text_encoder_url:
        os.environ["TEXT_ENCODER_URL"] = text_encoder_url

    selected_mode = text_encoder_mode or os.environ.get("TEXT_ENCODER_MODE") or "api"
    os.environ["TEXT_ENCODER_MODE"] = selected_mode
    return selected_mode


def _text_encoder_startup_error(mode: str) -> RuntimeError:
    url = os.environ.get("TEXT_ENCODER_URL", DEFAULT_TEXT_ENCODER_URL)
    if mode != "api":
        return RuntimeError(f"Failed to start Kimodo robot demo with TEXT_ENCODER_MODE={mode!r}.")
    return RuntimeError(
        "The Kimodo robot demo needs the text encoder server, but it is not reachable.\n\n"
        f"Start it in another terminal with:\n  {TEXT_ENCODER_SERVER_COMMAND}\n\n"
        f"Then run this app again. Current TEXT_ENCODER_URL is {url!r}.\n\n"
        "If you intentionally want the old in-process 8B LLM2Vec fallback, run with "
        "`--text-encoder-mode auto` or set `TEXT_ENCODER_MODE=local`, but that can be killed by the OS "
        "on machines without enough free memory."
    )


def _safe_clip_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._-")
    return value or "kimodo_motion"


def _default_output_root() -> Path:
    return MEMORIES_ROOT.expanduser().resolve()


def _scan_memory_stems(memories_root: Path) -> list[str]:
    bvh_root = memories_root / "bvh"
    csv_root = memories_root / "t2_csv"
    stems: set[str] = set()
    if bvh_root.is_dir():
        stems.update(str(path.relative_to(bvh_root).with_suffix("")) for path in bvh_root.rglob("*.bvh"))
    if csv_root.is_dir():
        stems.update(str(path.relative_to(csv_root).with_suffix("")) for path in csv_root.rglob("*.csv"))
    return sorted(stems)


def _memory_bvh_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "bvh" / Path(stem).with_suffix(".bvh")


def _memory_csv_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "t2_csv" / Path(stem).with_suffix(".csv")


def _memory_label(memories_root: Path, stem: str) -> str:
    has_bvh = _memory_bvh_path(memories_root, stem).is_file()
    has_csv = _memory_csv_path(memories_root, stem).is_file()
    if has_bvh and has_csv:
        state = "bvh+csv"
    elif has_bvh:
        state = "needs csv"
    elif has_csv:
        state = "csv only"
    else:
        state = "missing"
    return f"[{state}] {stem}"


def _stem_from_memory_label(label: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", str(label)).strip()


def _normalize_memory_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _resolve_memory_stem(requested: str, memories_root: Path) -> str:
    requested = _stem_from_memory_label(requested).strip().removesuffix(".bvh")
    if not requested:
        raise ValueError("Missing memory name")

    stems = _scan_memory_stems(memories_root)
    if not stems:
        raise FileNotFoundError(f"No BVH memories found in {memories_root / 'bvh'}")

    candidates = [requested]
    if requested.startswith("kimodo_"):
        candidates.append(f"generated/{requested}")

    exact = {stem: stem for stem in stems}
    lower = {stem.lower(): stem for stem in stems}
    names = {Path(stem).name.lower(): stem for stem in stems}

    for candidate in candidates:
        if candidate in exact:
            return exact[candidate]
        lowered = candidate.lower()
        if lowered in lower:
            return lower[lowered]
        if lowered in names:
            return names[lowered]

    normalized_requested = _normalize_memory_search(requested)
    normalized_matches = [
        stem
        for stem in stems
        if normalized_requested
        and (
            normalized_requested == _normalize_memory_search(stem)
            or normalized_requested == _normalize_memory_search(Path(stem).name)
        )
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    contains_matches = [
        stem
        for stem in stems
        if requested.lower() in stem.lower()
        or requested.lower() in Path(stem).name.lower()
        or (
            normalized_requested
            and normalized_requested in _normalize_memory_search(stem)
        )
    ]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1 or len(normalized_matches) > 1:
        matches = sorted(set(normalized_matches + contains_matches))
        raise ValueError(f"Ambiguous memory name. Matches: {', '.join(matches)}")

    available = ", ".join(stems) or "<none>"
    raise FileNotFoundError(f"Could not find memory {requested!r}. Available memories: {available}")


def _new_generated_stem() -> str:
    return f"generated/kimodo_{uuid.uuid4().hex[:10]}"


def _model_native_fps(demo: Demo, model_name: str, fallback: float) -> float:
    bundle = demo.models.get(model_name)
    if bundle is not None and bundle.model_fps and bundle.model_fps > 0.0:
        return float(bundle.model_fps)
    return float(fallback)


def _scan_world_scene_paths(worlds_root: Path) -> list[Path]:
    worlds_root = worlds_root.expanduser().resolve()
    if not worlds_root.is_dir():
        return []
    return sorted(path for path in worlds_root.rglob("*.ply") if path.is_file())


def _world_scene_label(path: Path, worlds_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(worlds_root.expanduser().resolve()))
    except ValueError:
        return str(path)


def _resolve_world_scene_path(requested: str, worlds_root: Path = WORLD_SCENES_ROOT) -> Path:
    worlds_root = worlds_root.expanduser().resolve()
    paths = _scan_world_scene_paths(worlds_root)
    labels = {_world_scene_label(path, worlds_root): path for path in paths}
    names = {path.name: path for path in paths}

    normalized = requested.strip()
    alias = WORLD_SCENE_ALIASES.get(normalized.lower(), normalized)
    candidates = [alias]
    if not alias.endswith(".ply"):
        candidates.append(f"{alias}.ply")

    for candidate in candidates:
        if candidate in labels:
            return labels[candidate]
        if candidate in names:
            return names[candidate]

    contains = [
        path
        for label, path in labels.items()
        if normalized.lower() in label.lower() or alias.lower() in label.lower()
    ]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        matches = ", ".join(_world_scene_label(path, worlds_root) for path in contains)
        raise ValueError(f"Ambiguous world name. Matches: {matches}")

    available = ", ".join(sorted(labels)) or "<none>"
    raise ValueError(f"World not found: {requested!r}. Available: {available}")


def _load_world_scene(path: Path) -> dict[str, np.ndarray | str]:
    """Load a world PLY without touching the human/robot scene nodes."""
    path = path.expanduser().resolve()
    plydata = PlyData.read(path)
    vertex_data = plydata["vertex"]
    vertex_fields = set(vertex_data.data.dtype.names or ())
    positions = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=-1).astype(np.float32)

    gaussian_fields = {
        "scale_0",
        "scale_1",
        "scale_2",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    if gaussian_fields.issubset(vertex_fields):
        scales = np.exp(
            np.stack([vertex_data["scale_0"], vertex_data["scale_1"], vertex_data["scale_2"]], axis=-1)
        ).astype(np.float32)
        wxyzs = np.stack(
            [vertex_data["rot_0"], vertex_data["rot_1"], vertex_data["rot_2"], vertex_data["rot_3"]],
            axis=1,
        ).astype(np.float32)
        colors = (
            0.5
            + GAUSSIAN_SH_C0
            * np.stack([vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"]], axis=1)
        ).astype(np.float32)
        colors = np.clip(colors, 0.0, 1.0)
        opacities = (1.0 / (1.0 + np.exp(-vertex_data["opacity"][:, None]))).astype(np.float32)
        rotations = tf.SO3(wxyzs).as_matrix().astype(np.float32)
        covariances = np.einsum(
            "nij,njk,nlk->nil",
            rotations,
            np.eye(3, dtype=np.float32)[None, :, :] * scales[:, None, :] ** 2,
            rotations,
            optimize=True,
        ).astype(np.float32)
        return {
            "kind": "gaussian_splats",
            "centers": positions,
            "rgbs": colors,
            "opacities": opacities,
            "covariances": covariances,
        }

    rgb_fields = {"red", "green", "blue"}
    if rgb_fields.issubset(vertex_fields):
        colors = np.stack([vertex_data["red"], vertex_data["green"], vertex_data["blue"]], axis=-1).astype(np.uint8)
    else:
        colors = np.array([170, 170, 170], dtype=np.uint8)
    return {"kind": "point_cloud", "points": positions, "colors": colors}


def _csv_motion_summary(csv_path: Path) -> str:
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return "Frames: `0`"

    def span(columns: list[str]) -> float:
        values = []
        for row in rows:
            for column in columns:
                if column in row:
                    values.append(float(row[column]))
        if not values:
            return 0.0
        return max(values) - min(values)

    root_spans_cm = [
        span(["root_translateX"]),
        span(["root_translateY"]),
        span(["root_translateZ"]),
    ]
    right_span_deg = span([f"right_joint{i}_dof" for i in range(1, 8)])
    left_span_deg = span([f"left_joint{i}_dof" for i in range(1, 8)])
    body_span_deg = span(
        [
            "waist_yaw_joint_dof",
            "waist_roll_joint_dof",
            "waist_pitch_joint_dof",
            "left_hip_pitch_joint_dof",
            "left_hip_roll_joint_dof",
            "left_hip_yaw_joint_dof",
            "left_knee_joint_dof",
            "left_ankle_roll_joint_dof",
            "left_ankle_pitch_joint_dof",
            "right_hip_pitch_joint_dof",
            "right_hip_roll_joint_dof",
            "right_hip_yaw_joint_dof",
            "right_knee_joint_dof",
            "right_ankle_roll_joint_dof",
            "right_ankle_pitch_joint_dof",
        ]
    )
    return (
        f"Frames: `{len(rows)}`\n\n"
        "Root span XYZ: "
        f"`{root_spans_cm[0] / 100.0:.2f}, "
        f"{root_spans_cm[1] / 100.0:.2f}, "
        f"{root_spans_cm[2] / 100.0:.2f} m`\n\n"
        f"Right arm span: `{right_span_deg:.1f} deg`\n\n"
        f"Left arm span: `{left_span_deg:.1f} deg`\n\n"
        f"Body/leg span: `{body_span_deg:.1f} deg`"
    )


@dataclass
class RobotWorkflowState:
    output_root: Path = field(default_factory=_default_output_root)
    memory_stem: str | None = None
    bvh_path: Path | None = None
    npz_path: Path | None = None
    csv_path: Path | None = None
    t2_motion: T2ViewerMotion | None = None
    arm_frames: list[ArmFrame] = field(default_factory=list)
    connection: T2NeroConnection = field(default_factory=T2NeroConnection)
    status_markdown: viser.GuiMarkdownHandle | None = None
    robot_markdown: viser.GuiMarkdownHandle | None = None
    retarget_running: bool = False
    placing_object: bool = False
    placed_object_handles: list[viser.SceneHandle] = field(default_factory=list)
    placed_object_count: int = 0

    def clear_t2_preview(self) -> None:
        if self.t2_motion is not None:
            self.t2_motion.clear()
            self.t2_motion = None
        self.arm_frames.clear()
        self.csv_path = None

    def clear_placed_objects(self) -> None:
        for handle in self.placed_object_handles:
            handle.remove()
        self.placed_object_handles.clear()
        self.placed_object_count = 0


class RobotDemo(Demo):
    """Kimodo demo with a robot production workflow mounted as a separate panel."""

    def __init__(
        self,
        default_model_name: str = DEFAULT_MODEL,
        world_scene_path: Path | None = DEFAULT_WORLD_SCENE_PATH,
    ):
        super().__init__(default_model_name=default_model_name)
        self.robot_workflows: dict[int, RobotWorkflowState] = {}
        self.world_scene_path = world_scene_path.expanduser().resolve() if world_scene_path is not None else None
        self._world_scene_data: dict[str, np.ndarray | str] | None = None
        self._world_scene_data_path: Path | None = None
        self._world_scene_lock = threading.Lock()
        self.world_scene_handles: dict[int, viser.SceneHandle] = {}
        self.world_scene_client_transforms: dict[int, tuple[Path, float, np.ndarray, np.ndarray]] = {}
        self._world_scene_sync_callbacks: dict[int, Callable[[Path], None]] = {}
        self._memory_sync_callbacks: dict[int, Callable[[str], str]] = {}
        self._memory_list_callbacks: dict[int, Callable[[], dict[str, object]]] = {}
        self._generate_retarget_callbacks: dict[int, Callable[[dict[str, object]], dict[str, object]]] = {}
        self._control_jobs: dict[str, dict[str, object]] = {}
        self._control_jobs_lock = threading.Lock()
        self._control_server: http.server.ThreadingHTTPServer | None = None

    def _setup_demo_for_client(self, client: viser.ClientHandle) -> None:
        super()._setup_demo_for_client(client)
        self._add_world_scene_to_client(client)
        self._create_world_scene_gui(client)
        self._hide_examples_folder(client)
        self.robot_workflows[client.client_id] = RobotWorkflowState()
        self._create_robot_pipeline_gui(client)

    def start_control_server(self, host: str, port: int) -> None:
        demo = self

        class ControlHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args: object) -> None:
                return

            def _send_json(self, status: int, payload: dict[str, object]) -> None:
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _request_payload(self) -> dict[str, object]:
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                payload: dict[str, object] = {
                    key: values[-1] for key, values in query.items() if values
                }
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    body = self.rfile.read(length)
                    if body:
                        body_payload = json.loads(body.decode("utf-8"))
                        if isinstance(body_payload, dict):
                            payload.update(body_payload)
                return payload

            def _load_memory(self) -> None:
                payload = self._request_payload()
                requested = str(payload.get("memory") or payload.get("stem") or "").strip()
                if not requested:
                    self._send_json(400, {"ok": False, "error": "Missing memory name"})
                    return
                callbacks = list(demo._memory_sync_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                errors: list[str] = []
                loaded_clients: list[int] = []
                loaded_stems: list[str] = []
                for client_id, sync_memory in callbacks:
                    try:
                        loaded_stems.append(sync_memory(requested))
                        loaded_clients.append(client_id)
                    except Exception as exc:
                        errors.append(f"{client_id}: {exc}")
                status = 200 if not errors else 500
                self._send_json(
                    status,
                    {
                        "ok": not errors,
                        "stem": loaded_stems[0] if loaded_stems else requested,
                        "requested": requested,
                        "loaded_stems": loaded_stems,
                        "loaded_clients": loaded_clients,
                        "errors": errors,
                    },
                )

            def _load_world(self) -> None:
                payload = self._request_payload()
                requested = str(payload.get("world") or payload.get("path") or "").strip()
                if not requested:
                    self._send_json(400, {"ok": False, "error": "Missing world name"})
                    return
                try:
                    path = _resolve_world_scene_path(requested)
                except Exception as exc:
                    self._send_json(404, {"ok": False, "error": str(exc)})
                    return
                callbacks = list(demo._world_scene_sync_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                errors: list[str] = []
                loaded_clients: list[int] = []
                for client_id, sync_world in callbacks:
                    try:
                        sync_world(path)
                        loaded_clients.append(client_id)
                    except Exception as exc:
                        errors.append(f"{client_id}: {exc}")
                status = 200 if not errors else 500
                self._send_json(
                    status,
                    {
                        "ok": not errors,
                        "world": _world_scene_label(path, WORLD_SCENES_ROOT),
                        "path": str(path),
                        "loaded_clients": loaded_clients,
                        "errors": errors,
                    },
                )

            def _worlds(self) -> None:
                worlds_root = WORLD_SCENES_ROOT.expanduser().resolve()
                worlds = [
                    _world_scene_label(path, worlds_root)
                    for path in _scan_world_scene_paths(worlds_root)
                ]
                self._send_json(200, {"ok": True, "worlds_root": str(worlds_root), "worlds": worlds})

            def _memories(self) -> None:
                callbacks = list(demo._memory_list_callbacks.items())
                if callbacks:
                    clients = []
                    for client_id, list_memories in callbacks:
                        payload = list_memories()
                        payload["client_id"] = client_id
                        clients.append(payload)
                    self._send_json(200, {"ok": True, "clients": clients})
                    return

                memories_root = MEMORIES_ROOT.expanduser().resolve()
                stems = _scan_memory_stems(memories_root)
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "memories_root": str(memories_root),
                        "stems": stems,
                        "labels": [_memory_label(memories_root, stem) for stem in stems],
                    },
                )

            def _generate_retarget(self) -> None:
                payload = self._request_payload()
                prompt = str(payload.get("prompt") or "").strip()
                if not prompt:
                    self._send_json(400, {"ok": False, "error": "Missing prompt"})
                    return
                callbacks = list(demo._generate_retarget_callbacks.items())
                if not callbacks:
                    self._send_json(503, {"ok": False, "error": "No connected Viser clients"})
                    return
                client_id, generate_retarget = callbacks[0]
                try:
                    response = generate_retarget(payload)
                except Exception as exc:
                    self._send_json(500, {"ok": False, "error": str(exc), "client_id": client_id})
                    return
                response["client_id"] = client_id
                self._send_json(202, {"ok": True, **response})

            def _job(self) -> None:
                payload = self._request_payload()
                job_id = str(payload.get("job_id") or payload.get("id") or "").strip()
                if not job_id:
                    self._send_json(400, {"ok": False, "error": "Missing job_id"})
                    return
                with demo._control_jobs_lock:
                    job = dict(demo._control_jobs.get(job_id) or {})
                if not job:
                    self._send_json(404, {"ok": False, "error": f"Unknown job_id: {job_id}"})
                    return
                self._send_json(200, {"ok": True, "job": job})

            def _client_status(self) -> list[dict[str, object]]:
                clients: list[dict[str, object]] = []
                for client_id in sorted(demo.client_sessions.keys()):
                    session = demo.client_sessions.get(client_id)
                    workflow = demo.robot_workflows.get(client_id)
                    world_transform = demo.world_scene_client_transforms.get(client_id)
                    current_world = None
                    if world_transform is not None:
                        current_world = _world_scene_label(world_transform[0], WORLD_SCENES_ROOT)
                    if session is None:
                        continue
                    clients.append(
                        {
                            "client_id": client_id,
                            "current_memory": workflow.memory_stem if workflow is not None else None,
                            "current_world": current_world,
                            "playing": bool(session.playing),
                            "frame": int(session.frame_idx),
                            "max_frame": int(session.max_frame_idx),
                            "model_fps": float(session.model_fps),
                            "duration_seconds": float(session.cur_duration),
                            "has_t2_preview": bool(workflow is not None and workflow.t2_motion is not None),
                            "has_robot_frames": bool(workflow is not None and workflow.arm_frames),
                        }
                    )
                return clients

            def do_GET(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                if path == "/status":
                    clients = self._client_status()
                    self._send_json(
                        200,
                        {
                            "ok": True,
                            "memory_clients": sorted(demo._memory_sync_callbacks.keys()),
                            "world_clients": sorted(demo._world_scene_sync_callbacks.keys()),
                            "clients": clients,
                        },
                    )
                    return
                if path == "/worlds":
                    self._worlds()
                    return
                if path == "/memories":
                    self._memories()
                    return
                if path == "/job":
                    self._job()
                    return
                if path == "/load-memory":
                    self._load_memory()
                    return
                if path == "/load-world":
                    self._load_world()
                    return
                if path == "/generate-retarget":
                    self._generate_retarget()
                    return
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

            def do_POST(self) -> None:
                path = urllib.parse.urlparse(self.path).path
                if path == "/load-memory":
                    self._load_memory()
                    return
                if path == "/load-world":
                    self._load_world()
                    return
                if path == "/generate-retarget":
                    self._generate_retarget()
                    return
                self._send_json(404, {"ok": False, "error": f"Unknown endpoint: {path}"})

        self._control_server = http.server.ThreadingHTTPServer((host, port), ControlHandler)
        thread = threading.Thread(
            target=self._control_server.serve_forever,
            name="kimodo-control-server",
            daemon=True,
        )
        thread.start()
        print(f"Kimodo control server listening on http://{host}:{port}")

    def _get_world_scene_data(self, world_scene_path: Path | None = None) -> dict[str, np.ndarray | str] | None:
        path = (world_scene_path or self.world_scene_path)
        if path is None:
            return None
        path = path.expanduser().resolve()
        if not path.is_file():
            return None
        with self._world_scene_lock:
            if self._world_scene_data is None or self._world_scene_data_path != path:
                self._world_scene_data = _load_world_scene(path)
                self._world_scene_data_path = path
            return self._world_scene_data

    def _add_world_scene_handle(
        self,
        client: viser.ClientHandle,
        world_scene_data: dict[str, np.ndarray | str],
        rotation_deg: np.ndarray,
        position: np.ndarray,
        scale: float,
    ) -> viser.SceneHandle:
        wxyz = _world_scene_wxyz(rotation_deg)
        scale = float(scale)
        if world_scene_data["kind"] == "gaussian_splats":
            return client.scene.add_gaussian_splats(
                "/world/scene",
                centers=world_scene_data["centers"] * scale,
                rgbs=world_scene_data["rgbs"],
                opacities=world_scene_data["opacities"],
                covariances=world_scene_data["covariances"] * (scale * scale),
                wxyz=wxyz,
                position=position,
            )
        return client.scene.add_point_cloud(
            "/world/scene",
            points=world_scene_data["points"] * scale,
            colors=world_scene_data["colors"],
            point_size=max(0.001, 0.02 * scale),
            point_shape="circle",
            precision="float32",
            wxyz=wxyz,
            position=position,
        )

    def _add_world_scene_to_client(self, client: viser.ClientHandle) -> None:
        try:
            world_scene_data = self._get_world_scene_data()
            if world_scene_data is None:
                return
            scale, rotation_deg, position = _world_scene_transform_for_path(self.world_scene_path)
            handle = self._add_world_scene_handle(
                client,
                world_scene_data,
                rotation_deg,
                position,
                scale,
            )
            self.world_scene_handles[client.client_id] = handle
            self.world_scene_client_transforms[client.client_id] = (
                self.world_scene_path,
                scale,
                rotation_deg.copy(),
                position.copy(),
            )
            client.add_notification(
                title="World loaded",
                body=str(self.world_scene_path),
                auto_close_seconds=4.0,
                color="green",
            )
        except Exception as exc:
            print(f"Failed to load world scene {self.world_scene_path}: {exc}")
            client.add_notification(
                title="World load failed",
                body=str(exc),
                auto_close_seconds=8.0,
                color="red",
            )

    def _create_world_scene_gui(self, client: viser.ClientHandle) -> None:
        world_handle = self.world_scene_handles.get(client.client_id)
        if world_handle is None:
            return
        worlds_root = WORLD_SCENES_ROOT.expanduser().resolve()
        world_paths = _scan_world_scene_paths(worlds_root)
        world_labels = [_world_scene_label(path, worlds_root) for path in world_paths] if world_paths else ["<no worlds>"]
        selected_world_path = self.world_scene_path
        if selected_world_path is not None:
            selected_world_path = selected_world_path.expanduser().resolve()
        initial_world_label = (
            _world_scene_label(selected_world_path, worlds_root)
            if selected_world_path in world_paths
            else world_labels[0]
        )
        initial_world_path = selected_world_path if selected_world_path in world_paths else (world_paths[0] if world_paths else None)
        initial_scale, initial_rotation_deg, initial_position = _world_scene_transform_for_path(initial_world_path)
        initial_x_limits, initial_y_limits, initial_z_limits = _world_scene_move_limits_for_path(initial_world_path)

        with client.gui.add_folder("World Scene", expand_by_default=True):
            worlds_root_text = client.gui.add_text("Root", initial_value=str(worlds_root))
            world_dropdown = client.gui.add_dropdown(
                "World",
                options=world_labels,
                initial_value=initial_world_label,
            )
            refresh_worlds_button = client.gui.add_button("Refresh Worlds")
            world_scale = client.gui.add_slider(
                "World Scale",
                min=WORLD_SCENE_SCALE_LIMITS[0],
                max=WORLD_SCENE_SCALE_LIMITS[1],
                step=0.01,
                initial_value=initial_scale,
            )
            x_rotation = client.gui.add_slider(
                "Pitch X",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_rotation_deg[0]),
            )
            y_rotation = client.gui.add_slider(
                "Turn Y",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_rotation_deg[1]),
            )
            z_rotation = client.gui.add_slider(
                "Roll Z",
                min=-180.0,
                max=180.0,
                step=1.0,
                initial_value=float(initial_rotation_deg[2]),
            )
            x_position = client.gui.add_slider(
                "Move X",
                min=initial_x_limits[0],
                max=initial_x_limits[1],
                step=0.01,
                initial_value=float(initial_position[0]),
            )
            y_position = client.gui.add_slider(
                "Height Y",
                min=initial_y_limits[0],
                max=initial_y_limits[1],
                step=0.01,
                initial_value=float(initial_position[1]),
            )
            z_position = client.gui.add_slider(
                "Move Z",
                min=initial_z_limits[0],
                max=initial_z_limits[1],
                step=0.01,
                initial_value=float(initial_position[2]),
            )

        applying_world_preset = False

        def current_worlds_root() -> Path:
            return Path(worlds_root_text.value).expanduser().resolve()

        def current_rotation_deg() -> np.ndarray:
            return np.array(
                [x_rotation.value, y_rotation.value, z_rotation.value],
                dtype=np.float64,
            )

        def current_position() -> np.ndarray:
            return np.array(
                [x_position.value, y_position.value, z_position.value],
                dtype=np.float64,
            )

        def selected_world() -> Path | None:
            value = str(world_dropdown.value)
            if value == "<no worlds>":
                return None
            return (current_worlds_root() / value).expanduser().resolve()

        def apply_move_limits(path: Path | None) -> None:
            x_limits, y_limits, z_limits = _world_scene_move_limits_for_path(path)
            x_position.min = x_limits[0]
            x_position.max = x_limits[1]
            y_position.min = y_limits[0]
            y_position.max = y_limits[1]
            z_position.min = z_limits[0]
            z_position.max = z_limits[1]

        def apply_world_preset(path: Path) -> None:
            scale, rotation_deg, position = _world_scene_transform_for_path(path)
            apply_move_limits(path)
            world_scale.value = scale
            x_rotation.value = float(rotation_deg[0])
            y_rotation.value = float(rotation_deg[1])
            z_rotation.value = float(rotation_deg[2])
            x_position.value = float(position[0])
            y_position.value = float(position[1])
            z_position.value = float(position[2])

        def reload_world(path: Path) -> None:
            nonlocal world_handle
            world_scene_data = self._get_world_scene_data(path)
            if world_scene_data is None:
                raise FileNotFoundError(path)
            world_handle.remove()
            world_handle = self._add_world_scene_handle(
                client,
                world_scene_data,
                current_rotation_deg(),
                current_position(),
                float(world_scale.value),
            )
            self.world_scene_handles[client.client_id] = world_handle
            self.world_scene_path = path.expanduser().resolve()
            self.world_scene_client_transforms[client.client_id] = (
                self.world_scene_path,
                float(world_scale.value),
                current_rotation_deg(),
                current_position(),
            )

        def sync_world_from_external_change(path: Path) -> None:
            nonlocal applying_world_preset
            root = current_worlds_root()
            label = _world_scene_label(path, root)
            try:
                applying_world_preset = True
                if label not in world_dropdown.options:
                    refresh_world_options(path)
                if label in world_dropdown.options:
                    world_dropdown.value = label
                apply_world_preset(path)
                reload_world(path)
            finally:
                applying_world_preset = False

        self._world_scene_sync_callbacks[client.client_id] = sync_world_from_external_change

        def refresh_world_options(select_path: Path | None = None) -> None:
            root = current_worlds_root()
            paths = _scan_world_scene_paths(root)
            labels = [_world_scene_label(path, root) for path in paths] if paths else ["<no worlds>"]
            world_dropdown.options = labels
            if select_path is not None and select_path.resolve() in [path.resolve() for path in paths]:
                world_dropdown.value = _world_scene_label(select_path, root)
            elif str(world_dropdown.value) not in labels:
                world_dropdown.value = labels[0]

        def update_world_transform() -> None:
            world_handle.wxyz = _world_scene_wxyz(current_rotation_deg())
            world_handle.position = current_position()
            path = selected_world()
            if path is not None:
                self.world_scene_client_transforms[client.client_id] = (
                    path.expanduser().resolve(),
                    float(world_scale.value),
                    current_rotation_deg(),
                    current_position(),
                )

        @world_scale.on_update
        def _(event: viser.GuiEvent) -> None:
            if applying_world_preset:
                return
            path = selected_world()
            if path is None:
                return
            try:
                reload_world(path)
            except Exception as exc:
                event.client.add_notification(
                    title="World scale failed",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @world_dropdown.on_update
        def _(event: viser.GuiEvent) -> None:
            nonlocal applying_world_preset
            if applying_world_preset:
                return
            path = selected_world()
            if path is None:
                return
            try:
                applying_world_preset = True
                apply_world_preset(path)
                applying_world_preset = False
                reload_world(path)
                event.client.add_notification(
                    title="World loaded",
                    body=str(path),
                    auto_close_seconds=4.0,
                    color="green",
                )
                for client_id, sync_world in list(self._world_scene_sync_callbacks.items()):
                    if client_id == client.client_id:
                        continue
                    try:
                        sync_world(path)
                    except Exception as exc:
                        print(f"Failed to sync world scene for client {client_id}: {exc}")
            except Exception as exc:
                if event.client is not None:
                    event.client.add_notification(
                        title="World load failed",
                        body=str(exc),
                        auto_close_seconds=8.0,
                        color="red",
                    )
            finally:
                applying_world_preset = False

        @refresh_worlds_button.on_click
        def _(event: viser.GuiEvent) -> None:
            refresh_world_options(selected_world())
            event.client.add_notification(
                title="Worlds refreshed",
                body=str(current_worlds_root()),
                auto_close_seconds=3.0,
                color="blue",
            )

        for control in (x_rotation, y_rotation, z_rotation, x_position, y_position, z_position):
            control.on_update(lambda _event: None if applying_world_preset else update_world_transform())

    @staticmethod
    def _hide_examples_folder(client: viser.ClientHandle) -> None:
        root = client.gui._container_handle_from_uuid.get("root")
        if root is None:
            return
        for child in list(root._children.values()):
            props = getattr(getattr(child, "_impl", None), "props", None)
            if getattr(props, "label", None) == "Examples":
                child.visible = False

    def on_client_disconnect(self, client: viser.ClientHandle) -> None:
        workflow = self.robot_workflows.pop(client.client_id, None)
        if workflow is not None:
            workflow.connection.disconnect()
            workflow.clear_t2_preview()
            workflow.clear_placed_objects()
        self.world_scene_handles.pop(client.client_id, None)
        self.world_scene_client_transforms.pop(client.client_id, None)
        self._world_scene_sync_callbacks.pop(client.client_id, None)
        self._memory_sync_callbacks.pop(client.client_id, None)
        self._memory_list_callbacks.pop(client.client_id, None)
        self._generate_retarget_callbacks.pop(client.client_id, None)
        super().on_client_disconnect(client)

    def set_frame(self, client_id: int, frame_idx: int, update_timeline: bool = True):
        super().set_frame(client_id, frame_idx, update_timeline=update_timeline)
        workflow = self.robot_workflows.get(client_id)
        if workflow is None:
            return
        if workflow.t2_motion is not None:
            workflow.t2_motion.set_frame(frame_idx)
        if workflow.connection.is_connected() and workflow.arm_frames:
            try:
                workflow.connection.send_frame(self._arm_frame_for_index(workflow, frame_idx))
            except Exception as exc:
                workflow.connection.disconnect()
                if workflow.robot_markdown is not None:
                    workflow.robot_markdown.content = f"Streaming stopped.\n\n`{exc}`"

    def _create_robot_pipeline_gui(self, client: viser.ClientHandle) -> None:
        workflow = self.robot_workflows[client.client_id]
        memories_root_default = workflow.output_root
        memory_stems = _scan_memory_stems(memories_root_default)
        memory_labels = (
            [_memory_label(memories_root_default, stem) for stem in memory_stems]
            if memory_stems
            else ["<no memories>"]
        )

        with client.gui.add_folder("Memories", expand_by_default=True):
            memories_root_text = client.gui.add_text("Root", initial_value=str(memories_root_default))
            memory_dropdown = client.gui.add_dropdown(
                "Memory",
                options=memory_labels,
                initial_value=memory_labels[0],
            )
            memory_status = client.gui.add_markdown("Select a memory.")
            refresh_memories_button = client.gui.add_button("Refresh Memories")
            load_memory_button = client.gui.add_button("Load Memory")
            retarget_memory_button = client.gui.add_button("Retarget Memory to T2", color="green")
            save_generated_memory_button = client.gui.add_button(
                "Save Memories",
                color="blue",
                hint="Save the current SOMA motion into the BVH memories folder.",
            )

        with client.gui.add_folder("Preview Visibility", expand_by_default=True):
            show_human_checkbox = client.gui.add_checkbox(
                "Show Human",
                initial_value=self.client_sessions[client.client_id].gui_elements.gui_viz_skinned_mesh_checkbox.value,
            )
            show_t2_checkbox = client.gui.add_checkbox("Show T2 Robot", initial_value=True)
            show_floor_grid_checkbox = client.gui.add_checkbox("Show Floor Grid", initial_value=False)

        with client.gui.add_folder("Scene Objects", expand_by_default=True):
            object_shape_dropdown = client.gui.add_dropdown(
                "Object",
                options=["Sphere", "Cylinder", "Box", "Water Bottle"],
                initial_value="Sphere",
            )
            place_object_button = client.gui.add_button("Place Object", color="green")
            object_position_markdown = client.gui.add_markdown(
                "Click **Place Object**, then click the Kimodo/Viser scene."
            )
            clear_objects_button = client.gui.add_button("Clear Objects", color="red")

        with client.gui.add_folder("Sync to Real Robot", expand_by_default=True):
            workflow.robot_markdown = client.gui.add_markdown("Disconnected.")
            dry_run_checkbox = client.gui.add_checkbox(
                "Dry run",
                initial_value=True,
                hint="Print streamed frames instead of publishing ROS commands.",
            )
            connect_button = client.gui.add_button("Connect")
            disconnect_button = client.gui.add_button("Disconnect")
            send_frame_button = client.gui.add_button("Send Current Frame")
            play_robot_button = client.gui.add_button("Play on Robot", color="green")
            stop_robot_button = client.gui.add_button("Stop Robot", color="red")

        with client.gui.add_folder("Prompt-to-Robot Job", expand_by_default=True):
            prompt_job_markdown = client.gui.add_markdown("No prompt job running.")
        prompt_job_status_targets: list[viser.GuiMarkdownHandle] = [prompt_job_markdown]

        with client.gui.add_folder("Robot Pipeline", expand_by_default=False):
            workflow.status_markdown = client.gui.add_markdown("No robot artifact generated yet.")
            clip_name_text = client.gui.add_text("Clip Name", initial_value="kimodo_motion")
            output_root_text = client.gui.add_text("Output Root", initial_value=str(workflow.output_root))
            retargeter_root_text = client.gui.add_text(
                "soma-retargeter",
                initial_value=str(default_soma_retargeter_root()),
            )
            conda_env_text = client.gui.add_text("Conda Env", initial_value="soma-retargeter")
            standard_tpose_checkbox = client.gui.add_checkbox(
                "Standard T-pose BVH",
                initial_value=False,
                hint="Use Kimodo's standard SOMA T-pose when exporting the BVH for retargeting.",
            )
            arms_only_checkbox = client.gui.add_checkbox(
                "Freeze T2 root/body",
                initial_value=False,
                hint="When enabled, preview only arms/grippers and keep the T2 body at frame 0.",
            )

            save_bvh_button = client.gui.add_button("Save SOMA BVH", color="blue")
            retarget_button = client.gui.add_button("Retarget to T2", color="green")
            csv_path_text = client.gui.add_text("T2 CSV", initial_value="")
            load_t2_button = client.gui.add_button("Load T2 Preview")

        def update_status(message: str) -> None:
            if workflow.status_markdown is not None:
                workflow.status_markdown.content = message
            client.flush()

        def update_prompt_job(message: str) -> None:
            for target in list(prompt_job_status_targets):
                try:
                    target.content = message
                except Exception:
                    prompt_job_status_targets.remove(target)
            client.flush()

        def update_robot_status(message: str) -> None:
            if workflow.robot_markdown is not None:
                workflow.robot_markdown.content = message

        def current_memories_root() -> Path:
            return Path(memories_root_text.value).expanduser().resolve()

        def selected_memory_stem() -> str | None:
            value = str(memory_dropdown.value)
            if value == "<no memories>":
                return None
            return _stem_from_memory_label(value)

        def refresh_memory_options(select_stem: str | None = None) -> list[str]:
            root = current_memories_root()
            stems = _scan_memory_stems(root)
            labels = [_memory_label(root, stem) for stem in stems] if stems else ["<no memories>"]
            memory_dropdown.options = labels
            if select_stem is not None and select_stem in stems:
                memory_dropdown.value = _memory_label(root, select_stem)
            elif str(memory_dropdown.value) not in labels:
                memory_dropdown.value = labels[0]
            update_memory_status()
            return stems

        def set_human_mesh_visible(visible: bool) -> None:
            session = self.client_sessions[client.client_id]
            session.gui_elements.gui_viz_skinned_mesh_checkbox.value = visible
            for motion in session.motions.values():
                motion.character.set_skinned_mesh_visibility(visible)

        def set_t2_visible(visible: bool) -> None:
            if workflow.t2_motion is not None:
                workflow.t2_motion.set_mesh_visibility(visible)

        def set_floor_grid_visible(visible: bool) -> None:
            grid_handle = self.grid_handles.get(client.client_id)
            if grid_handle is not None:
                grid_handle.visible = visible

        show_floor_grid_checkbox.value = False
        set_floor_grid_visible(False)

        def format_scene_position(position: np.ndarray) -> str:
            return f"X `{position[0]:.3f}`, Y `{position[1]:.3f}`, Z `{position[2]:.3f}`"

        def pointer_floor_intersection(
            origin: np.ndarray,
            direction: np.ndarray,
        ) -> np.ndarray | None:
            if abs(float(direction[1])) < 1e-8:
                return None
            t = -float(origin[1]) / float(direction[1])
            if t < 0.0:
                return None
            point = origin + t * direction
            point[1] = 0.0
            return point

        def pointer_world_intersection(
            origin: np.ndarray,
            direction: np.ndarray,
        ) -> np.ndarray | None:
            world_transform = self.world_scene_client_transforms.get(client.client_id)
            if world_transform is None:
                return None
            world_path, scale, rotation_deg, position = world_transform
            world_scene_data = self._get_world_scene_data(world_path)
            if world_scene_data is None:
                return None
            points_key = "centers" if world_scene_data["kind"] == "gaussian_splats" else "points"
            points = np.asarray(world_scene_data[points_key], dtype=np.float64)
            if points.size == 0:
                return None
            rotation = tf.SO3(_world_scene_wxyz(rotation_deg)).as_matrix()
            scene_points = (rotation @ (points * float(scale)).T).T + np.asarray(position, dtype=np.float64)
            ray_t = (scene_points - origin) @ direction
            in_front = ray_t > 0.0
            if not np.any(in_front):
                return None
            candidate_points = scene_points[in_front]
            candidate_t = ray_t[in_front]
            closest_on_ray = origin[None, :] + candidate_t[:, None] * direction[None, :]
            distances = np.linalg.norm(candidate_points - closest_on_ray, axis=1)
            nearest_idx = int(np.argmin(distances))
            if float(distances[nearest_idx]) > WORLD_OBJECT_PICK_MAX_DISTANCE:
                return None
            return candidate_points[nearest_idx]

        def pointer_scene_position(event: viser.ScenePointerEvent) -> np.ndarray | None:
            if event.ray_origin is None or event.ray_direction is None:
                return None
            origin = np.asarray(event.ray_origin, dtype=np.float64)
            direction = np.asarray(event.ray_direction, dtype=np.float64)
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                return None
            direction = direction / norm
            world_hit = pointer_world_intersection(origin, direction)
            if world_hit is not None:
                return world_hit
            return pointer_floor_intersection(origin, direction)

        def add_scene_object(base_position: np.ndarray, shape: str) -> list[viser.SceneHandle]:
            workflow.placed_object_count += 1
            name = f"/placed_objects/object_{workflow.placed_object_count}"
            shape = str(shape)
            if shape == "Cylinder":
                return [
                    client.scene.add_cylinder(
                        name,
                        radius=0.08,
                        height=0.28,
                        color=(60, 140, 245),
                        position=base_position + np.array([0.0, 0.14, 0.0], dtype=np.float64),
                    )
                ]
            if shape == "Box":
                return [
                    client.scene.add_box(
                        name,
                        dimensions=(0.18, 0.18, 0.18),
                        color=(245, 160, 55),
                        position=base_position + np.array([0.0, 0.09, 0.0], dtype=np.float64),
                    )
                ]
            if shape == "Water Bottle":
                body = client.scene.add_cylinder(
                    f"{name}/body",
                    radius=0.055,
                    height=0.26,
                    color=(80, 170, 255),
                    opacity=0.62,
                    position=base_position + np.array([0.0, 0.13, 0.0], dtype=np.float64),
                )
                neck = client.scene.add_cylinder(
                    f"{name}/neck",
                    radius=0.032,
                    height=0.08,
                    color=(90, 185, 255),
                    opacity=0.68,
                    position=base_position + np.array([0.0, 0.30, 0.0], dtype=np.float64),
                )
                cap = client.scene.add_cylinder(
                    f"{name}/cap",
                    radius=0.036,
                    height=0.035,
                    color=(35, 90, 210),
                    position=base_position + np.array([0.0, 0.3575, 0.0], dtype=np.float64),
                )
                return [body, neck, cap]
            return [
                client.scene.add_icosphere(
                    name,
                    radius=0.10,
                    color=(80, 210, 125),
                    position=base_position + np.array([0.0, 0.10, 0.0], dtype=np.float64),
                )
            ]

        def place_object_at(position: np.ndarray) -> None:
            handles = add_scene_object(position, str(object_shape_dropdown.value))
            workflow.placed_object_handles.extend(handles)
            object_position_markdown.content = (
                f"Placed `{object_shape_dropdown.value}` at Kimodo/Viser coordinates:\n\n"
                f"{format_scene_position(position)}"
            )

        def enable_one_click_object_placement() -> None:
            @client.scene.on_pointer_event("click")
            def _(event: viser.ScenePointerEvent) -> None:
                if not workflow.placing_object:
                    client.scene.remove_pointer_callback()
                    return
                position = pointer_scene_position(event)
                if position is None:
                    object_position_markdown.content = "Could not place object from this click."
                    client.scene.remove_pointer_callback()
                    workflow.placing_object = False
                    return
                place_object_at(position)
                workflow.placing_object = False
                client.scene.remove_pointer_callback()

        def current_human_root_position() -> np.ndarray | None:
            session = self.client_sessions[client.client_id]
            if not session.motions:
                return None
            motion = list(session.motions.values())[0]
            frame_idx = min(session.frame_idx, motion.joints_pos.shape[0] - 1)
            return (
                motion.joints_pos[frame_idx, session.skeleton.root_idx, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        def t2_to_human_offset(csv_path: Path) -> np.ndarray:
            human_root = current_human_root_position()
            if human_root is None:
                return np.zeros(3, dtype=np.float64)
            session = self.client_sessions[client.client_id]
            t2_motion = load_tara_motion_csv(csv_path, x_offset=0.0)
            t2_frame_idx = min(session.frame_idx, t2_motion.root_positions.shape[0] - 1)
            offset = np.zeros(3, dtype=np.float64)
            offset[0] = float(human_root[0] - t2_motion.root_positions[t2_frame_idx, 0])
            offset[2] = float(human_root[2] - t2_motion.root_positions[t2_frame_idx, 2])
            return offset

        def update_memory_status() -> None:
            root = current_memories_root()
            stem = selected_memory_stem()
            if stem is None:
                memory_status.content = f"No BVH memories found in `{root / 'bvh'}`."
                return
            bvh_path = _memory_bvh_path(root, stem)
            csv_path = _memory_csv_path(root, stem)
            csv_state = "available" if csv_path.is_file() else "missing"
            memory_status.content = (
                f"BVH: `{bvh_path}`\n\n"
                f"T2 CSV: `{csv_state}`\n\n"
                f"`{csv_path}`"
            )

        def save_current_bvh(
            stem: str | None = None,
            output_root: Path | None = None,
            fps: float | None = None,
        ) -> Path:
            session = self.client_sessions[client.client_id]
            if "soma" not in session.model_name.lower():
                raise ValueError("Retargeting requires a SOMA model. Select a Kimodo-SOMA model first.")
            if not session.motions:
                raise ValueError("Generate or load a SOMA motion first.")

            motion = list(session.motions.values())[0]
            output_root = output_root or Path(output_root_text.value).expanduser().resolve()
            relative_stem = Path(stem) if stem is not None else Path(_safe_clip_name(clip_name_text.value))
            bvh_path = output_root / "bvh" / relative_stem.with_suffix(".bvh")
            npz_path = output_root / "kimodo_npz" / relative_stem.with_suffix(".npz")
            bvh_path.parent.mkdir(parents=True, exist_ok=True)
            npz_path.parent.mkdir(parents=True, exist_ok=True)

            save_motion_bvh(
                str(bvh_path),
                motion.joints_local_rot,
                motion.joints_pos[:, session.skeleton.root_idx, :],
                skeleton=session.skeleton,
                fps=float(fps if fps is not None else _model_native_fps(self, session.model_name, session.model_fps)),
                standard_tpose=bool(standard_tpose_checkbox.value),
            )
            motion_data = {
                "posed_joints": motion.joints_pos.detach().cpu().numpy(),
                "global_rot_mats": motion.joints_rot.detach().cpu().numpy(),
                "local_rot_mats": motion.joints_local_rot.detach().cpu().numpy(),
                "root_positions": motion.joints_pos[:, session.skeleton.root_idx, :].detach().cpu().numpy(),
            }
            if motion.foot_contacts is not None:
                motion_data["foot_contacts"] = motion.foot_contacts.detach().cpu().numpy()
            save_kimodo_npz(str(npz_path), motion_data)

            metadata = {
                "stem": str(relative_stem),
                "model_name": session.model_name,
                "fps": float(fps if fps is not None else _model_native_fps(self, session.model_name, session.model_fps)),
                "num_frames": int(motion.joints_pos.shape[0]),
                "bvh_path": str(bvh_path),
                "npz_path": str(npz_path),
                "standard_tpose_bvh": bool(standard_tpose_checkbox.value),
            }
            meta_path = output_root / "metadata" / relative_stem.with_suffix(".json")
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            workflow.output_root = output_root
            workflow.memory_stem = str(relative_stem)
            workflow.bvh_path = bvh_path
            workflow.npz_path = npz_path
            update_status(f"Saved SOMA BVH:\n\n`{bvh_path}`")
            return bvh_path

        def load_bvh_memory(bvh_path: Path) -> None:
            session = self.client_sessions[client.client_id]
            joints_pos, joints_rot, foot_contacts, skeleton = load_motion_file(bvh_path, self.device)
            self.clear_motions(client.client_id)
            session.skeleton = skeleton
            session.max_frame_idx = int(joints_pos.shape[0]) - 1
            try:
                session.model_fps = 1.0 / read_bvh_frame_time_seconds(bvh_path)
            except Exception:
                session.model_fps = _model_native_fps(self, session.model_name, session.model_fps or 30.0)
            client.timeline.set_fps(session.model_fps)
            session.cur_duration = (session.max_frame_idx + 1) / float(session.model_fps)
            self.add_character_motion(client, skeleton, joints_pos, joints_rot, foot_contacts)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            self.set_frame(client.client_id, 0)

        def load_memory(stem: str) -> None:
            root = current_memories_root()
            stem = _resolve_memory_stem(stem, root)
            bvh_path = _memory_bvh_path(root, stem)
            csv_path = _memory_csv_path(root, stem)
            if not bvh_path.is_file() and not csv_path.is_file():
                raise FileNotFoundError(f"No BVH or T2 CSV found for memory `{stem}`.")
            if bvh_path.is_file():
                load_bvh_memory(bvh_path)
            workflow.output_root = root
            workflow.memory_stem = stem
            workflow.bvh_path = bvh_path if bvh_path.is_file() else None
            workflow.npz_path = None
            if csv_path.is_file():
                load_t2_preview(csv_path)
                if bvh_path.is_file():
                    update_status(f"Loaded memory with T2 CSV:\n\n`{stem}`")
                else:
                    update_status(f"Loaded T2 CSV-only robot memory:\n\n`{stem}`")
            else:
                workflow.clear_t2_preview()
                update_status(f"Loaded BVH memory without T2 CSV:\n\n`{stem}`")
            csv_path_text.value = str(csv_path)
            output_root_text.value = str(root)
            clip_name_text.value = Path(stem).name
            session = self.client_sessions.get(client.client_id)
            if session is not None:
                session.play_once = True
                session.playing = True

        def sync_memory_from_external_change(stem: str) -> str:
            root = current_memories_root()
            stem = _resolve_memory_stem(stem, root)
            refresh_memory_options(stem)
            load_memory(stem)
            return stem

        def list_memories_for_control_api() -> dict[str, object]:
            root = current_memories_root()
            stems = _scan_memory_stems(root)
            return {
                "memories_root": str(root),
                "stems": stems,
                "labels": [_memory_label(root, stem) for stem in stems],
            }

        self._memory_sync_callbacks[client.client_id] = sync_memory_from_external_change
        self._memory_list_callbacks[client.client_id] = list_memories_for_control_api

        def retarget_bvh_to_memory_csv(bvh_path: Path, output_root: Path, notify_client: viser.ClientHandle) -> None:
            if workflow.retarget_running:
                notify_client.add_notification(
                    title="Retarget already running",
                    body="Wait for the current soma-retargeter job to finish.",
                    auto_close_seconds=4.0,
                    color="orange",
                )
                return

            job = SomaT2RetargetJob(
                retargeter_root=Path(retargeter_root_text.value).expanduser().resolve(),
                bvh_path=bvh_path,
                output_root=output_root,
                conda_env=str(conda_env_text.value).strip() or "soma-retargeter",
            )
            workflow.retarget_running = True
            retarget_memory_button.disabled = True
            retarget_button.disabled = True
            update_status(f"Retargeting `{job.relative_stem}` to `{job.csv_path}`...")
            retarget_notif = notify_client.add_notification(
                title="Retargeting started",
                body="soma-retargeter is running headless.",
                loading=True,
                with_close_button=False,
            )

            def run_job() -> None:
                try:
                    result = job.run()
                    log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".retarget.log")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log_path.write_text(result.stdout or "", encoding="utf-8")
                    if result.returncode != 0:
                        raise RuntimeError(f"soma-retargeter failed with exit code {result.returncode}. Log: {log_path}")
                    if not job.csv_path.is_file():
                        raise FileNotFoundError(f"Expected retarget CSV was not created: {job.csv_path}")
                    workflow.output_root = output_root
                    workflow.memory_stem = str(job.relative_stem)
                    workflow.bvh_path = bvh_path
                    load_memory(str(job.relative_stem))
                    refresh_memory_options(str(job.relative_stem))
                    retarget_notif.title = "Retargeting finished"
                    retarget_notif.body = str(job.csv_path)
                    retarget_notif.loading = False
                    retarget_notif.with_close_button = True
                    retarget_notif.auto_close_seconds = 5.0
                    retarget_notif.color = "green"
                except Exception as exc:
                    update_status(f"Retarget failed:\n\n`{exc}`")
                    retarget_notif.title = "Retargeting failed"
                    retarget_notif.body = str(exc)
                    retarget_notif.loading = False
                    retarget_notif.with_close_button = True
                    retarget_notif.auto_close_seconds = 9.0
                    retarget_notif.color = "red"
                finally:
                    workflow.retarget_running = False
                    retarget_memory_button.disabled = False
                    retarget_button.disabled = False

            threading.Thread(target=run_job, daemon=True).start()

        def load_t2_preview(csv_path: Path) -> None:
            csv_path = csv_path.expanduser().resolve()
            if not csv_path.is_file():
                raise FileNotFoundError(csv_path)

            workflow.clear_t2_preview()
            retargeter_root = Path(retargeter_root_text.value).expanduser().resolve()
            urdf_path = retargeter_root / "antt_t2" / "T2_serial_nero_arms.urdf"
            workflow.t2_motion = T2ViewerMotion(
                name=f"t2_preview_{client.client_id}",
                server=client,
                csv_path=csv_path,
                urdf_path=urdf_path if urdf_path.is_file() else None,
                x_offset=0.0,
                position_offset=t2_to_human_offset(csv_path),
                color=(145, 145, 145),
                arms_only=bool(arms_only_checkbox.value),
            )
            workflow.t2_motion.set_frame(self.client_sessions[client.client_id].frame_idx)
            workflow.t2_motion.set_mesh_visibility(bool(show_t2_checkbox.value))
            workflow.arm_frames = load_arm_frames(csv_path, start_frame=0, max_frames=None, frame_stride=1)
            workflow.csv_path = csv_path
            csv_path_text.value = str(csv_path)
            session = self.client_sessions[client.client_id]
            session.max_frame_idx = max(session.max_frame_idx, workflow.t2_motion.length - 1)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            update_status(
                "Loaded T2 preview and robot arm frames:\n\n"
                f"`{csv_path}`\n\n"
                f"{_csv_motion_summary(csv_path)}"
            )

        def control_job_id(prompt: str) -> str:
            digest = hashlib.sha1(f"{client.client_id}:{time.time()}:{prompt}".encode("utf-8")).hexdigest()[:12]
            return f"generate_retarget_{digest}"

        def set_control_job(job_id: str, **updates: object) -> None:
            with self._control_jobs_lock:
                current = dict(self._control_jobs.get(job_id) or {})
                current.update(updates)
                current["job_id"] = job_id
                current["updated_at"] = time.time()
                self._control_jobs[job_id] = current

        def run_retarget_bvh_to_memory_csv_sync(bvh_path: Path, output_root: Path) -> tuple[SomaT2RetargetJob, Path]:
            job = SomaT2RetargetJob(
                retargeter_root=Path(retargeter_root_text.value).expanduser().resolve(),
                bvh_path=bvh_path,
                output_root=output_root,
                conda_env=str(conda_env_text.value).strip() or "soma-retargeter",
            )
            result = job.run()
            log_path = job.output_root / "logs" / job.relative_stem.with_suffix(".retarget.log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(result.stdout or "", encoding="utf-8")
            if result.returncode != 0:
                raise RuntimeError(f"soma-retargeter failed with exit code {result.returncode}. Log: {log_path}")
            if not job.csv_path.is_file():
                raise FileNotFoundError(f"Expected retarget CSV was not created: {job.csv_path}")
            return job, log_path

        def generate_save_retarget_from_control_api(payload: dict[str, object]) -> dict[str, object]:
            prompt = str(payload.get("prompt") or "").strip()
            if not prompt:
                raise ValueError("Missing prompt")
            session = self.client_sessions[client.client_id]
            if "soma" not in session.model_name.lower():
                raise ValueError("Generate-retarget requires a SOMA model. Select a Kimodo-SOMA model first.")
            duration = float(payload.get("duration_seconds") or session.cur_duration or DEFAULT_CUR_DURATION)
            duration = max(0.1, duration)
            seed = int(payload.get("seed") or 42)
            diffusion_steps = int(payload.get("diffusion_steps") or 100)
            stem = str(payload.get("stem") or "").strip() or _new_generated_stem()
            output_root = Path(str(payload.get("output_root") or current_memories_root())).expanduser().resolve()
            num_frames = max(1, int(round(duration * float(session.model_fps))))
            job_id = control_job_id(prompt)
            session.cur_duration = num_frames / float(session.model_fps)
            session.max_frame_idx = num_frames - 1
            client.timeline.clear_prompts()
            client.timeline.add_prompt(prompt, 0, session.max_frame_idx, color=(64, 124, 186))
            client.timeline.set_current_frame(0)
            client.timeline.set_zoom_settings(max_frames_zoom=max(session.max_frame_idx + 1, 1000))
            self.set_frame(client.client_id, 0)

            client.flush()

            set_control_job(
                job_id,
                status="queued",
                prompt=prompt,
                stem=stem,
                output_root=str(output_root),
                client_id=client.client_id,
            )
            update_prompt_job(
                "**Prompt-to-Robot Job**\n\n"
                "Status: `queued`\n\n"
                f"Prompt: `{prompt}`\n\n"
                f"Job: `{job_id}`\n\n"
                f"Target stem: `{stem}`\n\n"
                "Progress:\n\n"
                "- [x] Queued\n"
                "- [ ] Generate SOMA motion\n"
                "- [ ] Save memory\n"
                "- [ ] Retarget to T2\n"
                "- [ ] Load result"
            )
            progress_notif = client.add_notification(
                title="Prompt-to-robot queued",
                body=f"`{prompt}`",
                loading=True,
                with_close_button=False,
            )
            client.flush()

            def show_control_stage(
                title: str,
                body: str,
                status: str,
                progress_lines: list[str],
                **job_updates: object,
            ) -> None:
                set_control_job(job_id, status=status, **job_updates)
                update_prompt_job(
                    "**Prompt-to-Robot Job**\n\n"
                    f"Status: `{status}`\n\n"
                    f"Prompt: `{prompt}`\n\n"
                    f"Job: `{job_id}`\n\n"
                    f"Stage: **{title}**\n\n"
                    f"{body}\n\n"
                    "Progress:\n\n"
                    + "\n".join(progress_lines)
                )
                update_status(
                    f"{title}\n\n"
                    f"Prompt: `{prompt}`\n\n"
                    f"Job: `{job_id}`\n\n"
                    f"{body}"
                )
                progress_notif.title = title
                progress_notif.body = body.replace("\n\n", "\n")
                progress_notif.loading = status not in {"done", "error"}
                client.flush()

            def worker() -> None:
                try:
                    show_control_stage(
                        "Generating SOMA motion",
                        f"Frames: `{num_frames}`\n\nSeed: `{seed}`\n\nDenoising steps: `{diffusion_steps}`",
                        "generating",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [ ] Save memory",
                            "- [ ] Retarget to T2",
                            "- [ ] Load result",
                        ],
                    )
                    self.generate(
                        client,
                        [prompt],
                        [num_frames],
                        1,
                        seed,
                        diffusion_steps,
                        cfg_weight=[2.0, 2.0],
                        cfg_type="separated",
                        postprocess_parameters={
                            "post_processing": "g1" not in session.model_name.lower(),
                            "root_margin": 0.04,
                        },
                        transitions_parameters={"num_transition_frames": NB_TRANSITION_FRAMES},
                        real_robot_rotations=False,
                    )
                    session.cur_duration = num_frames / float(session.model_fps)
                    session.max_frame_idx = num_frames - 1
                    self.set_frame(client.client_id, 0)

                    show_control_stage(
                        "Saving generated memory",
                        f"Writing BVH/NPZ under:\n\n`{output_root}`\n\nStem: `{stem}`",
                        "saving",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [x] Save memory",
                            "- [ ] Retarget to T2",
                            "- [ ] Load result",
                        ],
                    )
                    bvh_path = save_current_bvh(
                        stem=stem,
                        output_root=output_root,
                        fps=_model_native_fps(self, session.model_name, session.model_fps),
                    )
                    refresh_memory_options(stem)

                    show_control_stage(
                        "Retargeting to T2 robot",
                        f"BVH: `{bvh_path}`",
                        "retargeting",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [x] Save memory",
                            "- [x] Retarget to T2",
                            "- [ ] Load result",
                        ],
                        bvh_path=str(bvh_path),
                    )
                    retarget_job, log_path = run_retarget_bvh_to_memory_csv_sync(bvh_path, output_root)
                    workflow.output_root = output_root
                    workflow.memory_stem = str(retarget_job.relative_stem)
                    workflow.bvh_path = bvh_path
                    load_memory(str(retarget_job.relative_stem))
                    refresh_memory_options(str(retarget_job.relative_stem))
                    show_control_stage(
                        "Prompt-to-robot complete",
                        "Generated memory is loaded in Viser and ready for playback.\n\n"
                        f"Stem: `{retarget_job.relative_stem}`\n\n"
                        f"BVH: `{bvh_path}`\n\n"
                        f"T2 CSV: `{retarget_job.csv_path}`\n\n"
                        f"Log: `{log_path}`",
                        "done",
                        [
                            "- [x] Queued",
                            "- [x] Generate SOMA motion",
                            "- [x] Save memory",
                            "- [x] Retarget to T2",
                            "- [x] Load result",
                        ],
                        stem=str(retarget_job.relative_stem),
                        bvh_path=str(bvh_path),
                        csv_path=str(retarget_job.csv_path),
                        log_path=str(log_path),
                    )
                    progress_notif.loading = False
                    progress_notif.with_close_button = True
                    progress_notif.auto_close_seconds = 8.0
                    progress_notif.color = "green"
                except Exception as exc:
                    show_control_stage(
                        "Prompt-to-robot failed",
                        f"Error: `{exc}`",
                        "error",
                        [
                            "- [x] Queued",
                            "- [ ] Generate SOMA motion",
                            "- [ ] Save memory",
                            "- [ ] Retarget to T2",
                            "- [ ] Load result",
                        ],
                        error=str(exc),
                    )
                    progress_notif.loading = False
                    progress_notif.with_close_button = True
                    progress_notif.auto_close_seconds = 10.0
                    progress_notif.color = "red"

            threading.Thread(target=worker, name=f"generate-retarget-{job_id}", daemon=True).start()
            return {"job_id": job_id, "status": "queued", "stem": stem}

        self._generate_retarget_callbacks[client.client_id] = generate_save_retarget_from_control_api

        @show_human_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_human_mesh_visible(bool(show_human_checkbox.value))

        @show_t2_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_t2_visible(bool(show_t2_checkbox.value))

        @show_floor_grid_checkbox.on_update
        def _(_event: viser.GuiEvent) -> None:
            set_floor_grid_visible(bool(show_floor_grid_checkbox.value))

        @place_object_button.on_click
        def _(event: viser.GuiEvent) -> None:
            workflow.placing_object = True
            enable_one_click_object_placement()
            object_position_markdown.content = (
                f"Placing `{object_shape_dropdown.value}`. Click the Kimodo/Viser scene."
            )
            event.client.add_notification(
                title="Place object",
                body="Click in the Viser scene to place the selected object.",
                auto_close_seconds=3.0,
                color="blue",
            )

        @clear_objects_button.on_click
        def _(_event: viser.GuiEvent) -> None:
            workflow.clear_placed_objects()
            object_position_markdown.content = "No placed objects."

        @arms_only_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            if workflow.csv_path is None:
                return
            try:
                load_t2_preview(workflow.csv_path)
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to reload T2 preview",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )

        @memory_dropdown.on_update
        def _(_event: viser.GuiEvent) -> None:
            update_memory_status()

        @refresh_memories_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stems = refresh_memory_options()
            event.client.add_notification(
                title="Memories refreshed",
                body=f"Found {len(stems)} BVH memories.",
                auto_close_seconds=3.0,
                color="blue",
            )

        @load_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No memory selected",
                    body="Refresh memories or choose a BVH memory.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            try:
                load_memory(stem)
                for client_id, sync_memory in list(self._memory_sync_callbacks.items()):
                    if client_id == client.client_id:
                        continue
                    try:
                        sync_memory(stem)
                    except Exception as exc:
                        print(f"Failed to sync memory for client {client_id}: {exc}")
                event.client.add_notification(
                    title="Memory loaded",
                    body=stem,
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to load memory",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @retarget_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = selected_memory_stem()
            if stem is None:
                event.client.add_notification(
                    title="No memory selected",
                    body="Choose a BVH memory before retargeting.",
                    auto_close_seconds=4.0,
                    color="red",
                )
                return
            root = current_memories_root()
            bvh_path = _memory_bvh_path(root, stem)
            if not bvh_path.is_file():
                event.client.add_notification(
                    title="Memory BVH missing",
                    body=str(bvh_path),
                    auto_close_seconds=5.0,
                    color="red",
                )
                return
            retarget_bvh_to_memory_csv(bvh_path, root, event.client)

        @save_generated_memory_button.on_click
        def _(event: viser.GuiEvent) -> None:
            stem = _new_generated_stem()
            root = current_memories_root()
            session = self.client_sessions[client.client_id]
            generated_fps = _model_native_fps(self, session.model_name, session.model_fps)
            try:
                bvh_path = save_current_bvh(stem=stem, output_root=root, fps=generated_fps)
                refresh_memory_options(stem)
                csv_path_text.value = str(_memory_csv_path(root, stem))
                output_root_text.value = str(root)
                clip_name_text.value = Path(stem).name
                event.client.add_notification(
                    title="Memory saved",
                    body=str(bvh_path),
                    auto_close_seconds=5.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to save generated memory",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @save_bvh_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                bvh_path = save_current_bvh()
                event.client.add_notification(
                    title="SOMA BVH saved",
                    body=str(bvh_path),
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to save BVH",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )

        @retarget_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                bvh_path = workflow.bvh_path if workflow.bvh_path is not None else save_current_bvh()
            except Exception as exc:
                event.client.add_notification(
                    title="Cannot start retarget",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )
                return

            retarget_bvh_to_memory_csv(bvh_path, Path(output_root_text.value).expanduser().resolve(), event.client)

        @load_t2_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                load_t2_preview(Path(csv_path_text.value))
                event.client.add_notification(
                    title="T2 preview loaded",
                    body=str(workflow.csv_path),
                    auto_close_seconds=4.0,
                    color="green",
                )
            except Exception as exc:
                event.client.add_notification(
                    title="Failed to load T2 preview",
                    body=str(exc),
                    auto_close_seconds=7.0,
                    color="red",
                )

        update_memory_status()

        def connect_robot(event: viser.GuiEvent) -> None:
            if not workflow.arm_frames:
                event.client.add_notification(
                    title="No T2 CSV loaded",
                    body="Retarget or load a T2 CSV before connecting.",
                    auto_close_seconds=5.0,
                    color="red",
                )
                return

            workflow.connection.dry_run = bool(dry_run_checkbox.value)
            workflow.connection.stream_script = default_t2_stream_script()

            def on_status(title: str, body: str) -> None:
                update_robot_status(f"{title}.\n\n`{body[-900:]}`")

            try:
                workflow.connection.connect(on_status=on_status)
                update_robot_status("Connecting...")
                current_frame = self.client_sessions[client.client_id].frame_idx
                workflow.connection.send_frame(self._arm_frame_for_index(workflow, current_frame))
                event.client.add_notification(
                    title="T2 Nero connecting",
                    body="Current visualizer frames will stream to the robot.",
                    auto_close_seconds=4.0,
                    color="blue",
                )
            except Exception as exc:
                update_robot_status(f"Connection failed.\n\n`{exc}`")
                event.client.add_notification(
                    title="Robot connection failed",
                    body=str(exc),
                    auto_close_seconds=8.0,
                    color="red",
                )

        @connect_button.on_click
        def _(event: viser.GuiEvent) -> None:
            connect_robot(event)

        @disconnect_button.on_click
        def _(event: viser.GuiEvent) -> None:
            workflow.connection.disconnect()
            update_robot_status("Disconnected.")
            event.client.add_notification(
                title="T2 Nero disconnected",
                body="Robot streaming is off.",
                auto_close_seconds=3.0,
                color="blue",
            )

        @send_frame_button.on_click
        def _(event: viser.GuiEvent) -> None:
            try:
                workflow.connection.send_frame(
                    self._arm_frame_for_index(workflow, self.client_sessions[client.client_id].frame_idx)
                )
                update_robot_status(f"Sent frame `{self.client_sessions[client.client_id].frame_idx}`.")
            except Exception as exc:
                update_robot_status(f"Send failed.\n\n`{exc}`")
                event.client.add_notification(
                    title="Send current frame failed",
                    body=str(exc),
                    auto_close_seconds=6.0,
                    color="red",
                )

        @play_robot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            connect_robot(event)
            session = self.client_sessions[client.client_id]
            session.play_once = False
            session.playing = True
            update_robot_status("Playing through the visualizer timeline.")

        @stop_robot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            session = self.client_sessions[client.client_id]
            session.play_once = False
            session.playing = False
            workflow.connection.disconnect()
            update_robot_status("Stopped and disconnected.")

    @staticmethod
    def _arm_frame_for_index(workflow: RobotWorkflowState, frame_idx: int) -> ArmFrame:
        if not workflow.arm_frames:
            raise RuntimeError("No T2 arm frames are loaded.")
        return workflow.arm_frames[max(0, min(int(frame_idx), len(workflow.arm_frames) - 1))]

    def run(self) -> None:
        try:
            update_counter = 0
            cuda_check_interval = 300
            while True:
                last_update_time = time.time()
                fps_candidates = [bundle.model_fps for bundle in self.models.values()]
                fps_candidates.extend(
                    session.model_fps
                    for session in self.client_sessions.values()
                    if session.model_fps and session.model_fps > 0.0
                )
                playback_fps = max(fps_candidates, default=60.0) * 2.0

                for client_id, session in list(self.client_sessions.items()):
                    if not session.model_fps or session.model_fps <= 0.0:
                        continue
                    playback_speed = max(float(session.playback_speed), 1e-6)
                    update_interval = max(1, int(playback_fps / (playback_speed * session.model_fps)))
                    new_frame_idx = session.frame_idx
                    if session.playing and update_counter % update_interval == 0:
                        if session.frame_idx >= session.max_frame_idx:
                            if session.play_once:
                                session.playing = False
                                session.play_once = False
                                new_frame_idx = session.max_frame_idx
                            else:
                                new_frame_idx = 0
                        else:
                            new_frame_idx = session.frame_idx + 1

                        if self.client_active(client_id):
                            self.set_frame(client_id, new_frame_idx)

                if update_counter % cuda_check_interval == 0:
                    self.check_cuda_health()

                time_remaining = max(0, 1.0 / playback_fps - (time.time() - last_update_time))
                time.sleep(time_remaining)
                update_counter += 1
                update_counter %= max(1, int(playback_fps))
        finally:
            for workflow in self.robot_workflows.values():
                workflow.connection.disconnect()
                workflow.clear_t2_preview()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kimodo robot workflow demo UI.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Default model to load. A SOMA model is required for the retarget workflow.",
    )
    parser.add_argument(
        "--text-encoder-mode",
        choices=("api", "local", "auto"),
        default=None,
        help="Text encoder mode. The robot demo defaults to api to avoid loading the 8B fallback in-process.",
    )
    parser.add_argument(
        "--text-encoder-url",
        default=None,
        help=f"Text encoder server URL. Defaults to TEXT_ENCODER_URL or {DEFAULT_TEXT_ENCODER_URL}.",
    )
    parser.add_argument(
        "--world-scene-path",
        default=os.environ.get("KIMODO_WORLD_SCENE_PATH", str(DEFAULT_WORLD_SCENE_PATH)),
        help="Optional world PLY to load into the Viser scene without changing the human or robot meshes.",
    )
    parser.add_argument(
        "--control-host",
        default=os.environ.get("KIMODO_CONTROL_HOST", "127.0.0.1"),
        help="Host for the lightweight Kimodo control API. Use 0.0.0.0 for LAN access.",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=int(os.environ.get("KIMODO_CONTROL_PORT", "8787")),
        help="Port for the lightweight Kimodo control API.",
    )
    args = parser.parse_args()

    text_encoder_mode = _configure_text_encoder_runtime(args.text_encoder_mode, args.text_encoder_url)
    resolved = resolve_model_name(args.model, "Kimodo")
    world_scene_path = Path(args.world_scene_path).expanduser() if args.world_scene_path else None
    try:
        demo = RobotDemo(default_model_name=resolved, world_scene_path=world_scene_path)
    except Exception:
        raise SystemExit(_text_encoder_startup_error(text_encoder_mode)) from None
    demo.start_control_server(args.control_host, args.control_port)
    demo.run()


if __name__ == "__main__":
    main()
