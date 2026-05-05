# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot-focused Kimodo demo.

This app keeps the regular Kimodo authoring UI intact and adds a separate
SOMA -> T2 -> real robot workflow panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import viser

from kimodo.demo.app import Demo
from kimodo.demo.config import DEFAULT_MODEL
from kimodo.exports.bvh import read_bvh_frame_time_seconds, save_motion_bvh
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.registry import DEFAULT_TEXT_ENCODER_URL, resolve_model_name
from kimodo.motion_io import load_motion_file
from kimodo.retarget.soma_t2 import SomaT2RetargetJob, default_soma_retargeter_root
from kimodo.robot.t2_nero import T2NeroConnection, default_t2_stream_script
from kimodo.scripts.t2_csv_arm_publisher import ArmFrame, load_arm_frames
from kimodo.viz.tara_rig import T2ViewerMotion


MEMORIES_ROOT = Path("/home/jony/important/soma-retargeter/assets/motions")
TEXT_ENCODER_SERVER_COMMAND = "python -m kimodo.scripts.run_text_encoder_server"


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
    if not bvh_root.is_dir():
        return []
    return sorted(str(path.relative_to(bvh_root).with_suffix("")) for path in bvh_root.rglob("*.bvh"))


def _memory_bvh_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "bvh" / Path(stem).with_suffix(".bvh")


def _memory_csv_path(memories_root: Path, stem: str) -> Path:
    return memories_root / "t2_csv" / Path(stem).with_suffix(".csv")


def _memory_label(memories_root: Path, stem: str) -> str:
    state = "csv" if _memory_csv_path(memories_root, stem).is_file() else "needs csv"
    return f"[{state}] {stem}"


def _stem_from_memory_label(label: str) -> str:
    return re.sub(r"^\[[^\]]+\]\s*", "", str(label)).strip()


def _new_generated_stem() -> str:
    return f"generated/kimodo_{uuid.uuid4().hex[:10]}"


def _model_native_fps(demo: Demo, model_name: str, fallback: float) -> float:
    bundle = demo.models.get(model_name)
    if bundle is not None and bundle.model_fps and bundle.model_fps > 0.0:
        return float(bundle.model_fps)
    return float(fallback)


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

    def clear_t2_preview(self) -> None:
        if self.t2_motion is not None:
            self.t2_motion.clear()
            self.t2_motion = None
        self.arm_frames.clear()
        self.csv_path = None


class RobotDemo(Demo):
    """Kimodo demo with a robot production workflow mounted as a separate panel."""

    def __init__(self, default_model_name: str = DEFAULT_MODEL):
        super().__init__(default_model_name=default_model_name)
        self.robot_workflows: dict[int, RobotWorkflowState] = {}

    def _setup_demo_for_client(self, client: viser.ClientHandle) -> None:
        super()._setup_demo_for_client(client)
        self._hide_examples_folder(client)
        self.robot_workflows[client.client_id] = RobotWorkflowState()
        self._create_robot_pipeline_gui(client)

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
            bvh_path = _memory_bvh_path(root, stem)
            csv_path = _memory_csv_path(root, stem)
            if not bvh_path.is_file():
                raise FileNotFoundError(bvh_path)
            load_bvh_memory(bvh_path)
            workflow.output_root = root
            workflow.memory_stem = stem
            workflow.bvh_path = bvh_path
            workflow.npz_path = None
            if csv_path.is_file():
                load_t2_preview(csv_path)
                update_status(f"Loaded memory with T2 CSV:\n\n`{stem}`")
            else:
                workflow.clear_t2_preview()
                update_status(f"Loaded BVH memory without T2 CSV:\n\n`{stem}`")
            csv_path_text.value = str(csv_path)
            output_root_text.value = str(root)
            clip_name_text.value = Path(stem).name

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
                x_offset=1.4,
                color=(145, 145, 145),
                arms_only=bool(arms_only_checkbox.value),
            )
            workflow.t2_motion.set_frame(self.client_sessions[client.client_id].frame_idx)
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
            session.playing = True
            update_robot_status("Playing through the visualizer timeline.")

        @stop_robot_button.on_click
        def _(event: viser.GuiEvent) -> None:
            session = self.client_sessions[client.client_id]
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
    args = parser.parse_args()

    text_encoder_mode = _configure_text_encoder_runtime(args.text_encoder_mode, args.text_encoder_url)
    resolved = resolve_model_name(args.model, "Kimodo")
    try:
        demo = RobotDemo(default_model_name=resolved)
    except Exception:
        raise SystemExit(_text_encoder_startup_error(text_encoder_mode)) from None
    demo.run()


if __name__ == "__main__":
    main()
