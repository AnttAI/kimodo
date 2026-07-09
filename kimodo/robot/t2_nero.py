# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T2 Nero arm streaming helper."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from kimodo.scripts.t2_csv_arm_publisher import ArmFrame


StatusCallback = Callable[[str, str], None]


def default_t2_stream_script() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "scripts" / "stream_t2_robot_sync.sh"


@dataclass
class T2NeroConnection:
    """Manage a long-lived stdin-driven T2 robot stream process."""

    stream_script: Path = field(default_factory=default_t2_stream_script)
    dry_run: bool = False
    require_base: bool = True
    process: subprocess.Popen[str] | None = None
    last_output: list[str] = field(default_factory=list)
    _last_payload_data: dict[str, object] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _ready_event: threading.Event = field(default_factory=threading.Event)
    _startup_error: str | None = None

    def is_connected(self) -> bool:
        with self._lock:
            return self.process is not None and self.process.poll() is None

    def connect(self, on_status: StatusCallback | None = None) -> None:
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return

            script = self.stream_script.expanduser().resolve()
            if not script.is_file():
                raise FileNotFoundError(f"T2 robot stream script not found: {script}")

            env = os.environ.copy()
            if self.dry_run:
                env["KIMODO_ROBOT_DRY_RUN"] = "1"
            env["KIMODO_ROBOT_REQUIRE_BASE"] = "1" if self.require_base else "0"

            process = subprocess.Popen(
                [str(script)],
                cwd=str(script.parent.parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            self.process = process
            self.last_output.clear()
            self._last_payload_data = None
            self._startup_error = None
            self._ready_event.clear()

        def _watch() -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    clean = line.rstrip()
                    self.last_output.append(clean)
                    self.last_output[:] = self.last_output[-20:]
                    print(f"[T2 ROBOT] {clean}", flush=True)
                    if clean.startswith("[READY]"):
                        self._ready_event.set()
                        if on_status is not None:
                            on_status("Connected", clean)
            return_code = process.wait()
            with self._lock:
                if self.process is process:
                    self.process = None
                if not self._ready_event.is_set():
                    self._startup_error = "\n".join(self.last_output[-6:]) or f"Exit code {return_code}"
                    self._ready_event.set()
            if on_status is not None:
                tail = "\n".join(self.last_output[-6:]) or f"Exit code {return_code}"
                on_status("Disconnected", tail)

        threading.Thread(target=_watch, daemon=True).start()

    def wait_until_ready(self, timeout: float = 10.0) -> None:
        """Wait until ROS subscribers are confirmed before streaming frames."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        remaining = max(0.0, deadline - time.monotonic())
        if not self._ready_event.wait(remaining):
            self.disconnect()
            raise TimeoutError("Timed out waiting for the T3 ROS stream to become ready.")
        if self._startup_error is not None or not self.is_connected():
            error = self._startup_error or "T3 ROS stream exited before becoming ready."
            raise RuntimeError(error)

    def disconnect(self) -> None:
        with self._lock:
            process = self.process
            self.process = None
        if process is None or process.poll() is not None:
            self._ready_event.set()
            return
        if process.stdin is not None:
            try:
                if self._last_payload_data is not None:
                    if (
                        "base_csv" in self._last_payload_data
                        or "base_csv_action" in self._last_payload_data
                    ):
                        stop_payload = {"base_csv_action": "stop"}
                    else:
                        stop_payload = dict(self._last_payload_data)
                        stop_payload["base_wheel_rpm"] = [0.0, 0.0]
                    process.stdin.write(json.dumps(stop_payload, separators=(",", ":")) + "\n")
                    process.stdin.flush()
                process.stdin.close()
            except OSError:
                pass
        try:
            # Let the ROS publisher consume the explicit zero-RPM frame and
            # exit cleanly on stdin EOF before forcing termination.
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.terminate()

    def send_frame(
        self,
        frame: ArmFrame,
        gripper: float | None = None,
        gripper_effort: float = 1.0,
        base_wheel_rpm: tuple[float, float] | None = None,
    ) -> None:
        with self._lock:
            process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("T2 Nero stream is not connected.")
        payload_data = {
            "frame_index": frame.frame_index,
            "right": frame.right,
            "left": frame.left,
        }
        if gripper is not None:
            payload_data["gripper"] = float(gripper)
            payload_data["gripper_effort"] = float(gripper_effort)
        if base_wheel_rpm is not None:
            payload_data["base_wheel_rpm"] = [float(base_wheel_rpm[0]), float(base_wheel_rpm[1])]
        self._last_payload_data = payload_data
        payload = json.dumps(payload_data, separators=(",", ":"))
        try:
            process.stdin.write(payload + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.disconnect()
            raise RuntimeError("T2 Nero stream disconnected while sending a frame.") from exc

    def send_base_frame(self, frame_index: int, base_wheel_rpm: tuple[float, float]) -> None:
        with self._lock:
            process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("T3 ROS stream is not connected.")
        payload_data: dict[str, object] = {
            "frame_index": int(frame_index),
            "base_wheel_rpm": [float(base_wheel_rpm[0]), float(base_wheel_rpm[1])],
        }
        self._last_payload_data = payload_data
        try:
            process.stdin.write(json.dumps(payload_data, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.disconnect()
            raise RuntimeError("T3 ROS stream disconnected while sending a base frame.") from exc

    def send_base_csv(self, csv_text: str, options: dict[str, object]) -> None:
        """Ask the Tara ROS node to play a complete CSV with hardware-side timing."""
        payload_data: dict[str, object] = {
            "base_csv": {
                "csv_text": str(csv_text),
                "options": dict(options),
            }
        }
        self._write_payload(payload_data, "TaraBase CSV")

    def stop_base_csv(self) -> None:
        """Stop hardware-side TaraBase CSV playback without replaying the CSV."""
        self._write_payload({"base_csv_action": "stop"}, "TaraBase CSV stop")

    def _write_payload(self, payload_data: dict[str, object], label: str) -> None:
        with self._lock:
            process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("T3 ROS stream is not connected.")
        self._last_payload_data = payload_data
        try:
            process.stdin.write(json.dumps(payload_data, separators=(",", ":")) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.disconnect()
            raise RuntimeError(f"T3 ROS stream disconnected while sending {label}.") from exc
