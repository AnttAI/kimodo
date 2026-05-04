# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T2 Nero arm streaming helper."""

from __future__ import annotations

import json
import os
import subprocess
import threading
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
    process: subprocess.Popen[str] | None = None
    last_output: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

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

        def _watch() -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    clean = line.rstrip()
                    self.last_output.append(clean)
                    self.last_output[:] = self.last_output[-20:]
                    print(f"[T2 ROBOT] {clean}", flush=True)
                    if on_status is not None and clean.startswith("[READY]"):
                        on_status("Connected", clean)
            return_code = process.wait()
            with self._lock:
                if self.process is process:
                    self.process = None
            if on_status is not None:
                tail = "\n".join(self.last_output[-6:]) or f"Exit code {return_code}"
                on_status("Disconnected", tail)

        threading.Thread(target=_watch, daemon=True).start()

    def disconnect(self) -> None:
        with self._lock:
            process = self.process
            self.process = None
        if process is None or process.poll() is not None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        process.terminate()

    def send_frame(self, frame: ArmFrame) -> None:
        with self._lock:
            process = self.process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("T2 Nero stream is not connected.")
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
        except (BrokenPipeError, OSError) as exc:
            self.disconnect()
            raise RuntimeError("T2 Nero stream disconnected while sending a frame.") from exc
