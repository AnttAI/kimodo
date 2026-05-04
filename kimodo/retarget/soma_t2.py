# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for running SOMA BVH to T2 CSV retargeting."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path


def default_soma_retargeter_root() -> Path:
    """Return the most likely local soma-retargeter checkout."""

    candidates = [
        Path(os.environ.get("SOMA_RETARGETER_ROOT", "")),
        Path("/home/jony/Downloads/soma-retargeter"),
        Path("/home/jony/code/soma-retargeter"),
        Path.cwd().parent / "soma-retargeter",
    ]
    for candidate in candidates:
        if candidate and (candidate / "app" / "bvh_to_csv_converter.py").is_file():
            return candidate.expanduser().resolve()
    return (Path.cwd().parent / "soma-retargeter").expanduser().resolve()


@dataclass(frozen=True)
class SomaT2RetargetJob:
    """A one-clip headless soma-retargeter batch job."""

    retargeter_root: Path
    bvh_path: Path
    output_root: Path
    conda_env: str = "soma-retargeter"
    batch_size: int = 1
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def bvh_root(self) -> Path:
        return self.output_root / "bvh"

    @property
    def csv_root(self) -> Path:
        return self.output_root / "t2_csv"

    @property
    def relative_stem(self) -> Path:
        return self.bvh_path.relative_to(self.bvh_root).with_suffix("")

    @property
    def csv_path(self) -> Path:
        return self.csv_root / self.relative_stem.with_suffix(".csv")

    @property
    def job_root(self) -> Path:
        return self.output_root / ".kimodo_retarget_jobs" / f"{self.bvh_path.stem}_{self.job_id}"

    @property
    def job_import_root(self) -> Path:
        return self.job_root / "bvh"

    @property
    def job_export_root(self) -> Path:
        return self.job_root / "t2_csv"

    @property
    def job_bvh_path(self) -> Path:
        return self.job_import_root / self.relative_stem.with_suffix(".bvh")

    @property
    def job_csv_path(self) -> Path:
        return self.job_export_root / self.relative_stem.with_suffix(".csv")

    @property
    def config_path(self) -> Path:
        return self.job_root / "retarget_config.json"

    def write_config(self) -> Path:
        self.job_bvh_path.parent.mkdir(parents=True, exist_ok=True)
        self.job_export_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.bvh_path, self.job_bvh_path)
        config = {
            "import_folder": str(self.job_import_root),
            "export_folder": str(self.job_export_root),
            "batch_size": int(self.batch_size),
            "retargeter": "Newton",
            "retarget_source": "soma",
            "retarget_target": "t2",
            "viewer_robot": "t2",
            "retarget_source_facing_direction": "Mujoco",
            "viewer_initial_camera": {
                "position": [0.0, -6.0, 1.8],
                "pitch": 0.0,
                "yaw": 90.0,
            },
        }
        self.job_root.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return self.config_path

    def command(self) -> list[str]:
        return [
            "conda",
            "run",
            "-n",
            self.conda_env,
            "python",
            str(self.retargeter_root / "app" / "bvh_to_csv_converter.py"),
            "--config",
            str(self.config_path),
            "--viewer",
            "null",
        ]

    def run(self) -> subprocess.CompletedProcess[str]:
        converter = self.retargeter_root / "app" / "bvh_to_csv_converter.py"
        if not converter.is_file():
            raise FileNotFoundError(f"soma-retargeter converter not found: {converter}")
        if not self.bvh_path.is_file():
            raise FileNotFoundError(f"BVH file not found: {self.bvh_path}")
        try:
            self.bvh_path.relative_to(self.bvh_root)
        except ValueError as exc:
            raise ValueError(f"BVH path must live under {self.bvh_root}: {self.bvh_path}") from exc
        self.write_config()
        result = subprocess.run(
            self.command(),
            cwd=str(self.retargeter_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode == 0 and self.job_csv_path.is_file():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.job_csv_path, self.csv_path)
        return result
