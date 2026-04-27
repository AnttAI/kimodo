# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for loading motion files used by Kimodo viewers and demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.skeleton import SkeletonBase
from kimodo.skeleton.bvh import parse_bvh_motion
from kimodo.skeleton.registry import build_skeleton


def _load_motion_npz(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SkeletonBase]:
    with np.load(path, allow_pickle=False) as data:
        if "joints_pos" in data:
            joints_pos = torch.from_numpy(data["joints_pos"]).to(device)
        elif "posed_joints" in data:
            joints_pos = torch.from_numpy(data["posed_joints"]).to(device)
        else:
            raise ValueError(f"{path}: missing 'joints_pos' or 'posed_joints'")

        if "joints_rot" in data:
            joints_rot = torch.from_numpy(data["joints_rot"]).to(device)
        elif "global_rot_mats" in data:
            joints_rot = torch.from_numpy(data["global_rot_mats"]).to(device)
        else:
            raise ValueError(f"{path}: missing 'joints_rot' or 'global_rot_mats'")

        foot_contacts = torch.from_numpy(data["foot_contacts"]).to(device) if "foot_contacts" in data else None

    if joints_pos.ndim == 4:
        joints_pos = joints_pos[0]
    if joints_rot.ndim == 5:
        joints_rot = joints_rot[0]
    if foot_contacts is not None and foot_contacts.ndim == 3:
        foot_contacts = foot_contacts[0]

    if joints_pos.ndim != 3 or joints_rot.ndim != 4:
        raise ValueError(f"{path}: unexpected tensor shapes for motion data")

    skeleton = build_skeleton(joints_pos.shape[1])
    return joints_pos, joints_rot, foot_contacts, skeleton


def _load_motion_bvh(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SkeletonBase]:
    local_rot_mats, root_trans, _fps = parse_bvh_motion(str(path))
    skeleton = build_skeleton(local_rot_mats.shape[1]).to(device)

    # BVH local rotations are authored in the source BVH rest-pose convention.
    # Convert them into Kimodo's standard T-pose convention before FK.
    local_rot_mats = local_rot_mats.to(device)
    local_rot_mats, _ = skeleton.to_standard_tpose(local_rot_mats)
    root_trans = root_trans.to(device=device, dtype=local_rot_mats.dtype)

    joints_rot, joints_pos, _ = skeleton.fk(local_rot_mats, root_trans)
    return joints_pos, joints_rot, None, skeleton


def _load_motion_csv(path: Path, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SkeletonBase]:
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")

    expected_header_prefix = [
        "Frame",
        "root_translateX",
        "root_translateY",
        "root_translateZ",
        "root_rotateX",
        "root_rotateY",
        "root_rotateZ",
    ]
    if header[: len(expected_header_prefix)] != expected_header_prefix:
        raise ValueError(
            f"{path}: unsupported CSV format. Expected soma-retargeter G1 CSV header starting with "
            f"{expected_header_prefix!r}."
        )

    csv_data = np.loadtxt(path, delimiter=",", skiprows=1)
    if csv_data.ndim == 1:
        csv_data = csv_data[None, :]
    if csv_data.shape[1] != 36:
        raise ValueError(f"{path}: expected 36 CSV columns, got {csv_data.shape[1]}.")

    skeleton = build_skeleton(34).to(device)
    converter = MujocoQposConverter(skeleton)

    root_positions_mujoco = torch.from_numpy(csv_data[:, 1:4] * 0.01).to(device=device, dtype=torch.float32)
    root_rot_mujoco = torch.from_numpy(
        Rotation.from_euler("xyz", np.deg2rad(csv_data[:, 4:7])).as_matrix()
    ).to(device=device, dtype=torch.float32)

    mujoco_to_kimodo = converter.mujoco_to_kimodo_matrix.to(device=device, dtype=torch.float32)
    kimodo_to_mujoco = converter.kimodo_to_mujoco_matrix.to(device=device, dtype=torch.float32)

    root_positions = torch.matmul(mujoco_to_kimodo[None, ...], root_positions_mujoco[..., None]).squeeze(-1)
    root_rot = torch.matmul(
        torch.matmul(mujoco_to_kimodo[None, ...], root_rot_mujoco),
        kimodo_to_mujoco[None, ...],
    )

    # soma-retargeter CSVs are expressed in a frame that is yawed 90 degrees relative
    # to the SOMA BVH clips we compare against in Kimodo. Rotate the whole root motion
    # into Kimodo's forward convention so the human and G1 face/move the same way.
    csv_to_kimodo_yaw = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    root_positions = torch.matmul(csv_to_kimodo_yaw[None, ...], root_positions[..., None]).squeeze(-1)
    root_rot = torch.matmul(csv_to_kimodo_yaw[None, ...], root_rot)

    joint_dofs = torch.from_numpy(np.deg2rad(csv_data[:, 7:])).to(device=device, dtype=torch.float32)[None, ...]
    local_rot_mats = torch.eye(3, device=device, dtype=torch.float32)[None, None, None, ...].repeat(
        1, joint_dofs.shape[1], skeleton.nbjoints, 1, 1
    )
    local_rot_mats[:, :, skeleton.root_idx] = root_rot[None, ...]
    local_rot_mats = converter._joint_dofs_to_local_rot_mats(
        joint_dofs,
        local_rot_mats,
        device=torch.device(device),
        dtype=local_rot_mats.dtype,
        use_relative=True,
    )

    joints_rot, joints_pos, _ = skeleton.fk(local_rot_mats.squeeze(0), root_positions)
    return joints_pos, joints_rot, None, skeleton


def load_motion_file(path: str | Path, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, SkeletonBase]:
    """Load a motion clip from Kimodo NPZ, compatible BVH, or soma-retargeter G1 CSV.

    Returns `(joints_pos, joints_rot, foot_contacts, skeleton)`.
    """
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_motion_npz(path, device)
    if suffix == ".bvh":
        return _load_motion_bvh(path, device)
    if suffix == ".csv":
        return _load_motion_csv(path, device)
    raise ValueError(f"Unsupported motion file format: {path.suffix}. Expected .npz, .bvh, or .csv")
