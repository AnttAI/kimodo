# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI model server for remote Kimodo generation.

This keeps the heavy Kimodo model/text-encoder process separate from the Viser
viewer process. The viewer sends prompts and already-serialized Kimodo
constraints, and this server returns generated Kimodo NPZ files.
"""

from __future__ import annotations

import base64
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kimodo import DEFAULT_MODEL, load_model
from kimodo.constraints import load_constraints_lst
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.model.cfg import CFG_TYPES
from kimodo.model.registry import get_model_info
from kimodo.tools import seed_everything


class GenerateRequest(BaseModel):
    prompts: list[str] = Field(default_factory=lambda: [""])
    num_frames: list[int]
    model: str = DEFAULT_MODEL
    constraints: list[dict[str, Any]] = Field(default_factory=list)
    num_samples: int = 1
    diffusion_steps: int = 100
    num_transition_frames: int = 5
    seed: Optional[int] = None
    cfg_type: Optional[str] = Field(default=None)
    cfg_weight: Optional[list[float]] = None
    post_processing: bool = True
    root_margin: float = 0.04
    return_npz_base64: bool = True


class GeneratedMotion(BaseModel):
    filename: str
    npz_base64: Optional[str] = None
    path: Optional[str] = None


class GenerateResponse(BaseModel):
    ok: bool
    model: str
    display_name: str
    fps: float
    motions: list[GeneratedMotion]


app = FastAPI(title="Kimodo Model Server")
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
_models: dict[str, tuple[object, str]] = {}
_model_lock = threading.Lock()
_generation_lock = threading.Lock()


def _get_model(model_name: str):
    with _model_lock:
        if model_name not in _models:
            model, resolved = load_model(
                model_name,
                device=_device,
                default_family="Kimodo",
                return_resolved_name=True,
            )
            _models[model_name] = (model, resolved)
        return _models[model_name]


def _single_sample(output: dict[str, Any], sample_idx: int, n_samples: int) -> dict[str, Any]:
    return {
        key: (
            value[sample_idx]
            if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples
            else value
        )
        for key, value in output.items()
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "device": _device, "loaded_models": sorted(_models)}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if not req.num_frames:
        raise HTTPException(status_code=400, detail="num_frames must not be empty")
    if len(req.prompts) != len(req.num_frames):
        raise HTTPException(status_code=400, detail="prompts and num_frames must have the same length")
    if req.cfg_type is not None and req.cfg_type not in CFG_TYPES:
        raise HTTPException(status_code=400, detail=f"cfg_type must be one of {CFG_TYPES}")

    try:
        model, resolved = _get_model(req.model)
        info = get_model_info(resolved)
        display_name = info.display_name if info else resolved
        if req.seed is not None:
            seed_everything(req.seed)

        constraint_lst = load_constraints_lst(req.constraints, model.skeleton) if req.constraints else []
        cfg_kwargs: dict[str, Any] = {}
        if req.cfg_type:
            cfg_kwargs["cfg_type"] = req.cfg_type
        if req.cfg_weight is not None:
            cfg_kwargs["cfg_weight"] = req.cfg_weight

        with _generation_lock:
            output = model(
                req.prompts,
                req.num_frames,
                constraint_lst=constraint_lst,
                num_denoising_steps=req.diffusion_steps,
                num_samples=req.num_samples,
                multi_prompt=True,
                num_transition_frames=req.num_transition_frames,
                post_processing=req.post_processing,
                root_margin=req.root_margin,
                return_numpy=True,
                **cfg_kwargs,
            )

        n_samples = int(output["posed_joints"].shape[0])
        out_dir = Path(tempfile.mkdtemp(prefix="kimodo_model_server_"))
        motions: list[GeneratedMotion] = []
        for idx in range(n_samples):
            filename = f"motion_{idx:02d}.npz" if n_samples > 1 else "motion.npz"
            path = out_dir / filename
            save_kimodo_npz(str(path), _single_sample(output, idx, n_samples))
            encoded = None
            if req.return_npz_base64:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            motions.append(
                GeneratedMotion(
                    filename=filename,
                    npz_base64=encoded,
                    path=str(path) if not req.return_npz_base64 else None,
                )
            )

        return GenerateResponse(
            ok=True,
            model=resolved,
            display_name=display_name,
            fps=float(model.fps),
            motions=motions,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "kimodo.server.model_server:app",
        host=os.environ.get("KIMODO_MODEL_SERVER_HOST", "0.0.0.0"),
        port=int(os.environ.get("KIMODO_MODEL_SERVER_PORT", "8000")),
    )
