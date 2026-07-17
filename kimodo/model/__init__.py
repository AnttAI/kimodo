# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo model package: main model class, text encoders, and loading utilities."""

from .common import resolve_target
from .registry import AVAILABLE_MODELS, DEFAULT_MODEL, DEFAULT_TEXT_ENCODER_URL, MODEL_NAMES

__all__ = [
    "Kimodo",
    "LLM2VecEncoder",
    "TMR",
    "TwostageDenoiser",
    "load_model",
    "load_checkpoint_state_dict",
    "resolve_target",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_ENCODER_URL",
    "MODEL_NAMES",
]


def __getattr__(name: str):
    if name == "Kimodo":
        from .kimodo_model import Kimodo

        return Kimodo
    if name == "LLM2VecEncoder":
        from .llm2vec import LLM2VecEncoder

        return LLM2VecEncoder
    if name == "TMR":
        from .tmr import TMR

        return TMR
    if name == "TwostageDenoiser":
        from .twostage_denoiser import TwostageDenoiser

        return TwostageDenoiser
    if name == "load_model":
        from .load_model import load_model

        return load_model
    if name == "load_checkpoint_state_dict":
        from .loading import load_checkpoint_state_dict

        return load_checkpoint_state_dict
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
