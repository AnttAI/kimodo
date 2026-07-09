# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM2Vec encoder wrapper for Kimodo text conditioning."""

import os

import numpy as np
import torch

from .llm2vec import LLM2Vec


class LLM2VecEncoder:
    """LLM2Vec text embeddings."""

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        dtype: str,
        llm_dim: int,
        device: str | torch.device = "auto",
    ) -> None:
        torch_dtype = getattr(torch, dtype)
        self.llm_dim = llm_dim
        requested_device = os.environ.get("TEXT_ENCODER_DEVICE", str(device)).strip().lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(requested_device)

        cache_dir = os.environ.get("HUGGINGFACE_CACHE_DIR")

        if "TEXT_ENCODERS_DIR" in os.environ:
            base_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], base_model_name_or_path)
            peft_model_name_or_path = os.path.join(os.environ["TEXT_ENCODERS_DIR"], peft_model_name_or_path)

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name_or_path,
            peft_model_name_or_path=peft_model_name_or_path,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device: torch.device | str):
        # An explicit TEXT_ENCODER_DEVICE keeps the large encoder off the
        # motion model's GPU even when the parent Kimodo model calls .to().
        requested_device = os.environ.get("TEXT_ENCODER_DEVICE", str(device))
        if requested_device.strip().lower() == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(requested_device)
        self.model = self.model.to(self._device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def get_device(self):
        return self._device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        with torch.no_grad():
            encoded_text = self.model.encode(
                text,
                batch_size=len(text),
                show_progress_bar=False,
                device=str(self._device),
            )

        assert len(encoded_text.shape)
        assert self.llm_dim == encoded_text.shape[-1]

        encoded_text = encoded_text[:, None]
        lengths = np.ones(len(encoded_text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        encoded_text = torch.tensor(encoded_text).to(self.get_device())
        return encoded_text, lengths


class DummyTextEncoder:
    """Zero-vector text encoder for constraint-only generation without LLM weights.

    Activated by setting TEXT_ENCODER_MODE=dummy. Returns zero embeddings
    of the correct shape (llm_dim=4096), which the model treats as
    unconditional (same as empty-text in classifier-free guidance training).

    This allows running Kimodo on GPUs with <17GB VRAM and without
    Llama-3 access, using only kinematic constraints for motion control.
    """

    def __init__(self, llm_dim: int = 4096, device: str = "cuda:0") -> None:
        self.llm_dim = llm_dim
        self._device = torch.device(device)
        print(f"[Kimodo] Using DummyTextEncoder (zero embeddings, dim={llm_dim})")
        print("[Kimodo] Text prompts will be ignored. Use constraints for motion control.")

    def to(self, device: torch.device):
        self._device = torch.device(device)
        return self

    def eval(self):
        return self

    def get_device(self):
        return self._device

    def __call__(self, text: list[str] | str):
        is_string = False
        if isinstance(text, str):
            text = [text]
            is_string = True

        encoded_text = torch.zeros(len(text), 1, self.llm_dim, device=self._device)
        lengths = np.ones(len(text), dtype=int).tolist()

        if is_string:
            encoded_text = encoded_text[0]
            lengths = lengths[0]

        return encoded_text, lengths
