# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-2-2b-it text encoder for PiD.

Loads from a local directory (or HuggingFace ID as fallback).
Replaces PiD's _TEXT_ENCODER_DICT / _load_text_encoder.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GemmaTextEncoder:
    """Frozen Gemma-2-2b-it decoder for PiD caption encoding.

    Args:
        model_id: Local path or HF ID for gemma-2-2b-it.
        device: Target device.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "right"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            .get_decoder()
            .to(device)
        )
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def encode(self, captions: list[str]) -> torch.Tensor:
        """Encode captions -> (B, seq_len, 2304) hidden states."""
        tokens = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300,
        ).to(self.model.device)
        outputs = self.model(**tokens, output_hidden_states=True)
        return outputs.last_hidden_state
