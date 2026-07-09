# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma-2-2b-it text encoder for PiD.

Loads from a local directory (or HuggingFace ID as fallback).
Replaces PiD's _TEXT_ENCODER_DICT / _load_text_encoder.

Replicates the chi-prompt prefixing + max_length padding + select_index slicing
from PixelDiTModel._encode_text_raw so the caption embedding fed to PidNet
matches the training distribution exactly.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chi-prompt prefix used by the SFT-distill experiments. Every caption is
# prefixed with this prompt-engineering string before encoding so the Gemma
# hidden states match the training distribution. MUST stay in sync with
# PiD/pid/_src/configs/pid/experiment/shared_config.py::_CHI_PROMPT.
_CHI_PROMPT = [
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
]

# Matches PidNet's txt_max_length (see config.py _SHARED_BACKBONE).
_MODEL_MAX_LENGTH = 300


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

        # Chi-prompt prefix joined into a single string, matching
        # PixelDiTModel.__init__ which does "\n".join(config.chi_prompt).
        self._chi_prompt_str = "\n".join(_CHI_PROMPT)
        self._num_chi_tokens = len(self.tokenizer.encode(self._chi_prompt_str))

    @torch.no_grad()
    def encode(self, captions: list[str]) -> torch.Tensor:
        """Encode captions -> (B, model_max_length, 2304) hidden states.

        Replicates PixelDiTModel._encode_text_raw: prepend chi-prompt, pad to
        max_length, run Gemma decoder, then slice via select_index to keep BOS +
        the last (model_max_length - 1) tokens. The PiD student was trained with
        this fixed 300-token layout (chi-prompt tail + caption + right-padding)
        and no attention mask is applied downstream, so matching it is required
        for correct image quality.
        """
        prompts_all = [self._chi_prompt_str + cap for cap in captions]
        max_length_all = self._num_chi_tokens + _MODEL_MAX_LENGTH - 2

        caption_token = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)

        caption_embs = self.model(
            caption_token.input_ids,
            caption_token.attention_mask,
        )[0]

        select_index = [0] + list(range(-_MODEL_MAX_LENGTH + 1, 0))
        caption_embs = caption_embs[:, select_index]
        return caption_embs
