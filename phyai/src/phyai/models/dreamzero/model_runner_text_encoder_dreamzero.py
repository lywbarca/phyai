"""DreamZero text encoder runner."""

from __future__ import annotations

import os

import torch

from phyai.models.dreamzero.text_encoder_wan import DreamZeroWanTextEncoder
from phyai.runtime.model_runner import ModelRunner


class DreamZeroTextEncoderRunner(ModelRunner):
    """Wraps the DreamZero Wan2.1 UMT5 text encoder."""

    def __init__(
        self,
        text_encoder: DreamZeroWanTextEncoder,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.text_encoder = text_encoder
        self.device = torch.device(device)
        self.dtype = dtype
        self._compiled_forward = None
        self._compile_scope = "both"

    def setup(self) -> None:
        if os.getenv("DREAMZERO_COMPILE_TEXT_ENCODER", "false").lower() == "true":
            compile_mode = os.getenv(
                "DREAMZERO_TEXT_ENCODER_COMPILE_MODE", "reduce-overhead"
            )
            self._compile_scope = os.getenv(
                "DREAMZERO_TEXT_ENCODER_COMPILE_SCOPE", "both"
            ).lower()
            if self._compile_scope not in {"both", "positive", "negative"}:
                raise ValueError(
                    "DREAMZERO_TEXT_ENCODER_COMPILE_SCOPE must be one of "
                    "both, positive, or negative; "
                    f"got {self._compile_scope!r}."
                )
            compiled_forward = torch.compile(
                self.text_encoder.forward,
                mode=compile_mode,
                fullgraph=True,
                dynamic=False,
            )
            if self._compile_scope == "both":
                self.text_encoder.forward = compiled_forward
            else:
                self._compiled_forward = compiled_forward

    def _encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        prompt_kind: str | None,
    ) -> torch.Tensor:
        if input_ids.shape != attention_mask.shape:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match "
                f"attention_mask shape {tuple(attention_mask.shape)}."
            )
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        attention_mask = attention_mask.to(device=self.device)
        if self._compiled_forward is not None and prompt_kind == self._compile_scope:
            prompt_emb = self._compiled_forward(input_ids, attention_mask)
        else:
            prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=self.dtype)
        return prompt_emb.masked_fill(
            attention_mask.to(torch.bool).unsqueeze(-1) == 0, 0
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_prompt(input_ids, attention_mask, prompt_kind=None)

    @torch.no_grad()
    def encode_positive_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_prompt(input_ids, attention_mask, prompt_kind="positive")

    @torch.no_grad()
    def encode_negative_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self._encode_prompt(input_ids, attention_mask, prompt_kind="negative")

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_prompt(input_ids, attention_mask)


__all__ = ["DreamZeroTextEncoderRunner"]
