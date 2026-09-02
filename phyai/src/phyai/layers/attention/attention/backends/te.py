"""Transformer Engine/cuDNN backend for dense no-cache attention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from phyai.layers.attention.attention.base import (
    AttentionBackend,
    AttentionLayerProto,
    AttnCtx,
    AttnMetadata,
    AttnPlanHandle,
)
from phyai.layers.attention.attention.registry import register_backend


if TYPE_CHECKING:
    from phyai.runtime.model_runner import ModelRunner


@dataclass(frozen=True)
class TransformerEngineAttentionPlan(AttnPlanHandle):
    """Empty plan handle; dense TE attention plans inside cuDNN."""


@register_backend("te")
class TransformerEngineAttentionBackend(AttentionBackend):
    """Official DreamZero Transformer Engine F16 fused-attention path."""

    def __init__(self, runner: "ModelRunner | None" = None) -> None:
        del runner
        try:
            from phyai.layers.attention.attention.backends.cudnn_attention import (
                DotProductAttention,
            )
        except ImportError as exc:
            raise ImportError(
                "backend='te' requires transformer-engine and transformer-engine-torch."
            ) from exc
        self._attention_cls: type[torch.nn.Module] = DotProductAttention
        self._attention: torch.nn.Module | None = None
        self._config: tuple[int, int, bool] | None = None

    def init_forward_metadata(self, meta: AttnMetadata) -> AttnPlanHandle:
        if meta.mode.is_idle():
            return TransformerEngineAttentionPlan()
        if not meta.layout.is_padded():
            raise NotImplementedError(
                "Transformer Engine attention supports padded 4-D BSHD input only."
            )
        layer = meta.extras.get("layer_proto")
        if layer is None:
            raise ValueError(
                "TransformerEngineAttentionBackend.init_forward_metadata requires "
                "extras['layer_proto']."
            )
        self._ensure_attention(layer)
        return TransformerEngineAttentionPlan()

    def forward(
        self,
        layer: AttentionLayerProto,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx: AttnCtx,
    ) -> torch.Tensor:
        if ctx.mode.is_idle():
            return q.new_zeros(q.shape)
        if not ctx.layout.is_padded():
            raise NotImplementedError(
                "Transformer Engine attention supports padded 4-D BSHD input only."
            )
        self._validate_inputs(q, k, v)
        attention = self._ensure_attention(layer)
        return attention(query_layer=q, key_layer=k, value_layer=v)

    def _ensure_attention(self, layer: AttentionLayerProto) -> torch.nn.Module:
        self._validate_layer(layer)
        config = (layer.num_heads, layer.head_dim, layer.causal)
        if self._attention is None:
            self._attention = self._attention_cls(
                num_attention_heads=layer.num_heads,
                kv_channels=layer.head_dim,
                qkv_format="bshd",
                attn_mask_type="causal" if layer.causal else "no_mask",
                attention_dropout=0.0,
            )
            self._attention.eval()
            self._config = config
        elif self._config != config:
            raise RuntimeError(
                f"TE backend was initialized for {self._config}, got {config}."
            )
        return self._attention

    @staticmethod
    def _validate_layer(layer: AttentionLayerProto) -> None:
        expected_scale = 1.0 / math.sqrt(layer.head_dim)
        if not math.isclose(layer.scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
            raise NotImplementedError(
                "Transformer Engine backend requires the default 1/sqrt(head_dim) scale."
            )
        if layer.sliding_window is not None:
            raise NotImplementedError(
                "Transformer Engine backend does not support sliding_window here."
            )
        if layer.logits_soft_cap is not None:
            raise NotImplementedError(
                "Transformer Engine backend does not support logits_soft_cap."
            )

    @staticmethod
    def _validate_inputs(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        if q.device.type != "cuda":
            raise ValueError("Transformer Engine attention requires CUDA tensors.")
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "Transformer Engine attention requires float16 or bfloat16 tensors."
            )
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError(
                f"q/k/v dtypes must match, got {q.dtype}, {k.dtype}, {v.dtype}."
            )


__all__ = [
    "TransformerEngineAttentionBackend",
    "TransformerEngineAttentionPlan",
]
