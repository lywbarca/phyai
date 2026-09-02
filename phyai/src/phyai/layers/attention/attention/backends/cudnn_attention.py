"""Forward-only cuDNN fused attention wrapper matching DreamZero upstream.

This is the inference subset of NVIDIA DreamZero's
``modules/cudnn_attention.py``. It intentionally calls Transformer Engine's
low-level ``fused_attn_fwd`` entry point with the same argument ordering and
forces the same F16 arbitrary-sequence-length implementation selected by the
upstream wrapper. Only dense BSHD FP16/BF16 inference is supported.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import transformer_engine
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import (
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    QKVLayout,
    fused_attn_fwd,
)


_TE_VERSION = tuple(int(part) for part in transformer_engine.__version__.split(".")[:2])

try:
    import transformer_engine.pytorch.attention.dot_product_attention.utils as dpa_utils
except ImportError:
    import transformer_engine.pytorch.dot_product_attention.utils as dpa_utils

if _TE_VERSION >= (2, 8):
    from transformer_engine.pytorch.cpp_extensions.fused_attn import SoftmaxType


_F16_ARB_ELTS_PER_THREAD = 16


class DotProductAttention(torch.nn.Module):
    """Dense BSHD dot-product attention matching the official DreamZero path."""

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        *,
        attention_dropout: float = 0.0,
        qkv_format: str = "bshd",
        attn_mask_type: str = "no_mask",
    ) -> None:
        super().__init__()
        if qkv_format != "bshd":
            raise ValueError(f"Only bshd layout is supported, got {qkv_format!r}.")
        if attn_mask_type not in ("causal", "no_mask"):
            raise ValueError(
                f"attn_mask_type must be 'causal' or 'no_mask', got {attn_mask_type!r}."
            )
        self.num_attention_heads = int(num_attention_heads)
        self.softmax_scale = 1.0 / math.sqrt(kv_channels)
        self.attention_dropout = float(attention_dropout)
        self.attn_mask_type = attn_mask_type
        self.window_size = dpa_utils.check_set_window_size(attn_mask_type)

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_seqlen_q = query_layer.shape[:2]
        max_seqlen_kv = key_layer.shape[1]
        device = query_layer.device

        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_q,
            step=max_seqlen_q,
            dtype=torch.int32,
            device=device,
        )
        cu_seqlens_kv = torch.arange(
            0,
            (batch_size + 1) * max_seqlen_kv,
            step=max_seqlen_kv,
            dtype=torch.int32,
            device=device,
        )
        outputs = _fused_attn_forward(
            query_layer.contiguous(),
            key_layer.contiguous(),
            value_layer.contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask_type=self.attn_mask_type,
            window_size=self.window_size,
            softmax_scale=self.softmax_scale,
            dropout=self.attention_dropout if self.training else 0.0,
            is_training=self.training,
        )
        return outputs[0]


def _fused_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    attn_mask_type: str,
    window_size: Any,
    softmax_scale: float,
    dropout: float,
    is_training: bool,
) -> tuple[torch.Tensor | None, ...]:
    if _TE_VERSION >= (2, 18):
        return fused_attn_fwd(
            is_training,
            max_seqlen_q,
            max_seqlen_kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            q,
            k,
            v,
            q.dtype,
            FusedAttnBackend["F16_arbitrary_seqlen"],
            attn_scale=softmax_scale,
            dropout=dropout,
            qkv_layout="bshd_bshd_bshd",
            o_format="bshd",
            attn_bias_type="no_bias",
            attn_mask_type=attn_mask_type,
            softmax_type="vanilla",
            window_size=tuple(window_size),
        )

    args: tuple[Any, ...] = (
        max_seqlen_q,
        max_seqlen_kv,
        is_training,
        softmax_scale,
        dropout,
        True,
        QKVLayout["bshd_bshd_bshd"],
        AttnBiasType["no_bias"],
        AttnMaskType[attn_mask_type],
    )
    if _TE_VERSION >= (2, 8):
        args += (SoftmaxType["vanilla"],)
    args += (
        tuple(window_size),
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        q.dtype,
        cu_seqlens_q,
        cu_seqlens_kv,
        None,
        None,
        None,
        None,
        None,
    )
    if _TE_VERSION >= (2, 8):
        args += (None,)
    args += (None, _F16_ARB_ELTS_PER_THREAD)
    if _TE_VERSION >= (2, 9):
        args += (False,)
    if _TE_VERSION >= (2, 10):
        args += (False,)
    return tex.fused_attn_fwd(*args)


__all__ = ["DotProductAttention"]
