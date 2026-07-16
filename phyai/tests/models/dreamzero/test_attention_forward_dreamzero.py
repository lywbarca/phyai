from __future__ import annotations

import torch
import torch.nn as nn

import phyai.layers.linear as L
from phyai.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroCrossAttention,
    DreamZeroDiTConfig,
    DreamZeroSelfAttention,
)


def _init_linear_dispatcher() -> None:
    L.init(register_flashinfer=False, validate=False)


def _tiny_config() -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=4,
        action_horizon=3,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        input_embedding_dim=1536,
        num_frame_per_block=1,
        dit=DreamZeroDiTConfig(
            dim=8,
            ffn_dim=12,
            frame_seqlen=4,
            freq_dim=4,
            in_dim=36,
            num_action_per_block=3,
            num_frame_per_block=1,
            num_heads=2,
            num_layers=1,
            num_state_per_block=1,
            out_dim=2,
        ),
    )


def _disable_qk_norm(module: nn.Module) -> None:
    if hasattr(module, "norm_q"):
        module.norm_q = nn.Identity()
    if hasattr(module, "norm_k"):
        module.norm_k = nn.Identity()
    if hasattr(module, "norm_k_img"):
        module.norm_k_img = nn.Identity()


def test_self_attention_forward_returns_output_and_stateless_cache(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    attn = DreamZeroSelfAttention(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
        prefix="blocks.0.self_attn",
    )
    _disable_qk_norm(attn)
    for param in attn.parameters():
        param.data.normal_(std=0.05)

    x = torch.randn(2, 4, 8)
    out, cache = attn(x, use_cache=True)

    assert out.shape == x.shape
    assert cache is not None
    assert cache.shape == (2, 2, 4, 2, 4)
    assert not hasattr(attn, "kv_cache")


def test_self_attention_forward_appends_external_cache(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    attn = DreamZeroSelfAttention(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
    )
    _disable_qk_norm(attn)
    for param in attn.parameters():
        param.data.normal_(std=0.05)

    x0 = torch.randn(2, 3, 8)
    _, cache0 = attn(x0, use_cache=True)
    assert cache0 is not None

    x1 = torch.randn(2, 2, 8)
    out1, cache1 = attn(x1, kv_cache=cache0)

    assert out1.shape == x1.shape
    assert cache1 is not None
    assert cache1.shape == (2, 2, 5, 2, 4)
    torch.testing.assert_close(cache1[:, :, :3], cache0)


def test_cross_attention_forward_splits_image_and_text_context(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    attn = DreamZeroCrossAttention(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
        prefix="blocks.0.cross_attn",
    )
    _disable_qk_norm(attn)
    for param in attn.parameters():
        param.data.normal_(std=0.05)

    x = torch.randn(2, 3, 8)
    context = torch.randn(2, 7, 8)
    out, cache = attn(x, context, image_context_tokens=2, use_cache=True)

    assert out.shape == x.shape
    assert cache is not None
    assert cache.shape == (2, 2, 5, 2, 4)
    assert not hasattr(attn, "crossattn_cache")


def test_cross_attention_forward_reuses_external_text_cache(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    attn = DreamZeroCrossAttention(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
    )
    _disable_qk_norm(attn)
    for param in attn.parameters():
        param.data.normal_(std=0.05)

    x = torch.randn(2, 3, 8)
    context = torch.randn(2, 7, 8)
    _, cache0 = attn(x, context, image_context_tokens=2, use_cache=True)
    assert cache0 is not None

    changed_text_context = torch.randn(2, 7, 8)
    out, cache1 = attn(
        x,
        changed_text_context,
        image_context_tokens=2,
        crossattn_cache=cache0,
    )

    assert out.shape == x.shape
    assert cache1 is not None
    torch.testing.assert_close(cache1, cache0)
