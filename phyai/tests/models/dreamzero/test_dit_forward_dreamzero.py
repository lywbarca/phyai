from __future__ import annotations

import torch
import torch.nn as nn

import phyai.layers.linear as L
from phyai.layers import LayerNorm, RMSNorm
from phyai.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroDiT,
    DreamZeroDiTConfig,
    DreamZeroDiTForwardBatch,
    DreamZeroDiTRunner,
)


def _init_linear_dispatcher() -> None:
    L.init(register_flashinfer=False, validate=False)


def _tiny_config(*, num_layers: int = 1) -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=4,
        action_horizon=2,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        input_embedding_dim=1536,
        num_frame_per_block=1,
        dit=DreamZeroDiTConfig(
            dim=16,
            ffn_dim=32,
            frame_seqlen=4,
            freq_dim=8,
            in_dim=36,
            max_chunk_size=-1,
            num_action_per_block=2,
            num_frame_per_block=1,
            num_heads=2,
            num_layers=num_layers,
            num_state_per_block=1,
            out_dim=2,
        ),
    )


def _replace_norms_with_identity(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (LayerNorm, RMSNorm)):
            module._modules[name] = nn.Identity()
        else:
            _replace_norms_with_identity(child)


def _init_tiny_model(model: nn.Module) -> None:
    _replace_norms_with_identity(model)
    for param in model.parameters():
        if param.dim() == 1:
            param.data.zero_()
        else:
            param.data.normal_(mean=0.0, std=0.02)


def test_dreamzero_dit_full_forward_with_action_registers(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
    )
    _init_tiny_model(model)

    x = torch.randn(1, 36, 2, 4, 4)
    timestep = torch.tensor([[0.2, 0.7]], dtype=torch.float32)
    context = torch.randn(1, 3, 4096)
    clip_feature = torch.randn(1, 2, 1280)
    action = torch.randn(1, 2, 4)
    timestep_action = torch.tensor([[0.3, 0.6]], dtype=torch.float32)
    state = torch.randn(1, 1, 5)

    video, action_out, kv_cache, crossattn_cache = model(
        x,
        timestep,
        context,
        clip_feature=clip_feature,
        action=action,
        timestep_action=timestep_action,
        state=state,
        image_context_tokens=2,
    )

    assert video.shape == (1, 2, 2, 4, 4)
    assert action_out is not None
    assert action_out.shape == (1, 2, 4)
    assert kv_cache == [None]
    assert crossattn_cache == [None]


def test_dreamzero_dit_runner_owns_and_updates_kv_cache(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_config(num_layers=2),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
    )
    _init_tiny_model(model)
    runner = DreamZeroDiTRunner(model, device="cpu")

    batch = DreamZeroDiTForwardBatch(
        x=torch.randn(1, 36, 1, 4, 4),
        timestep=torch.tensor([[0.4]], dtype=torch.float32),
        context=torch.randn(1, 3, 4096),
        clip_feature=torch.randn(1, 2, 1280),
        action=torch.randn(1, 2, 4),
        timestep_action=torch.tensor([[0.4, 0.4]], dtype=torch.float32),
        state=torch.randn(1, 1, 5),
        image_context_tokens=2,
    )

    output = runner.forward(batch)

    assert output.video.shape == (1, 2, 1, 4, 4)
    assert output.action is not None
    assert output.action.shape == (1, 2, 4)
    assert len(runner.kv_cache) == 2
    assert runner._kv_cache is not None
    assert runner._kv_cache.seq_len == 4
    assert all(cache is not None for cache in runner.kv_cache)
    for cache in runner.kv_cache:
        assert cache is not None
        assert cache.shape == (2, 1, 4, 2, 8)

    runner.reset()
    assert runner.kv_cache == []
    assert runner.crossattn_cache == []
    assert runner._kv_cache is not None
    assert runner._kv_cache.seq_len == 0


def test_dreamzero_dit_runner_rejects_cache_capacity_overflow(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_config(num_layers=1),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
    )
    _init_tiny_model(model)
    runner = DreamZeroDiTRunner(model, device="cpu", max_kv_cache_tokens=2)

    batch = DreamZeroDiTForwardBatch(
        x=torch.randn(1, 36, 1, 4, 4),
        timestep=torch.tensor([[0.4]], dtype=torch.float32),
        context=torch.randn(1, 3, 4096),
        clip_feature=torch.randn(1, 2, 1280),
        action=torch.randn(1, 2, 4),
        timestep_action=torch.tensor([[0.4, 0.4]], dtype=torch.float32),
        state=torch.randn(1, 1, 5),
        image_context_tokens=2,
    )

    import pytest

    with pytest.raises(ValueError, match="exceeds capacity"):
        runner.forward(batch)
