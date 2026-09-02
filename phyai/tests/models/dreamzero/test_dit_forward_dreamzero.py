from __future__ import annotations

import torch
import torch.nn as nn

import phyai.layers.linear as L
import phyai.models.dreamzero.modeling_dreamzero as modeling
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


def test_dreamzero_fa2_cross_attention_matches_official_call(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_flash_attn_varlen_func(**kwargs):
        captured.update(kwargs)
        return kwargs["q"].clone()

    monkeypatch.setattr(
        modeling,
        "_get_flash_attn_varlen_func",
        lambda: fake_flash_attn_varlen_func,
    )
    query = torch.randn(2, 3, 2, 4, dtype=torch.float64)
    key = torch.randn(2, 5, 2, 4, dtype=torch.float32)
    value = torch.randn(2, 5, 2, 4, dtype=torch.float16)

    output = modeling._dreamzero_fa2_cross_attention(query, key, value)

    assert output.dtype == query.dtype
    assert output.shape == query.shape
    torch.testing.assert_close(
        output,
        query.flatten(0, 1).to(value.dtype).unflatten(0, (2, 3)).to(query.dtype),
        rtol=0,
        atol=0,
    )
    assert captured["q"].shape == (6, 2, 4)
    assert captured["k"].shape == (10, 2, 4)
    assert captured["v"].shape == (10, 2, 4)
    assert captured["q"].dtype == value.dtype
    assert captured["k"].dtype == value.dtype
    torch.testing.assert_close(
        captured["cu_seqlens_q"], torch.tensor([0, 3, 6], dtype=torch.int32)
    )
    torch.testing.assert_close(
        captured["cu_seqlens_k"], torch.tensor([0, 5, 10], dtype=torch.int32)
    )
    assert captured["max_seqlen_q"] == 3
    assert captured["max_seqlen_k"] == 5
    assert captured["dropout_p"] == 0.0
    assert captured["softmax_scale"] is None
    assert captured["causal"] is False
    assert captured["window_size"] == (-1, -1)
    assert captured["deterministic"] is False


def test_dreamzero_cross_attention_uses_fa2_only_for_te(fake_mesh, monkeypatch) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()

    class RecordingAttention(nn.Module):
        def __init__(self, *args, backend: str, **kwargs) -> None:
            super().__init__()
            self.backend = backend
            self.calls = 0

        def forward(
            self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            self.calls += 1
            return query

    fa2_calls: list[tuple[torch.Size, torch.Size, torch.Size]] = []

    def fake_fa2(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        fa2_calls.append((query.shape, key.shape, value.shape))
        return query

    monkeypatch.setattr(modeling, "Attention", RecordingAttention)
    monkeypatch.setattr(modeling, "_dreamzero_fa2_cross_attention", fake_fa2)
    x = torch.randn(1, 4, 16)
    context = torch.randn(1, 3, 16)

    te_attention = modeling.DreamZeroCrossAttention(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="te",
        norm_backend="torch",
    )
    te_attention(x, context, image_context_tokens=2)

    assert len(fa2_calls) == 2
    assert te_attention.attn.calls == 0

    eager_attention = modeling.DreamZeroCrossAttention(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="torch",
    )
    eager_attention(x, context, image_context_tokens=2)

    assert len(fa2_calls) == 2
    assert eager_attention.attn.calls == 2


def test_dreamzero_image_projection_matches_official_layer_norm_epsilon(
    fake_mesh,
) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="torch",
    )

    assert model.img_emb["proj_0_norm"].variance_epsilon == 1e-5
    assert model.img_emb["proj_4_norm"].variance_epsilon == 1e-5


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
    assert all(cache is not None for cache in runner.kv_cache)
    for cache in runner.kv_cache:
        assert cache is not None
        assert cache.shape == (2, 1, 4, 2, 8)

    runner.reset()
    assert runner.kv_cache == []
    assert runner.crossattn_cache == []


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

    with pytest.raises(ValueError, match="exceeds configured"):
        runner.forward(batch)
