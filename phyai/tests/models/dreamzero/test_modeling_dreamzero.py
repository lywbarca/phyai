from __future__ import annotations

import torch

import phyai.layers.linear as L
from phyai.models.dreamzero import DreamZeroConfig, DreamZeroDiT, DreamZeroDiTConfig


def _init_linear_dispatcher() -> None:
    L.init(register_flashinfer=False, validate=False)


def _tiny_tp4_config() -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=8,
        action_horizon=4,
        max_action_dim=8,
        max_state_dim=16,
        hidden_size=16,
        input_embedding_dim=1536,
        num_frame_per_block=2,
        dit=DreamZeroDiTConfig(
            dim=64,
            ffn_dim=128,
            frame_seqlen=8,
            freq_dim=16,
            in_dim=36,
            num_action_per_block=4,
            num_frame_per_block=2,
            num_heads=4,
            num_layers=2,
            num_state_per_block=1,
            out_dim=16,
        ),
    )


def test_dreamzero_dit_tp4_skeleton_shapes(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 4})
    _init_linear_dispatcher()
    cfg = _tiny_tp4_config()

    model = DreamZeroDiT(cfg, params_dtype=torch.bfloat16, device="cpu")

    assert len(model.blocks) == 2
    block0 = model.blocks[0]
    assert block0.self_attn.num_local_heads == 1
    assert block0.cross_attn.num_local_heads == 1

    assert block0.self_attn.q.weight.shape == (16, 64)
    assert block0.self_attn.k.weight.shape == (16, 64)
    assert block0.self_attn.v.weight.shape == (16, 64)
    assert block0.self_attn.o.weight.shape == (64, 16)

    assert block0.cross_attn.q.weight.shape == (16, 64)
    assert block0.cross_attn.k_img.weight.shape == (16, 64)
    assert block0.cross_attn.o.weight.shape == (64, 16)

    assert block0.ffn.fc1.weight.shape == (32, 64)
    assert block0.ffn.fc2.weight.shape == (64, 32)
    assert block0.modulation.shape == (1, 6, 64)

    assert model.patch_embedding.weight.shape == (64, 36, 1, 2, 2)
    assert model.head.head.weight.shape == (64, 64)
    assert model.action_encoder.W1.W.shape == (1, 8, 64)
    assert model.action_decoder.layer2.W.shape == (1, 16, 8)


def test_dreamzero_dit_tp4_attaches_sharded_weight_loaders(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 2})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_tp4_config(),
        params_dtype=torch.bfloat16,
        device="cpu",
    )

    q = model.blocks[0].self_attn.q
    o = model.blocks[0].self_attn.o

    assert q.weight.hf_keys == [("blocks.0.self_attn.q.weight", None)]
    assert q.bias.hf_keys == [("blocks.0.self_attn.q.bias", None)]
    assert o.weight.hf_keys == [("blocks.0.self_attn.o.weight", None)]
    assert o.bias.hf_keys == [("blocks.0.self_attn.o.bias", None)]

    q_src = torch.arange(64 * 64, dtype=torch.bfloat16).reshape(64, 64)
    q.weight.weight_loader(q.weight, q_src, None)
    torch.testing.assert_close(q.weight, q_src.narrow(0, 32, 16))

    o_src = torch.arange(64 * 64, dtype=torch.bfloat16).reshape(64, 64)
    o.weight.weight_loader(o.weight, o_src, None)
    torch.testing.assert_close(o.weight, o_src.narrow(1, 32, 16))


def test_dreamzero_dit_tp4_replicates_non_tp_parameters(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 4}, ranks={"tp": 3})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_tp4_config(),
        params_dtype=torch.bfloat16,
        device="cpu",
    )

    assert model.patch_embedding.weight.hf_keys == [("patch_embedding.weight", None)]
    assert model.blocks[0].modulation.hf_keys == [("blocks.0.modulation", None)]
    assert model.action_encoder.W1.W.hf_keys == [("action_encoder.W1.W", None)]
