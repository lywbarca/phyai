from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

import phyai.layers.linear as L
from phyai.models.dreamzero import (
    DreamZeroActionEncoder,
    DreamZeroCategorySpecificLinear,
    DreamZeroCategorySpecificMLP,
    DreamZeroConfig,
    DreamZeroDiTConfig,
    DreamZeroMLP,
    DreamZeroSinusoidalPositionalEncoding,
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


def test_category_specific_linear_matches_selected_bmm() -> None:
    layer = DreamZeroCategorySpecificLinear(
        num_categories=2,
        input_dim=2,
        output_dim=3,
        params_dtype=torch.float32,
        device="cpu",
    )
    layer.W.data = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    layer.b.data = torch.tensor(
        [
            [0.5, 1.0, 1.5],
            [2.0, 2.5, 3.0],
        ],
        dtype=torch.float32,
    )
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    cat_ids = torch.tensor([0, 1])

    out = layer(x, cat_ids)
    expected = torch.bmm(x, layer.W[cat_ids]) + layer.b[cat_ids].unsqueeze(1)

    torch.testing.assert_close(out, expected)


def test_category_specific_linear_collapses_single_category_ids() -> None:
    layer = DreamZeroCategorySpecificLinear(
        num_categories=1,
        input_dim=2,
        output_dim=1,
        params_dtype=torch.float32,
        device="cpu",
    )
    layer.W.data.fill_(2.0)
    layer.b.data.fill_(0.5)

    x = torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]])
    out = layer(x, torch.tensor([0, 17]))

    torch.testing.assert_close(out, torch.tensor([[[8.5]], [[12.5]]]))


def test_category_specific_linear_rejects_out_of_range_multi_category_ids() -> None:
    layer = DreamZeroCategorySpecificLinear(
        num_categories=2,
        input_dim=2,
        output_dim=1,
        params_dtype=torch.float32,
        device="cpu",
    )

    with pytest.raises(ValueError, match=r"cat_ids must be in range"):
        layer(torch.zeros(1, 1, 2), torch.tensor([2]))


def test_category_specific_mlp_matches_reference_relu_stack() -> None:
    mlp = DreamZeroCategorySpecificMLP(
        num_categories=1,
        input_dim=2,
        hidden_dim=3,
        output_dim=2,
        params_dtype=torch.float32,
        device="cpu",
    )
    mlp.layer1.W.data.fill_(0.25)
    mlp.layer1.b.data.fill_(-0.5)
    mlp.layer2.W.data.fill_(0.5)
    mlp.layer2.b.data.fill_(1.0)

    x = torch.tensor([[[1.0, 2.0], [-1.0, 1.0]]])
    cat_ids = torch.tensor([0])

    hidden = F.relu(torch.bmm(x, mlp.layer1.W[cat_ids]) + mlp.layer1.b[:, None])
    expected = torch.bmm(hidden, mlp.layer2.W[cat_ids]) + mlp.layer2.b[:, None]

    torch.testing.assert_close(mlp(x, cat_ids), expected)


def test_action_encoder_accepts_batch_or_token_timesteps() -> None:
    encoder = DreamZeroActionEncoder(
        action_dim=4,
        hidden_size=6,
        num_embodiments=1,
        params_dtype=torch.float32,
        device="cpu",
    )
    encoder.W1.W.data.fill_(0.1)
    encoder.W1.b.data.zero_()
    encoder.W2.W.data.fill_(0.05)
    encoder.W2.b.data.zero_()
    encoder.W3.W.data.fill_(0.2)
    encoder.W3.b.data.zero_()

    actions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10.0
    cat_ids = torch.tensor([0, 0])
    batch_timesteps = torch.tensor([1.0, 2.0])
    token_timesteps = batch_timesteps[:, None].expand(-1, actions.shape[1])

    from_batch = encoder(actions, batch_timesteps, cat_ids)
    from_tokens = encoder(actions, token_timesteps, cat_ids)

    assert from_batch.shape == (2, 3, 6)
    torch.testing.assert_close(from_batch, from_tokens)


def test_sinusoidal_positional_encoding_matches_formula() -> None:
    encoding = DreamZeroSinusoidalPositionalEncoding(embedding_dim=4)
    timesteps = torch.tensor([[0.0, 1.0]])

    out = encoding(timesteps)
    exponent = -torch.arange(2, dtype=torch.float32) * (
        torch.log(torch.tensor(10000.0)) / 2
    )
    freqs = timesteps.unsqueeze(-1) * exponent.exp()
    expected = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)

    torch.testing.assert_close(out, expected)


def test_dreamzero_mlp_forward_matches_linear_gelu_linear(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    mlp = DreamZeroMLP(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        prefix="blocks.0.ffn",
    )
    mlp.fc1.weight.data.normal_(std=0.05)
    mlp.fc1.bias.data.normal_(std=0.05)
    mlp.fc2.weight.data.normal_(std=0.05)
    mlp.fc2.bias.data.normal_(std=0.05)

    x = torch.randn(2, 5, 8)
    out = mlp(x)
    expected = F.linear(
        F.gelu(F.linear(x, mlp.fc1.weight, mlp.fc1.bias), approximate="tanh"),
        mlp.fc2.weight,
        mlp.fc2.bias,
    )

    torch.testing.assert_close(out, expected)
