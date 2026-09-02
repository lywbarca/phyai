from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from phyai.models.dreamzero import (
    DreamZeroTextEncoderConfig,
    DreamZeroTextEncoderRunner,
    DreamZeroWanTextEncoder,
)
from phyai.models.dreamzero.text_encoder_wan import T5Attention


def tiny_text_encoder_config() -> DreamZeroTextEncoderConfig:
    return DreamZeroTextEncoderConfig(
        vocab=64,
        dim=16,
        dim_attn=16,
        dim_ffn=32,
        num_heads=4,
        num_layers=2,
        num_buckets=8,
        dropout=0.0,
        max_length=6,
    )


def test_dreamzero_wan_text_encoder_forward_shape() -> None:
    cfg = tiny_text_encoder_config()
    model = DreamZeroWanTextEncoder(cfg)
    input_ids = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 6, 7, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0]])

    out = model(input_ids, attention_mask)

    assert out.shape == (2, 6, 16)
    assert out.dtype == torch.float32


def test_dreamzero_text_encoder_runner_zeroes_padding_tokens() -> None:
    cfg = tiny_text_encoder_config()
    model = DreamZeroWanTextEncoder(cfg)
    runner = DreamZeroTextEncoderRunner(model, device="cpu", dtype=torch.float32)
    input_ids = torch.tensor([[1, 2, 3, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])

    out = runner.encode_prompt(input_ids, attention_mask)

    assert out.shape == (1, 6, 16)
    torch.testing.assert_close(out[:, 3:], torch.zeros_like(out[:, 3:]))
    assert not torch.allclose(out[:, :3], torch.zeros_like(out[:, :3]))


def test_t5_attention_matches_official_einsum_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(7)
    module = T5Attention(dim=8, dim_attn=8, num_heads=2, dropout=0.0).eval()
    x = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ]
    )
    pos_bias = torch.randn(1, 2, 5, 5)

    batch_size = x.size(0)
    q = module.q(x).view(batch_size, -1, module.num_heads, module.head_dim)
    k = module.k(x).view(batch_size, -1, module.num_heads, module.head_dim)
    v = module.v(x).view(batch_size, -1, module.num_heads, module.head_dim)
    attn_bias = x.new_zeros(batch_size, module.num_heads, q.size(1), k.size(1))
    attn_bias += pos_bias
    attn_bias.masked_fill_(
        mask.view(batch_size, 1, 1, -1) == 0,
        torch.finfo(x.dtype).min,
    )
    attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
    attn = F.softmax(attn.float(), dim=-1).type_as(attn)
    expected = torch.einsum("bnij,bjnc->binc", attn, v)
    expected = module.o(expected.reshape(batch_size, -1, module.dim_attn))

    def fail_sdpa(*args: object, **kwargs: object) -> None:
        raise AssertionError("DreamZero T5 attention must use the official einsum path")

    monkeypatch.setattr(F, "scaled_dot_product_attention", fail_sdpa)
    actual = module(x, mask=mask, pos_bias=pos_bias)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
