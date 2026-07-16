from __future__ import annotations

import torch

from phyai.models.dreamzero import (
    DreamZeroTextEncoderConfig,
    DreamZeroTextEncoderRunner,
    DreamZeroWanTextEncoder,
)


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
