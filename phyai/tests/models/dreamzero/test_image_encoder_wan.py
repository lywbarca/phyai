from __future__ import annotations

import torch

from phyai.models.dreamzero import (
    DreamZeroImageEncoderConfig,
    DreamZeroImageEncoderRunner,
    DreamZeroWanImageEncoder,
)
from phyai.models.dreamzero.image_encoder_wan import (
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
)


def tiny_image_encoder_config() -> DreamZeroImageEncoderConfig:
    return DreamZeroImageEncoderConfig(
        embed_dim=8,
        image_size=8,
        patch_size=4,
        vision_dim=16,
        vision_heads=4,
        vision_layers=3,
        vision_mlp_ratio=2,
        feature_layer=2,
    )


def test_dreamzero_wan_image_encoder_forward_returns_feature_tokens() -> None:
    cfg = tiny_image_encoder_config()
    model = DreamZeroWanImageEncoder(cfg)
    pixel_values = torch.randn(2, 3, 8, 8)

    out = model(pixel_values)

    assert out.shape == (2, 5, 16)


def test_dreamzero_wan_image_encoder_uses_feature_layer_before_final_block() -> None:
    cfg = tiny_image_encoder_config()
    model = DreamZeroWanImageEncoder(cfg)
    pixel_values = torch.randn(1, 3, 8, 8)

    default_out = model.model.visual(pixel_values, use_31_block=True)
    two_block_out = model.model.visual(pixel_values, feature_layer=2)
    full_out = model.model.visual(pixel_values, use_31_block=False)

    torch.testing.assert_close(default_out, two_block_out)
    assert not torch.allclose(default_out, full_out)


def test_dreamzero_wan_image_encoder_preprocess_matches_clip_grid() -> None:
    cfg = tiny_image_encoder_config()
    model = DreamZeroWanImageEncoder(cfg)
    videos = torch.zeros(2, 1, 3, 4, 6)

    pixel_values = model.preprocess(videos)

    assert pixel_values.shape == (2, 3, 8, 8)
    assert pixel_values.dtype == videos.dtype


def test_image_preprocess_keeps_clip_constants_at_compute_dtype() -> None:
    cfg = tiny_image_encoder_config()
    model = DreamZeroWanImageEncoder(cfg).to(dtype=torch.bfloat16)
    videos = torch.zeros(1, 1, 3, 8, 8, dtype=torch.float32)

    pixel_values = model.preprocess(videos)

    mean = torch.tensor(CLIP_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_IMAGE_STD).view(1, 3, 1, 1)
    expected = ((torch.zeros_like(pixel_values) + 0.5) - mean) / std
    torch.testing.assert_close(pixel_values, expected, rtol=0, atol=0)


def test_dreamzero_image_encoder_runner_accepts_preprocessed_pixels() -> None:
    cfg = tiny_image_encoder_config()
    model = DreamZeroWanImageEncoder(cfg)
    runner = DreamZeroImageEncoderRunner(model, device="cpu", dtype=torch.float32)
    pixel_values = torch.randn(1, 3, 8, 8)

    out = runner.encode_image(pixel_values, preprocessed=True)

    assert out.shape == (1, 5, 16)
