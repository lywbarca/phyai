from __future__ import annotations

import pytest

from phyai.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroDiTConfig,
    DreamZeroImageEncoderConfig,
    DreamZeroTextEncoderConfig,
)


def test_dreamzero_config_loads_upstream_nested_shape() -> None:
    cfg = DreamZeroConfig.from_dict(
        {
            "action_dim": 32,
            "action_horizon": 24,
            "torch_dtype": "bfloat16",
            "model_dtype": "float32",
            "action_head_cfg": {
                "config": {
                    "hidden_size": 64,
                    "input_embedding_dim": 1536,
                    "max_action_dim": 32,
                    "max_state_dim": 64,
                    "num_frames": 33,
                    "num_frame_per_block": 2,
                    "num_inference_timesteps": 4,
                    "num_timestep_buckets": 1000,
                    "tiled": False,
                    "diffusion_model_cfg": {
                        "diffusion_model_pretrained_path": "/checkpoint/wan",
                        "dim": 5120,
                        "eps": 1e-6,
                        "ffn_dim": 13824,
                        "frame_seqlen": 880,
                        "freq_dim": 256,
                        "in_dim": 36,
                        "max_chunk_size": 4,
                        "model_type": "i2v",
                        "num_action_per_block": 24,
                        "num_frame_per_block": 2,
                        "num_heads": 40,
                        "num_layers": 40,
                        "num_state_per_block": 1,
                        "out_dim": 16,
                    },
                    "text_encoder_cfg": {
                        "text_encoder_pretrained_path": "/checkpoint/t5.pth",
                    },
                    "image_encoder_cfg": {
                        "image_encoder_pretrained_path": "/checkpoint/clip.pth",
                    },
                    "vae_cfg": {
                        "vae_pretrained_path": "/checkpoint/vae.pth",
                    },
                }
            },
        }
    )

    assert cfg.action_dim == 32
    assert cfg.action_horizon == 24
    assert cfg.hidden_size == 1024
    assert cfg.dit.dim == 5120
    assert cfg.dit.num_layers == 40
    assert cfg.dit.head_dim == 128
    assert cfg.dit.pretrained_path == "/checkpoint/wan"
    assert cfg.text_encoder.pretrained_path == "/checkpoint/t5.pth"
    assert cfg.image_encoder.pretrained_path == "/checkpoint/clip.pth"
    assert cfg.vae.pretrained_path == "/checkpoint/vae.pth"


def test_dreamzero_config_uses_diffusion_hidden_size_when_present() -> None:
    cfg = DreamZeroConfig.from_dict(
        {
            "hidden_size": 0,
            "action_head_cfg": {
                "config": {
                    "diffusion_model_cfg": {
                        "hidden_size": 2048,
                    },
                }
            },
        }
    )

    assert cfg.hidden_size == 2048


def test_dreamzero_dit_config_validates_tp4() -> None:
    cfg = DreamZeroDiTConfig()
    cfg.validate_tp_size(4)


def test_dreamzero_dit_config_rejects_invalid_tp() -> None:
    cfg = DreamZeroDiTConfig(num_heads=40, ffn_dim=13824, dim=5120)

    with pytest.raises(ValueError, match="num_heads=40"):
        cfg.validate_tp_size(3)


def test_dreamzero_image_encoder_config_matches_wan21_clip_defaults() -> None:
    cfg = DreamZeroImageEncoderConfig()

    assert cfg.image_size == 224
    assert cfg.patch_size == 14
    assert cfg.vision_dim == 1280
    assert cfg.vision_layers == 32
    assert cfg.feature_layer == 31
    assert cfg.num_patches == 256
    assert cfg.head_dim == 80


def test_dreamzero_text_encoder_config_matches_wan21_umt5_defaults() -> None:
    cfg = DreamZeroTextEncoderConfig()

    assert cfg.vocab == 256384
    assert cfg.dim == 4096
    assert cfg.dim_attn == 4096
    assert cfg.dim_ffn == 10240
    assert cfg.num_heads == 64
    assert cfg.num_layers == 24
    assert cfg.num_buckets == 32
    assert cfg.shared_pos is False
    assert cfg.max_length == 512
    assert cfg.head_dim == 64


def test_dreamzero_text_encoder_config_rejects_invalid_attention_shape() -> None:
    with pytest.raises(ValueError, match="dim_attn=10"):
        DreamZeroTextEncoderConfig(dim_attn=10, num_heads=4)


def test_dreamzero_image_encoder_config_rejects_invalid_feature_layer() -> None:
    with pytest.raises(ValueError, match="feature_layer=33"):
        DreamZeroImageEncoderConfig(feature_layer=33)


def test_dreamzero_config_rejects_action_horizon_mismatch() -> None:
    with pytest.raises(ValueError, match="action_horizon=16"):
        DreamZeroConfig(action_horizon=16, dit=DreamZeroDiTConfig())
