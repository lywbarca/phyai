from __future__ import annotations

from phyai.models.dreamzero import (
    dreamzero_component_from_key,
    dreamzero_dit_weight_remap,
    dreamzero_image_encoder_weight_remap,
    dreamzero_text_encoder_weight_remap,
    dreamzero_vae_weight_remap,
)


def test_dreamzero_dit_weight_remap_strips_action_head_model_prefix() -> None:
    assert (
        dreamzero_dit_weight_remap("action_head.model.blocks.0.self_attn.q.weight")
        == "blocks.0.self_attn.q.weight"
    )
    assert (
        dreamzero_dit_weight_remap("action_head.model.action_encoder.W1.W")
        == "action_encoder.W1.W"
    )


def test_dreamzero_dit_weight_remap_drops_non_dit_components() -> None:
    assert (
        dreamzero_dit_weight_remap(
            "action_head.text_encoder.blocks.0.self_attn.q.weight"
        )
        is None
    )
    assert (
        dreamzero_dit_weight_remap(
            "action_head.image_encoder.model.visual.patch_embedding.weight"
        )
        is None
    )
    assert (
        dreamzero_dit_weight_remap("action_head.vae.model.encoder.conv.weight") is None
    )
    assert dreamzero_dit_weight_remap("backbone.some_weight") is None


def test_dreamzero_component_from_key_classifies_checkpoint_subtrees() -> None:
    assert (
        dreamzero_component_from_key("action_head.model.blocks.0.self_attn.q.weight")
        == "dit"
    )
    assert (
        dreamzero_component_from_key("action_head.text_encoder.blocks.0.ffn.0.weight")
        == "text_encoder"
    )
    assert (
        dreamzero_component_from_key(
            "action_head.image_encoder.model.visual.patch_embedding.weight"
        )
        == "image_encoder"
    )
    assert (
        dreamzero_component_from_key("action_head.vae.model.encoder.conv.weight")
        == "vae"
    )
    assert dreamzero_component_from_key("action_head.some_other_weight") == (
        "action_head_other"
    )
    assert dreamzero_component_from_key("backbone.some_weight") is None


def test_dreamzero_vae_weight_remap_strips_action_head_vae_prefix() -> None:
    assert (
        dreamzero_vae_weight_remap("action_head.vae.model.encoder.conv1.weight")
        == "model.encoder.conv1.weight"
    )
    assert (
        dreamzero_vae_weight_remap("action_head.vae.model.decoder.head.2.bias")
        == "model.decoder.head.2.bias"
    )
    assert dreamzero_vae_weight_remap("model.conv1.weight") == "model.conv1.weight"
    assert dreamzero_vae_weight_remap("action_head.model.blocks.0.weight") is None


def test_dreamzero_image_encoder_weight_remap_supports_checkpoint_prefix() -> None:
    assert (
        dreamzero_image_encoder_weight_remap(
            "action_head.image_encoder.model.visual.patch_embedding.weight"
        )
        == "model.visual.patch_embedding.weight"
    )
    assert (
        dreamzero_image_encoder_weight_remap(
            "action_head.image_encoder.model.log_scale"
        )
        == "model.log_scale"
    )
    assert (
        dreamzero_image_encoder_weight_remap("visual.transformer.0.attn.to_qkv.weight")
        == "model.visual.transformer.0.attn.to_qkv.weight"
    )
    assert (
        dreamzero_image_encoder_weight_remap("action_head.vae.model.conv.weight")
        is None
    )


def test_dreamzero_text_encoder_weight_remap_supports_checkpoint_prefix() -> None:
    assert (
        dreamzero_text_encoder_weight_remap(
            "action_head.text_encoder.blocks.0.attn.q.weight"
        )
        == "blocks.0.attn.q.weight"
    )
    assert (
        dreamzero_text_encoder_weight_remap(
            "action_head.text_encoder.token_embedding.weight"
        )
        == "token_embedding.weight"
    )
    assert (
        dreamzero_text_encoder_weight_remap("blocks.0.pos_embedding.embedding.weight")
        == "blocks.0.pos_embedding.embedding.weight"
    )
    assert (
        dreamzero_text_encoder_weight_remap("action_head.image_encoder.model.weight")
        is None
    )
