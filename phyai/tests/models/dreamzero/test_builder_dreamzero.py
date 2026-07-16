from __future__ import annotations

import torch
import torch.nn as nn

import phyai.models.dreamzero.builder_dreamzero as builder
from phyai.models.dreamzero import (
    DreamZeroBuildOptions,
    DreamZeroConfig,
    DreamZeroDiTConfig,
    build_dreamzero_minimal_pipeline,
)


def _tiny_config() -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=4,
        action_horizon=2,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        num_frames=5,
        num_inference_timesteps=1,
        num_frame_per_block=1,
        cfg_scale=1.5,
        dit=DreamZeroDiTConfig(
            dim=16,
            ffn_dim=32,
            frame_seqlen=4,
            freq_dim=8,
            in_dim=22,
            max_chunk_size=-1,
            num_action_per_block=2,
            num_frame_per_block=1,
            num_heads=2,
            num_layers=1,
            num_state_per_block=1,
            out_dim=2,
        ),
    )


class _FakeModule(nn.Module):
    def __init__(self, name: str, *args, **kwargs) -> None:
        super().__init__()
        self.name = name
        self.args = args
        self.kwargs = kwargs


class _FakeScheduler:
    def __init__(self, model, *, device, use_cfg_runner) -> None:
        self.model = model
        self.device = torch.device(device)
        self.use_cfg_runner = use_cfg_runner
        self.setup_called = False

    def setup(self) -> None:
        self.setup_called = True


def test_build_dreamzero_minimal_pipeline_wires_components(
    monkeypatch, tmp_path
) -> None:
    cfg = _tiny_config()
    load_calls = []

    monkeypatch.setattr(builder, "load_config", lambda path, cls: cfg)
    monkeypatch.setattr(
        builder,
        "DreamZeroDiT",
        lambda *args, **kwargs: _FakeModule("dit", *args, **kwargs),
    )
    monkeypatch.setattr(
        builder,
        "DreamZeroWanTextEncoder",
        lambda *args, **kwargs: _FakeModule("text", *args, **kwargs),
    )
    monkeypatch.setattr(
        builder,
        "DreamZeroWanImageEncoder",
        lambda *args, **kwargs: _FakeModule("image", *args, **kwargs),
    )
    monkeypatch.setattr(
        builder,
        "DreamZeroWanVAE",
        lambda *args, **kwargs: _FakeModule("vae", *args, **kwargs),
    )
    monkeypatch.setattr(builder, "DreamZeroWS1Scheduler", _FakeScheduler)

    def fake_load_pretrained(model, source, *, remap, strict, progress):
        load_calls.append((model.name, source, remap, strict, progress))
        return builder.LoadReport(loaded=[model.name])

    monkeypatch.setattr(builder, "load_pretrained", fake_load_pretrained)

    bundle = build_dreamzero_minimal_pipeline(
        DreamZeroBuildOptions(
            checkpoint_dir=tmp_path,
            dtype=torch.float32,
            device="cpu",
            seed=7,
            weight_strict=True,
            progress=False,
            attn_backend="eager",
            norm_backend="torch",
            use_cfg_runner=False,
        )
    )

    assert bundle.config is cfg
    assert bundle.pipeline.config is cfg
    assert bundle.pipeline.seed == 7
    assert bundle.scheduler.setup_called
    assert bundle.scheduler.use_cfg_runner is False
    assert bundle.dit.kwargs["attn_backend"] == "eager"
    assert bundle.dit.kwargs["norm_backend"] == "torch"
    assert bundle.text_encoder.kwargs["params_dtype"] is torch.float32
    assert bundle.image_encoder.kwargs["params_dtype"] is torch.float32
    assert [name for name, *_ in load_calls] == [
        "dit",
        "text",
        "image",
        "vae",
    ]
    assert load_calls[0][2] is builder.dreamzero_dit_weight_remap
    assert load_calls[1][2] is builder.dreamzero_text_encoder_weight_remap
    assert load_calls[2][2] is builder.dreamzero_image_encoder_weight_remap
    assert load_calls[3][2] is builder.dreamzero_vae_weight_remap
    assert set(bundle.load_reports) == {"dit", "text_encoder", "image_encoder", "vae"}


def test_build_dreamzero_rejects_unimplemented_encoder_strategy(tmp_path) -> None:
    options = DreamZeroBuildOptions(
        checkpoint_dir=tmp_path,
        config=_tiny_config(),
        device="cpu",
        encoder_strategy="tp_rank0_broadcast",  # type: ignore[arg-type]
    )

    try:
        build_dreamzero_minimal_pipeline(options)
    except ValueError as exc:
        assert "encoder_strategy='replicated'" in str(exc)
    else:
        raise AssertionError("expected unsupported encoder strategy to fail")
