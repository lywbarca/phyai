from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroDiTConfig,
    DreamZeroMinimalPipeline,
    DreamZeroSchedulerOutput,
    dreamzero_images_to_video_tensor,
)


def _tiny_config(*, num_frame_per_block: int = 1) -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=4,
        action_horizon=2,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        input_embedding_dim=1536,
        num_frames=5,
        num_inference_timesteps=1,
        num_frame_per_block=num_frame_per_block,
        cfg_scale=1.5,
        sigma_shift=1.0,
        dit=DreamZeroDiTConfig(
            dim=16,
            ffn_dim=32,
            frame_seqlen=4,
            freq_dim=8,
            in_dim=22,
            max_chunk_size=-1,
            num_action_per_block=2,
            num_frame_per_block=num_frame_per_block,
            num_heads=2,
            num_layers=1,
            num_state_per_block=1,
            out_dim=2,
        ),
    )


@dataclass
class _Processed:
    images: torch.Tensor
    state: torch.Tensor
    embodiment_id: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    negative_input_ids: torch.Tensor
    negative_attention_mask: torch.Tensor


class _TextRunner:
    text_encoder = torch.nn.Identity()

    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4096)


class _ImageRunner:
    image_encoder = torch.nn.Identity()

    def __init__(self) -> None:
        self.last_videos = None

    def encode_image(
        self,
        videos: torch.Tensor,
        *,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        assert not preprocessed
        self.last_videos = videos.clone()
        return torch.ones(videos.shape[0], 2, 1280, dtype=videos.dtype)


class _VAERunner:
    vae = torch.nn.Identity()

    def __init__(self) -> None:
        self.last_pixels = None
        self.last_use_autocast = None

    def encode(
        self,
        pixels: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
        use_autocast: bool = True,
    ) -> torch.Tensor:
        del tiled, tile_size, tile_stride
        self.last_pixels = pixels.clone()
        self.last_use_autocast = use_autocast
        bsz = pixels.shape[0]
        return torch.arange(
            bsz * 2 * 2 * 4 * 4,
            dtype=pixels.dtype,
            device=pixels.device,
        ).reshape(bsz, 2, 2, 4, 4)


class _Scheduler:
    def __init__(self) -> None:
        self.request = None
        self.current_start_frame = 0
        self.model = torch.nn.Identity()
        self.local_attn_size = -1
        self.reset_calls = 0

    def reset_sequence(self) -> None:
        self.reset_calls += 1
        self.current_start_frame = 0

    def step(self, request):
        self.request = request
        start_frame = (
            1 if request.current_start_frame == 0 else request.current_start_frame
        )
        self.current_start_frame = start_frame + request.video.shape[2]
        return DreamZeroSchedulerOutput(
            video=request.video + 1,
            action=request.action + 2,
            last_video_pred=None,
            last_action_pred=None,
            cond_kv_cache=[],
            uncond_kv_cache=None,
            current_start_frame=self.current_start_frame,
        )


def test_images_to_video_tensor_matches_official_uint8_scaling() -> None:
    images = torch.tensor(
        [[[[[0, 127, 255]]], [[[255, 127, 0]]]]],
        dtype=torch.uint8,
    )

    videos = dreamzero_images_to_video_tensor(images, dtype=torch.float32)

    assert videos.shape == (1, 3, 2, 1, 1)
    mid = 127 * (2.0 / 255.0) - 1.0
    torch.testing.assert_close(videos[0, :, 0, 0, 0], torch.tensor([-1.0, mid, 1.0]))
    torch.testing.assert_close(videos[0, :, 1, 0, 0], torch.tensor([1.0, mid, -1.0]))


def test_images_to_video_tensor_matches_official_bfloat16_rounding_order() -> None:
    images = (
        torch.arange(256, dtype=torch.uint8)
        .reshape(1, 1, 1, 256, 1)
        .expand(-1, -1, -1, -1, 3)
        .contiguous()
    )

    videos = dreamzero_images_to_video_tensor(images, dtype=torch.bfloat16)

    expected = images.to(torch.bfloat16).div(255.0).mul(2.0).sub(1.0)
    expected = expected.permute(0, 4, 1, 2, 3).contiguous()
    old_fp32_order = (
        images.to(torch.float32).mul(2.0 / 255.0).sub(1.0).to(torch.bfloat16)
    )
    old_fp32_order = old_fp32_order.permute(0, 4, 1, 2, 3).contiguous()
    assert torch.equal(videos, expected)
    assert not torch.equal(videos, old_fp32_order)


def test_first_frame_condition_uses_latest_frame_for_real_world_chunk() -> None:
    cfg = _tiny_config()
    image_runner = _ImageRunner()
    vae_runner = _VAERunner()
    pipeline = DreamZeroMinimalPipeline(
        config=cfg,
        text_encoder=_TextRunner(),
        image_encoder=image_runner,
        vae=vae_runner,
        scheduler=_Scheduler(),
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )
    videos = torch.arange(1 * 3 * 4 * 8 * 8, dtype=torch.float32).reshape(1, 3, 4, 8, 8)

    pipeline.encode_first_frame_condition(videos)

    assert vae_runner.last_use_autocast is True
    assert image_runner.last_videos is not None
    torch.testing.assert_close(
        image_runner.last_videos,
        videos[:, :, -1:].transpose(1, 2),
    )


def test_minimal_pipeline_builds_scheduler_request() -> None:
    cfg = _tiny_config()
    scheduler = _Scheduler()
    pipeline = DreamZeroMinimalPipeline(
        config=cfg,
        text_encoder=_TextRunner(),
        image_encoder=_ImageRunner(),
        vae=_VAERunner(),
        scheduler=scheduler,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )
    processed = _Processed(
        images=torch.zeros(1, 4, 8, 8, 3, dtype=torch.uint8),
        state=torch.ones(1, 1, 5),
        embodiment_id=torch.tensor([17], dtype=torch.long),
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        negative_input_ids=torch.zeros(1, 4, dtype=torch.long),
        negative_attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    output = pipeline(processed)
    request = scheduler.request

    assert request is not None
    assert output.video.shape == (1, 2, 1, 4, 4)
    assert output.action.shape == (1, 2, 4)
    assert request.video.shape == (1, 2, 1, 4, 4)
    assert request.action.shape == (1, 2, 4)
    assert request.state.shape == (1, 1, 5)
    torch.testing.assert_close(request.embodiment_id, torch.tensor([17]))
    assert request.context.shape == (1, 4, 4096)
    assert request.uncond_context is not None
    assert request.clip_feature.shape == (1, 2, 1280)
    assert request.uncond_clip_feature is request.clip_feature
    assert request.clean_video.shape == (1, 2, 1, 4, 4)
    assert request.reference_video is None
    assert request.y.shape == (1, 6, 2, 4, 4)
    assert request.seq_len == 4
    assert request.current_start_frame == 0
    assert request.concat_first_frame_latent
    assert request.prefill_clean_cache is None


def test_reference_encoder_trace_preserves_warmup_frame(tmp_path, monkeypatch) -> None:
    cfg = _tiny_config()
    scheduler = _Scheduler()
    pipeline = DreamZeroMinimalPipeline(
        config=cfg,
        text_encoder=_TextRunner(),
        image_encoder=_ImageRunner(),
        vae=_VAERunner(),
        scheduler=scheduler,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )
    tensors = {
        "DiT_step_00.branch0.model.raw_text_context.pt": torch.full((1, 4, 4096), 1.0),
        "DiT_step_00.branch0.model.raw_clip_feature.pt": torch.full((1, 2, 1280), 2.0),
        "DiT_step_00.branch0.model.video.condition_y.pt": torch.full(
            (1, 6, 2, 4, 4), 3.0
        ),
        "first-frame_KV_warmup.branch0.model.video.condition_y.pt": torch.full(
            (1, 6, 1, 4, 4), 4.0
        ),
        "first-frame_KV_warmup.branch0.model.video.input.pt": torch.full(
            (1, 2, 1, 4, 4), 5.0
        ),
    }
    for name, tensor in tensors.items():
        torch.save(tensor, tmp_path / name)
    monkeypatch.setenv("DREAMZERO_REFERENCE_ENCODER_TRACE_DIR", str(tmp_path))
    processed = _Processed(
        images=torch.zeros(1, 4, 8, 8, 3, dtype=torch.uint8),
        state=torch.ones(1, 1, 5),
        embodiment_id=torch.tensor([17], dtype=torch.long),
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        negative_input_ids=torch.zeros(1, 4, dtype=torch.long),
        negative_attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    pipeline(processed)
    request = scheduler.request

    assert request is not None
    torch.testing.assert_close(
        request.context, tensors["DiT_step_00.branch0.model.raw_text_context.pt"]
    )
    torch.testing.assert_close(
        request.clip_feature, tensors["DiT_step_00.branch0.model.raw_clip_feature.pt"]
    )
    torch.testing.assert_close(
        request.clean_video,
        tensors["first-frame_KV_warmup.branch0.model.video.input.pt"],
    )
    assert request.y.shape == (1, 6, 3, 4, 4)
    torch.testing.assert_close(
        request.y[:, :, :1],
        tensors["first-frame_KV_warmup.branch0.model.video.condition_y.pt"],
    )
    torch.testing.assert_close(
        request.y[:, :, 1:], tensors["DiT_step_00.branch0.model.video.condition_y.pt"]
    )


def test_minimal_pipeline_builds_reference_video_for_later_chunks() -> None:
    cfg = _tiny_config()
    scheduler = _Scheduler()
    vae_runner = _VAERunner()
    pipeline = DreamZeroMinimalPipeline(
        config=cfg,
        text_encoder=_TextRunner(),
        image_encoder=_ImageRunner(),
        vae=vae_runner,
        scheduler=scheduler,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )
    processed = _Processed(
        images=torch.zeros(1, 4, 8, 8, 3, dtype=torch.uint8),
        state=torch.ones(1, 1, 5),
        embodiment_id=torch.tensor([17], dtype=torch.long),
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        negative_input_ids=torch.zeros(1, 4, dtype=torch.long),
        negative_attention_mask=torch.ones(1, 4, dtype=torch.long),
    )
    for frame, value in enumerate((0, 64, 128, 255)):
        processed.images[:, frame].fill_(value)

    pipeline(processed)
    pipeline(processed)
    request = scheduler.request

    assert request is not None
    assert request.current_start_frame == 2
    assert request.clean_video.shape == (1, 2, 1, 4, 4)
    assert request.reference_video is not None
    assert request.reference_video.shape == (1, 2, 1, 4, 4)
    assert request.reset_kv_cache is None
    assert vae_runner.last_pixels is not None
    assert vae_runner.last_pixels.shape[2] == 5
    videos = dreamzero_images_to_video_tensor(
        processed.images,
        device="cpu",
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        vae_runner.last_pixels,
        torch.cat([videos[:, :, :1], videos], dim=2),
    )


def test_reference_video_expands_four_frames_to_official_nine_frame_input(
    tmp_path, monkeypatch
) -> None:
    cfg = _tiny_config(num_frame_per_block=2)
    vae_runner = _VAERunner()
    pipeline = DreamZeroMinimalPipeline(
        config=cfg,
        text_encoder=_TextRunner(),
        image_encoder=_ImageRunner(),
        vae=vae_runner,
        scheduler=_Scheduler(),
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )
    videos = torch.arange(1 * 3 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 3, 4, 2, 2)
    monkeypatch.setenv("DREAMZERO_DIT_DETAIL_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("DREAMZERO_DIT_TRACE_REFERENCE", "1")

    pipeline.encode_reference_video(videos)

    assert vae_runner.last_use_autocast is False
    assert vae_runner.last_pixels is not None
    repeated = torch.repeat_interleave(videos, 2, dim=2)
    expected = torch.cat([repeated[:, :, :1], repeated], dim=2)
    assert expected.shape[2] == 9
    torch.testing.assert_close(vae_runner.last_pixels, expected)
    torch.testing.assert_close(
        torch.load(tmp_path / "reference_VAE.input.pt", weights_only=True),
        expected,
    )
    torch.testing.assert_close(
        torch.load(tmp_path / "reference_VAE.output_full.pt", weights_only=True),
        torch.arange(1 * 2 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 2, 2, 4, 4),
    )


def test_minimal_pipeline_reconditions_before_local_attention_reset() -> None:
    cfg = _tiny_config()
    scheduler = _Scheduler()
    scheduler.local_attn_size = 2
    pipeline = DreamZeroMinimalPipeline(
        config=cfg,
        text_encoder=_TextRunner(),
        image_encoder=_ImageRunner(),
        vae=_VAERunner(),
        scheduler=scheduler,
        device="cpu",
        dtype=torch.float32,
        seed=123,
    )
    processed = _Processed(
        images=torch.zeros(1, 4, 8, 8, 3, dtype=torch.uint8),
        state=torch.ones(1, 1, 5),
        embodiment_id=torch.tensor([17], dtype=torch.long),
        input_ids=torch.ones(1, 4, dtype=torch.long),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        negative_input_ids=torch.zeros(1, 4, dtype=torch.long),
        negative_attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    pipeline(processed)
    assert scheduler.current_start_frame == 2
    previous_condition = pipeline._condition
    pipeline(processed)
    request = scheduler.request

    assert request is not None
    assert request.current_start_frame == 0
    assert request.reset_kv_cache is True
    assert request.reference_video is None
    assert pipeline._condition is not previous_condition
    assert scheduler.reset_calls == 2
