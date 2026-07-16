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


def _tiny_config() -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=4,
        action_horizon=2,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        input_embedding_dim=1536,
        num_frames=5,
        num_inference_timesteps=1,
        num_frame_per_block=1,
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
            num_frame_per_block=1,
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
    def encode_prompt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del attention_mask
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, -1, 4096)


class _ImageRunner:
    def encode_image(
        self,
        videos: torch.Tensor,
        *,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        assert not preprocessed
        return torch.ones(videos.shape[0], 2, 1280, dtype=videos.dtype)


class _VAERunner:
    def encode(
        self,
        pixels: torch.Tensor,
        *,
        tiled: bool = False,
        tile_size: tuple[int, int] = (34, 34),
        tile_stride: tuple[int, int] = (18, 16),
    ) -> torch.Tensor:
        del tiled, tile_size, tile_stride
        bsz = pixels.shape[0]
        return torch.arange(
            bsz * 2 * 2 * 4 * 4,
            dtype=pixels.dtype,
            device=pixels.device,
        ).reshape(bsz, 2, 2, 4, 4)


class _Scheduler:
    def __init__(self) -> None:
        self.request = None

    def step(self, request):
        self.request = request
        return DreamZeroSchedulerOutput(
            video=request.video + 1,
            action=request.action + 2,
            last_video_pred=None,
            last_action_pred=None,
            cond_kv_cache=[],
            uncond_kv_cache=None,
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
        images=torch.zeros(1, 3, 8, 8, 3, dtype=torch.uint8),
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
    assert request.y.shape == (1, 6, 2, 4, 4)
    assert request.seq_len == 4
    assert request.current_start_frame == 1
    assert request.concat_first_frame_latent
    assert request.prefill_clean_cache
