"""Minimal DreamZero inference pipeline.

This module stays at the library boundary: callers provide an already-built
processor output, encoder runners, and WS1 scheduler. Weight loading and raw
tokenization remain outside this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.models.dreamzero.model_runner_image_encoder_dreamzero import (
    DreamZeroImageEncoderRunner,
)
from phyai.models.dreamzero.model_runner_text_encoder_dreamzero import (
    DreamZeroTextEncoderRunner,
)
from phyai.models.dreamzero.model_runner_vae_dreamzero import DreamZeroVAERunner
from phyai.models.dreamzero.scheduler_ws1_dreamzero import (
    DreamZeroRequest,
    DreamZeroSchedulerOutput,
    DreamZeroWS1Scheduler,
)


def _get_field(inputs: Any, name: str, default: Any = None) -> Any:
    if isinstance(inputs, dict):
        return inputs.get(name, default)
    return getattr(inputs, name, default)


def _reference_vae_trace_tensor(name: str, value: torch.Tensor) -> None:
    trace_dir = os.getenv("DREAMZERO_DIT_DETAIL_TRACE_DIR")
    enabled = os.getenv("DREAMZERO_DIT_TRACE_REFERENCE", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if not trace_dir or not enabled:
        return
    os.makedirs(trace_dir, exist_ok=True)
    torch.save(
        value.detach().cpu(),
        os.path.join(trace_dir, f"reference_VAE.{name}.pt"),
    )


def _encoder_boundary_trace_tensor(
    request_index: int, name: str, value: torch.Tensor
) -> None:
    trace_dir = os.getenv("DREAMZERO_ENCODER_BOUNDARY_TRACE_DIR")
    if not trace_dir:
        return
    os.makedirs(trace_dir, exist_ok=True)
    torch.save(
        value.detach().cpu(),
        os.path.join(trace_dir, f"request{request_index:02d}.{name}.pt"),
    )


def dreamzero_images_to_video_tensor(
    images: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert processor images `[B, T, H, W, C]` to `[B, C, T, H, W]`.

    The DreamZero processor emits uint8 image grids. Official inference maps
    those bytes to the VAE/image-encoder range `[-1, 1]`.
    """

    if images.dim() != 5:
        raise ValueError(
            f"images must have shape [B, T, H, W, C], got {tuple(images.shape)}."
        )
    if images.shape[-1] != 3:
        raise ValueError(f"images last dimension must be 3, got {images.shape[-1]}.")
    if device is None:
        device = images.device
    if images.dtype == torch.uint8:
        video = images.to(device=device, dtype=dtype)
        video.div_(255.0).mul_(2.0).sub_(1.0)
    else:
        video = images.to(device=device, dtype=dtype)
    return video.permute(0, 4, 1, 2, 3).contiguous()


def dreamzero_generate_noise(
    shape: tuple[int, ...],
    *,
    seed: int | None,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = (
        None if seed is None else torch.Generator(device=device).manual_seed(seed)
    )
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)


@dataclass
class DreamZeroFirstFrameCondition:
    clip_feature: torch.Tensor
    y: torch.Tensor
    clean_video: torch.Tensor


@dataclass
class DreamZeroPipelineOutput:
    video: torch.Tensor
    action: torch.Tensor
    request: DreamZeroRequest
    scheduler_output: DreamZeroSchedulerOutput


class DreamZeroMinimalPipeline:
    """Connect DreamZero processor output to text/image/VAE encoders and WS1 DiT."""

    def __init__(
        self,
        *,
        config: DreamZeroConfig,
        text_encoder: DreamZeroTextEncoderRunner | None,
        image_encoder: DreamZeroImageEncoderRunner | None,
        vae: DreamZeroVAERunner | None,
        scheduler: DreamZeroWS1Scheduler,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
        seed: int | None = None,
        sequential_cpu_offload: bool = False,
        encoder_rank0_broadcast: bool = False,
    ) -> None:
        self.config = config
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.dtype = dtype
        self.seed = seed
        self.sequential_cpu_offload = bool(sequential_cpu_offload)
        self.encoder_rank0_broadcast = bool(encoder_rank0_broadcast)
        self._condition: DreamZeroFirstFrameCondition | None = None
        self._encoder_boundary_trace_index = 0

    @property
    def _is_encoder_rank(self) -> bool:
        return not self.encoder_rank0_broadcast or dist.get_rank() == 0

    def _broadcast_encoder_tensor(self, value: torch.Tensor | None) -> torch.Tensor:
        if not self.encoder_rank0_broadcast:
            if value is None:
                raise RuntimeError("encoder returned no tensor")
            return value.to(self.device, self.dtype)
        metadata = torch.zeros(9, dtype=torch.int64, device=self.device)
        if self._is_encoder_rank:
            if value is None:
                raise RuntimeError("rank0 encoder returned no tensor")
            value = value.to(self.device, self.dtype).contiguous()
            metadata[0] = value.dim()
            metadata[1 : value.dim() + 1] = torch.tensor(
                value.shape, dtype=torch.int64, device=self.device
            )
        dist.broadcast(metadata, src=0)
        ndim = int(metadata[0].item())
        shape = tuple(int(item) for item in metadata[1 : ndim + 1].tolist())
        if not self._is_encoder_rank:
            value = torch.empty(shape, dtype=self.dtype, device=self.device)
        assert value is not None
        dist.broadcast(value, src=0)
        return value

    def _activate(self, module: torch.nn.Module) -> None:
        if self.sequential_cpu_offload:
            module.to(device=self.device, dtype=self.dtype)

    def _offload(self, module: torch.nn.Module) -> None:
        if self.sequential_cpu_offload:
            module.to(device="cpu", dtype=self.dtype)
            torch.cuda.empty_cache()

    def encode_text(self, processed: Any) -> tuple[torch.Tensor, torch.Tensor]:
        context = None
        uncond_context = None
        if self._is_encoder_rank:
            if self.text_encoder is None:
                raise RuntimeError("encoder rank has no text encoder")
            self._activate(self.text_encoder.text_encoder)
            try:
                encode_positive = getattr(
                    self.text_encoder,
                    "encode_positive_prompt",
                    self.text_encoder.encode_prompt,
                )
                encode_negative = getattr(
                    self.text_encoder,
                    "encode_negative_prompt",
                    self.text_encoder.encode_prompt,
                )
                context = encode_positive(
                    _get_field(processed, "input_ids"),
                    _get_field(processed, "attention_mask"),
                )
                uncond_context = encode_negative(
                    _get_field(processed, "negative_input_ids"),
                    _get_field(processed, "negative_attention_mask"),
                )
            finally:
                self._offload(self.text_encoder.text_encoder)
        return (
            self._broadcast_encoder_tensor(context),
            self._broadcast_encoder_tensor(uncond_context),
        )

    def encode_first_frame_condition(
        self,
        videos: torch.Tensor,
    ) -> DreamZeroFirstFrameCondition:
        if videos.dim() != 5:
            raise ValueError(
                f"videos must have shape [B, C, T, H, W], got {tuple(videos.shape)}."
            )
        # Official real-world inference reconditions on the latest observed
        # frame when a multi-frame request starts a new attention window.
        frame = videos[:, :, -1:] if videos.shape[2] in (4, 9) else videos[:, :, :1]
        first_frame = frame.transpose(1, 2).contiguous()
        batch_size, _, channels, height, width = first_frame.shape
        if channels != 3:
            raise ValueError(f"first frame must have 3 channels, got {channels}.")

        clip_feature = None
        if self._is_encoder_rank:
            if self.image_encoder is None:
                raise RuntimeError("encoder rank has no image encoder")
            self._activate(self.image_encoder.image_encoder)
            try:
                clip_feature = self.image_encoder.encode_image(
                    first_frame, preprocessed=False
                )
            finally:
                self._offload(self.image_encoder.image_encoder)
        clip_feature = self._broadcast_encoder_tensor(clip_feature)
        image_input = first_frame.transpose(1, 2).contiguous()
        image_zeros = torch.zeros(
            batch_size,
            channels,
            self.config.num_frames - 1,
            height,
            width,
            dtype=self.dtype,
            device=self.device,
        )
        vae_input = torch.cat(
            [image_input.to(self.device, self.dtype), image_zeros],
            dim=2,
        )
        latents = None
        if self._is_encoder_rank:
            if self.vae is None:
                raise RuntimeError("encoder rank has no VAE")
            self._activate(self.vae.vae)
            try:
                latents = self.vae.encode(
                    vae_input,
                    tiled=self.config.tiled,
                    tile_size=(
                        self.config.tile_size_height,
                        self.config.tile_size_width,
                    ),
                    tile_stride=(
                        self.config.tile_stride_height,
                        self.config.tile_stride_width,
                    ),
                    use_autocast=True,
                )
            finally:
                self._offload(self.vae.vae)
        latents = self._broadcast_encoder_tensor(latents)
        mask = torch.zeros(
            batch_size,
            4,
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
            dtype=latents.dtype,
            device=latents.device,
        )
        mask[:, :, 0:1] = 1
        return DreamZeroFirstFrameCondition(
            clip_feature=clip_feature.to(self.device, self.dtype),
            y=torch.cat([mask, latents], dim=1).to(self.device, self.dtype),
            clean_video=latents[:, :, 0:1].to(self.device, self.dtype),
        )

    def encode_reference_video(self, videos: torch.Tensor) -> torch.Tensor:
        if videos.dim() != 5:
            raise ValueError(
                f"videos must have shape [B, C, T, H, W], got {tuple(videos.shape)}."
            )
        if videos.shape[2] <= 1:
            raise ValueError("reference video requires more than one input frame.")

        vae_input = videos
        num_frames = int(vae_input.shape[2])
        frames_per_block = int(self.config.num_frame_per_block)
        if (num_frames - 1) // 4 != frames_per_block:
            observed_blocks = num_frames // 4
            if observed_blocks <= 0:
                raise ValueError(
                    "reference video must contain at least four observed frames; "
                    f"got {num_frames}."
                )
            if observed_blocks != frames_per_block:
                if frames_per_block % observed_blocks != 0:
                    raise ValueError(
                        "reference video frames cannot be aligned to "
                        f"num_frame_per_block={frames_per_block}: got {num_frames}."
                    )
                vae_input = torch.repeat_interleave(
                    vae_input,
                    frames_per_block // observed_blocks,
                    dim=2,
                )
            vae_input = torch.cat([vae_input[:, :, :1], vae_input], dim=2)
        vae_input = vae_input.to(self.device, self.dtype)
        _reference_vae_trace_tensor("input", vae_input)
        latents = None
        if self._is_encoder_rank:
            if self.vae is None:
                raise RuntimeError("encoder rank has no VAE")
            self._activate(self.vae.vae)
            try:
                latents = self.vae.encode(
                    vae_input,
                    tiled=self.config.tiled,
                    tile_size=(
                        self.config.tile_size_height,
                        self.config.tile_size_width,
                    ),
                    tile_stride=(
                        self.config.tile_stride_height,
                        self.config.tile_stride_width,
                    ),
                    use_autocast=False,
                )
            finally:
                self._offload(self.vae.vae)
        latents = self._broadcast_encoder_tensor(latents)
        _reference_vae_trace_tensor("output_full", latents)
        return latents[:, :, -self.config.num_frame_per_block :].to(
            self.device, self.dtype
        )

    def build_request(
        self,
        processed: Any,
        *,
        guidance_scale: float | None = None,
        current_start_frame: int | None = None,
        num_inference_steps: int | None = None,
        dynamic_dit: bool = False,
        dynamic_dit_scheduler_steps: int = 16,
        prefill_clean_cache: bool | None = None,
        reset_kv_cache: bool | None = None,
    ) -> DreamZeroRequest:
        images = _get_field(processed, "images")
        if images is None:
            raise ValueError("DreamZero minimal pipeline requires processed.images.")
        state = _get_field(processed, "state")
        if state is None:
            raise ValueError("DreamZero minimal pipeline requires processed.state.")
        embodiment_id = _get_field(processed, "embodiment_id")

        videos = dreamzero_images_to_video_tensor(
            images,
            device=self.device,
            dtype=self.dtype,
        )
        encoder_trace_request_index = self._encoder_boundary_trace_index
        if os.getenv("DREAMZERO_ENCODER_BOUNDARY_TRACE_DIR"):
            self._encoder_boundary_trace_index += 1
        context, uncond_context = self.encode_text(processed)
        _encoder_boundary_trace_tensor(
            encoder_trace_request_index, "text_context0", context
        )
        _encoder_boundary_trace_tensor(
            encoder_trace_request_index, "text_context1", uncond_context
        )
        reference_encoder_dir = os.getenv("DREAMZERO_REFERENCE_ENCODER_TRACE_DIR")
        reference_encoder_components = {
            component.strip().lower()
            for component in os.getenv(
                "DREAMZERO_REFERENCE_ENCODER_COMPONENTS", "text,clip,vae"
            ).split(",")
            if component.strip()
        }
        unknown_reference_components = reference_encoder_components - {
            "text",
            "clip",
            "vae",
        }
        if unknown_reference_components:
            raise ValueError(
                "Unknown DREAMZERO_REFERENCE_ENCODER_COMPONENTS: "
                f"{sorted(unknown_reference_components)}"
            )
        if reference_encoder_dir and "text" in reference_encoder_components:
            reference_path = Path(reference_encoder_dir)
            positive_path = reference_path / "request00.text_context0.pt"
            negative_path = reference_path / "request00.text_context1.pt"
            if positive_path.exists() or negative_path.exists():
                if not positive_path.exists() or not negative_path.exists():
                    raise FileNotFoundError(
                        "Reference text boundary trace must contain both "
                        "request00.text_context0.pt and request00.text_context1.pt."
                    )
                context = torch.load(
                    positive_path,
                    map_location=self.device,
                    weights_only=True,
                ).to(self.device, self.dtype)
                uncond_context = torch.load(
                    negative_path,
                    map_location=self.device,
                    weights_only=True,
                ).to(self.device, self.dtype)
            else:
                context = torch.load(
                    reference_path / "DiT_step_00.branch0.model.raw_text_context.pt",
                    map_location=self.device,
                    weights_only=True,
                ).to(self.device, self.dtype)
        start_frame = (
            self.scheduler.current_start_frame
            if current_start_frame is None
            else int(current_start_frame)
        )
        if (
            reset_kv_cache is None
            and self.scheduler.local_attn_size != -1
            and start_frame >= self.scheduler.local_attn_size
        ):
            reset_kv_cache = True
            start_frame = 0
        if reset_kv_cache is None and self._condition is None:
            reset_kv_cache = True
            start_frame = 0
        if reset_kv_cache is None and start_frame != 0 and videos.shape[2] == 1:
            reset_kv_cache = True
            start_frame = 0
        reset_sequence = start_frame == 0 if reset_kv_cache is None else reset_kv_cache
        if reset_sequence:
            # Release the previous attention window before image/VAE
            # reconditioning. Waiting until scheduler.step() keeps both the old
            # multi-gigabyte KV cache and the new encoder working set alive.
            self.scheduler.reset_sequence()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        if reset_sequence or self._condition is None:
            condition = self.encode_first_frame_condition(videos)
            if reference_encoder_dir and reference_encoder_components & {
                "clip",
                "vae",
            }:
                reference_path = Path(reference_encoder_dir)
                clip_feature = condition.clip_feature
                y = condition.y
                clean_video = condition.clean_video
                if "clip" in reference_encoder_components:
                    clip_feature = torch.load(
                        reference_path
                        / "DiT_step_00.branch0.model.raw_clip_feature.pt",
                        map_location=self.device,
                        weights_only=True,
                    ).to(self.device, self.dtype)
                if "vae" in reference_encoder_components:
                    denoise_y = torch.load(
                        reference_path
                        / "DiT_step_00.branch0.model.video.condition_y.pt",
                        map_location=self.device,
                        weights_only=True,
                    ).to(self.device, self.dtype)
                    warmup_y_path = (
                        reference_path
                        / "first-frame_KV_warmup.branch0.model.video.condition_y.pt"
                    )
                    warmup_video_path = (
                        reference_path
                        / "first-frame_KV_warmup.branch0.model.video.input.pt"
                    )
                    if warmup_y_path.exists() != warmup_video_path.exists():
                        raise FileNotFoundError(
                            "Reference encoder trace must contain both warmup condition_y "
                            "and warmup video input."
                        )
                    if warmup_y_path.exists():
                        warmup_y = torch.load(
                            warmup_y_path,
                            map_location=self.device,
                            weights_only=True,
                        ).to(self.device, self.dtype)
                        clean_video = torch.load(
                            warmup_video_path,
                            map_location=self.device,
                            weights_only=True,
                        ).to(self.device, self.dtype)
                        if warmup_y.shape[2] != 1 or clean_video.shape[2] != 1:
                            raise ValueError(
                                "Reference warmup tensors must contain exactly one frame."
                            )
                        y = torch.cat([warmup_y, denoise_y], dim=2)
                    else:
                        y = denoise_y
                        clean_video = y[:, 4:, :1]
                condition = DreamZeroFirstFrameCondition(
                    clip_feature=clip_feature,
                    y=y,
                    clean_video=clean_video,
                )
            self._condition = condition
            start_frame = 0 if reset_sequence else start_frame
        else:
            condition = self._condition
        _encoder_boundary_trace_tensor(
            encoder_trace_request_index, "clip_feature", condition.clip_feature
        )
        _encoder_boundary_trace_tensor(
            encoder_trace_request_index, "condition_y", condition.y
        )
        _encoder_boundary_trace_tensor(
            encoder_trace_request_index, "clean_video", condition.clean_video
        )

        reference_video = None
        if not reset_sequence and start_frame != 1 and videos.shape[2] > 1:
            reference_video = self.encode_reference_video(videos)

        batch_size = videos.shape[0]
        latent_h, latent_w = condition.clean_video.shape[3:5]
        noise_video = dreamzero_generate_noise(
            (
                batch_size,
                condition.clean_video.shape[1],
                self.config.num_frame_per_block,
                latent_h,
                latent_w,
            ),
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        noise_action = dreamzero_generate_noise(
            (batch_size, self.config.action_horizon, self.config.action_dim),
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        seq_len = self.config.num_frame_per_block * (latent_h // 2) * (latent_w // 2)
        use_cfg = (
            self.config.cfg_scale if guidance_scale is None else guidance_scale
        ) != 1.0
        return DreamZeroRequest(
            video=noise_video,
            action=noise_action,
            state=state.to(self.device, self.dtype),
            context=context,
            embodiment_id=(
                embodiment_id.to(self.device, torch.long)
                if isinstance(embodiment_id, torch.Tensor)
                else None
            ),
            clip_feature=condition.clip_feature,
            y=condition.y,
            clean_video=condition.clean_video,
            reference_video=reference_video,
            uncond_context=uncond_context if use_cfg else None,
            uncond_clip_feature=condition.clip_feature if use_cfg else None,
            seq_len=seq_len,
            current_start_frame=start_frame,
            concat_first_frame_latent=True,
            image_context_tokens=condition.clip_feature.shape[1],
            num_inference_steps=num_inference_steps,
            dynamic_dit=dynamic_dit,
            dynamic_dit_scheduler_steps=dynamic_dit_scheduler_steps,
            guidance_scale=guidance_scale,
            sigma_shift=self.config.sigma_shift,
            decouple_inference_noise=self.config.decouple_inference_noise,
            video_inference_final_noise=self.config.video_inference_final_noise,
            prefill_clean_cache=prefill_clean_cache,
            reset_kv_cache=reset_kv_cache,
        )

    @torch.no_grad()
    def __call__(
        self,
        processed: Any,
        *,
        guidance_scale: float | None = None,
        current_start_frame: int | None = None,
        num_inference_steps: int | None = None,
        dynamic_dit: bool = False,
        dynamic_dit_scheduler_steps: int = 16,
        reset_kv_cache: bool | None = None,
    ) -> DreamZeroPipelineOutput:
        request = self.build_request(
            processed,
            guidance_scale=guidance_scale,
            current_start_frame=current_start_frame,
            num_inference_steps=num_inference_steps,
            dynamic_dit=dynamic_dit,
            dynamic_dit_scheduler_steps=dynamic_dit_scheduler_steps,
            reset_kv_cache=reset_kv_cache,
        )
        self._activate(self.scheduler.model)
        try:
            scheduler_output = self.scheduler.step(request)
        finally:
            self._offload(self.scheduler.model)
        return DreamZeroPipelineOutput(
            video=scheduler_output.video,
            action=scheduler_output.action,
            request=request,
            scheduler_output=scheduler_output,
        )


__all__ = [
    "DreamZeroFirstFrameCondition",
    "DreamZeroMinimalPipeline",
    "DreamZeroPipelineOutput",
    "dreamzero_generate_noise",
    "dreamzero_images_to_video_tensor",
]
