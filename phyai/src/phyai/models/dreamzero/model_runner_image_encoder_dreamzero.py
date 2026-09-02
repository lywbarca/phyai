"""DreamZero image encoder runner."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from torch._inductor.runtime.cache_dir_utils import cache_dir, temporary_cache_dir

from phyai.models.dreamzero.image_encoder_wan import DreamZeroWanImageEncoder
from phyai.models.dreamzero.inductor_cache_dreamzero import (
    OFFICIAL_CLIP_EMBEDDED_CACHE_ROOT,
    is_official_clip_inductor_runtime,
    seed_official_clip_inductor_configs,
)
from phyai.runtime.cuda_graph_manager import CudaGraph
from phyai.runtime.model_runner import ModelRunner
from phyai.utils import this_rank_log


logger = logging.getLogger(__name__)


class DreamZeroImageEncoderRunner(ModelRunner):
    """Wraps the DreamZero Wan2.1 CLIP image encoder."""

    def __init__(
        self,
        image_encoder: DreamZeroWanImageEncoder,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        use_cuda_graph: bool = False,
        graph_batch_size: int = 1,
    ) -> None:
        self.image_encoder = image_encoder
        self.device = torch.device(device)
        self.dtype = dtype
        self.use_cuda_graph = bool(use_cuda_graph)
        self.graph_batch_size = int(graph_batch_size)
        if self.graph_batch_size <= 0:
            raise ValueError(
                f"graph_batch_size={self.graph_batch_size} must be positive."
            )
        self.graph: CudaGraph | None = None
        self._reseed_official_configs_before_forward = False
        self._official_clip_cache_root: Path | None = None
        self._clip_trace_index = 0

    def _seed_official_configs(self) -> tuple[Path, ...]:
        if self._official_clip_cache_root is None:
            return ()
        roots = (
            self._official_clip_cache_root,
            OFFICIAL_CLIP_EMBEDDED_CACHE_ROOT,
        )
        installed = []
        seen = set()
        for root in roots:
            normalized = root.resolve()
            if normalized in seen:
                continue
            seen.add(normalized)
            installed.extend(seed_official_clip_inductor_configs(root))
        return tuple(installed)

    def _compile_visual(self) -> None:
        compile_mode = os.getenv(
            "DREAMZERO_IMAGE_ENCODER_COMPILE_MODE", "reduce-overhead"
        )
        self.image_encoder.model.visual.forward = torch.compile(
            self.image_encoder.model.visual.forward,
            mode=compile_mode,
            fullgraph=True,
            dynamic=False,
        )

    def _warm_compiled_visual(self) -> None:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=self.device.type == "cuda",
            ),
        ):
            pixel_values = torch.zeros(
                self.graph_batch_size,
                self.image_encoder.config.num_channels,
                self.image_encoder.config.image_size,
                self.image_encoder.config.image_size,
                dtype=self.dtype,
                device=self.device,
            )
            self.image_encoder.encode_image(
                pixel_values,
                preprocessed=True,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def setup(self) -> None:
        if os.getenv("DREAMZERO_COMPILE_IMAGE_ENCODER", "false").lower() == "true":
            if is_official_clip_inductor_runtime(self.device):
                self._reseed_official_configs_before_forward = True
                configured_root = os.getenv("DREAMZERO_CLIP_INDUCTOR_CACHE_DIR")
                self._official_clip_cache_root = (
                    Path(configured_root)
                    if configured_root
                    else Path(f"{cache_dir()}.dreamzero_clip")
                )
                with temporary_cache_dir(str(self._official_clip_cache_root)):
                    installed = self._seed_official_configs()
                    this_rank_log(
                        logger,
                        logging.INFO,
                        "Seeded %d official DreamZero CLIP Inductor configs in %s.",
                        len(installed),
                        self._official_clip_cache_root,
                    )
                    self._compile_visual()
                    self._seed_official_configs()
                    self._warm_compiled_visual()
            else:
                self._compile_visual()
        if not self.use_cuda_graph or self.device.type != "cuda":
            return
        example = {
            "pixel_values": torch.zeros(
                self.graph_batch_size,
                self.image_encoder.config.num_channels,
                self.image_encoder.config.image_size,
                self.image_encoder.config.image_size,
                dtype=self.dtype,
                device=self.device,
            ),
        }
        if self._reseed_official_configs_before_forward:
            self._seed_official_configs()
        self.graph = CudaGraph()
        self.graph.capture(self._forward_preprocessed, example)

    def _forward_preprocessed(self, *, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(pixel_values)

    @torch.no_grad()
    def encode_image(
        self,
        videos: torch.Tensor | list[torch.Tensor],
        *,
        preprocessed: bool = False,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.device.type == "cuda",
        ):
            if preprocessed:
                pixel_values = videos
                if not isinstance(pixel_values, torch.Tensor):
                    raise TypeError("preprocessed videos must be a Tensor.")
                pixel_values = pixel_values.to(self.device, self.dtype)
            else:
                pixel_values = self.image_encoder.preprocess(videos).to(
                    self.device, self.dtype
                )
            # Text compilation can overwrite a CLIP reduction config in the
            # embedded cache after setup but before the first image request.
            if self._reseed_official_configs_before_forward:
                self._seed_official_configs()
            if self.graph is not None:
                output = self.graph.replay({"pixel_values": pixel_values}).clone()
            else:
                output = self.image_encoder.encode_image(
                    pixel_values, preprocessed=True
                )
            # The compiled visual graph can return before its output is safe for
            # the separately compiled VAE/DiT path on Thor.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            trace_dir = os.getenv("DREAMZERO_CLIP_RUNNER_TRACE_DIR") or os.getenv(
                "DREAMZERO_DIT_DETAIL_TRACE_DIR"
            )
            if trace_dir:
                trace_path = Path(trace_dir)
                trace_path.mkdir(parents=True, exist_ok=True)
                traced_output = output.detach().cpu()
                request_path = trace_path / (
                    f"CLIP.runner_output.request{self._clip_trace_index:02d}.pt"
                )
                input_path = trace_path / (
                    f"CLIP.runner_input.request{self._clip_trace_index:02d}.pt"
                )
                self._clip_trace_index += 1
                torch.save(pixel_values.detach().cpu(), input_path)
                torch.save(traced_output, request_path)
                torch.save(traced_output, trace_path / "CLIP.runner_output.pt")
            return output

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.encode_image(pixel_values, preprocessed=True)


__all__ = ["DreamZeroImageEncoderRunner"]
