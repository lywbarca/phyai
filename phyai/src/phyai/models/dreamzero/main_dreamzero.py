"""DreamZero engine plugin entry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from phyai.engine import Engine, Entry, EntryArgs
from phyai.engine_config import get_engine_config
from phyai.models.dreamzero.builder_dreamzero import (
    DreamZeroBuildOptions,
    DreamZeroEncoderStrategy,
    DreamZeroPipelineBundle,
    build_dreamzero_minimal_pipeline,
)
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.models.dreamzero.pipeline_dreamzero import DreamZeroPipelineOutput


@dataclass
class DreamZeroArgs(EntryArgs):
    """Args bundle for the DreamZero minimal inference plugin."""

    checkpoint_dir: str | Path | None = None
    config: DreamZeroConfig | None = None
    seed: int | None = None
    weight_strict: bool = True
    progress: bool | None = None
    encoder_strategy: DreamZeroEncoderStrategy = "replicated"
    use_cfg_runner: bool = True
    image_use_cuda_graph: bool = False
    image_graph_batch_size: int = 1
    sequential_cpu_offload: bool = False
    skip_parameter_init: bool = False
    num_inference_steps: int | None = None
    dynamic_dit: bool = False
    dynamic_dit_scheduler_steps: int = 16


@Engine.register
class DreamZeroEntry(Entry):
    """DreamZero minimal pipeline entry.

    Requests must already be preprocessed by ``phyai-utils-tools`` or an
    equivalent caller-owned processor. The main ``phyai`` package deliberately
    does not own tokenizer or action denormalization logic.
    """

    name: ClassVar[str] = "dreamzero"
    args_cls: ClassVar[type[EntryArgs]] = DreamZeroArgs

    def __init__(self) -> None:
        self.bundle: DreamZeroPipelineBundle | None = None
        self.num_inference_steps: int | None = None
        self.dynamic_dit = False
        self.dynamic_dit_scheduler_steps = 16

    def setup(self, args: DreamZeroArgs) -> None:  # type: ignore[override]
        if args.checkpoint_dir is None:
            raise ValueError("DreamZeroArgs.checkpoint_dir is required.")
        self.num_inference_steps = args.num_inference_steps
        self.dynamic_dit = args.dynamic_dit
        self.dynamic_dit_scheduler_steps = args.dynamic_dit_scheduler_steps
        eng = get_engine_config()
        self.bundle = build_dreamzero_minimal_pipeline(
            DreamZeroBuildOptions(
                checkpoint_dir=args.checkpoint_dir,
                config=args.config,
                dtype=eng.device.params_dtype,
                device=eng.device.target,
                seed=args.seed,
                weight_strict=args.weight_strict,
                progress=args.progress,
                attn_backend=eng.backends.attn,
                norm_backend=eng.backends.norm,
                use_cfg_runner=args.use_cfg_runner,
                encoder_strategy=args.encoder_strategy,
                image_use_cuda_graph=args.image_use_cuda_graph,
                image_graph_batch_size=args.image_graph_batch_size,
                sequential_cpu_offload=args.sequential_cpu_offload,
                skip_parameter_init=args.skip_parameter_init,
            )
        )

    def step(self, request: Any) -> DreamZeroPipelineOutput:  # type: ignore[override]
        if self.bundle is None:
            raise RuntimeError("DreamZeroEntry.step called before setup.")
        with torch.inference_mode():
            return self.bundle.pipeline(
                request,
                num_inference_steps=self.num_inference_steps,
                dynamic_dit=self.dynamic_dit,
                dynamic_dit_scheduler_steps=self.dynamic_dit_scheduler_steps,
            )

    def close(self) -> None:
        self.bundle = None
        self.num_inference_steps = None
        self.dynamic_dit = False
        self.dynamic_dit_scheduler_steps = 16

    def dump_targets(self) -> dict[str, torch.nn.Module]:  # type: ignore[override]
        if self.bundle is None:
            return {}
        return {"dit": self.bundle.dit}


__all__ = ["DreamZeroArgs", "DreamZeroEntry"]
