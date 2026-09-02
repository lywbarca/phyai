from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import (
    BackendConfig,
    DeviceConfig,
    EngineConfig,
    ParallelConfig,
    RuntimeConfig,
)
from phyai.models.dreamzero.main_dreamzero import DreamZeroArgs
from phyai_utils_tools.models.dreamzero import DreamZeroPolicy, DreamZeroProcessor


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def resolve_device(name: str) -> str:
    if name != "cuda":
        return name
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return f"cuda:{local_rank}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument("--cfg-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=4,
        help="Number of raw video frames for chunks after the first one.",
    )
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--prompt", default="Pick up the cube.")
    parser.add_argument("--attn-backend", default="flashinfer")
    parser.add_argument("--norm-backend", default="flashinfer")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    world_size = args.cfg_size * args.tp_size
    config = EngineConfig(
        backends=BackendConfig(attn=args.attn_backend, norm=args.norm_backend),
        device=DeviceConfig(target=device, params_dtype=dtype),
        parallel=ParallelConfig(
            world_size=world_size,
            cfg_size=args.cfg_size,
            tp_size=args.tp_size,
        ),
        runtime=RuntimeConfig(use_cuda_graph=False),
    )

    engine = Engine(
        EngineArgs(
            plugin="dreamzero",
            plugin_args=DreamZeroArgs(
                checkpoint_dir=args.ckpt_dir,
                seed=args.seed,
                weight_strict=not args.non_strict,
                progress=True,
                sequential_cpu_offload=args.sequential_cpu_offload,
                num_inference_steps=args.num_inference_steps,
            ),
            config=config,
        )
    )
    try:
        bundle = engine.entry.bundle  # type: ignore[attr-defined]
        processor = DreamZeroProcessor.from_pretrained(
            args.ckpt_dir,
            tokenizer_name=args.tokenizer,
            max_length=bundle.config.text_encoder.max_length,
            max_state_dim=bundle.config.max_state_dim,
            max_action_dim=bundle.config.max_action_dim,
            action_horizon=bundle.config.action_horizon,
            num_views=args.views,
        )
        state = torch.zeros(1, 1, args.state_dim, dtype=torch.float32)
        policy = DreamZeroPolicy(processor=processor, infer=engine.step)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        for chunk_idx in range(args.num_chunks):
            frames = args.frames if chunk_idx == 0 else args.chunk_frames
            video = torch.randint(
                0,
                256,
                (1, frames, args.views, args.height, args.width, 3),
                dtype=torch.uint8,
            )
            policy_out = policy.act(
                {"video": video, "task": [args.prompt], "state": state}
            )
            out = policy_out.model_output
            scheduler_out = out.scheduler_output
            request = out.request
            cond_lens = [
                None if cache is None else int(cache.shape[2])
                for cache in scheduler_out.cond_kv_cache
            ]
            if rank == 0:
                non_empty_lens = [length for length in cond_lens if length is not None]
                min_cache_len = min(non_empty_lens) if non_empty_lens else 0
                max_cache_len = max(non_empty_lens) if non_empty_lens else 0
                print(
                    "DreamZero chunk passed: "
                    f"chunk={chunk_idx} raw_frames={frames} "
                    f"request_start={request.current_start_frame} "
                    f"next_start={scheduler_out.current_start_frame} "
                    f"reference_video={request.reference_video is not None} "
                    f"video={tuple(out.video.shape)} action={tuple(out.action.shape)} "
                    f"post_action={tuple(policy_out.action.shape)} "
                    f"kv_len_range=({min_cache_len},{max_cache_len}) "
                    f"dtype={out.action.dtype} device={out.action.device}",
                    flush=True,
                )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
