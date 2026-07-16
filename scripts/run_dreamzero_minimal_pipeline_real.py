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
    parser.add_argument("--ckpt-dir", default="/data/share/DreamZero-DROID")
    parser.add_argument("--tokenizer", default="/data/share/google-umt5-xxl")
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
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--prompt", default="Pick up the cube.")
    parser.add_argument("--attn-backend", default="flashinfer")
    parser.add_argument("--norm-backend", default="flashinfer")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
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
        video = torch.randint(
            0,
            256,
            (1, args.frames, args.views, args.height, args.width, 3),
            dtype=torch.uint8,
        )
        state = torch.zeros(1, 1, args.state_dim, dtype=torch.float32)
        policy = DreamZeroPolicy(processor=processor, infer=engine.step)
        policy_out = policy.act({"video": video, "task": [args.prompt], "state": state})
        out = policy_out.model_output
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank == 0:
            print(
                "DreamZero minimal pipeline real smoke passed: "
                f"video={tuple(out.video.shape)} action={tuple(out.action.shape)} "
                f"post_action={tuple(policy_out.action.shape)} "
                f"dtype={out.action.dtype} device={out.action.device}"
            )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
