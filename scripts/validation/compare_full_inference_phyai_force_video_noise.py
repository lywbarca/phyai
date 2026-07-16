from __future__ import annotations

import argparse
import os
import signal
import shutil
import sys
import traceback
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def progress(msg: str) -> None:
    print(f"[phyai-full-compare] {msg}", flush=True)


def install_signal_diagnostics() -> None:
    def report(signum, frame):
        print(
            f"[phyai-full-compare] received signal={signum} pid={os.getpid()} "
            f"rank={os.environ.get('RANK', '?')} local_rank={os.environ.get('LOCAL_RANK', '?')}",
            flush=True,
        )
        traceback.print_stack(frame)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, report)


def install_phyai_ext_stub() -> None:
    if "phyai_ext.radix_cache" in sys.modules:
        return
    phyai_ext = types.ModuleType("phyai_ext")
    radix_cache = types.ModuleType("phyai_ext.radix_cache")

    class _UnavailablePhyAIExt:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("phyai_ext is not available in this validation script")

    class PrefixCache(_UnavailablePhyAIExt):
        pass

    class MatchResult(_UnavailablePhyAIExt):
        pass

    class Tier:
        GPU = "GPU"
        CPU = "CPU"

    def __getattr__(name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        placeholder = type(name, (_UnavailablePhyAIExt,), {})
        setattr(radix_cache, name, placeholder)
        return placeholder

    radix_cache.PrefixCache = PrefixCache
    radix_cache.MatchResult = MatchResult
    radix_cache.Tier = Tier
    radix_cache.__file__ = "<validation-stub>"
    radix_cache.__getattr__ = __getattr__
    phyai_ext.radix_cache = radix_cache
    sys.modules.setdefault("phyai_ext", phyai_ext)
    sys.modules["phyai_ext.radix_cache"] = radix_cache


install_phyai_ext_stub()

progress("before phyai imports")
from phyai.engine import Engine, EngineArgs  # noqa: E402
from phyai.engine_config import (  # noqa: E402
    BackendConfig,
    DeviceConfig,
    EngineConfig,
    ParallelConfig,
    RuntimeConfig,
)
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig  # noqa: E402
from phyai.models.dreamzero.main_dreamzero import DreamZeroArgs  # noqa: E402
from phyai.utils import load_config  # noqa: E402
from phyai_utils_tools.models.dreamzero import DreamZeroPolicy, DreamZeroProcessor  # noqa: E402
progress("after phyai imports")


PROMPT = "Pick up the blue cube and place it into the bowl."


def set_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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


def make_raw_obs() -> dict:
    b, t, v, h, w, c = 1, 1, 3, 176, 320, 3
    video = (np.arange(b * t * v * h * w * c, dtype=np.uint32) * 17 + 23) % 256
    video = video.astype(np.uint8).reshape(b, t, v, h, w, c)
    state = np.linspace(-0.35, 0.45, num=b * 1 * 8, dtype=np.float32).reshape(b, 1, 8)
    return {
        "video": torch.from_numpy(video),
        "state": torch.from_numpy(state),
        "state.joint_position": torch.from_numpy(state[..., :7].copy()),
        "state.gripper_position": torch.from_numpy(state[..., 7:8].copy()),
        "task": [PROMPT],
        "annotation.language.action_text": [PROMPT],
    }


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, object]:
    ref = ref.detach().cpu().float()
    got = got.detach().cpu().float()
    diff = got - ref
    return {
        "same_shape": tuple(ref.shape) == tuple(got.shape),
        "shape": tuple(got.shape),
        "max_abs": diff.abs().max().item() if diff.numel() else 0.0,
        "mean_abs": diff.abs().mean().item() if diff.numel() else 0.0,
        "rms_abs": diff.pow(2).mean().sqrt().item() if diff.numel() else 0.0,
        "max_rel": (diff.abs() / ref.abs().clamp_min(1e-3)).max().item() if diff.numel() else 0.0,
    }


def print_metrics(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    m = metrics(ref, got)
    print(
        f"{name}: shape={m['shape']} same_shape={m['same_shape']} "
        f"max_abs={m['max_abs']:.9g} mean_abs={m['mean_abs']:.9g} "
        f"rms_abs={m['rms_abs']:.9g} max_rel={m['max_rel']:.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="/data/share/DreamZero-DROID")
    parser.add_argument("--tokenizer", default="/data/share/google-umt5-xxl")
    parser.add_argument("--official-dump", default="/tmp/dreamzero_full_official.pt")
    parser.add_argument("--output", default="/tmp/dreamzero_full_phyai_tp2_cfg2.pt")
    parser.add_argument("--trace-dir", default="/tmp/dreamzero_full_phyai_tp2_cfg2_trace")
    parser.add_argument("--force-video-noise", default=None)
    parser.add_argument("--step-trace", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16", "float16"))
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--cfg-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--num-inference-steps", type=int, default=16)
    parser.add_argument("--attn-backend", default="flashinfer")
    parser.add_argument("--norm-backend", default="flashinfer")
    parser.add_argument("--non-strict", action="store_true")
    args = parser.parse_args()

    install_signal_diagnostics()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    progress(f"cpu threads={torch.get_num_threads()}")
    progress("set determinism")
    set_determinism(args.seed)
    device = resolve_device(args.device)
    dtype = dtype_from_name(args.dtype)
    world_size = args.tp_size * args.cfg_size

    progress("load config")
    config_obj: DreamZeroConfig = load_config(args.ckpt_dir, DreamZeroConfig)
    config_obj = replace(config_obj, num_inference_timesteps=args.num_inference_steps)

    engine_config = EngineConfig(
        backends=BackendConfig(attn=args.attn_backend, norm=args.norm_backend),
        device=DeviceConfig(target=device, params_dtype=dtype),
        parallel=ParallelConfig(
            world_size=world_size,
            cfg_size=args.cfg_size,
            tp_size=args.tp_size,
        ),
        runtime=RuntimeConfig(use_cuda_graph=False),
    )
    progress("before engine init")
    engine = Engine(
        EngineArgs(
            plugin="dreamzero",
            plugin_args=DreamZeroArgs(
                checkpoint_dir=args.ckpt_dir,
                config=config_obj,
                seed=args.seed,
                weight_strict=not args.non_strict,
                progress=True,
                encoder_strategy="rank0_broadcast",
                sequential_cpu_offload=True,
                skip_parameter_init=True,
            ),
            config=engine_config,
        )
    )
    progress("after engine init")
    try:
        bundle = engine.entry.bundle  # type: ignore[attr-defined]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        trace_dir = Path(args.trace_dir)
        if rank == 0:
            shutil.rmtree(trace_dir, ignore_errors=True)
            trace_dir.mkdir(parents=True, exist_ok=True)

        def save_trace(name: str, value) -> None:
            if rank == 0:
                torch.save(value.detach().cpu(), trace_dir / f"{name}.pt")

        pipeline = bundle.pipeline
        if args.step_trace:
            import phyai.models.dreamzero.scheduler_ws1_dreamzero as sched_mod

            step_trace_dir = trace_dir / "steps"
            if rank == 0:
                step_trace_dir.mkdir(parents=True, exist_ok=True)
            dist.barrier()

            def save_step_tensor(name: str, value) -> None:
                if not isinstance(value, torch.Tensor):
                    return
                if rank == 0:
                    torch.save(value.detach().cpu(), step_trace_dir / f"{name}.pt")

            def tensor_stats(value: torch.Tensor) -> dict[str, object]:
                value = value.detach()
                finite = torch.isfinite(value)
                as_float = value.float()
                return {
                    "shape": tuple(value.shape),
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                    "mean": as_float.mean().item() if value.numel() else 0.0,
                    "std": as_float.std().item() if value.numel() > 1 else 0.0,
                    "abs_max": as_float.abs().max().item() if value.numel() else 0.0,
                    "finite": bool(finite.all().item()) if value.numel() else True,
                }

            def kv_cache_stats(cache) -> list[dict[str, object] | None]:
                out = []
                for item in cache or []:
                    out.append(None if item is None else tensor_stats(item))
                return out

            original_run_runner = pipeline.scheduler._run_runner
            run_call_index = {"value": 0}

            def traced_run_runner(*rr_args, **rr_kwargs):
                call_index = run_call_index["value"]
                run_call_index["value"] += 1
                runner = rr_args[0] if rr_args else None
                if runner is not None and hasattr(runner, "model"):
                    runner.model._dz_trace_enabled = (
                        os.getenv("DREAMZERO_DIT_DETAIL_TRACE_DIR") is not None
                        and call_index == 1
                    )
                    runner.model._dz_trace_label = f"DiT_step_{call_index - 1:02d}"
                    runner.model._dz_trace_branch = 0
                output = original_run_runner(*rr_args, **rr_kwargs)
                if runner is not None and hasattr(runner, "model"):
                    runner.model._dz_trace_enabled = False
                update_kv_cache = bool(rr_kwargs.get("update_kv_cache", False))
                action = rr_kwargs.get("action", None)
                prefix = f"run_runner_{call_index:02d}_rank{rank}"
                if rank == 0:
                    if isinstance(output.video, torch.Tensor):
                        torch.save(output.video.detach().cpu(), step_trace_dir / f"{prefix}.video.pt")
                    if isinstance(output.action, torch.Tensor):
                        torch.save(output.action.detach().cpu(), step_trace_dir / f"{prefix}.action.pt")
                    meta = {
                        "call_index": call_index,
                        "rank": rank,
                        "update_kv_cache": update_kv_cache,
                        "has_action_input": isinstance(action, torch.Tensor),
                        "video": tensor_stats(output.video),
                        "action": tensor_stats(output.action) if isinstance(output.action, torch.Tensor) else None,
                        "cond_kv_cache": kv_cache_stats(pipeline.scheduler.cond_runner.kv_cache),
                    }
                    torch.save(meta, step_trace_dir / f"{prefix}.meta.pt")
                return output

            pipeline.scheduler._run_runner = traced_run_runner

            original_combine = pipeline.scheduler._combine_cfg_parallel
            combine_index = {"value": 0}

            def traced_combine_cfg_parallel(*combine_args, **combine_kwargs):
                video_pred, action_pred = original_combine(*combine_args, **combine_kwargs)
                step_index = combine_index["value"]
                combine_index["value"] += 1
                save_step_tensor(f"step_{step_index:02d}.video_pred_cfg", video_pred)
                save_step_tensor(f"step_{step_index:02d}.action_pred_cfg", action_pred)
                return video_pred, action_pred

            pipeline.scheduler._combine_cfg_parallel = traced_combine_cfg_parallel

            original_stepper_step = sched_mod.DreamZeroFlowStepper.step
            stepper_call_index = {"video": 0, "action": 0}

            def traced_stepper_step(self, *, model_output, sample, step_index, timestep=None):
                result = original_stepper_step(
                    self,
                    model_output=model_output,
                    sample=sample,
                    step_index=step_index,
                    timestep=timestep,
                )
                kind = "video" if sample.ndim == 5 else "action"
                call_index = stepper_call_index[kind]
                stepper_call_index[kind] += 1
                save_step_tensor(f"step_{call_index:02d}.{kind}_sample_before", sample)
                save_step_tensor(f"step_{call_index:02d}.{kind}_model_output", model_output)
                save_step_tensor(f"step_{call_index:02d}.{kind}_sample_after", result)
                if rank == 0:
                    torch.save(
                        {
                            "kind": kind,
                            "call_index": call_index,
                            "step_index": step_index,
                            "timestep": None
                            if timestep is None
                            else timestep.detach().cpu(),
                            "sample_before": tensor_stats(sample),
                            "model_output": tensor_stats(model_output),
                            "sample_after": tensor_stats(result),
                        },
                        step_trace_dir / f"step_{call_index:02d}.{kind}_meta.pt",
                    )
                return result

            sched_mod.DreamZeroFlowStepper.step = traced_stepper_step

        original_encode_text = pipeline.encode_text

        def traced_encode_text(processed):
            context, uncond_context = original_encode_text(processed)
            save_trace("text_cond_input_ids", processed.input_ids)
            save_trace("text_cond_attention_mask", processed.attention_mask)
            save_trace("text_uncond_input_ids", processed.negative_input_ids)
            save_trace("text_uncond_attention_mask", processed.negative_attention_mask)
            save_trace("text_cond_embedding", context)
            save_trace("text_uncond_embedding", uncond_context)
            return context, uncond_context

        original_encode_first_frame_condition = pipeline.encode_first_frame_condition

        def traced_encode_first_frame_condition(videos):
            condition = original_encode_first_frame_condition(videos)
            save_trace("videos_bcthw", videos)
            save_trace("first_frame_btc_hw", videos[:, :, :1].transpose(1, 2))
            save_trace("clip_feature", condition.clip_feature)
            save_trace("vae_condition_y", condition.y)
            save_trace("clean_video", condition.clean_video)
            return condition

        original_build_request = pipeline.build_request

        def traced_build_request(processed, **kwargs):
            request = original_build_request(processed, **kwargs)
            if args.force_video_noise:
                forced_video = torch.load(args.force_video_noise, map_location="cpu")
                request.video = forced_video.to(
                    device=request.video.device,
                    dtype=request.video.dtype,
                )
            save_trace("processed_input.images", processed.images)
            save_trace("processed_input.state", processed.state)
            save_trace("processed_input.embodiment_id", processed.embodiment_id)
            save_trace("initial_video_noise", request.video)
            save_trace("initial_action_noise", request.action)
            save_trace("dit_state", request.state)
            if request.embodiment_id is not None:
                save_trace("dit_embodiment_id", request.embodiment_id)
            return request

        pipeline.encode_text = traced_encode_text
        pipeline.encode_first_frame_condition = traced_encode_first_frame_condition
        pipeline.build_request = traced_build_request
        progress("before processor init")
        processor = DreamZeroProcessor.from_pretrained(
            args.ckpt_dir,
            tokenizer_name=args.tokenizer,
            max_length=bundle.config.text_encoder.max_length,
            max_state_dim=bundle.config.max_state_dim,
            max_action_dim=bundle.config.max_action_dim,
            action_horizon=bundle.config.action_horizon,
            num_views=3,
        )
        progress("after processor init")
        policy = DreamZeroPolicy(processor=processor, infer=engine.step)
        obs = make_raw_obs()
        progress("before policy.act")
        out = policy.act(obs)
        progress("after policy.act")
        if rank == 0:
            phyai_action = out.model_output.action.detach().cpu().float()
            phyai_final_action = out.action.detach().cpu().float()
            phyai_video = out.model_output.video.detach().cpu().float()
            payload = {
                "meta": {
                    "seed": args.seed,
                    "num_inference_steps": args.num_inference_steps,
                    "tp_size": args.tp_size,
                    "cfg_size": args.cfg_size,
                    "prompt": PROMPT,
                },
                "outputs": {
                    "normalized_action": phyai_action,
                    "final_action": phyai_final_action,
                    "video": phyai_video,
                },
                "trace_dir": str(trace_dir),
            }
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.output)
            print(f"saved {args.output}")
            print(f"normalized_action={tuple(phyai_action.shape)}")
            print(f"final_action={tuple(phyai_final_action.shape)}")
            print(f"video={tuple(phyai_video.shape)}")

            if Path(args.official_dump).exists():
                official = torch.load(args.official_dump, map_location="cpu")
                off_action = official["outputs"]["normalized_action"].float()
                off_final = official["outputs"]["final_action"].float()
                off_video = official["outputs"]["video_pred"].float()
                print_metrics("normalized_action", off_action, phyai_action)
                print_metrics("final_action", off_final, phyai_final_action)
                print_metrics("final_action.joint", off_final[..., :7], phyai_final_action[..., :7])
                print_metrics("final_action.gripper", off_final[..., 7:8], phyai_final_action[..., 7:8])
                if off_video.shape[2] != phyai_video.shape[2] and off_video.shape[2] > phyai_video.shape[2]:
                    off_video_cmp = off_video[:, :, -phyai_video.shape[2] :]
                else:
                    off_video_cmp = off_video
                print_metrics("video_latent", off_video_cmp, phyai_video)
            else:
                print(f"official dump not found: {args.official_dump}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
