#!/usr/bin/env python3
"""Profile one in-process DreamZero chunk with recorded observations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import (
    BackendConfig,
    DeviceConfig,
    EngineConfig,
    ParallelConfig,
    RuntimeConfig,
)
from phyai.models.dreamzero.main_dreamzero import DreamZeroArgs
from phyai_utils_tools.models.dreamzero import DreamZeroProcessor


TRACE_ENV_VARS = (
    "DREAMZERO_CLIP_RUNNER_TRACE_DIR",
    "DREAMZERO_DIT_DETAIL_TRACE_DIR",
    "DREAMZERO_DIT_DETAIL_TRACE_BLOCKS",
    "DREAMZERO_DIT_DETAIL_TRACE_STAGE_BLOCKS",
    "DREAMZERO_DIT_TRACE_REFERENCE",
    "DREAMZERO_DIT_TRACE_UNCOND",
    "DREAMZERO_DIT_TRACE_WARMUP",
    "DREAMZERO_DYNAMIC_DIT_TRACE",
    "DREAMZERO_ENCODER_BOUNDARY_TRACE_DIR",
    "DREAMZERO_REFERENCE_ENCODER_TRACE_DIR",
    "DREAMZERO_STEP_TRACE_DIR",
    "DREAMZERO_SYNC_DIT_OUTPUTS",
)

REQUIRED_TRACE_KEYS = (
    "prompt",
    "request_environment_step",
    "request_right_image",
    "request_left_image",
    "request_wrist_image",
    "request_joint_position",
    "request_gripper_position",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile exactly one in-process DreamZero engine.step call. Input "
            "preprocessing and action postprocessing are outside the measured scope."
        )
    )
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--chunk-index",
        type=int,
        default=0,
        help="Zero-based target chunk. Earlier chunks are replayed outside profiling.",
    )
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1140)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--dynamic-dit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dynamic-dit-scheduler-steps", type=int, default=16)
    parser.add_argument("--attn-backend", default="flashinfer")
    parser.add_argument("--norm-backend", default="flashinfer")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--validate-input-only",
        action="store_true",
        help="Validate and summarize the recorded trace without loading the model.",
    )
    parser.add_argument(
        "--sequential-cpu-offload",
        action="store_true",
        help=(
            "Offload inactive modules to CPU. This is required by the current "
            "resident model on one 48 GB A40, but transfer time remains in the "
            "engine.step measurement."
        ),
    )
    parser.add_argument(
        "--torch-profiler",
        action="store_true",
        help="Capture the target chunk with torch.profiler and export a Chrome trace.",
    )
    parser.add_argument(
        "--profiler-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--profiler-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--profiler-with-stack", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help=(
            "Call cudaProfilerStart/Stop around only the target engine.step. Use "
            "with Nsight Systems capture-range=cudaProfilerApi or Nsight Compute "
            "profile-from-start=off."
        ),
    )
    parser.add_argument(
        "--allow-trace-env",
        action="store_true",
        help="Allow active DreamZero tensor/debug trace environment variables.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.chunk_index < 0:
        raise ValueError("--chunk-index must be non-negative.")
    if args.frames_per_chunk <= 0:
        raise ValueError("--frames-per-chunk must be positive.")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative.")
    if args.dynamic_dit_scheduler_steps < 2:
        raise ValueError("--dynamic-dit-scheduler-steps must be at least 2.")
    if args.torch_profiler and args.repetitions != 1:
        raise ValueError("Use --repetitions 1 with --torch-profiler.")
    if args.torch_profiler and args.cuda_profiler_range:
        raise ValueError(
            "Use --torch-profiler and --cuda-profiler-range in separate runs."
        )
    if not args.validate_input_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DreamZero chunk profiling.")

    active_trace_env = {
        name: os.environ[name]
        for name in TRACE_ENV_VARS
        if name in os.environ and os.environ[name].lower() not in {"", "0", "false", "no", "off"}
    }
    if active_trace_env and not args.allow_trace_env:
        names = ", ".join(sorted(active_trace_env))
        raise RuntimeError(
            f"Refusing to profile with active tensor/debug trace variables: {names}. "
            "Unset them or pass --allow-trace-env intentionally."
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_frames(values: np.ndarray, index: int, count: int) -> np.ndarray:
    start = max(0, index - count + 1)
    frames = [values[position] for position in range(start, index + 1)]
    while len(frames) < count:
        frames.insert(0, frames[0])
    return np.stack(frames, axis=0)


def load_observations(path: Path, frames_per_chunk: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as source:
        missing = sorted(set(REQUIRED_TRACE_KEYS) - set(source.files))
        if missing:
            raise ValueError(f"Input trace is missing keys: {missing}")
        arrays = {key: source[key] for key in REQUIRED_TRACE_KEYS}

    request_count = int(len(arrays["request_environment_step"]))
    prompt = str(arrays["prompt"].item())
    observations: list[dict[str, Any]] = []
    for index in range(request_count):
        frame_count = 1 if index == 0 else frames_per_chunk
        views = [
            latest_frames(arrays["request_right_image"], index, frame_count),
            latest_frames(arrays["request_left_image"], index, frame_count),
            latest_frames(arrays["request_wrist_image"], index, frame_count),
        ]
        video = np.stack(views, axis=1)
        state = np.concatenate(
            [
                np.asarray(arrays["request_joint_position"][index]).reshape(-1),
                np.asarray(arrays["request_gripper_position"][index]).reshape(-1),
            ]
        ).astype(np.float32)
        observations.append(
            {
                "video": video[None, ...],
                "state": torch.from_numpy(state.reshape(1, 1, -1)),
                "task": [prompt],
            }
        )
    metadata = {
        "prompt": prompt,
        "request_count": request_count,
        "environment_steps": arrays["request_environment_step"].tolist(),
    }
    return observations, metadata


def build_engine_and_processor(args: argparse.Namespace) -> tuple[Engine, DreamZeroProcessor]:
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    config = EngineConfig(
        backends=BackendConfig(attn=args.attn_backend, norm=args.norm_backend),
        device=DeviceConfig(target="cuda:0", params_dtype=torch.bfloat16),
        parallel=ParallelConfig(world_size=1, cfg_size=1, tp_size=1),
        runtime=RuntimeConfig(use_cuda_graph=False),
    )
    engine = Engine(
        EngineArgs(
            plugin="dreamzero",
            plugin_args=DreamZeroArgs(
                checkpoint_dir=args.ckpt_dir,
                seed=args.seed,
                weight_strict=not args.non_strict,
                progress=args.progress,
                sequential_cpu_offload=args.sequential_cpu_offload,
                num_inference_steps=args.num_inference_steps,
                dynamic_dit=args.dynamic_dit,
                dynamic_dit_scheduler_steps=args.dynamic_dit_scheduler_steps,
            ),
            config=config,
        )
    )
    bundle = engine.entry.bundle  # type: ignore[attr-defined]
    processor = DreamZeroProcessor.from_pretrained(
        args.ckpt_dir,
        tokenizer_name=args.tokenizer,
        max_length=bundle.config.text_encoder.max_length,
        max_state_dim=bundle.config.max_state_dim,
        max_action_dim=bundle.config.max_action_dim,
        action_horizon=bundle.config.action_horizon,
        num_views=3,
    )
    return engine, processor


def reset_sequence(engine: Engine) -> None:
    bundle = engine.entry.bundle  # type: ignore[attr-defined]
    bundle.pipeline._condition = None
    bundle.pipeline.scheduler.reset_sequence()


def prepare_target_state(engine: Engine, processed: list[Any], target: int) -> None:
    reset_sequence(engine)
    for index in range(target):
        engine.step(processed[index])


def cache_lengths(output: Any) -> list[int | None]:
    return [
        None if cache is None else int(cache.shape[2])
        for cache in output.scheduler_output.cond_kv_cache
    ]


def measure_target(
    engine: Engine,
    processed: Any,
    *,
    chunk_index: int,
    cuda_profiler_range: bool,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cudart = torch.cuda.cudart() if cuda_profiler_range else None
    if cudart is not None:
        result = cudart.cudaProfilerStart()
        if result != 0:
            raise RuntimeError(f"cudaProfilerStart failed with error code {result}.")

    torch.cuda.nvtx.range_push(f"dreamzero.engine_step.chunk_{chunk_index}")
    try:
        start_event.record()
        wall_started = time.perf_counter_ns()
        output = engine.step(processed)
        end_event.record()
        end_event.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_started) / 1e6
    finally:
        torch.cuda.nvtx.range_pop()
        if cudart is not None:
            result = cudart.cudaProfilerStop()
            if result != 0:
                raise RuntimeError(f"cudaProfilerStop failed with error code {result}.")

    action = output.action
    lengths = cache_lengths(output)
    non_empty_lengths = [length for length in lengths if length is not None]
    return {
        "chunk_index": chunk_index,
        "wall_ms": wall_ms,
        "cuda_ms": float(start_event.elapsed_time(end_event)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "action_shape": list(action.shape),
        "action_min": float(action.min().item()),
        "action_max": float(action.max().item()),
        "action_finite": bool(torch.isfinite(action).all().item()),
        "request_start_frame": output.request.current_start_frame,
        "next_start_frame": output.scheduler_output.current_start_frame,
        "kv_min": min(non_empty_lengths) if non_empty_lengths else 0,
        "kv_max": max(non_empty_lengths) if non_empty_lengths else 0,
        "dit_compute_steps": int(output.scheduler_output.dit_compute_steps),
        "scheduler_steps": int(output.scheduler_output.scheduler_steps),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_manifest(args: argparse.Namespace) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short", "--untracked-files=no"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flashinfer": package_version("flashinfer-python"),
        "transformers": package_version("transformers"),
        "gpu": properties.name,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_total_memory_mib": properties.total_memory / (1024**2),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf32": args.allow_tf32,
        "trace_environment": {
            name: os.environ[name] for name in TRACE_ENV_VARS if name in os.environ
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    wall = np.asarray([row["wall_ms"] for row in rows], dtype=np.float64)
    cuda = np.asarray([row["cuda_ms"] for row in rows], dtype=np.float64)
    wall_mean = float(np.mean(wall))
    return {
        "samples": len(rows),
        "wall_mean_ms": wall_mean,
        "wall_median_ms": float(np.median(wall)),
        "wall_p95_ms": float(np.percentile(wall, 95)),
        "wall_min_ms": float(np.min(wall)),
        "wall_max_ms": float(np.max(wall)),
        "wall_stdev_ms": float(statistics.stdev(wall)) if len(rows) > 1 else 0.0,
        "wall_cv": (
            float(statistics.stdev(wall) / wall_mean)
            if len(rows) > 1 and wall_mean
            else 0.0
        ),
        "cuda_mean_ms": float(np.mean(cuda)),
        "cuda_median_ms": float(np.median(cuda)),
        "cuda_p95_ms": float(np.percentile(cuda, 95)),
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    observations, input_metadata = load_observations(
        args.input, args.frames_per_chunk
    )
    if args.chunk_index >= len(observations):
        raise ValueError(
            f"--chunk-index {args.chunk_index} is outside the trace with "
            f"{len(observations)} requests."
        )
    input_summary = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "chunk_index": args.chunk_index,
        "input_frames": int(observations[args.chunk_index]["video"].shape[1]),
        "video_shape": list(observations[args.chunk_index]["video"].shape),
        "state_shape": list(observations[args.chunk_index]["state"].shape),
        "prefix_chunks": args.chunk_index,
        "metadata": input_metadata,
    }
    if args.validate_input_only:
        print(json.dumps(input_summary, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    engine, processor = build_engine_and_processor(args)
    try:
        processed = [processor.preprocess(observation) for observation in observations]
        for _ in range(args.warmup_runs):
            prepare_target_state(engine, processed, args.chunk_index)
            engine.step(processed[args.chunk_index])
            torch.cuda.synchronize()

        rows: list[dict[str, Any]] = []
        profiler_table_path: Path | None = None
        profiler_trace_path: Path | None = None
        if args.torch_profiler:
            prepare_target_state(engine, processed, args.chunk_index)
            with torch.profiler.profile(
                activities=(
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ),
                record_shapes=args.profiler_record_shapes,
                profile_memory=args.profiler_memory,
                with_stack=args.profiler_with_stack,
            ) as profiler:
                rows.append(
                    measure_target(
                        engine,
                        processed[args.chunk_index],
                        chunk_index=args.chunk_index,
                        cuda_profiler_range=False,
                    )
                )
            profiler_trace_path = args.output_dir / "torch_trace.json"
            profiler_table_path = args.output_dir / "torch_key_averages.txt"
            profiler.export_chrome_trace(str(profiler_trace_path))
            profiler_table_path.write_text(
                profiler.key_averages(group_by_input_shape=True).table(
                    sort_by="self_cuda_time_total",
                    row_limit=100,
                ),
                encoding="utf-8",
            )
        else:
            for repetition in range(args.repetitions):
                prepare_target_state(engine, processed, args.chunk_index)
                row = measure_target(
                    engine,
                    processed[args.chunk_index],
                    chunk_index=args.chunk_index,
                    cuda_profiler_range=args.cuda_profiler_range,
                )
                row["repetition"] = repetition
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

        summary = {
            "scope": "in-process engine.step(processed); preprocessing excluded",
            "input": input_summary["input"],
            "input_sha256": input_summary["input_sha256"],
            "input_metadata": input_metadata,
            "chunk_index": args.chunk_index,
            "input_frames": int(observations[args.chunk_index]["video"].shape[1]),
            "prefix_chunks_replayed_outside_profile": args.chunk_index,
            "warmup_runs": args.warmup_runs,
            "configuration": {
                "seed": args.seed,
                "num_inference_steps": args.num_inference_steps,
                "dynamic_dit": args.dynamic_dit,
                "dynamic_dit_scheduler_steps": args.dynamic_dit_scheduler_steps,
                "attn_backend": args.attn_backend,
                "norm_backend": args.norm_backend,
                "dtype": "bfloat16",
                "world_size": 1,
                "cuda_graph": False,
                "sequential_cpu_offload": args.sequential_cpu_offload,
            },
            "environment": environment_manifest(args),
            "aggregate": aggregate(rows),
            "rows": rows,
            "torch_profiler_trace": (
                str(profiler_trace_path) if profiler_trace_path is not None else None
            ),
            "torch_profiler_table": (
                str(profiler_table_path) if profiler_table_path is not None else None
            ),
        }
        summary_path = args.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary["aggregate"], indent=2), flush=True)
        print(f"Wrote {summary_path}", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
