from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

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

try:
    from tfrecord.reader import tfrecord_loader
except ImportError as exc:  # pragma: no cover - exercised only without optional dep.
    raise SystemExit(
        "This evaluator requires the optional 'tfrecord' package. "
        "Install it in the active environment, e.g. `uv pip install tfrecord`."
    ) from exc


IMAGE_KEYS = {
    "observation/exterior_image_0_left": "steps/observation/exterior_image_1_left",
    "observation/exterior_image_1_left": "steps/observation/exterior_image_2_left",
    "observation/wrist_image_left": "steps/observation/wrist_image_left",
}
LANGUAGE_KEYS = (
    "steps/language_instruction",
    "steps/language_instruction_2",
    "steps/language_instruction_3",
)
RELATIVE_OFFSETS = (-23, -16, -8, 0)
ACTION_HORIZON = 24


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


def _decode_text(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_instruction(record: dict[str, Any]) -> str:
    for key in LANGUAGE_KEYS:
        if key not in record:
            continue
        values = record[key]
        for item in values[: min(len(values), 5)]:
            text = _decode_text(item).strip()
            if text:
                return text
    return "pick up the object"


def _decode_jpeg(buf: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(buf)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _decode_image_sequence(
    record: dict[str, Any],
    key: str,
    indices: list[int],
) -> np.ndarray:
    return np.stack([_decode_jpeg(record[key][i]) for i in indices], axis=0)


def _episode_length(record: dict[str, Any]) -> int:
    return int(len(record["steps/is_first"]))


def _reshape(record: dict[str, Any], key: str, length: int, dim: int) -> np.ndarray:
    return np.asarray(record[key]).reshape(length, dim)


def _make_frame_schedule(
    total_frames: int,
    chunks: int,
    *,
    single_frame_chunks: bool = False,
) -> list[tuple[str, list[int], int]]:
    schedule = [("initial", [0], 0)]
    current_frame = 23
    for chunk_idx in range(chunks):
        if single_frame_chunks:
            indices = [current_frame]
        else:
            indices = [max(current_frame + offset, 0) for offset in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        schedule.append((f"chunk{chunk_idx}", indices, indices[-1]))
        current_frame += ACTION_HORIZON
    return schedule


def _iter_records(data_dir: Path):
    files = sorted(data_dir.glob("*.tfrecord-*"))
    if not files:
        files = sorted((data_dir / "1.0.0").glob("*.tfrecord-*"))
    if not files:
        raise FileNotFoundError(f"no tfrecord shards found under {data_dir}")
    for path in files:
        for record in tfrecord_loader(
            str(path),
            None,
            description=None,
            sequence_description=None,
        ):
            yield path, record


def _build_observation(
    record: dict[str, Any],
    frame_indices: list[int],
    current_index: int,
    prompt: str,
    joint_obs: np.ndarray,
    grip_obs: np.ndarray,
) -> dict[str, Any]:
    video = np.stack(
        [
            _decode_image_sequence(record, record_key, frame_indices)
            for record_key in IMAGE_KEYS.values()
        ],
        axis=1,
    )
    state = np.concatenate(
        [joint_obs[current_index], grip_obs[current_index]],
        axis=-1,
    ).astype(np.float32)
    return {
        "video": video[None, ...],
        "state": torch.from_numpy(state.reshape(1, 1, -1)),
        "task": [prompt],
    }


def _ground_truth_actions(
    record: dict[str, Any],
    start: int,
    horizon: int,
) -> np.ndarray:
    length = _episode_length(record)
    joint_cmd = _reshape(record, "steps/action_dict/joint_position", length, 7)
    grip_cmd = np.asarray(record["steps/action_dict/gripper_position"]).reshape(
        length,
        1,
    )
    end = min(start + horizon, length)
    gt = np.concatenate([joint_cmd[start:end], grip_cmd[start:end]], axis=-1)
    if gt.shape[0] < horizon:
        gt = np.concatenate(
            [gt, np.repeat(gt[-1:], horizon - gt.shape[0], axis=0)],
            axis=0,
        )
    return gt.astype(np.float32)


def _as_action_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().to(torch.float32).numpy()
    else:
        value = np.asarray(value, dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    return value.astype(np.float32, copy=False)


def _abs_metrics(diff: np.ndarray) -> dict[str, float]:
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.reshape(-1).astype(np.float64)
    b64 = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(a64) * np.linalg.norm(b64)
    if denom == 0:
        return 1.0 if np.linalg.norm(a64 - b64) == 0 else 0.0
    return float(np.dot(a64, b64) / denom)


def _action_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    diff = pred - ref
    metrics = _abs_metrics(diff)
    metrics["cos"] = _cosine(pred, ref)
    metrics["joint_mean_abs"] = float(np.mean(np.abs(diff[..., :7])))
    metrics["gripper_mean_abs"] = float(np.mean(np.abs(diff[..., 7:])))
    return metrics


def _load_reference(
    path: str | None,
) -> tuple[np.ndarray, np.ndarray | None, list[Any]]:
    if path is None:
        return np.empty((0,), dtype=np.float32), None, []
    data = np.load(path, allow_pickle=True)
    ref_pred = np.asarray(data["pred"], dtype=np.float32)
    ref_gt = np.asarray(data["gt"], dtype=np.float32) if "gt" in data.files else None
    rows = list(data["rows"]) if "rows" in data.files else []
    return ref_pred, ref_gt, rows


def _check_reference_rows(
    rows: list[dict[str, Any]],
    reference_rows: list[Any],
) -> list[str]:
    warnings = []
    for index, row in enumerate(rows[: len(reference_rows)]):
        ref_row = reference_rows[index]
        if hasattr(ref_row, "item"):
            ref_row = ref_row.item()
        if not isinstance(ref_row, dict):
            continue
        for key in ("episode", "query", "current_index"):
            if row.get(key) != ref_row.get(key):
                warnings.append(
                    f"reference row {index} {key} mismatch: "
                    f"phyai={row.get(key)!r} ref={ref_row.get(key)!r}"
                )
    return warnings


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def evaluate(args: argparse.Namespace) -> dict[str, Any] | None:
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
                progress=args.progress,
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
        policy = DreamZeroPolicy(processor=processor, infer=engine.step)
        rank = _rank()

        pred_rows: list[np.ndarray] = []
        normalized_rows: list[np.ndarray] = []
        gt_rows: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        episodes_done = 0
        queries_done = 0

        for shard_path, record in _iter_records(Path(args.data)):
            length = _episode_length(record)
            if length < ACTION_HORIZON + 1:
                continue

            prompt = args.prompt or _first_instruction(record)
            joint_obs = _reshape(record, "steps/observation/joint_position", length, 7)
            grip_obs = np.asarray(record["steps/observation/gripper_position"]).reshape(
                length,
                1,
            )
            schedule = _make_frame_schedule(
                length,
                args.chunks_per_episode,
                single_frame_chunks=args.single_frame_chunks,
            )
            if rank == 0 and args.verbose:
                print(
                    f"episode {episodes_done} shard={shard_path.name} "
                    f"T={length} prompt={prompt!r}",
                    flush=True,
                )

            for name, frame_indices, current_index in schedule:
                if current_index + ACTION_HORIZON > length:
                    continue
                if args.max_queries is not None and queries_done >= args.max_queries:
                    break

                obs = _build_observation(
                    record,
                    frame_indices,
                    current_index,
                    prompt,
                    joint_obs,
                    grip_obs,
                )
                policy_out = policy.act(obs)
                pred = _as_action_array(policy_out.action)
                normalized = _as_action_array(
                    policy_out.postprocessed.normalized_action
                )
                gt = _ground_truth_actions(record, current_index, ACTION_HORIZON)
                if pred.shape != gt.shape:
                    raise RuntimeError(
                        f"shape mismatch for {name}: pred={pred.shape} gt={gt.shape}"
                    )

                model_output = policy_out.model_output
                scheduler_output = model_output.scheduler_output
                request = model_output.request
                cond_lens = [
                    None if cache is None else int(cache.shape[2])
                    for cache in scheduler_output.cond_kv_cache
                ]
                non_empty_lens = [length for length in cond_lens if length is not None]
                row = {
                    "episode": episodes_done,
                    "query": name,
                    "current_index": current_index,
                    "frame_indices": list(frame_indices),
                    "prompt": prompt,
                    "request_start": request.current_start_frame,
                    "next_start": scheduler_output.current_start_frame,
                    "reference_video": request.reference_video is not None,
                    "kv_min_len": min(non_empty_lens) if non_empty_lens else 0,
                    "kv_max_len": max(non_empty_lens) if non_empty_lens else 0,
                    "pred_min": float(np.min(pred)),
                    "pred_max": float(np.max(pred)),
                    "gt_min": float(np.min(gt)),
                    "gt_max": float(np.max(gt)),
                    "nan_count": int(np.isnan(pred).sum()),
                    "inf_count": int(np.isinf(pred).sum()),
                }
                row.update(
                    {
                        f"gt_{key}": value
                        for key, value in _action_metrics(pred, gt).items()
                    }
                )

                if rank == 0:
                    pred_rows.append(pred)
                    normalized_rows.append(normalized)
                    gt_rows.append(gt)
                    rows.append(row)
                    print(
                        f"ep={episodes_done} {name:8s} idx={current_index:3d} "
                        f"start={row['request_start']}->{row['next_start']} "
                        f"ref={row['reference_video']} "
                        f"kv=({row['kv_min_len']},{row['kv_max_len']}) "
                        f"gt_mae={row['gt_mean_abs']:.5f} "
                        f"gt_joint={row['gt_joint_mean_abs']:.5f} "
                        f"gt_grip={row['gt_gripper_mean_abs']:.5f}",
                        flush=True,
                    )
                queries_done += 1

            episodes_done += 1
            if args.max_queries is not None and queries_done >= args.max_queries:
                break
            if episodes_done >= args.episodes:
                break

        if rank != 0:
            return None
        if not rows:
            raise RuntimeError("no evaluation rows produced")

        pred_arr = np.stack(pred_rows, axis=0)
        normalized_arr = np.stack(normalized_rows, axis=0)
        gt_arr = np.stack(gt_rows, axis=0)
        gt_metrics = _action_metrics(pred_arr, gt_arr)
        summary: dict[str, Any] = {
            "data": str(args.data),
            "ckpt_dir": str(args.ckpt_dir),
            "episodes": episodes_done,
            "queries": len(rows),
            "chunks_per_episode": args.chunks_per_episode,
            "single_frame_chunks": args.single_frame_chunks,
            "num_inference_steps": args.num_inference_steps,
            "cfg_size": args.cfg_size,
            "tp_size": args.tp_size,
            "dtype": args.dtype,
            "attn_backend": args.attn_backend,
            "norm_backend": args.norm_backend,
            "pred_shape": list(pred_arr.shape),
            "gt_shape": list(gt_arr.shape),
            "gt_metrics": gt_metrics,
            "rows": rows,
        }

        ref_pred, ref_gt, ref_rows = _load_reference(args.reference_npz)
        if args.reference_npz is not None:
            if ref_pred.shape[0] < pred_arr.shape[0]:
                raise RuntimeError(
                    f"reference has fewer rows than PhyAI output: "
                    f"{ref_pred.shape[0]} < {pred_arr.shape[0]}"
                )
            ref_pred = ref_pred[: pred_arr.shape[0]]
            summary["reference_npz"] = args.reference_npz
            summary["reference_pred_shape"] = list(ref_pred.shape)
            summary["reference_metrics"] = _action_metrics(pred_arr, ref_pred)
            warnings = _check_reference_rows(rows, ref_rows)
            if ref_gt is not None:
                ref_gt = ref_gt[: gt_arr.shape[0]]
                summary["reference_gt_metrics"] = _action_metrics(gt_arr, ref_gt)
            if warnings:
                summary["reference_warnings"] = warnings

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            pred=pred_arr,
            normalized_action=normalized_arr,
            gt=gt_arr,
            rows=np.array(rows, dtype=object),
            summary=json.dumps(summary, indent=2),
        )
        json_path = output.with_suffix(".json")
        json_path.write_text(json.dumps(summary, indent=2))
        print(
            "summary:",
            json.dumps(
                {key: value for key, value in summary.items() if key != "rows"},
                indent=2,
            ),
            flush=True,
        )
        print(f"saved: {output}", flush=True)
        print(f"saved: {json_path}", flush=True)
        return summary
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PhyAI DreamZero on real DROID TFRecord samples."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-npz", default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--chunks-per-episode", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--single-frame-chunks", action="store_true")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument("--cfg-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--attn-backend", default="flashinfer")
    parser.add_argument("--norm-backend", default="flashinfer")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
