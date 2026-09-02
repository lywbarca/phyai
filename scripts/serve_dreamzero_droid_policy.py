from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import os
import traceback
from collections import deque
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
from phyai_utils_tools.models.dreamzero import DreamZeroPolicy, DreamZeroProcessor

LOGGER = logging.getLogger("phyai.dreamzero.websocket")

ROBOARENA_IMAGE_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
)


@dataclasses.dataclass
class PolicyServerConfig:
    image_resolution: tuple[int, int] | None = (180, 320)
    needs_wrist_camera: bool = True
    n_external_cameras: int = 2
    needs_stereo_camera: bool = False
    needs_session_id: bool = True
    action_space: str = "joint_position"


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


class MsgpackCodec:
    def __init__(self) -> None:
        try:
            from openpi_client import msgpack_numpy
        except ImportError:
            msgpack_numpy = None
        self._msgpack_numpy = msgpack_numpy
        if msgpack_numpy is None:
            try:
                import msgpack
            except ImportError as exc:
                raise RuntimeError(
                    "This server needs either openpi_client or msgpack installed. "
                    "Install the lightweight deps with `uv pip install msgpack websockets` "
                    "in the PhyAI environment, or run it inside a sim_evals/openpi env."
                ) from exc
            self._msgpack = msgpack
        else:
            self._msgpack = None
            self._packer = msgpack_numpy.Packer()

    def pack(self, value: Any) -> bytes:
        if self._msgpack_numpy is not None:
            return self._packer.pack(value)
        return self._msgpack.packb(
            value,
            default=self._encode_numpy,
            use_bin_type=True,
        )

    def unpack(self, value: bytes) -> Any:
        if self._msgpack_numpy is not None:
            return self._msgpack_numpy.unpackb(value)
        return self._msgpack.unpackb(
            value,
            raw=False,
            object_hook=self._decode_numpy,
        )

    @staticmethod
    def _encode_numpy(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"dtype": value.dtype.str,
                b"shape": value.shape,
                b"data": value.tobytes(),
            }
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, torch.Tensor):
            return MsgpackCodec._encode_numpy(value.detach().cpu().numpy())
        raise TypeError(f"cannot msgpack encode {type(value)!r}")

    @staticmethod
    def _decode_numpy(value: dict[Any, Any]) -> Any:
        marker = value.get(b"__ndarray__", value.get("__ndarray__"))
        if marker is True:
            dtype = np.dtype(value.get(b"dtype", value.get("dtype")))
            shape = tuple(value.get(b"shape", value.get("shape")))
            data = value.get(b"data", value.get("data"))
            return np.ndarray(buffer=data, dtype=dtype, shape=shape)

        marker = value.get(b"__npgeneric__", value.get("__npgeneric__"))
        if marker is True:
            dtype = np.dtype(value.get(b"dtype", value.get("dtype")))
            data = value.get(b"data", value.get("data"))
            return dtype.type(data)

        marker = value.get(b"nd", value.get("nd"))  # codespell:ignore nd
        if marker is not True:
            return value
        dtype = np.dtype(value.get(b"type", value.get("type")))
        shape = tuple(value.get(b"shape", value.get("shape")))
        data = value.get(b"data", value.get("data"))
        return np.frombuffer(data, dtype=dtype).reshape(shape)


class PhyAIDreamZeroDroidPolicy:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(args.device)
        self.dtype = dtype_from_name(args.dtype)
        world_size = args.cfg_size * args.tp_size
        if world_size != 1:
            raise ValueError(
                "serve_dreamzero_droid_policy.py currently supports only "
                "cfg_size=1,tp_size=1. Keep the websocket process single-rank "
                "or add a distributed worker loop."
            )
        config = EngineConfig(
            backends=BackendConfig(attn=args.attn_backend, norm=args.norm_backend),
            device=DeviceConfig(target=self.device, params_dtype=self.dtype),
            parallel=ParallelConfig(
                world_size=world_size,
                cfg_size=args.cfg_size,
                tp_size=args.tp_size,
            ),
            runtime=RuntimeConfig(use_cuda_graph=False),
        )
        self.engine = Engine(
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
            ),
        )
        bundle = self.engine.entry.bundle  # type: ignore[attr-defined]
        processor = DreamZeroProcessor.from_pretrained(
            args.ckpt_dir,
            tokenizer_name=args.tokenizer,
            max_length=bundle.config.text_encoder.max_length,
            max_state_dim=bundle.config.max_state_dim,
            max_action_dim=bundle.config.max_action_dim,
            action_horizon=bundle.config.action_horizon,
            num_views=3,
        )
        self.policy = DreamZeroPolicy(processor=processor, infer=self.engine.step)
        self.frame_buffers: dict[str, deque[np.ndarray]] = {
            key: deque(maxlen=args.frames_per_chunk) for key in ROBOARENA_IMAGE_KEYS
        }
        self.current_session_id: str | None = None
        self.is_first_call = True
        self.request_count = 0
        self.sequence_request_count = 0

    def close(self) -> None:
        self.engine.close()

    def _reset_model_sequence(self, *, reason: str, clear_frame_buffers: bool) -> None:
        if clear_frame_buffers:
            self.frame_buffers = {
                key: deque(maxlen=self.args.frames_per_chunk)
                for key in ROBOARENA_IMAGE_KEYS
            }
        self.is_first_call = True
        self.sequence_request_count = 0
        bundle = self.engine.entry.bundle  # type: ignore[attr-defined]
        pipeline = bundle.pipeline
        pipeline._condition = None
        pipeline.scheduler.reset_sequence()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        LOGGER.info(
            "reset DreamZero sequence reason=%s clear_frame_buffers=%s",
            reason,
            clear_frame_buffers,
        )

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        if reset_info is not None:
            self.current_session_id = reset_info.get("session_id")
        self.request_count = 0
        self._reset_model_sequence(reason="session reset", clear_frame_buffers=True)

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self.current_session_id:
            LOGGER.info("new session_id=%s; resetting DreamZero state", session_id)
            self.current_session_id = str(session_id)
            self.reset({"session_id": self.current_session_id})

        for key in ROBOARENA_IMAGE_KEYS:
            if key not in obs:
                raise KeyError(f"missing required image key {key!r}")
            self._append_frames(key, obs[key])

        if (
            self.args.reset_every_chunks > 0
            and self.sequence_request_count >= self.args.reset_every_chunks
        ):
            self._reset_model_sequence(
                reason=(f"periodic chunk limit " f"{self.args.reset_every_chunks}"),
                clear_frame_buffers=False,
            )

        raw_obs = self._build_dreamzero_observation(obs)
        result = self.policy.act(raw_obs)
        actions = self._as_action_array(result.action)
        self.request_count += 1
        self.sequence_request_count += 1
        self.is_first_call = False

        scheduler_output = result.model_output.scheduler_output
        kv_lens = [
            None if cache is None else int(cache.shape[2])
            for cache in scheduler_output.cond_kv_cache
        ]
        non_empty = [length for length in kv_lens if length is not None]
        LOGGER.info(
            (
                "request=%d sequence_request=%d session=%s actions=%s "
                "range=[%.4f,%.4f] kv=(%s,%s) dit_steps=%d/%d"
            ),
            self.request_count,
            self.sequence_request_count,
            self.current_session_id,
            actions.shape,
            float(np.min(actions)),
            float(np.max(actions)),
            min(non_empty) if non_empty else 0,
            max(non_empty) if non_empty else 0,
            scheduler_output.dit_compute_steps,
            scheduler_output.scheduler_steps,
        )
        return {"actions": actions}

    def _append_frames(self, key: str, value: Any) -> None:
        arr = self._as_uint8_image_array(value)
        if arr.ndim == 3:
            self.frame_buffers[key].append(arr)
        elif arr.ndim == 4:
            for frame in arr:
                self.frame_buffers[key].append(frame)
        else:
            raise ValueError(f"{key} must be HWC or THWC, got shape {arr.shape}")

    def _build_dreamzero_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        num_frames = 1 if self.is_first_call else self.args.frames_per_chunk
        per_view = [
            self._latest_frames(self.frame_buffers[key], num_frames)
            for key in ROBOARENA_IMAGE_KEYS
        ]
        video = np.stack(per_view, axis=1)
        joint = self._as_vector(obs.get("observation/joint_position"), 7)
        gripper = self._as_vector(obs.get("observation/gripper_position"), 1)
        state = np.concatenate([joint, gripper], axis=-1).astype(np.float32)
        prompt = obs.get("prompt") or self.args.default_prompt
        return {
            "video": video[None, ...],
            "state": torch.from_numpy(state.reshape(1, 1, -1)),
            "task": [str(prompt)],
        }

    @staticmethod
    def _latest_frames(buffer: deque[np.ndarray], count: int) -> np.ndarray:
        if not buffer:
            raise ValueError(
                "cannot build DreamZero observation from empty frame buffer"
            )
        frames = list(buffer)[-count:]
        while len(frames) < count:
            frames.insert(0, frames[0])
        return np.stack(frames, axis=0)

    @staticmethod
    def _as_uint8_image_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value)
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        min_value = float(np.nanmin(arr))
        max_value = float(np.nanmax(arr))
        if min_value >= -1.1 and max_value <= 1.1:
            arr = (arr + 1.0) * 127.5 if min_value < -0.01 else arr * 255.0
        elif max_value > min_value:
            arr = (arr - min_value) / (max_value - min_value) * 255.0
        return arr.clip(0, 255).astype(np.uint8)

    @staticmethod
    def _as_vector(value: Any, dim: int) -> np.ndarray:
        if value is None:
            return np.zeros((dim,), dtype=np.float32)
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] < dim:
            arr = np.pad(arr, (0, dim - arr.shape[0]))
        return arr[:dim]

    @staticmethod
    def _as_action_array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().to(torch.float32).numpy()
        else:
            value = np.asarray(value, dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2 or value.shape[-1] != 8:
            raise ValueError(f"expected action shape [N,8], got {value.shape}")
        return value.astype(np.float32, copy=False)


class WebsocketPolicyServer:
    def __init__(
        self,
        *,
        policy: PhyAIDreamZeroDroidPolicy,
        server_config: PolicyServerConfig,
        host: str,
        port: int,
    ) -> None:
        self.policy = policy
        self.server_config = server_config
        self.host = host
        self.port = port
        self.codec = MsgpackCodec()

    def serve_forever(self) -> None:
        try:
            asyncio.run(self.run())
        finally:
            self.policy.close()

    async def run(self) -> None:
        try:
            from websockets.asyncio.server import serve
            import websockets.frames
        except ImportError:
            try:
                from websockets import serve
                import websockets.frames
            except ImportError as exc:
                raise RuntimeError(
                    "This server requires websockets. Install it with "
                    "`uv pip install websockets` in the active environment."
                ) from exc
        self._close_code = websockets.frames.CloseCode.INTERNAL_ERROR
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            ping_interval=None,
        ):
            LOGGER.info(
                "serving PhyAI DreamZero DROID policy on %s:%d", self.host, self.port
            )
            await asyncio.Future()

    async def _handler(self, websocket: Any, *_args: Any) -> None:
        remote = getattr(websocket, "remote_address", None)
        LOGGER.info("connection opened from %s", remote)
        await websocket.send(self.codec.pack(dataclasses.asdict(self.server_config)))
        while True:
            try:
                obs = self.codec.unpack(await websocket.recv())
                endpoint = obs.pop("endpoint", "infer")
                if endpoint == "reset":
                    self.policy.reset(obs)
                    await websocket.send("reset successful")
                    continue
                result = self.policy.infer(obs)
                await websocket.send(self.codec.pack(result))
            except Exception as exc:
                if "ConnectionClosed" in exc.__class__.__name__:
                    LOGGER.info("connection closed from %s", remote)
                    break
                error = traceback.format_exc()
                await websocket.send(error)
                await websocket.close(
                    code=self._close_code,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a PhyAI DreamZero-DROID policy over the RoboArena websocket protocol."
    )
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float32", "bfloat16", "float16"),
    )
    parser.add_argument("--cfg-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--dynamic-dit", action="store_true")
    parser.add_argument("--dynamic-dit-scheduler-steps", type=int, default=16)
    parser.add_argument("--attn-backend", default="flashinfer")
    parser.add_argument("--norm-backend", default="flashinfer")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--sequential-cpu-offload", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--frames-per-chunk", type=int, default=4)
    parser.add_argument(
        "--reset-every-chunks",
        type=int,
        default=0,
        help=(
            "Reset DreamZero sequence/KV after this many action chunks. "
            "Use 0 to disable periodic resets."
        ),
    )
    parser.add_argument("--default-prompt", default="pick up the object")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    if args.reset_every_chunks < 0:
        raise ValueError("--reset-every-chunks must be non-negative.")
    if args.dynamic_dit_scheduler_steps < 2:
        raise ValueError("--dynamic-dit-scheduler-steps must be at least 2.")
    if args.device == "cuda" and not args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    policy = PhyAIDreamZeroDroidPolicy(args)
    server = WebsocketPolicyServer(
        policy=policy,
        server_config=PolicyServerConfig(),
        host=args.host,
        port=args.port,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
