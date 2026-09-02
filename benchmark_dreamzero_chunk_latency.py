#!/usr/bin/env python3
"""Benchmark end-to-end DreamZero policy latency with recorded observations."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


class WebsocketClientPolicy:
    """Minimal client compatible with the DreamZero policy servers."""

    def __init__(self, host: str, port: int) -> None:
        self._socket = websockets.sync.client.connect(
            f"ws://{host}:{port}",
            compression=None,
            max_size=None,
            ping_interval=60,
            ping_timeout=600,
        )
        self.metadata = self._unpack(self._socket.recv())

    @staticmethod
    def _encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                b"__ndarray__": True,
                b"dtype": value.dtype.str,
                b"shape": value.shape,
                b"data": value.tobytes(),
            }
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"cannot msgpack encode {type(value)!r}")

    @staticmethod
    def _decode(value: dict[Any, Any]) -> Any:
        marker = value.get(b"__ndarray__", value.get("__ndarray__"))
        if marker is not True:
            return value
        dtype = np.dtype(value.get(b"dtype", value.get("dtype")))
        shape = tuple(value.get(b"shape", value.get("shape")))
        data = value.get(b"data", value.get("data"))
        return np.ndarray(buffer=data, dtype=dtype, shape=shape)

    @classmethod
    def _pack(cls, value: Any) -> bytes:
        return msgpack.packb(value, default=cls._encode, use_bin_type=True)

    @classmethod
    def _unpack(cls, value: bytes | str) -> Any:
        if isinstance(value, str):
            raise RuntimeError(value)
        return msgpack.unpackb(value, raw=False, object_hook=cls._decode)

    def infer(self, observation: dict[str, Any]) -> Any:
        observation["endpoint"] = "infer"
        self._socket.send(self._pack(observation))
        return self._unpack(self._socket.recv())

    def reset(self, reset_info: dict[str, Any]) -> None:
        reset_info["endpoint"] = "reset"
        self._socket.send(self._pack(reset_info))
        self._socket.recv()

    def close(self) -> None:
        self._socket.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--inter-query-pause-seconds", type=float, default=6.0)
    args = parser.parse_args()

    with np.load(args.input, allow_pickle=False) as source:
        trace = {key: source[key] for key in source.files}

    session_id = str(uuid.uuid4())
    prompt = str(trace["prompt"].item())
    request_count = len(trace["request_environment_step"])
    rows: list[dict[str, object]] = []
    client = WebsocketClientPolicy(args.host, args.port)
    try:
        client.reset({"session_id": session_id})
        for index in range(request_count):
            request = {
                "observation/exterior_image_0_left": trace["request_right_image"][
                    index
                ],
                "observation/exterior_image_1_left": trace["request_left_image"][index],
                "observation/wrist_image_left": trace["request_wrist_image"][index],
                "observation/joint_position": trace["request_joint_position"][index],
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),
                "observation/gripper_position": trace["request_gripper_position"][
                    index
                ],
                "prompt": prompt,
                "session_id": session_id,
            }
            started = time.perf_counter_ns()
            result = client.infer(request)
            latency_seconds = (time.perf_counter_ns() - started) / 1e9
            actions = result["actions"] if isinstance(result, dict) else result
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 8:
                raise ValueError(f"expected action chunk [N,8], got {actions.shape}")
            row = {
                "request_index": index + 1,
                "environment_step": int(trace["request_environment_step"][index]),
                "input_frames": 1 if index == 0 else 4,
                "latency_seconds": latency_seconds,
                "action_shape": list(actions.shape),
                "action_min": float(actions.min()),
                "action_max": float(actions.max()),
            }
            rows.append(row)
            print(
                f"request={index + 1}/{request_count} "
                f"step={row['environment_step']} frames={row['input_frames']} "
                f"latency={latency_seconds:.6f}s",
                flush=True,
            )
            if index + 1 < request_count and args.inter_query_pause_seconds > 0:
                time.sleep(args.inter_query_pause_seconds)
    finally:
        client.close()

    all_latencies = [float(row["latency_seconds"]) for row in rows]
    stable_latencies = all_latencies[1:]
    summary = {
        "input": str(args.input),
        "host": args.host,
        "port": args.port,
        "session_id": session_id,
        "requests": request_count,
        "timing_scope": "websocket infer request to complete action response",
        "inter_query_pause_seconds_excluded": args.inter_query_pause_seconds,
        "initial_request_seconds": all_latencies[0],
        "stable_chunks": len(stable_latencies),
        "stable_mean_seconds": statistics.fmean(stable_latencies),
        "stable_median_seconds": statistics.median(stable_latencies),
        "stable_min_seconds": min(stable_latencies),
        "stable_max_seconds": max(stable_latencies),
        "stable_stdev_seconds": statistics.stdev(stable_latencies),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
