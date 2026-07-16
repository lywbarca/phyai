from __future__ import annotations

import os
import socket
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import phyai.parallel as P
from phyai.models.dreamzero.modeling_dreamzero import DreamZeroTensorParallelRMSNorm


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _worker(rank: int, world_size: int, port: int, err_queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        P.init(
            layout=(world_size,), mesh_dim_names=("tp",), device="cpu", backend="gloo"
        )

        torch.manual_seed(7)
        hidden_size = 16
        local_size = hidden_size // world_size
        full_x = torch.randn(2, 3, hidden_size, dtype=torch.float32)
        full_weight = torch.randn(hidden_size, dtype=torch.float32)
        local_x = full_x.narrow(-1, rank * local_size, local_size).contiguous()
        local_weight = full_weight.narrow(0, rank * local_size, local_size).contiguous()

        norm = DreamZeroTensorParallelRMSNorm(
            hidden_size,
            local_size,
            eps=1e-6,
            params_dtype=torch.float32,
            device="cpu",
        )
        norm.weight.data.copy_(local_weight)
        out = norm(local_x)

        expected = (
            full_x
            * torch.rsqrt(full_x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            * full_weight
        ).narrow(-1, rank * local_size, local_size)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    except BaseException as exc:
        err_queue.put((rank, repr(exc), traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_dreamzero_tp_rmsnorm_matches_full_dim_rmsnorm() -> None:
    world_size = 4
    ctx = mp.get_context("fork")
    err_queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, port, err_queue))
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise TimeoutError("DreamZero TP RMSNorm worker hung.")

    errors = []
    while not err_queue.empty():
        errors.append(err_queue.get_nowait())
    if errors:
        rank, error, tb = errors[0]
        raise AssertionError(f"worker rank={rank} failed: {error}\n{tb}")
    for rank, proc in enumerate(procs):
        if proc.exitcode != 0:
            raise AssertionError(f"worker rank={rank} exited with {proc.exitcode}.")
