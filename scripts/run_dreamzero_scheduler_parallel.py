from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import phyai.layers.layer_norm as layer_norm
import phyai.layers.linear as linear_layers
import phyai.parallel as P
from phyai.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroDiT,
    DreamZeroDiTConfig,
    DreamZeroRequest,
    DreamZeroWS1Scheduler,
)


def enable_torch_norm_backend() -> None:
    layer_norm._VALID_BACKENDS = ("flashinfer", "phyai-kernel", "torch")

    def resolve_backend(name: str) -> str:
        canonical = name.replace("_", "-").lower()
        if canonical not in layer_norm._VALID_BACKENDS:
            raise ValueError(f"Unknown norm backend {name!r}.")
        return canonical

    def rms_kernel(x, weight, eps):
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
        return (y * weight.float()).to(dtype=x.dtype)

    def fused_add_rms_kernel(x, residual, weight, eps):
        added = x + residual
        return rms_kernel(added, weight, eps), added

    def layer_kernel(x, weight, bias, eps):
        return F.layer_norm(x, (x.shape[-1],), weight, bias, eps)

    layer_norm._resolve_backend = resolve_backend
    layer_norm.RMSNorm._load_kernels = staticmethod(
        lambda backend: (rms_kernel, fused_add_rms_kernel)
    )
    layer_norm.RMSNorm._load_kernel = staticmethod(lambda backend: rms_kernel)
    layer_norm.LayerNorm._load_kernel = staticmethod(lambda backend: layer_kernel)


def init_distributed(*, cfg_size: int, tp_size: int, device: torch.device) -> None:
    world_size = cfg_size * tp_size
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
            dist.init_process_group(backend=backend, device_id=device)
        else:
            dist.init_process_group(backend=backend)
    if cfg_size > 1:
        layout = (cfg_size, tp_size)
        names = ("cfg", "tp")
    else:
        layout = (tp_size,)
        names = ("tp",)
    P.init(
        layout=layout,
        mesh_dim_names=names,
        device=device.type,
        backend="nccl" if device.type == "cuda" else "gloo",
        enable_pynccl=False,
    )


def tiny_config() -> DreamZeroConfig:
    return DreamZeroConfig(
        action_dim=4,
        action_horizon=2,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        input_embedding_dim=1536,
        num_inference_timesteps=1,
        num_frame_per_block=1,
        cfg_scale=1.5,
        sigma_shift=1.0,
        dit=DreamZeroDiTConfig(
            dim=32,
            ffn_dim=64,
            frame_seqlen=4,
            freq_dim=8,
            in_dim=36,
            max_chunk_size=-1,
            model_type="i2v",
            num_action_per_block=2,
            num_frame_per_block=1,
            num_heads=4,
            num_layers=1,
            num_state_per_block=1,
            out_dim=2,
        ),
    )


def replace_norms_with_identity(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (layer_norm.LayerNorm, layer_norm.RMSNorm)):
            module._modules[name] = nn.Identity()
        else:
            replace_norms_with_identity(child)


def init_tiny_model(model: nn.Module) -> None:
    torch.manual_seed(11)
    replace_norms_with_identity(model)
    for param in model.parameters():
        if param.dim() == 1:
            param.data.zero_()
        else:
            param.data.normal_(mean=0.0, std=0.02)


def make_request(device: torch.device, *, do_cfg: bool) -> DreamZeroRequest:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(23)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device)

    return DreamZeroRequest(
        video=randn(1, 2, 1, 4, 4),
        action=randn(1, 2, 4),
        state=randn(1, 1, 5),
        context=randn(1, 3, 4096),
        uncond_context=randn(1, 3, 4096) if do_cfg else None,
        clip_feature=randn(1, 2, 1280),
        uncond_clip_feature=randn(1, 2, 1280) if do_cfg else None,
        y=randn(1, 34, 1, 4, 4),
        clean_video=randn(1, 2, 1, 4, 4),
        concat_first_frame_latent=True,
        image_context_tokens=2,
        num_inference_steps=1,
        guidance_scale=1.5 if do_cfg else 1.0,
    )


def assert_replicated_across_cfg(tensor: torch.Tensor, *, name: str) -> None:
    mesh = P.default_mesh()
    names = mesh.torch_mesh.mesh_dim_names or ()
    if "cfg" not in names or mesh.axis_size("cfg") == 1:
        return
    gathered = P.all_gather(tensor.unsqueeze(0), axis="cfg", dim=0)
    torch.testing.assert_close(gathered[0], gathered[1], rtol=1e-5, atol=1e-5)
    if mesh.axis_local_rank("tp") == 0 and mesh.axis_local_rank("cfg") == 0:
        print(f"{name}: cfg replicas match")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if args.cfg_size not in (1, 2):
        raise ValueError("--cfg-size must be 1 or 2.")
    if args.tp_size <= 0:
        raise ValueError("--tp-size must be positive.")

    device = torch.device(args.device)
    if device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
    linear_layers.init(register_flashinfer=False, validate=False)
    enable_torch_norm_backend()
    init_distributed(cfg_size=args.cfg_size, tp_size=args.tp_size, device=device)

    mesh = P.default_mesh()
    tp_rank = mesh.axis_local_rank("tp")
    cfg_rank = mesh.axis_local_rank("cfg") if args.cfg_size > 1 else 0

    model = DreamZeroDiT(
        tiny_config(),
        params_dtype=torch.float32,
        device=device,
        attn_backend="eager",
        norm_backend="torch",
    ).eval()
    init_tiny_model(model)
    scheduler = DreamZeroWS1Scheduler(model, device=device, use_cfg_runner=True)
    scheduler.setup()
    out = scheduler.step(make_request(device, do_cfg=args.cfg_size > 1))

    assert out.video.shape == (1, 2, 1, 4, 4)
    assert out.action.shape == (1, 2, 4)
    assert out.last_video_pred is not None
    assert out.last_action_pred is not None
    assert out.last_video_pred.shape == out.video.shape
    assert out.last_action_pred.shape == out.action.shape
    assert out.cond_kv_cache or (args.cfg_size == 2 and cfg_rank == 1)
    if args.cfg_size == 2:
        assert out.uncond_kv_cache or cfg_rank == 0
        assert_replicated_across_cfg(out.video, name="video")
        assert_replicated_across_cfg(out.action, name="action")

    if tp_rank == 0 and cfg_rank == 0:
        print(
            "DreamZero scheduler parallel validation passed: "
            f"cfg={args.cfg_size}, tp={args.tp_size}, device={device.type}"
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
