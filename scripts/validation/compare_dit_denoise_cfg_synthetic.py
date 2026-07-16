from __future__ import annotations

import argparse
import os
import sys
import types

import torch
import torch.distributed as dist
import torch.nn.functional as F


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

import phyai.layers.linear as linear_layers
import phyai.layers.layer_norm as layer_norm
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


def init_parallel(tp_size: int, cfg_size: int, device: torch.device) -> None:
    world_size = tp_size * cfg_size
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if cfg_size > 1:
        layout = (cfg_size, tp_size)
        mesh_dim_names = ("cfg", "tp")
    else:
        layout = (tp_size,)
        mesh_dim_names = ("tp",)
    P.init(
        layout=layout,
        mesh_dim_names=mesh_dim_names,
        device=device.type,
        backend="nccl" if device.type == "cuda" else "gloo",
        enable_pynccl=False,
    )


def build_model(device: torch.device) -> DreamZeroDiT:
    cfg = DreamZeroConfig(
        action_dim=4,
        action_horizon=2,
        max_action_dim=4,
        max_state_dim=5,
        hidden_size=6,
        input_embedding_dim=1536,
        num_inference_timesteps=3,
        num_frame_per_block=1,
        cfg_scale=1.5,
        sigma_shift=1.0,
        decouple_inference_noise=False,
        dit=DreamZeroDiTConfig(
            dim=48,
            ffn_dim=96,
            frame_seqlen=4,
            freq_dim=8,
            in_dim=2,
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
    return DreamZeroDiT(
        cfg,
        params_dtype=torch.float32,
        device=device,
        attn_backend="eager",
        norm_backend="torch",
    )


def metrics(ref: torch.Tensor, got: torch.Tensor) -> dict[str, float | bool | tuple[int, ...]]:
    ref = ref.detach().cpu().float()
    got = got.detach().cpu().float()
    diff = got - ref
    return {
        "same_shape": tuple(ref.shape) == tuple(got.shape),
        "shape": tuple(got.shape),
        "max_abs": diff.abs().max().item(),
        "mean_abs": diff.abs().mean().item(),
        "rms_abs": diff.pow(2).mean().sqrt().item(),
        "max_rel": (diff.abs() / ref.abs().clamp_min(1e-3)).max().item(),
    }


def print_metrics(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    m = metrics(ref, got)
    print(
        f"{name}: shape={m['shape']} same_shape={m['same_shape']} "
        f"max_abs={m['max_abs']:.8g} mean_abs={m['mean_abs']:.8g} "
        f"rms_abs={m['rms_abs']:.8g} max_rel={m['max_rel']:.8g}"
    )


def remap_official_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    replacements = (
        ("text_embedding.2.", "text_embedding.1."),
        ("time_embedding.2.", "time_embedding.1."),
        ("time_projection.1.", "time_projection.0."),
        ("img_emb.proj.0.", "img_emb.proj_0_norm."),
        ("img_emb.proj.1.", "img_emb.proj_1."),
        ("img_emb.proj.3.", "img_emb.proj_3."),
        ("img_emb.proj.4.", "img_emb.proj_4_norm."),
        (".ffn.0.", ".ffn.fc1."),
        (".ffn.2.", ".ffn.fc2."),
    )
    for key, value in state_dict.items():
        new_key = key
        for src, dst in replacements:
            new_key = new_key.replace(src, dst)
        remapped[new_key] = value
    return remapped


def tp_slice_state_dict(
    state_dict: dict[str, torch.Tensor],
    target_state: dict[str, torch.Tensor],
    tp_size: int,
    tp_rank: int,
    num_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    if tp_size == 1:
        return dict(state_dict)
    out_sharded_tokens = (
        ".q.weight", ".q.bias", ".k.weight", ".k.bias", ".v.weight", ".v.bias",
        ".k_img.weight", ".k_img.bias", ".v_img.weight", ".v_img.bias",
        ".fc1.weight", ".fc1.bias",
    )
    in_sharded_tokens = (".o.weight", ".fc2.weight")
    local_heads = num_heads // tp_size
    start_dim = tp_rank * local_heads * head_dim
    end_dim = start_dim + local_heads * head_dim
    sliced = {}
    for key, value in state_dict.items():
        target = target_state.get(key)
        if target is None:
            sliced[key] = value
        elif key.endswith((".norm_q.weight", ".norm_k.weight", ".norm_k_img.weight")):
            sliced[key] = value[start_dim:end_dim].clone()
        elif key.endswith(out_sharded_tokens):
            rows_per_rank = value.shape[0] // tp_size
            sliced[key] = value[tp_rank * rows_per_rank : (tp_rank + 1) * rows_per_rank].clone()
        elif key.endswith(in_sharded_tokens):
            cols_per_rank = value.shape[1] // tp_size
            sliced[key] = value[:, tp_rank * cols_per_rank : (tp_rank + 1) * cols_per_rank].clone()
        else:
            sliced[key] = value
    return sliced


def load_official_weights(model: DreamZeroDiT, payload, tp_size: int, tp_rank: int) -> None:
    state_dict = remap_official_state_dict(payload["state_dict"])
    target_state = model.state_dict()
    for key in target_state:
        if key.endswith(".norm1.weight") or key.endswith(".norm2.weight"):
            state_dict.setdefault(key, torch.ones_like(target_state[key]))
        elif key.endswith(".norm3.weight") or key == "head.norm.weight":
            state_dict.setdefault(key, torch.ones_like(target_state[key]))
        elif key.endswith(".norm3.bias"):
            state_dict.setdefault(key, torch.zeros_like(target_state[key]))
    state_dict = tp_slice_state_dict(
        state_dict,
        target_state,
        tp_size,
        tp_rank,
        num_heads=4,
        head_dim=12,
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"missing={missing[:20]}")
        print(f"unexpected={unexpected[:20]}")
        raise SystemExit(2)


def to_device_inputs(payload, device: torch.device) -> dict:
    inputs = {}
    for key, value in payload["inputs"].items():
        inputs[key] = value.to(device) if torch.is_tensor(value) else value
    return inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="/tmp/dreamzero_denoise_cfg_synthetic.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--cfg-size", type=int, default=1)
    args = parser.parse_args()

    linear_layers.init(register_flashinfer=False, validate=False)
    enable_torch_norm_backend()
    device = torch.device(args.device)
    init_parallel(args.tp_size, args.cfg_size, device)
    mesh = P.default_mesh()
    tp_rank = mesh.axis_local_rank("tp")
    cfg_rank = mesh.axis_local_rank("cfg") if args.cfg_size > 1 else 0

    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    model = build_model(device).eval()
    load_official_weights(model, payload, args.tp_size, tp_rank)
    scheduler = DreamZeroWS1Scheduler(model, device=device, use_cfg_runner=True)
    scheduler.setup()

    inputs = to_device_inputs(payload, device)
    request = DreamZeroRequest(
        video=inputs["video"],
        action=inputs["action"],
        state=inputs["state"],
        context=inputs["context"],
        clip_feature=inputs["clip_feature"],
        y=inputs["y"],
        uncond_context=inputs["uncond_context"],
        uncond_clip_feature=inputs["clip_feature"],
        seq_len=int(inputs["seq_len"]),
        current_start_frame=int(inputs["current_start_frame"]),
        concat_first_frame_latent=False,
        image_context_tokens=int(inputs["image_context_tokens"]),
        num_inference_steps=int(inputs["num_inference_steps"]),
        guidance_scale=float(inputs["guidance_scale"]),
        sigma_shift=float(inputs["sigma_shift"]),
        update_kv_cache=False,
        prefill_clean_cache=False,
    )
    with torch.no_grad():
        output = scheduler.step(request)

    if tp_rank == 0 and cfg_rank == 0:
        print(f"mesh: cfg_size={args.cfg_size} tp_size={args.tp_size}")
        print_metrics("final_video", payload["outputs"]["video"], output.video)
        print_metrics("final_action", payload["outputs"]["action"], output.action)
        if output.last_video_pred is not None:
            print_metrics(
                "last_cfg_video_pred",
                payload["trajectory"][-1]["cfg_video_pred"],
                output.last_video_pred,
            )
        if output.last_action_pred is not None:
            print_metrics(
                "last_action_pred",
                payload["trajectory"][-1]["action_pred"],
                output.last_action_pred,
            )


if __name__ == "__main__":
    main()
