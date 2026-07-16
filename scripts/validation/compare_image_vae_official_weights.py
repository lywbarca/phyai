from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from safetensors.torch import load_file


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

from phyai.models.dreamzero.configuration_dreamzero import (  # noqa: E402
    DreamZeroImageEncoderConfig,
    DreamZeroVAEConfig,
)
from phyai.models.dreamzero.image_encoder_wan import (  # noqa: E402
    DreamZeroWanImageEncoder,
    dreamzero_image_encoder_weight_remap,
)
from phyai.models.dreamzero.vae_wan import (  # noqa: E402
    DreamZeroWanVAE,
    dreamzero_vae_weight_remap,
)


def load_remapped_state_dict(
    module: torch.nn.Module,
    ckpt_dir: str,
    prefix: str,
    remap,
    *,
    allowed_missing_prefixes: tuple[str, ...] = (),
) -> None:
    ckpt = Path(ckpt_dir)
    with (ckpt / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    files = sorted({filename for key, filename in index.items() if key.startswith(prefix)})
    state = {}
    for filename in files:
        shard = load_file(str(ckpt / filename), device="cpu")
        for key, value in shard.items():
            mapped = remap(key)
            if mapped is not None:
                state[mapped] = value
    missing, unexpected = module.load_state_dict(state, strict=False)
    real_missing = [
        key for key in missing if not key.startswith(allowed_missing_prefixes)
    ]
    if real_missing or unexpected:
        print(f"missing={real_missing[:30]}")
        print(f"unexpected={unexpected[:30]}")
        raise SystemExit(2)
    print(f"loaded {len(state)} tensors from {prefix}")


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


def print_diagnostics(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    ref = ref.detach().cpu().float()
    got = got.detach().cpu().float()
    diff = (got - ref).abs()
    flat = diff.flatten()
    topk = min(10, flat.numel())
    vals, idx = flat.topk(topk)
    print(f"{name}: top_abs={vals.tolist()} flat_idx={idx.tolist()}")
    print(
        f"{name}: quantiles="
        f"{torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99, 0.999])).tolist()}"
    )
    if diff.dim() == 3:
        per_token = diff.amax(dim=-1).flatten()
        token_vals, token_idx = per_token.topk(min(10, per_token.numel()))
        print(
            f"{name}: per_token_top_abs={token_vals.tolist()} "
            f"token_idx={token_idx.tolist()}"
        )
        print(
            f"{name}: cls_max={diff[:, 0].max().item():.8g} "
            f"patch_max={diff[:, 1:].max().item():.8g} "
            f"cls_mean={diff[:, 0].mean().item():.8g} "
            f"patch_mean={diff[:, 1:].mean().item():.8g}"
        )


def print_weight_checks(module: torch.nn.Module, ckpt_dir: str, prefix: str, remap) -> None:
    ckpt = Path(ckpt_dir)
    with (ckpt / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    state = module.state_dict()
    checked = 0
    max_abs = 0.0
    max_key = ""
    for key, filename in sorted(index.items()):
        if not key.startswith(prefix):
            continue
        mapped = remap(key)
        if mapped is None or mapped not in state:
            continue
        ref = load_file(str(ckpt / filename), device="cpu")[key].float()
        got = state[mapped].detach().cpu().float()
        diff = (got - ref).abs().max().item()
        checked += 1
        if diff > max_abs:
            max_abs = diff
            max_key = key
    print(f"{prefix} weight_check: checked={checked} max_abs={max_abs:.8g} key={max_key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="/data/share/DreamZero-DROID")
    parser.add_argument("--dump", default="/tmp/dreamzero_image_vae_official.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    image_encoder = DreamZeroWanImageEncoder(
        DreamZeroImageEncoderConfig(), params_dtype=torch.float32
    ).to(device).eval()
    vae = DreamZeroWanVAE(DreamZeroVAEConfig()).to(device).eval()

    load_remapped_state_dict(
        image_encoder,
        args.ckpt_dir,
        "action_head.image_encoder.",
        dreamzero_image_encoder_weight_remap,
    )
    load_remapped_state_dict(
        vae,
        args.ckpt_dir,
        "action_head.vae.",
        dreamzero_vae_weight_remap,
    )

    image = payload["inputs"]["image"].to(device)
    vae_video = payload["inputs"]["vae_video"].to(device)
    vae_latent = payload["inputs"]["vae_latent"].to(device)
    with torch.no_grad():
        image_features = image_encoder.encode_image([image])
        vae_encoded = vae.encode(vae_video, tiled=False)
        vae_decoded = vae.decode(vae_latent, tiled=False)

    print(f"meta={payload['meta']}")
    print_metrics("image_features", payload["outputs"]["image_features"], image_features)
    print_metrics("vae_encoded", payload["outputs"]["vae_encoded"], vae_encoded)
    print_metrics("vae_decoded", payload["outputs"]["vae_decoded"], vae_decoded)
    if args.diagnostics:
        print_diagnostics(
            "image_features", payload["outputs"]["image_features"], image_features
        )
        print_diagnostics("vae_encoded", payload["outputs"]["vae_encoded"], vae_encoded)
        print_diagnostics("vae_decoded", payload["outputs"]["vae_decoded"], vae_decoded)
        print_weight_checks(
            image_encoder,
            args.ckpt_dir,
            "action_head.image_encoder.",
            dreamzero_image_encoder_weight_remap,
        )
        print_weight_checks(
            vae,
            args.ckpt_dir,
            "action_head.vae.",
            dreamzero_vae_weight_remap,
        )


if __name__ == "__main__":
    main()
