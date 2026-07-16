from __future__ import annotations

import argparse
import sys
import types

import torch


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

from phyai.models.dreamzero.configuration_dreamzero import DreamZeroVAEConfig
from phyai.models.dreamzero.vae_wan import DreamZeroWanVAE


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", default="/tmp/dreamzero_vae_synthetic.pt")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    model = DreamZeroWanVAE(DreamZeroVAEConfig()).to(device).eval()
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        print(f"missing={missing}")
        print(f"unexpected={unexpected}")
        raise SystemExit(2)

    video = payload["inputs"]["video"].to(device)
    latent = payload["inputs"]["latent"].to(device)
    with torch.no_grad():
        encoded = model.encode(video, tiled=False)
        decoded = model.decode(latent, tiled=False)

    print(f"meta={payload['meta']}")
    print_metrics("encoded", payload["outputs"]["encoded"], encoded)
    print_metrics("decoded", payload["outputs"]["decoded"], decoded)


if __name__ == "__main__":
    main()
