from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

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

from phyai_utils_tools.models.dreamzero.processor_dreamzero import DreamZeroProcessor  # noqa: E402
from phyai_utils_tools.models.dreamzero.steps_dreamzero import OBS  # noqa: E402
from phyai_utils_tools.processing.transition import ACTION  # noqa: E402


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


def maybe_pad_action(action: torch.Tensor, max_action_dim: int) -> torch.Tensor:
    if action.shape[-1] >= max_action_dim:
        return action
    pad = torch.zeros(*action.shape[:-1], max_action_dim - action.shape[-1], dtype=action.dtype)
    return torch.cat([action, pad], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dump", default="/tmp/dreamzero_action_postprocess_official.pt")
    parser.add_argument("--ckpt-dir", default="/data/share/DreamZero-DROID")
    parser.add_argument("--tokenizer", default="/data/share/google-umt5-xxl")
    parser.add_argument("--embodiment-tag", default="oxe_droid")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    payload = torch.load(args.official_dump, map_location="cpu")
    normalized_action = payload["inputs"]["normalized_action"].float()
    obs = payload["inputs"]["obs"]
    official_final = payload["outputs"]["action"].float()
    official_before_relative = payload["outputs"]["before_relative_action"].float()

    processor = DreamZeroProcessor.from_pretrained(
        args.ckpt_dir,
        tokenizer_name=args.tokenizer,
        embodiment_tag=args.embodiment_tag,
        relative_action=None,
        device="cpu",
    )
    print(f"processor action_keys={processor.action_keys}")
    print(f"processor state_keys={processor.state_keys}")
    print(f"processor relative_action_keys={processor.relative_action_keys}")
    print(f"processor raw_action_dim={processor._raw_action_dim()}")

    padded_action = maybe_pad_action(normalized_action, processor.max_action_dim)
    final = processor.postprocess(padded_action, obs=obs).action.float()
    before_relative = processor.postprocess(padded_action, obs=None).action.float()

    print_metrics("final_action", official_final, final)
    print_metrics("before_relative_action", official_before_relative, before_relative)
    print_metrics("joint.final", official_final[..., :7], final[..., :7])
    print_metrics("gripper.final", official_final[..., 7:8], final[..., 7:8])
    print_metrics("joint.before_relative", official_before_relative[..., :7], before_relative[..., :7])
    print_metrics("gripper.before_relative", official_before_relative[..., 7:8], before_relative[..., 7:8])

    rel_delta_ref = official_final - official_before_relative
    rel_delta_got = final - before_relative
    print_metrics("relative_delta", rel_delta_ref, rel_delta_got)

    transition_final = processor.postprocessor({ACTION: padded_action, OBS: obs}).action.float()
    print_metrics("transition_call_final", official_final, transition_final)

    out_path = Path(args.official_dump).with_name("dreamzero_action_postprocess_phyai_compare.pt")
    torch.save(
        {
            "official": payload,
            "phyai": {
                "final_action": final,
                "before_relative_action": before_relative,
                "normalized_action_padded": padded_action,
            },
        },
        out_path,
    )
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
