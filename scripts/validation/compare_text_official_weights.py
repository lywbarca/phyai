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

from phyai.models.dreamzero.configuration_dreamzero import DreamZeroTextEncoderConfig  # noqa: E402
from phyai.models.dreamzero.text_encoder_wan import (  # noqa: E402
    DreamZeroWanTextEncoder,
    T5Attention,
    dreamzero_text_encoder_weight_remap,
)
from phyai_utils_tools.models.dreamzero.processor_dreamzero import DreamZeroProcessor  # noqa: E402
from phyai_utils_tools.processing.transition import PROMPT, Transition  # noqa: E402


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_remapped_state_dict(
    module: torch.nn.Module,
    ckpt_dir: str,
    prefix: str,
    remap,
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
    if missing or unexpected:
        print(f"missing={missing[:30]}")
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


def print_masked_metrics(
    name: str,
    ref: torch.Tensor,
    got: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    ref = ref.detach().cpu().float()
    got = got.detach().cpu().float()
    mask = attention_mask.detach().cpu().to(torch.bool)
    while mask.dim() < ref.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(ref)
    if not mask.any():
        print(f"{name}_active: no active elements")
        return
    diff = (got - ref)[mask]
    denom = ref[mask].abs().clamp_min(1e-3)
    print(
        f"{name}_active: elements={diff.numel()} "
        f"max_abs={diff.abs().max().item():.8g} "
        f"mean_abs={diff.abs().mean().item():.8g} "
        f"rms_abs={diff.pow(2).mean().sqrt().item():.8g} "
        f"max_rel={(diff.abs() / denom).max().item():.8g}"
    )


def patch_t5_attention_to_official_einsum() -> None:
    def forward(self, x, context=None, mask=None, pos_bias=None):
        context = x if context is None else context
        batch_size = x.size(0)
        q = self.q(x).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k(context).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v(context).view(batch_size, -1, self.num_heads, self.head_dim)

        attn_bias = x.new_zeros(batch_size, self.num_heads, q.size(1), k.size(1))
        if pos_bias is not None:
            attn_bias = attn_bias + pos_bias.to(dtype=x.dtype, device=x.device)
        if mask is not None:
            if mask.ndim not in {2, 3}:
                raise ValueError(
                    f"mask must be 2-D or 3-D; got shape {tuple(mask.shape)}."
                )
            mask_view = (
                mask.view(batch_size, 1, 1, -1)
                if mask.ndim == 2
                else mask.unsqueeze(1)
            )
            attn_bias = attn_bias.masked_fill(
                mask_view.to(device=x.device) == 0,
                torch.finfo(x.dtype).min,
            )

        attn = torch.einsum("binc,bjnc->bnij", q, k) + attn_bias
        attn = torch.softmax(attn.float(), dim=-1).type_as(attn)
        out = torch.einsum("bnij,bjnc->binc", attn, v)
        out = out.reshape(batch_size, -1, self.dim_attn)
        return self.dropout(self.o(out))

    T5Attention.forward = forward


def print_diagnostics(name: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    ref = ref.detach().cpu().float()
    got = got.detach().cpu().float()
    diff = (got - ref).abs()
    flat = diff.flatten()
    vals, idx = flat.topk(min(10, flat.numel()))
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


def print_weight_check(module: torch.nn.Module, ckpt_dir: str, prefix: str, remap) -> None:
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
    parser.add_argument("--tokenizer", default="/data/share/google-umt5-xxl")
    parser.add_argument("--dump", default="/tmp/dreamzero_text_official.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--official-attention", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    payload = torch.load(args.dump, map_location="cpu", weights_only=False)
    if args.official_attention:
        patch_t5_attention_to_official_einsum()
        print("patched phyai T5Attention to official einsum attention")

    processor = DreamZeroProcessor(tokenizer_name=args.tokenizer, max_length=512)
    proc_out = processor.preprocessor(Transition({PROMPT: payload["inputs"]["prompts"]}))
    print_metrics("processor_input_ids", payload["inputs"]["input_ids"], proc_out.input_ids)
    print_metrics(
        "processor_attention_mask",
        payload["inputs"]["attention_mask"],
        proc_out.attention_mask,
    )
    print_metrics(
        "processor_negative_input_ids",
        payload["inputs"]["negative_input_ids"],
        proc_out.negative_input_ids,
    )
    print_metrics(
        "processor_negative_attention_mask",
        payload["inputs"]["negative_attention_mask"],
        proc_out.negative_attention_mask,
    )

    model = DreamZeroWanTextEncoder(
        DreamZeroTextEncoderConfig(dropout=0.0),
        params_dtype=dtype,
        device=device,
    ).eval()
    load_remapped_state_dict(
        model,
        args.ckpt_dir,
        "action_head.text_encoder.",
        dreamzero_text_encoder_weight_remap,
    )

    input_ids = payload["inputs"]["input_ids"].to(device=device, dtype=torch.long)
    attention_mask = payload["inputs"]["attention_mask"].to(device=device)
    negative_input_ids = payload["inputs"]["negative_input_ids"].to(
        device=device, dtype=torch.long
    )
    negative_attention_mask = payload["inputs"]["negative_attention_mask"].to(device=device)
    with torch.no_grad():
        text_features = model(input_ids, attention_mask)
        text_features = text_features.masked_fill(
            attention_mask.to(torch.bool).unsqueeze(-1) == 0,
            0,
        )
        negative_text_features = model(negative_input_ids, negative_attention_mask)
        negative_text_features = negative_text_features.masked_fill(
            negative_attention_mask.to(torch.bool).unsqueeze(-1) == 0,
            0,
        )

    print(f"meta={payload['meta']}")
    print_metrics("text_features", payload["outputs"]["text_features"], text_features)
    print_masked_metrics(
        "text_features",
        payload["outputs"]["text_features"],
        text_features,
        payload["inputs"]["attention_mask"],
    )
    print_metrics(
        "negative_text_features",
        payload["outputs"]["negative_text_features"],
        negative_text_features,
    )
    print_masked_metrics(
        "negative_text_features",
        payload["outputs"]["negative_text_features"],
        negative_text_features,
        payload["inputs"]["negative_attention_mask"],
    )
    if args.diagnostics:
        print_diagnostics("text_features", payload["outputs"]["text_features"], text_features)
        print_diagnostics(
            "negative_text_features",
            payload["outputs"]["negative_text_features"],
            negative_text_features,
        )
        print_weight_check(
            model,
            args.ckpt_dir,
            "action_head.text_encoder.",
            dreamzero_text_encoder_weight_remap,
        )


if __name__ == "__main__":
    main()
