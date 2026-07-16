from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from groot.vla.model.dreamzero.modules.wan_video_text_encoder import WanTextEncoder
from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer


PROMPTS = [
    "Pick up the blue cube and place it into the bowl.",
    "Open the drawer and then close it.",
]
NEGATIVE_PROMPT = ""


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def load_prefixed_state_dict(module: torch.nn.Module, ckpt_dir: str, prefix: str) -> None:
    ckpt = Path(ckpt_dir)
    with (ckpt / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    files = sorted({filename for key, filename in index.items() if key.startswith(prefix)})
    state = {}
    for filename in files:
        shard = load_file(str(ckpt / filename), device="cpu")
        for key, value in shard.items():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = value
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"missing={missing[:30]}")
        print(f"unexpected={unexpected[:30]}")
        raise SystemExit(2)
    print(f"loaded {len(state)} tensors from {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="/data/share/DreamZero-DROID")
    parser.add_argument("--tokenizer", default="/data/share/google-umt5-xxl")
    parser.add_argument("--output", default="/tmp/dreamzero_text_official.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    if os.environ.get("DREAMZERO_DROP_CACHES_FIRST") == "1":
        os.system("sudo -n /usr/local/sbin/dreamzero_drop_caches || true")

    set_seed(args.seed)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    tokenizer = HuggingfaceTokenizer(
        name=args.tokenizer, seq_len=512, clean="whitespace"
    )
    input_ids, attention_mask = tokenizer(
        PROMPTS, return_mask=True, add_special_tokens=True
    )
    negative_input_ids, negative_attention_mask = tokenizer(
        [NEGATIVE_PROMPT] * len(PROMPTS), return_mask=True, add_special_tokens=True
    )

    model = WanTextEncoder(dropout=0.0).to(device=device, dtype=dtype).eval()
    load_prefixed_state_dict(model, args.ckpt_dir, "action_head.text_encoder.")

    with torch.no_grad():
        text_features = model(
            input_ids.to(device=device, dtype=torch.long),
            attention_mask.to(device=device),
        )
        text_features = text_features.masked_fill(
            attention_mask.to(device=device, dtype=torch.bool).unsqueeze(-1) == 0,
            0,
        )
        negative_text_features = model(
            negative_input_ids.to(device=device, dtype=torch.long),
            negative_attention_mask.to(device=device),
        )
        negative_text_features = negative_text_features.masked_fill(
            negative_attention_mask.to(device=device, dtype=torch.bool).unsqueeze(-1) == 0,
            0,
        )

    payload = {
        "inputs": {
            "prompts": PROMPTS,
            "negative_prompt": NEGATIVE_PROMPT,
            "input_ids": input_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "negative_input_ids": negative_input_ids.detach().cpu(),
            "negative_attention_mask": negative_attention_mask.detach().cpu(),
        },
        "outputs": {
            "text_features": text_features.detach().cpu(),
            "negative_text_features": negative_text_features.detach().cpu(),
        },
        "meta": {
            "seed": args.seed,
            "ckpt_dir": args.ckpt_dir,
            "tokenizer": args.tokenizer,
            "dtype": args.dtype,
            "input_shape": tuple(input_ids.shape),
            "output_shape": tuple(text_features.shape),
        },
    }
    torch.save(payload, args.output)
    print(f"wrote {args.output}")
    print(f"input_ids {tuple(input_ids.shape)} text_features {tuple(text_features.shape)}")
    print(f"negative_text_features {tuple(negative_text_features.shape)}")


if __name__ == "__main__":
    main()
