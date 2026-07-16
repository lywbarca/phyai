from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from groot.vla.model.dreamzero.modules.wan_video_image_encoder import WanImageEncoder
from groot.vla.model.dreamzero.modules.wan_video_vae import WanVideoVAE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    used_missing = [k for k in missing if not k.startswith("model.textual.")]
    if used_missing or unexpected:
        print(f"missing_non_textual={used_missing[:20]}")
        print(f"unexpected={unexpected[:20]}")
        raise SystemExit(2)
    print(f"loaded {len(state)} tensors from {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default="/data/share/DreamZero-DROID")
    parser.add_argument("--output", default="/tmp/dreamzero_image_vae_official.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    if os.environ.get("DREAMZERO_DROP_CACHES_FIRST") == "1":
        os.system("sudo -n /usr/local/sbin/dreamzero_drop_caches || true")

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    image_encoder = WanImageEncoder().to(device).eval()
    vae = WanVideoVAE(z_dim=16).to(device).eval()
    load_prefixed_state_dict(
        image_encoder, args.ckpt_dir, "action_head.image_encoder."
    )
    load_prefixed_state_dict(vae, args.ckpt_dir, "action_head.vae.")

    image = torch.empty(1, 3, 224, 224, device=device).uniform_(-1, 1)
    vae_video = torch.empty(1, 3, 1, 8, 8, device=device).uniform_(-1, 1)
    vae_latent = torch.randn(1, 16, 1, 1, 1, device=device)

    with torch.no_grad():
        image_features = image_encoder.encode_image([image])
        vae_encoded = vae.encode(vae_video, tiled=False)
        vae_decoded = vae.decode(vae_latent, tiled=False)

    payload = {
        "inputs": {
            "image": image.detach().cpu(),
            "vae_video": vae_video.detach().cpu(),
            "vae_latent": vae_latent.detach().cpu(),
        },
        "outputs": {
            "image_features": image_features.detach().cpu(),
            "vae_encoded": vae_encoded.detach().cpu(),
            "vae_decoded": vae_decoded.detach().cpu(),
        },
        "meta": {
            "seed": args.seed,
            "ckpt_dir": args.ckpt_dir,
            "image_shape": tuple(image.shape),
            "image_features_shape": tuple(image_features.shape),
            "vae_video_shape": tuple(vae_video.shape),
            "vae_latent_shape": tuple(vae_latent.shape),
            "vae_encoded_shape": tuple(vae_encoded.shape),
            "vae_decoded_shape": tuple(vae_decoded.shape),
        },
    }
    torch.save(payload, args.output)
    print(f"wrote {args.output}")
    print(f"image_features {tuple(image_features.shape)}")
    print(f"vae_encoded {tuple(vae_encoded.shape)} vae_decoded {tuple(vae_decoded.shape)}")


if __name__ == "__main__":
    main()
