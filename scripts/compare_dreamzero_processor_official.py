from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def add_repo_to_path(path: str) -> None:
    repo = Path(path).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def tensorize(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def print_metrics(name: str, ref: Any, got: Any) -> None:
    ref_t = tensorize(ref)
    got_t = tensorize(got)
    same_shape = tuple(ref_t.shape) == tuple(got_t.shape)
    if ref_t.dtype == torch.bool or got_t.dtype == torch.bool:
        diff = ref_t.to(torch.int32) - got_t.to(torch.int32)
    else:
        diff = ref_t.to(torch.float32) - got_t.to(torch.float32)
    max_abs = diff.abs().max().item() if diff.numel() else 0
    mean_abs = diff.abs().float().mean().item() if diff.numel() else 0
    print(
        f"{name}: shape={tuple(got_t.shape)} same_shape={same_shape} "
        f"max_abs={max_abs:.8g} mean_abs={mean_abs:.8g}"
    )
    if not same_shape or max_abs != 0:
        raise AssertionError(f"{name} mismatch")


def build_payload(*, training: bool) -> dict[str, Any]:
    batch_size = 2
    video = np.arange(batch_size * 2 * 3 * 2 * 3 * 3, dtype=np.uint8).reshape(
        batch_size,
        2,
        3,
        2,
        3,
        3,
    )
    payload: dict[str, Any] = {
        "video": video,
        "state": np.array(
            [
                [[0.1, 0.2, 0.3]],
                [[1.1, 1.2, 1.3]],
            ],
            dtype=np.float32,
        ),
        "annotation.language.action_text": np.array(
            ["Pick up the cube", "Open the drawer"],
            dtype=object,
        ),
    }
    if training:
        payload["action"] = np.arange(batch_size * 24 * 2, dtype=np.float32).reshape(
            batch_size,
            24,
            2,
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-repo", default=os.environ.get("DREAMZERO_OFFICIAL_REPO")
    )
    parser.add_argument("--tokenizer", default="/data/share/google-umt5-xxl")
    parser.add_argument("--embodiment-tag", default="oxe_droid")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    if args.official_repo is None:
        raise SystemExit("Pass --official-repo or set DREAMZERO_OFFICIAL_REPO.")

    add_repo_to_path(args.official_repo)

    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import DreamTransform
    from phyai_utils_tools.models.dreamzero.processor_dreamzero import (
        DreamZeroProcessor,
    )
    from phyai_utils_tools.models.dreamzero.steps_dreamzero import (
        DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING,
    )

    tag = EmbodimentTag(args.embodiment_tag)
    mapping = DREAMZERO_DEFAULT_EMBODIMENT_TAG_MAPPING.copy()
    payload = build_payload(training=args.training)

    official = DreamTransform(
        default_instruction="Perform the default behavior.",
        language_dropout_prob=0.0,
        always_use_default_instruction=False,
        max_state_dim=64,
        max_action_dim=32,
        max_length=512,
        state_horizon=1,
        action_horizon=24,
        num_views=3,
        tokenizer_path=args.tokenizer,
        embodiment_tag_mapping=mapping,
        embodiment_tag=tag,
        training=args.training,
    )
    official_out = official({key: value.copy() for key, value in payload.items()})

    processor = DreamZeroProcessor(
        tokenizer_name=args.tokenizer,
        max_length=512,
        max_state_dim=64,
        max_action_dim=32,
        state_horizon=1,
        action_horizon=24,
        num_views=3,
        embodiment_tag=args.embodiment_tag,
        embodiment_tag_mapping=mapping,
        training=args.training,
    )
    phyai_out = processor.preprocess(payload)

    print_metrics("images", official_out["images"], phyai_out.images)
    print_metrics("state", official_out["state"], phyai_out.state)
    print_metrics("state_mask", official_out["state_mask"], phyai_out.state_mask)
    print_metrics(
        "embodiment_id", official_out["embodiment_id"], phyai_out.embodiment_id
    )
    print_metrics("text", official_out["text"], phyai_out.input_ids)
    print_metrics(
        "text_attention_mask",
        official_out["text_attention_mask"],
        phyai_out.attention_mask,
    )
    print_metrics(
        "text_negative", official_out["text_negative"], phyai_out.negative_input_ids
    )
    print_metrics(
        "text_attention_mask_negative",
        official_out["text_attention_mask_negative"],
        phyai_out.negative_attention_mask,
    )
    if args.training:
        print_metrics("action", official_out["action"], phyai_out.action)
        print_metrics("action_mask", official_out["action_mask"], phyai_out.action_mask)
    print("processor compare passed")


if __name__ == "__main__":
    main()
