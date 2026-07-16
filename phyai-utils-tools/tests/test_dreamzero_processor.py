"""Tests for DreamZeroProcessor with an offline tokenizer stub."""

from __future__ import annotations

import torch

import phyai_utils_tools.models.dreamzero.processor_dreamzero as proc_mod
from phyai_utils_tools.models.dreamzero import (
    DREAMZERO_DEFAULT_NEGATIVE_PROMPT,
    DreamZeroProcessedInputs,
    DreamZeroPolicy,
    DreamZeroProcessor,
    DreamZeroTextInputs,
)


class _StubTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        prompts,
        return_tensors,
        padding,
        truncation,
        max_length,
        add_special_tokens,
    ):
        self.calls.append(list(prompts))
        batch_size = len(prompts)
        ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        mask = torch.zeros(batch_size, max_length, dtype=torch.long)
        for idx, prompt in enumerate(prompts):
            real_len = min(max_length, max(1, len(prompt.split()) + 1))
            ids[idx, :real_len] = torch.arange(1, real_len + 1)
            mask[idx, :real_len] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_dreamzero_processor_tokenizes_prompt_and_negative(monkeypatch) -> None:
    tokenizer = _StubTokenizer()
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: tokenizer)
    proc = DreamZeroProcessor(max_length=8, negative_prompt="")

    out = proc.preprocess({"task": [" Pick   up   the cup ", "open drawer"]})

    assert isinstance(out, DreamZeroTextInputs)
    assert out.input_ids.shape == (2, 8)
    assert out.attention_mask.shape == (2, 8)
    assert out.negative_input_ids.shape == (2, 8)
    assert out.negative_attention_mask.shape == (2, 8)
    assert tokenizer.calls[0] == ["Pick up the cup", "open drawer"]
    assert tokenizer.calls[1] == ["", ""]
    assert out.images is None
    assert out.state is None


def test_make_dreamzero_processors_factory(monkeypatch) -> None:
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    pre, post = proc_mod.make_dreamzero_processors(max_length=4)

    assert pre.name == "dreamzero_preprocessor"
    assert post.name == "dreamzero_postprocessor"


def test_dreamzero_processor_prepares_oxe_droid_video_grid(monkeypatch) -> None:
    tokenizer = _StubTokenizer()
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: tokenizer)
    proc = DreamZeroProcessor(max_length=16)

    left = torch.tensor([[[1], [2]], [[3], [4]]], dtype=torch.uint8)
    right = torch.tensor([[[5], [6]], [[7], [8]]], dtype=torch.uint8)
    wrist = torch.tensor([[[9], [10]], [[11], [12]]], dtype=torch.uint8)
    video = torch.stack([left, right, wrist], dim=0).unsqueeze(0).unsqueeze(0)
    out = proc.preprocess(
        {
            "video": video,
            "task": ["Pick the cube"],
            "state": torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float32),
        }
    )

    assert isinstance(out, DreamZeroProcessedInputs)
    assert out.images is not None
    assert out.images.shape == (1, 1, 4, 4, 1)
    expected = (
        torch.tensor(
            [
                [
                    [9, 9, 10, 10],
                    [11, 11, 12, 12],
                    [1, 2, 5, 6],
                    [3, 4, 7, 8],
                ]
            ],
            dtype=torch.uint8,
        )
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    torch.testing.assert_close(out.images, expected)
    assert tokenizer.calls[0][0].startswith(
        "A multi-view video shows that a robot pick the cube"
    )
    assert (
        "top view shows the camera view from the robot's wrist" in tokenizer.calls[0][0]
    )
    assert tokenizer.calls[1] == [DREAMZERO_DEFAULT_NEGATIVE_PROMPT]


def test_dreamzero_processor_prepares_generic_multiview_grid(monkeypatch) -> None:
    tokenizer = _StubTokenizer()
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: tokenizer)
    proc = DreamZeroProcessor(max_length=16, embodiment_tag="agibot")

    views = [torch.full((1, 1, 1), value, dtype=torch.uint8) for value in (1, 2, 3)]
    video = torch.stack(views, dim=0).unsqueeze(0).unsqueeze(0)
    out = proc.preprocess({"video": video, "task": ["Wave"]})

    assert out.images is not None
    assert out.images.shape == (1, 1, 2, 2, 1)
    expected = (
        torch.tensor([[[1, 3], [2, 0]]], dtype=torch.uint8).unsqueeze(0).unsqueeze(-1)
    )
    torch.testing.assert_close(out.images, expected)
    assert (
        "top-left view shows the camera view from the robot's head"
        in tokenizer.calls[0][0]
    )


def test_dreamzero_processor_pads_state_and_action_masks(monkeypatch) -> None:
    tokenizer = _StubTokenizer()
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: tokenizer)
    proc = DreamZeroProcessor(
        max_length=8,
        max_state_dim=5,
        max_action_dim=4,
        action_horizon=2,
    )
    video = torch.zeros(1, 1, 1, 1, 1, 1, dtype=torch.uint8)
    state = torch.tensor([[[1.0, 2.0, 3.0]]])
    action = torch.tensor([[[0.5, 0.6], [0.7, 0.8]]])

    out = proc.preprocess(
        {
            "video": video,
            "task": ["Move"],
            "state": state,
            "action": action,
        }
    )

    assert out.state is not None
    assert out.state_mask is not None
    assert out.action is not None
    assert out.action_mask is not None
    torch.testing.assert_close(
        out.state,
        torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        out.state_mask,
        torch.tensor([[[True, True, True, False, False]]]),
    )
    torch.testing.assert_close(
        out.action,
        torch.tensor([[[0.5, 0.6, 0.0, 0.0], [0.7, 0.8, 0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        out.action_mask,
        torch.tensor([[[True, True, False, False], [True, True, False, False]]]),
    )


def test_dreamzero_postprocess_slices_and_unnormalizes_action(monkeypatch) -> None:
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    proc = DreamZeroProcessor(
        max_length=4,
        raw_action_dim=2,
        dataset_stats={
            "action": {
                "q01": torch.tensor([0.0, 10.0]),
                "q99": torch.tensor([2.0, 14.0]),
            }
        },
    )

    out = proc.postprocess(torch.tensor([[[-1.0, 0.0, 99.0]]]))

    torch.testing.assert_close(out.action, torch.tensor([[[0.0, 12.0]]]))
    torch.testing.assert_close(
        out.normalized_action,
        torch.tensor([[[-1.0, 0.0, 99.0]]]),
    )


def test_dreamzero_postprocess_relative_action_uses_last_observation_state(
    monkeypatch,
) -> None:
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    metadata = {
        "oxe_droid": {
            "statistics": {
                "state": {
                    "joint_position": {
                        "mean": [0.0, 0.0],
                        "q01": [0.0, 0.0],
                        "q99": [1.0, 1.0],
                    },
                    "gripper_position": {
                        "mean": [0.0],
                        "q01": [0.0],
                        "q99": [1.0],
                    },
                },
                "action": {
                    "joint_position": {
                        "mean": [0.0, 0.0],
                        "q01": [0.0, 10.0],
                        "q99": [2.0, 14.0],
                    },
                    "gripper_position": {
                        "mean": [0.0],
                        "q01": [0.0],
                        "q99": [1.0],
                    },
                },
            }
        }
    }
    proc = DreamZeroProcessor(
        max_length=4,
        raw_action_dim=3,
        dataset_stats={
            "state": {
                "q01": torch.tensor([0.0, 0.0, 0.0]),
                "q99": torch.tensor([1.0, 1.0, 1.0]),
            },
            "action": {
                "q01": torch.tensor([0.0, 10.0, 0.0]),
                "q99": torch.tensor([2.0, 14.0, 1.0]),
            },
        },
        metadata=metadata,
        state_keys=("joint_position", "gripper_position"),
        action_keys=("joint_position", "gripper_position"),
        relative_action_keys=("joint_position",),
    )
    obs = {"state": torch.tensor([[[100.0, 200.0, 0.25]]])}

    out = proc.postprocess(torch.tensor([[[-1.0, 0.0, 0.0]]]), obs=obs)

    torch.testing.assert_close(out.action, torch.tensor([[[100.0, 212.0, 0.5]]]))


def test_dreamzero_policy_wraps_preprocess_infer_and_postprocess(monkeypatch) -> None:
    monkeypatch.setattr(proc_mod, "get_tokenizer", lambda name: _StubTokenizer())
    proc = DreamZeroProcessor(max_length=4, raw_action_dim=2)
    calls = []

    def infer(processed):
        calls.append(processed)
        return {"action": torch.tensor([[[1.0, 2.0, 3.0]]])}

    policy = DreamZeroPolicy(processor=proc, infer=infer)
    video = torch.zeros(1, 1, 1, 1, 1, 1, dtype=torch.uint8)
    result = policy.act(
        {"video": video, "task": ["Move"], "state": torch.zeros(1, 1, 2)}
    )

    assert len(calls) == 1
    assert isinstance(result.processed, DreamZeroProcessedInputs)
    torch.testing.assert_close(result.action, torch.tensor([[[1.0, 2.0]]]))
