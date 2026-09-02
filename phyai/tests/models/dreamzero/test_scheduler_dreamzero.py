from __future__ import annotations

import torch
import torch.nn as nn

import phyai.layers.linear as L
import phyai.models.dreamzero.scheduler_ws1_dreamzero as scheduler_mod
from phyai.layers import LayerNorm, RMSNorm
from phyai.models.dreamzero import (
    DreamZeroConfig,
    DreamZeroDiT,
    DreamZeroDiTConfig,
    DreamZeroFlowStepper,
    DreamZeroRequest,
    DreamZeroSchedulerOutput,
    DreamZeroWS1Scheduler,
)


class _FakeRunner:
    def __init__(self) -> None:
        self.calls = []
        self.kv_cache = ["cache"]

    def setup(self) -> None:
        return None

    def reset(self) -> None:
        self.calls.clear()

    def forward(self, batch):
        self.calls.append(batch)
        video = torch.ones_like(batch.x) * (2.0 if batch.action is not None else 0.0)
        action = (
            torch.ones_like(batch.action) * 3.0 if batch.action is not None else None
        )
        return type(
            "Output",
            (),
            {
                "video": video,
                "action": action,
                "kv_cache": self.kv_cache,
                "crossattn_cache": [],
            },
        )()


class _ValueRunner:
    def __init__(self, *, video_value: float, action_value: float) -> None:
        self.video_value = video_value
        self.action_value = action_value
        self.calls = []
        self.kv_cache = [torch.tensor([video_value])]

    def setup(self) -> None:
        return None

    def reset(self) -> None:
        self.calls.clear()

    def forward(self, batch):
        self.calls.append(batch)
        video = torch.full_like(batch.x, self.video_value)
        action = (
            torch.full_like(batch.action, self.action_value)
            if batch.action is not None
            else None
        )
        return type(
            "Output",
            (),
            {
                "video": video,
                "action": action,
                "kv_cache": self.kv_cache,
                "crossattn_cache": [],
            },
        )()


class _FakeConfig:
    cfg_scale = 1.0
    sigma_shift = 1.0
    decouple_inference_noise = False
    video_inference_final_noise = 0.8
    num_inference_timesteps = 2


class _FakeModel:
    config = _FakeConfig()

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(()))


def _init_linear_dispatcher() -> None:
    L.init(register_flashinfer=False, validate=False)


def _tiny_config(*, num_layers: int = 1) -> DreamZeroConfig:
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
            dim=16,
            ffn_dim=32,
            frame_seqlen=4,
            freq_dim=8,
            in_dim=36,
            max_chunk_size=-1,
            num_action_per_block=2,
            num_frame_per_block=1,
            num_heads=2,
            num_layers=num_layers,
            num_state_per_block=1,
            out_dim=2,
        ),
    )


def _replace_norms_with_identity(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, (LayerNorm, RMSNorm)):
            module._modules[name] = nn.Identity()
        else:
            _replace_norms_with_identity(child)


def _init_tiny_model(model: nn.Module) -> None:
    _replace_norms_with_identity(model)
    for param in model.parameters():
        if param.dim() == 1:
            param.data.zero_()
        else:
            param.data.normal_(mean=0.0, std=0.02)


def test_flow_stepper_uses_official_flow_unipc_schedule() -> None:
    stepper = DreamZeroFlowStepper(shift=1.0)
    stepper.set_timesteps(2, device="cpu", dtype=torch.float32)
    assert stepper.timesteps is not None
    torch.testing.assert_close(stepper.timesteps, torch.tensor([999, 499]))

    sample = torch.tensor([[1.0]])
    sample = stepper.step(
        model_output=torch.tensor([[2.0]]),
        sample=sample,
        step_index=0,
    )
    sample = stepper.step(
        model_output=torch.tensor([[2.0]]),
        sample=sample,
        step_index=1,
    )

    torch.testing.assert_close(sample, torch.tensor([[-0.998]]))


def test_ws1_scheduler_prefills_then_denoises_with_runner() -> None:
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _FakeRunner()
    scheduler.setup()

    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        clean_video=torch.zeros(1, 2, 1, 2, 2),
        image_context_tokens=1,
        num_inference_steps=2,
    )
    out = scheduler.step(request)

    assert isinstance(out, DreamZeroSchedulerOutput)
    assert out.video.shape == request.video.shape
    assert out.action.shape == request.action.shape
    assert len(scheduler.cond_runner.calls) == 3
    assert scheduler.cond_runner.calls[0].action is None
    assert scheduler.cond_runner.calls[0].update_kv_cache
    assert all(not call.update_kv_cache for call in scheduler.cond_runner.calls[1:])
    assert out.cond_kv_cache == ["cache"]


def test_ws1_scheduler_dynamic_dit_matches_official_skip_schedule() -> None:
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _FakeRunner()
    scheduler.setup()

    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        image_context_tokens=1,
        num_inference_steps=1,
        dynamic_dit=True,
        dynamic_dit_scheduler_steps=16,
    )
    out = scheduler.step(request)

    assert out.scheduler_steps == 16
    assert out.dit_compute_steps == 4
    assert len(scheduler.cond_runner.calls) == 4
    assert all(call.action is not None for call in scheduler.cond_runner.calls)
    assert torch.isfinite(out.video).all()
    assert torch.isfinite(out.action).all()


def test_ws1_scheduler_slices_full_condition_for_prefill_and_denoise() -> None:
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _FakeRunner()
    scheduler.setup()

    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 2, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        y=torch.randn(1, 22, 4, 2, 2),
        clean_video=torch.zeros(1, 2, 1, 2, 2),
        current_start_frame=1,
        image_context_tokens=1,
        seq_len=8,
        num_inference_steps=1,
        prefill_clean_cache=True,
    )

    scheduler.step(request)

    prefill = scheduler.cond_runner.calls[0]
    denoise = scheduler.cond_runner.calls[1]
    assert prefill.action is None
    assert prefill.current_start_frame == 0
    assert prefill.seq_len is None
    assert prefill.y is not None
    assert prefill.y.shape == (1, 22, 1, 2, 2)
    torch.testing.assert_close(prefill.y, request.y[:, :, 0:1])
    assert denoise.action is not None
    assert denoise.seq_len == 8
    assert denoise.current_start_frame == 1
    assert denoise.y is not None
    assert denoise.y.shape == (1, 22, 2, 2, 2)
    torch.testing.assert_close(denoise.y, request.y[:, :, 1:3])


def test_ws1_scheduler_updates_reference_cache_between_chunks() -> None:
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _FakeRunner()
    scheduler.setup()
    y = torch.randn(1, 22, 4, 2, 2)

    first = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        y=y,
        clean_video=torch.zeros(1, 2, 1, 2, 2),
        image_context_tokens=1,
        num_inference_steps=1,
    )
    first_out = scheduler.step(first)

    assert first_out.current_start_frame == 2
    assert len(scheduler.cond_runner.calls) == 2
    assert scheduler.cond_runner.calls[0].action is None
    assert scheduler.cond_runner.calls[0].update_kv_cache
    assert scheduler.cond_runner.calls[0].current_start_frame == 0
    assert scheduler.cond_runner.calls[1].action is not None
    assert not scheduler.cond_runner.calls[1].update_kv_cache
    assert scheduler.cond_runner.calls[1].current_start_frame == 1

    second = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        y=y,
        reference_video=torch.ones(1, 2, 1, 2, 2),
        image_context_tokens=1,
        num_inference_steps=1,
    )
    second_out = scheduler.step(second)

    assert second_out.current_start_frame == 3
    assert len(scheduler.cond_runner.calls) == 4
    reference = scheduler.cond_runner.calls[2]
    denoise = scheduler.cond_runner.calls[3]
    assert reference.action is None
    assert reference.update_kv_cache
    assert reference.current_start_frame == 1
    assert reference.seq_len is None
    assert reference.y is not None
    torch.testing.assert_close(reference.y, y[:, :, 1:2])
    assert denoise.action is not None
    assert not denoise.update_kv_cache
    assert denoise.current_start_frame == 2
    assert denoise.y is not None
    torch.testing.assert_close(denoise.y, y[:, :, 2:3])


def test_reference_cache_trace_wraps_cond_runner(monkeypatch) -> None:
    model = _FakeModel()
    scheduler = DreamZeroWS1Scheduler(model, device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _FakeRunner()
    scheduler.setup()
    monkeypatch.setenv("DREAMZERO_DIT_TRACE_REFERENCE", "1")
    observed = []

    def capture(*args, **kwargs):
        del args, kwargs
        observed.append(
            (
                model._dz_trace_enabled,
                model._dz_trace_label,
                model._dz_trace_branch,
            )
        )

    scheduler._run_runner = capture
    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        y=torch.zeros(1, 22, 4, 2, 2),
    )

    scheduler._update_reference_cache(
        scheduler.cond_runner,
        request=request,
        reference_video=torch.zeros(1, 2, 1, 2, 2),
        context=request.context,
        clip_feature=None,
        current_start_frame=2,
    )

    assert observed == [(True, "reference_KV_update", 0)]
    assert not model._dz_trace_enabled


def test_ws1_scheduler_cfg_parallel_rank_uses_only_local_branch(fake_mesh) -> None:
    fake_mesh(sizes={"cfg": 2, "tp": 1}, ranks={"cfg": 1, "tp": 0})
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _ValueRunner(video_value=2.0, action_value=3.0)
    scheduler.uncond_runner = _ValueRunner(video_value=10.0, action_value=11.0)
    scheduler.setup()

    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        uncond_context=torch.ones(1, 3, 8),
        clean_video=None,
        image_context_tokens=1,
        num_inference_steps=1,
        guidance_scale=1.5,
    )
    runner, context, _ = scheduler._cfg_parallel_branch(
        request=request,
        context=request.context,
        clip_feature=None,
        use_cfg=True,
    )

    assert runner is scheduler.uncond_runner
    torch.testing.assert_close(context, request.uncond_context)
    assert scheduler.cond_runner.calls == []
    assert scheduler.uncond_runner.calls == []


def test_ws1_scheduler_cfg_parallel_gathers_and_combines(
    monkeypatch, fake_mesh
) -> None:
    fake_mesh(sizes={"cfg": 2, "tp": 1}, ranks={"cfg": 0, "tp": 0})
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _ValueRunner(video_value=2.0, action_value=3.0)
    scheduler.uncond_runner = _ValueRunner(video_value=10.0, action_value=11.0)
    scheduler.setup()

    def fake_all_gather(x: torch.Tensor, *, axis: str, dim: int, mesh="model"):
        del mesh
        assert axis == "cfg"
        assert dim == 0
        if x.dim() == 6:
            return torch.stack(
                [torch.full_like(x[0], 2.0), torch.full_like(x[0], 10.0)],
                dim=0,
            )
        return torch.stack(
            [torch.full_like(x[0], 3.0), torch.full_like(x[0], 11.0)],
            dim=0,
        )

    monkeypatch.setattr(scheduler_mod.P, "all_gather", fake_all_gather)
    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        uncond_context=torch.ones(1, 3, 8),
        clean_video=None,
        image_context_tokens=1,
        num_inference_steps=1,
        guidance_scale=1.5,
    )

    out = scheduler.step(request)

    assert len(scheduler.cond_runner.calls) == 1
    assert len(scheduler.uncond_runner.calls) == 0
    torch.testing.assert_close(
        out.last_video_pred, torch.full_like(request.video, -2.0)
    )
    torch.testing.assert_close(
        out.last_action_pred, torch.full_like(request.action, 3.0)
    )
    torch.testing.assert_close(out.video, torch.full_like(request.video, 1.998))
    torch.testing.assert_close(out.action, torch.full_like(request.action, -2.997))


def test_ws1_scheduler_action_cfg_uses_cond_only(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    scheduler = DreamZeroWS1Scheduler(_FakeModel(), device="cpu", use_cfg_runner=False)
    scheduler.cond_runner = _ValueRunner(video_value=2.0, action_value=3.0)
    scheduler.uncond_runner = _ValueRunner(video_value=10.0, action_value=11.0)
    scheduler.setup()

    request = DreamZeroRequest(
        video=torch.zeros(1, 2, 1, 2, 2),
        action=torch.zeros(1, 2, 4),
        state=torch.zeros(1, 1, 5),
        context=torch.zeros(1, 3, 8),
        uncond_context=torch.ones(1, 3, 8),
        clean_video=None,
        image_context_tokens=1,
        num_inference_steps=1,
        guidance_scale=1.5,
    )

    out = scheduler.step(request)

    assert len(scheduler.cond_runner.calls) == 1
    assert len(scheduler.uncond_runner.calls) == 1
    torch.testing.assert_close(
        out.last_video_pred, torch.full_like(request.video, -2.0)
    )
    torch.testing.assert_close(
        out.last_action_pred, torch.full_like(request.action, 3.0)
    )


def test_ws1_scheduler_runs_real_tiny_cfg_two_branch_path(fake_mesh) -> None:
    fake_mesh(sizes={"tp": 1})
    _init_linear_dispatcher()
    model = DreamZeroDiT(
        _tiny_config(),
        params_dtype=torch.float32,
        device="cpu",
        attn_backend="eager",
        norm_backend="phyai-kernel",
    )
    _init_tiny_model(model)

    scheduler = DreamZeroWS1Scheduler(model, device="cpu", use_cfg_runner=True)
    scheduler.setup()
    request = DreamZeroRequest(
        video=torch.randn(1, 2, 1, 4, 4),
        action=torch.randn(1, 2, 4),
        state=torch.randn(1, 1, 5),
        context=torch.randn(1, 3, 4096),
        uncond_context=torch.randn(1, 3, 4096),
        clip_feature=torch.randn(1, 2, 1280),
        uncond_clip_feature=torch.randn(1, 2, 1280),
        y=torch.randn(1, 34, 1, 4, 4),
        clean_video=torch.randn(1, 2, 1, 4, 4),
        concat_first_frame_latent=True,
        image_context_tokens=2,
        num_inference_steps=1,
        guidance_scale=1.5,
    )

    out = scheduler.step(request)

    assert out.video.shape == request.video.shape
    assert out.action.shape == request.action.shape
    assert out.last_video_pred is not None
    assert out.last_video_pred.shape == request.video.shape
    assert out.last_action_pred is not None
    assert out.last_action_pred.shape == request.action.shape
    assert out.cond_kv_cache
    assert out.uncond_kv_cache
