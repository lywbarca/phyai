"""DreamZero single-card latent-level scheduler.

This scheduler starts at the model-ready tensor boundary: text/image/VAE
encoding is handled by the caller or by a future plugin layer. The scheduler
owns DreamZero DiT runners, resets them per request, performs optional clean
context KV prefill, and runs the flow denoise loop through the runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

import phyai.parallel as P
from phyai.models.dreamzero.configuration_dreamzero import DreamZeroConfig
from phyai.models.dreamzero.model_runner_dreamzero import (
    DreamZeroDiTForwardBatch,
    DreamZeroDiTForwardOutput,
    DreamZeroDiTRunner,
)
from phyai.models.dreamzero.modeling_dreamzero import DreamZeroDiT
from phyai.runtime.schedule import Scheduler


_SEQ_LEN_UNSET = object()


def _axis_rank_size(axis: str) -> tuple[int, int]:
    try:
        mesh = P.default_mesh()
        names = mesh.torch_mesh.mesh_dim_names or ()
        if axis not in names:
            return 0, 1
        return mesh.axis_local_rank(axis), mesh.axis_size(axis)
    except Exception:
        return 0, 1


@dataclass
class DreamZeroRequest:
    """One model-ready DreamZero DiT inference request.

    Tensors are already encoded and placed on the desired device. Video latents
    use DreamZero DiT's `(B, C, T, H, W)` layout. `context` is the conditional
    text/image-context tensor before the DiT's internal context projections.
    """

    video: torch.Tensor
    action: torch.Tensor
    state: torch.Tensor
    context: torch.Tensor
    embodiment_id: torch.Tensor | None = None
    clip_feature: torch.Tensor | None = None
    y: torch.Tensor | None = None
    clean_video: torch.Tensor | None = None
    uncond_context: torch.Tensor | None = None
    uncond_clip_feature: torch.Tensor | None = None
    seq_len: int | None = None
    current_start_frame: int = 0
    concat_first_frame_latent: bool = False
    image_context_tokens: int = 257
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    sigma_shift: float | None = None
    decouple_inference_noise: bool | None = None
    video_inference_final_noise: float | None = None
    update_kv_cache: bool = False
    prefill_clean_cache: bool | None = None


@dataclass
class DreamZeroSchedulerOutput:
    video: torch.Tensor
    action: torch.Tensor
    last_video_pred: torch.Tensor | None
    last_action_pred: torch.Tensor | None
    cond_kv_cache: list[torch.Tensor | None]
    uncond_kv_cache: list[torch.Tensor | None] | None


class DreamZeroFlowStepper:
    """DreamZero Flow UniPC multistep scheduler.

    This is a local, dependency-light equivalent of DreamZero's official
    FlowUniPCMultistepScheduler for the settings used by inference:
    flow-prediction, x0 prediction, order-2 UniPC, bh2, lower-order final, and
    zero final sigma.
    """

    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        solver_order: int = 2,
    ) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.solver_order = int(solver_order)
        if self.solver_order <= 0:
            raise ValueError(f"solver_order must be positive, got {solver_order}.")
        alphas = torch.linspace(
            1.0 / self.num_train_timesteps,
            1.0,
            self.num_train_timesteps,
            dtype=torch.float32,
        )
        train_sigmas = 1.0 - alphas
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None
        self.sigma_min = train_sigmas[-1].item()
        self.sigma_max = train_sigmas[0].item()
        self.model_outputs: list[torch.Tensor | None] = [None] * self.solver_order
        self.timestep_list: list[torch.Tensor | None] = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample: torch.Tensor | None = None
        self.this_order = 1
        self.config = SimpleNamespace(
            num_train_timesteps=self.num_train_timesteps,
            solver_order=self.solver_order,
            prediction_type="flow_prediction",
            thresholding=False,
            predict_x0=True,
            solver_type="bh2",
            lower_order_final=True,
            final_sigmas_type="zero",
        )

    def set_timesteps(
        self,
        num_inference_steps: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        final_sigma: float = 0.0,
        shift: float | None = None,
    ) -> None:
        if num_inference_steps <= 0:
            raise ValueError(
                f"num_inference_steps must be positive, got {num_inference_steps}."
            )
        sigmas = torch.linspace(
            self.sigma_max,
            self.sigma_min,
            num_inference_steps + 1,
            dtype=torch.float32,
        )[:-1]
        sigma_shift = self.shift if shift is None else float(shift)
        sigmas = sigma_shift * sigmas / (1 + (sigma_shift - 1) * sigmas)
        sigmas = torch.cat([sigmas, sigmas.new_zeros(1)], dim=0)
        if final_sigma:
            sigma_max = sigmas[0]
            sigmas = sigmas * (sigma_max - final_sigma) / sigma_max + final_sigma
        del dtype
        self.sigmas = sigmas.to(device=device)
        self.timesteps = (self.sigmas[:-1] * self.num_train_timesteps).to(torch.int64)
        self.model_outputs = [None] * self.solver_order
        self.timestep_list = [None] * self.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        self.this_order = 1

    @staticmethod
    def _sigma_to_alpha_sigma_t(
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return 1 - sigma, sigma

    def _convert_model_output(
        self,
        *,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        if self.sigmas is None:
            raise RuntimeError("call set_timesteps() before step().")
        sigma_t = self.sigmas[step_index].to(device=sample.device, dtype=sample.dtype)
        return sample - sigma_t * model_output

    def _multistep_uni_p_bh_update(
        self,
        *,
        sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        if self.sigmas is None:
            raise RuntimeError("call set_timesteps() before step().")
        model_outputs = self.model_outputs
        m0 = model_outputs[-1]
        if m0 is None:
            raise RuntimeError("missing current model output for UniPC update.")

        sigma_t = self.sigmas[step_index + 1].to(device=sample.device)
        sigma_s0 = self.sigmas[step_index].to(device=sample.device)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        _alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(_alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_outputs[-(i + 1)]
            if mi is None:
                continue
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(
                self.sigmas[si].to(device=sample.device)
            )
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk.to(device=m0.device, dtype=m0.dtype))

        hh = -h
        h_phi_1 = torch.expm1(hh)
        b_h = torch.expm1(hh)

        x_t = sigma_t / sigma_s0 * sample - alpha_t * h_phi_1 * m0
        if d1s:
            d1s_tensor = torch.stack(d1s, dim=1)
            if order == 2:
                rhos_p = torch.full((1,), 0.5, dtype=sample.dtype, device=sample.device)
            else:
                rks.append(
                    torch.ones((), dtype=self.sigmas.dtype, device=sample.device)
                )
                rks_tensor = torch.stack(rks, dim=0)
                r_mat = []
                b_vec = []
                h_phi_k = h_phi_1 / hh - 1
                factorial_i = 1
                for i in range(1, order + 1):
                    r_mat.append(torch.pow(rks_tensor, i - 1))
                    b_vec.append(h_phi_k * factorial_i / b_h)
                    factorial_i *= i + 1
                    h_phi_k = h_phi_k / hh - 1 / factorial_i
                rhos_p = torch.linalg.solve_ex(
                    torch.stack(r_mat, dim=0)[:-1, :-1],
                    torch.stack(b_vec, dim=0)[:-1],
                )[0].to(device=sample.device, dtype=sample.dtype)
            pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s_tensor)
            x_t = x_t - alpha_t * b_h * pred_res
        return x_t.to(sample.dtype)

    def _multistep_uni_c_bh_update(
        self,
        *,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        if self.sigmas is None:
            raise RuntimeError("call set_timesteps() before step().")
        model_outputs = self.model_outputs
        m0 = model_outputs[-1]
        if m0 is None:
            raise RuntimeError("missing previous model output for UniPC correction.")

        sigma_t = self.sigmas[step_index].to(device=this_sample.device)
        sigma_s0 = self.sigmas[step_index - 1].to(device=this_sample.device)
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
        h = lambda_t - lambda_s0

        rks = []
        d1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_outputs[-(i + 1)]
            if mi is None:
                continue
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(
                self.sigmas[si].to(device=this_sample.device)
            )
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            d1s.append((mi - m0) / rk.to(device=m0.device, dtype=m0.dtype))
        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=this_sample.device))
        rks_tensor = torch.stack(rks, dim=0)

        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b_h = torch.expm1(hh)

        r_mat = []
        b_vec = []
        factorial_i = 1
        for i in range(1, order + 1):
            r_mat.append(torch.pow(rks_tensor, i - 1))
            b_vec.append(h_phi_k * factorial_i / b_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        if order == 1:
            rhos_c = torch.full(
                (1,),
                0.5,
                dtype=this_sample.dtype,
                device=this_sample.device,
            )
        else:
            rhos_c = torch.linalg.solve_ex(
                torch.stack(r_mat, dim=0),
                torch.stack(b_vec, dim=0),
            )[0].to(device=this_sample.device, dtype=this_sample.dtype)

        x_t = sigma_t / sigma_s0 * last_sample - alpha_t * h_phi_1 * m0
        if d1s:
            d1s_tensor = torch.stack(d1s, dim=1)
            corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s_tensor)
        else:
            corr_res = 0
        d1_t = this_model_output - m0
        x_t = x_t - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
        return x_t.to(this_sample.dtype)

    def step(
        self,
        *,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.sigmas is None or self.timesteps is None:
            raise RuntimeError("call set_timesteps() before step().")
        use_corrector = step_index > 0 and self.last_sample is not None

        converted = self._convert_model_output(
            model_output=model_output,
            sample=sample,
            step_index=step_index,
        )
        if use_corrector:
            sample = self._multistep_uni_c_bh_update(
                this_model_output=converted,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                step_index=step_index,
            ).clone()

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]
        self.model_outputs[-1] = converted
        self.timestep_list[-1] = (
            self.timesteps[step_index] if timestep is None else timestep
        )

        if self.config.lower_order_final:
            this_order = min(self.solver_order, len(self.timesteps) - step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        if self.this_order <= 0:
            raise RuntimeError("UniPC order must be positive.")

        self.last_sample = sample
        prev_sample = self._multistep_uni_p_bh_update(
            sample=sample,
            order=self.this_order,
            step_index=step_index,
        ).clone()
        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1
        return prev_sample


class DreamZeroWS1Scheduler(Scheduler):
    """Single-card DreamZero DiT denoise scheduler."""

    def __init__(
        self,
        model: DreamZeroDiT,
        *,
        device: torch.device | str | None = None,
        use_cfg_runner: bool = True,
    ) -> None:
        self.model = model
        self.cfg: DreamZeroConfig = model.config
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.cfg_rank, self.cfg_size = _axis_rank_size("cfg")
        self.cond_runner = DreamZeroDiTRunner(model, device=self.device)
        self.uncond_runner = (
            DreamZeroDiTRunner(model, device=self.device)
            if use_cfg_runner or self.cfg_size > 1
            else None
        )
        self._ready = False

    def setup(self) -> None:
        self.cond_runner.setup()
        if self.uncond_runner is not None:
            self.uncond_runner.setup()
        self._ready = True

    def _reset_runners(self) -> None:
        self.cond_runner.reset()
        if self.uncond_runner is not None:
            self.uncond_runner.reset()

    def _run_runner(
        self,
        runner: DreamZeroDiTRunner,
        *,
        video: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        y: torch.Tensor | None,
        request: DreamZeroRequest,
        update_kv_cache: bool,
        use_crossattn_cache: bool,
        update_crossattn_cache: bool,
        seq_len: int | None | object = _SEQ_LEN_UNSET,
        current_start_frame: int | None = None,
    ) -> DreamZeroDiTForwardOutput:
        return runner.forward(
            DreamZeroDiTForwardBatch(
                x=video,
                timestep=timestep,
                context=context,
                seq_len=request.seq_len if seq_len is _SEQ_LEN_UNSET else seq_len,
                current_start_frame=(
                    request.current_start_frame
                    if current_start_frame is None
                    else current_start_frame
                ),
                y=y,
                clip_feature=clip_feature,
                action=action,
                timestep_action=timestep_action,
                state=state,
                embodiment_id=embodiment_id,
                concat_first_frame_latent=request.concat_first_frame_latent,
                image_context_tokens=request.image_context_tokens,
                use_kv_cache=True,
                update_kv_cache=update_kv_cache,
                use_crossattn_cache=use_crossattn_cache,
                update_crossattn_cache=update_crossattn_cache,
            )
        )

    @staticmethod
    def _slice_temporal_condition(
        condition: torch.Tensor | None,
        *,
        start: int,
        length: int,
    ) -> torch.Tensor | None:
        if condition is None:
            return None
        if length <= 0:
            raise ValueError(f"condition length must be positive, got {length}.")
        total = int(condition.shape[2])
        if total == length:
            return condition
        if total <= 0:
            raise ValueError("condition tensor must have a non-empty temporal axis.")
        if start + length <= total:
            return condition[:, :, start : start + length]
        return condition[:, :, max(0, total - length) :]

    def _cfg_parallel_branch(
        self,
        *,
        request: DreamZeroRequest,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        use_cfg: bool,
    ) -> tuple[DreamZeroDiTRunner, torch.Tensor, torch.Tensor | None]:
        if self.cfg_size == 1 or not use_cfg:
            return self.cond_runner, context, clip_feature
        if self.cfg_size != 2:
            raise ValueError(
                "DreamZero CFG parallelism currently supports cfg_size=2; "
                f"got cfg_size={self.cfg_size}."
            )
        if request.uncond_context is None:
            raise ValueError("CFG parallelism requires uncond_context.")
        if self.cfg_rank == 0:
            return self.cond_runner, context, clip_feature
        if self.uncond_runner is None:
            raise RuntimeError("CFG requested but scheduler has no uncond runner.")
        uncond_context = request.uncond_context.to(self.device)
        uncond_clip = (
            request.uncond_clip_feature.to(self.device)
            if request.uncond_clip_feature is not None
            else None
        )
        return self.uncond_runner, uncond_context, uncond_clip

    def _combine_cfg_parallel(
        self,
        *,
        video_local: torch.Tensor,
        action_local: torch.Tensor,
        guidance_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg_size != 2:
            raise ValueError(
                "DreamZero CFG parallel combine requires cfg_size=2; "
                f"got cfg_size={self.cfg_size}."
            )
        # all_gather over cfg is rank ordered: rank 0 is cond, rank 1 is uncond.
        video_pair = P.all_gather(video_local.unsqueeze(0), axis="cfg", dim=0)
        action_pair = P.all_gather(action_local.unsqueeze(0), axis="cfg", dim=0)
        video_pred = video_pair[1] + guidance_scale * (video_pair[0] - video_pair[1])
        action_pred = action_pair[0]
        return video_pred, action_pred

    def _prefill_clean_cache(
        self,
        runner: DreamZeroDiTRunner,
        *,
        request: DreamZeroRequest,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
    ) -> None:
        if request.clean_video is None:
            return
        bsz = request.clean_video.shape[0]
        timestep = torch.zeros(
            bsz,
            request.clean_video.shape[2],
            dtype=torch.int64,
            device=request.clean_video.device,
        )
        self._run_runner(
            runner,
            video=request.clean_video,
            timestep=timestep,
            action=None,
            timestep_action=None,
            state=None,
            embodiment_id=None,
            context=context,
            clip_feature=clip_feature,
            y=self._slice_temporal_condition(
                request.y,
                start=0,
                length=request.clean_video.shape[2],
            ),
            request=request,
            update_kv_cache=True,
            use_crossattn_cache=False,
            update_crossattn_cache=True,
            seq_len=None,
            current_start_frame=0,
        )

    @torch.no_grad()
    def step(self, request: DreamZeroRequest) -> DreamZeroSchedulerOutput:
        if not self._ready:
            raise RuntimeError("call setup() before step().")
        self._reset_runners()

        video = request.video.to(self.device)
        action = request.action.to(self.device)
        state = request.state.to(self.device)
        embodiment_id = (
            request.embodiment_id.to(self.device)
            if request.embodiment_id is not None
            else None
        )
        context = request.context.to(self.device)
        clip_feature = (
            request.clip_feature.to(self.device)
            if request.clip_feature is not None
            else None
        )
        y = request.y.to(self.device) if request.y is not None else None
        denoise_y = self._slice_temporal_condition(
            y,
            start=request.current_start_frame,
            length=video.shape[2],
        )
        clean_video = (
            request.clean_video.to(self.device)
            if request.clean_video is not None
            else None
        )
        if clean_video is not request.clean_video:
            request = DreamZeroRequest(
                video=video,
                action=action,
                state=state,
                context=context,
                embodiment_id=embodiment_id,
                clip_feature=clip_feature,
                y=y,
                clean_video=clean_video,
                uncond_context=request.uncond_context,
                uncond_clip_feature=request.uncond_clip_feature,
                seq_len=request.seq_len,
                current_start_frame=request.current_start_frame,
                concat_first_frame_latent=request.concat_first_frame_latent,
                image_context_tokens=request.image_context_tokens,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                sigma_shift=request.sigma_shift,
                decouple_inference_noise=request.decouple_inference_noise,
                video_inference_final_noise=request.video_inference_final_noise,
                update_kv_cache=request.update_kv_cache,
                prefill_clean_cache=request.prefill_clean_cache,
            )

        guidance_scale = (
            self.cfg.cfg_scale
            if request.guidance_scale is None
            else float(request.guidance_scale)
        )
        use_cfg = request.uncond_context is not None and guidance_scale != 1.0
        if use_cfg and self.uncond_runner is None:
            raise RuntimeError("CFG requested but scheduler has no uncond runner.")
        cfg_parallel = self.cfg_size > 1
        branch_runner, branch_context, branch_clip = self._cfg_parallel_branch(
            request=request,
            context=context,
            clip_feature=clip_feature,
            use_cfg=use_cfg,
        )

        prefill_clean_cache = (
            request.current_start_frame == 0 and request.clean_video is not None
            if request.prefill_clean_cache is None
            else bool(request.prefill_clean_cache)
        )
        if prefill_clean_cache:
            if cfg_parallel:
                self._prefill_clean_cache(
                    branch_runner,
                    request=request,
                    context=branch_context,
                    clip_feature=branch_clip,
                )
            else:
                self._prefill_clean_cache(
                    self.cond_runner,
                    request=request,
                    context=context,
                    clip_feature=clip_feature,
                )
            if use_cfg and not cfg_parallel:
                assert self.uncond_runner is not None
                uncond_context = request.uncond_context.to(self.device)
                uncond_clip = (
                    request.uncond_clip_feature.to(self.device)
                    if request.uncond_clip_feature is not None
                    else None
                )
                self._prefill_clean_cache(
                    self.uncond_runner,
                    request=request,
                    context=uncond_context,
                    clip_feature=uncond_clip,
                )

        steps = request.num_inference_steps or self.cfg.num_inference_timesteps
        sigma_shift = request.sigma_shift or self.cfg.sigma_shift
        video_stepper = DreamZeroFlowStepper(shift=sigma_shift)
        action_stepper = DreamZeroFlowStepper(shift=sigma_shift)
        final_sigma = 0.0
        decouple = (
            self.cfg.decouple_inference_noise
            if request.decouple_inference_noise is None
            else bool(request.decouple_inference_noise)
        )
        if decouple:
            final_sigma = (
                self.cfg.video_inference_final_noise
                if request.video_inference_final_noise is None
                else float(request.video_inference_final_noise)
            )
        video_stepper.set_timesteps(
            steps,
            device=self.device,
            dtype=video.dtype,
            final_sigma=final_sigma,
        )
        action_stepper.set_timesteps(steps, device=self.device, dtype=action.dtype)
        assert video_stepper.timesteps is not None
        assert action_stepper.timesteps is not None

        last_video_pred: torch.Tensor | None = None
        last_action_pred: torch.Tensor | None = None
        for step_index, video_timestep in enumerate(video_stepper.timesteps):
            action_timestep = action_stepper.timesteps[step_index]
            timestep = torch.full(
                (video.shape[0], video.shape[2]),
                int(video_timestep.item()),
                dtype=torch.int64,
                device=self.device,
            )
            timestep_action = torch.full(
                (action.shape[0], action.shape[1]),
                int(action_timestep.item()),
                dtype=torch.int64,
                device=self.device,
            )
            if cfg_parallel:
                local = self._run_runner(
                    branch_runner,
                    video=video,
                    timestep=timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=branch_context,
                    clip_feature=branch_clip,
                    y=denoise_y,
                    request=request,
                    update_kv_cache=request.update_kv_cache,
                    use_crossattn_cache=True,
                    update_crossattn_cache=step_index == 0,
                )
                if local.action is None:
                    raise RuntimeError(
                        "DreamZero local CFG forward returned no action."
                    )
                if use_cfg:
                    video_pred, action_pred = self._combine_cfg_parallel(
                        video_local=local.video,
                        action_local=local.action,
                        guidance_scale=guidance_scale,
                    )
                else:
                    video_pred = local.video
                    action_pred = local.action
            else:
                cond = self._run_runner(
                    self.cond_runner,
                    video=video,
                    timestep=timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=context,
                    clip_feature=clip_feature,
                    y=denoise_y,
                    request=request,
                    update_kv_cache=request.update_kv_cache,
                    use_crossattn_cache=True,
                    update_crossattn_cache=step_index == 0,
                )
                if cond.action is None:
                    raise RuntimeError(
                        "DreamZero action denoise forward returned None."
                    )
                video_pred = cond.video
                action_pred = cond.action
            if use_cfg and not cfg_parallel:
                assert self.uncond_runner is not None
                uncond_context = request.uncond_context.to(self.device)
                uncond_clip = (
                    request.uncond_clip_feature.to(self.device)
                    if request.uncond_clip_feature is not None
                    else None
                )
                uncond = self._run_runner(
                    self.uncond_runner,
                    video=video,
                    timestep=timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=uncond_context,
                    clip_feature=uncond_clip,
                    y=denoise_y,
                    request=request,
                    update_kv_cache=request.update_kv_cache,
                    use_crossattn_cache=True,
                    update_crossattn_cache=step_index == 0,
                )
                if uncond.action is None:
                    raise RuntimeError("DreamZero uncond forward returned no action.")
                video_pred = uncond.video + guidance_scale * (cond.video - uncond.video)
                action_pred = cond.action
            if video_pred.shape != video.shape:
                raise ValueError(
                    "DreamZero video prediction shape must match the denoise "
                    f"sample shape; got pred={tuple(video_pred.shape)} and "
                    f"sample={tuple(video.shape)}. If the DiT uses first-frame "
                    "conditioning channels, pass y and "
                    "concat_first_frame_latent=True so only the generated latent "
                    "channels are stepped."
                )

            video = video_stepper.step(
                model_output=video_pred,
                sample=video,
                step_index=step_index,
                timestep=video_timestep,
            )
            action = action_stepper.step(
                model_output=action_pred,
                sample=action,
                step_index=step_index,
                timestep=action_timestep,
            )
            last_video_pred = video_pred
            last_action_pred = action_pred

        return DreamZeroSchedulerOutput(
            video=video,
            action=action,
            last_video_pred=last_video_pred,
            last_action_pred=last_action_pred,
            cond_kv_cache=self.cond_runner.kv_cache,
            uncond_kv_cache=self.uncond_runner.kv_cache if use_cfg else None,
        )


__all__ = [
    "DreamZeroFlowStepper",
    "DreamZeroRequest",
    "DreamZeroSchedulerOutput",
    "DreamZeroWS1Scheduler",
]
