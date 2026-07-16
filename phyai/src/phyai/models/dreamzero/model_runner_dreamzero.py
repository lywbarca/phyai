"""DreamZero DiT runner.

The runner owns per-request cache tensors and calls the stateless DiT model for
one forward pass. Scheduler code decides when to reset or reuse this runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from phyai.cache import KVCachePool
from phyai.models.dreamzero.modeling_dreamzero import DreamZeroDiT
from phyai.runtime.model_runner import ModelRunner


@dataclass
class DreamZeroDiTForwardBatch:
    x: torch.Tensor
    timestep: torch.Tensor
    context: torch.Tensor
    seq_len: int | None = None
    current_start_frame: int = 0
    y: torch.Tensor | None = None
    clip_feature: torch.Tensor | None = None
    action: torch.Tensor | None = None
    timestep_action: torch.Tensor | None = None
    state: torch.Tensor | None = None
    embodiment_id: torch.Tensor | None = None
    clean_x: torch.Tensor | None = None
    aug_t: torch.Tensor | None = None
    concat_first_frame_latent: bool = False
    image_context_tokens: int = 257
    use_kv_cache: bool = True
    update_kv_cache: bool = True
    use_crossattn_cache: bool = False
    update_crossattn_cache: bool = True


@dataclass
class DreamZeroDiTForwardOutput:
    video: torch.Tensor
    action: torch.Tensor | None
    kv_cache: list[torch.Tensor | None]
    crossattn_cache: list[torch.Tensor | None]


class DreamZeroLayerKVCache:
    """Fixed-capacity, per-layer DreamZero KV cache backed by KVCachePool.

    The current DreamZero modeling code still accepts dense per-layer tensors
    shaped ``(2, B, S, H, D)``. This class keeps the runner-owned persistent
    storage in PhyAI's paged-KV layout and materializes those dense views at
    the modeling boundary. It is intentionally runner-local runtime state.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if max_seq_len < 0:
            raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}.")
        if max_seq_len == 0:
            max_seq_len = 1
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.max_seq_len = int(max_seq_len)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.seq_len = 0
        self.pool = KVCachePool(
            num_layers=self.num_layers,
            num_slots=self.batch_size * self.max_seq_len,
            num_kv_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.slot_indices = torch.arange(
            self.batch_size * self.max_seq_len,
            dtype=torch.int64,
            device=self.device,
        ).reshape(self.batch_size, self.max_seq_len)

    def reset(self) -> None:
        self.seq_len = 0

    def empty_cache_list(self) -> list[torch.Tensor]:
        shape = (
            2,
            self.batch_size,
            0,
            self.num_heads,
            self.head_dim,
        )
        return [
            torch.empty(shape, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]

    def as_cache_list(self) -> list[torch.Tensor]:
        if self.seq_len == 0:
            return self.empty_cache_list()
        slots = self.slot_indices[:, : self.seq_len].reshape(-1)
        caches: list[torch.Tensor] = []
        for layer_id in range(self.num_layers):
            k, v = self.pool.gather_kv(layer_id, slots)
            k = k.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
            v = v.reshape(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
            caches.append(torch.stack([k, v], dim=0))
        return caches

    def _validate_updated(self, caches: list[torch.Tensor | None]) -> int:
        if len(caches) != self.num_layers:
            raise ValueError(
                f"cache list length {len(caches)} does not match "
                f"num_layers={self.num_layers}."
            )
        seq_len: int | None = None
        for layer_id, cache in enumerate(caches):
            if cache is None:
                raise ValueError(f"updated cache for layer {layer_id} is None.")
            expected_prefix = (2, self.batch_size)
            if cache.dim() != 5 or tuple(cache.shape[:2]) != expected_prefix:
                raise ValueError(
                    f"updated cache must start with {expected_prefix}; got "
                    f"{tuple(cache.shape)}."
                )
            if cache.shape[3] != self.num_heads or cache.shape[4] != self.head_dim:
                raise ValueError(
                    f"updated cache head shape must be "
                    f"({self.num_heads}, {self.head_dim}); got "
                    f"({cache.shape[3]}, {cache.shape[4]})."
                )
            layer_seq_len = int(cache.shape[2])
            if seq_len is None:
                seq_len = layer_seq_len
            elif seq_len != layer_seq_len:
                raise ValueError("all layer caches must have the same sequence length.")
        return int(seq_len or 0)

    def commit(self, caches: list[torch.Tensor | None]) -> None:
        seq_len = self._validate_updated(caches)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"updated cache seq_len={seq_len} exceeds capacity "
                f"max_seq_len={self.max_seq_len}."
            )
        if seq_len == 0:
            self.seq_len = 0
            return
        slots = self.slot_indices[:, :seq_len].reshape(-1)
        for layer_id, cache in enumerate(caches):
            assert cache is not None
            k = (
                cache[0]
                .to(dtype=self.dtype, device=self.device)
                .reshape(
                    self.batch_size * seq_len,
                    self.num_heads,
                    self.head_dim,
                )
            )
            v = (
                cache[1]
                .to(dtype=self.dtype, device=self.device)
                .reshape(
                    self.batch_size * seq_len,
                    self.num_heads,
                    self.head_dim,
                )
            )
            self.pool.write_kv(layer_id, slots, k, v)
        self.seq_len = seq_len


class DreamZeroDiTRunner(ModelRunner):
    """Owns DreamZero DiT KV caches for one scheduler branch."""

    def __init__(
        self,
        model: DreamZeroDiT,
        *,
        device: torch.device | str | None = None,
        max_kv_cache_tokens: int | None = None,
        max_crossattn_cache_tokens: int | None = None,
    ) -> None:
        self.model = model
        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device)
        self.max_kv_cache_tokens = max_kv_cache_tokens
        self.max_crossattn_cache_tokens = max_crossattn_cache_tokens
        self._kv_cache: DreamZeroLayerKVCache | None = None
        self._crossattn_cache: DreamZeroLayerKVCache | None = None

    def setup(self) -> None:
        return None

    def reset(self) -> None:
        if self._kv_cache is not None:
            self._kv_cache.reset()
        if self._crossattn_cache is not None:
            self._crossattn_cache.reset()

    @property
    def kv_cache(self) -> list[torch.Tensor | None]:
        if self._kv_cache is None or self._kv_cache.seq_len == 0:
            return []
        return self._kv_cache.as_cache_list()

    @property
    def crossattn_cache(self) -> list[torch.Tensor | None]:
        if self._crossattn_cache is None or self._crossattn_cache.seq_len == 0:
            return []
        return self._crossattn_cache.as_cache_list()

    def create_kv_caches(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
        seq_len: int = 0,
    ) -> list[torch.Tensor]:
        if device is None:
            device = self.device
        first_attn = self.model.blocks[0].self_attn
        shape = (
            2,
            batch_size,
            seq_len,
            first_attn.num_local_heads,
            first_attn.head_dim,
        )
        return [
            torch.empty(shape, dtype=dtype, device=device)
            for _ in range(len(self.model.blocks))
        ]

    def _cache_shape(self) -> tuple[int, int, int]:
        first_attn = self.model.blocks[0].self_attn
        return len(self.model.blocks), first_attn.num_local_heads, first_attn.head_dim

    def _new_cache(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        capacity: int,
    ) -> DreamZeroLayerKVCache:
        num_layers, num_heads, head_dim = self._cache_shape()
        return DreamZeroLayerKVCache(
            num_layers=num_layers,
            batch_size=batch_size,
            max_seq_len=capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )

    def _ensure_kv_cache(
        self, batch: DreamZeroDiTForwardBatch
    ) -> DreamZeroLayerKVCache:
        capacity = self.max_kv_cache_tokens or 1
        if (
            self._kv_cache is not None
            and self._kv_cache.batch_size == batch.x.shape[0]
            and self._kv_cache.dtype == batch.x.dtype
            and self._kv_cache.device == batch.x.device
            and self._kv_cache.max_seq_len >= capacity
        ):
            return self._kv_cache
        self._kv_cache = self._new_cache(
            batch_size=batch.x.shape[0],
            dtype=batch.x.dtype,
            device=batch.x.device,
            capacity=capacity,
        )
        return self._kv_cache

    def _ensure_crossattn_cache(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        capacity: int,
    ) -> DreamZeroLayerKVCache:
        target_capacity = self.max_crossattn_cache_tokens or capacity
        if (
            self._crossattn_cache is not None
            and self._crossattn_cache.batch_size == batch_size
            and self._crossattn_cache.dtype == dtype
            and self._crossattn_cache.device == device
            and self._crossattn_cache.max_seq_len >= target_capacity
        ):
            return self._crossattn_cache
        self._crossattn_cache = self._new_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            capacity=target_capacity,
        )
        return self._crossattn_cache

    def _grow_kv_cache(
        self, updated_cache: list[torch.Tensor | None]
    ) -> DreamZeroLayerKVCache:
        first = next(cache for cache in updated_cache if cache is not None)
        seq_len = int(first.shape[2])
        current_capacity = self._kv_cache.max_seq_len if self._kv_cache else 0
        if self.max_kv_cache_tokens is not None:
            capacity = max(current_capacity, self.max_kv_cache_tokens, 1)
        else:
            capacity = max(seq_len, current_capacity, 1)
        if self._kv_cache is None or capacity > current_capacity:
            self._kv_cache = self._new_cache(
                batch_size=int(first.shape[1]),
                dtype=first.dtype,
                device=first.device,
                capacity=capacity,
            )
        return self._kv_cache

    def _grow_crossattn_cache(
        self, updated_cache: list[torch.Tensor | None]
    ) -> DreamZeroLayerKVCache:
        first = next(cache for cache in updated_cache if cache is not None)
        seq_len = int(first.shape[2])
        current_capacity = (
            self._crossattn_cache.max_seq_len if self._crossattn_cache else 0
        )
        if self.max_crossattn_cache_tokens is not None:
            capacity = max(current_capacity, self.max_crossattn_cache_tokens, 1)
        else:
            capacity = max(seq_len, current_capacity, 1)
        if self._crossattn_cache is None or capacity > current_capacity:
            self._crossattn_cache = self._new_cache(
                batch_size=int(first.shape[1]),
                dtype=first.dtype,
                device=first.device,
                capacity=capacity,
            )
        return self._crossattn_cache

    def _input_kv_cache(
        self, batch: DreamZeroDiTForwardBatch
    ) -> list[torch.Tensor | None] | None:
        if not batch.use_kv_cache:
            return None
        cache = self._ensure_kv_cache(batch)
        if cache.seq_len == 0:
            return cache.empty_cache_list()
        return cache.as_cache_list()

    def _input_crossattn_cache(
        self, batch: DreamZeroDiTForwardBatch
    ) -> list[torch.Tensor | None] | None:
        del batch
        if self._crossattn_cache is None or self._crossattn_cache.seq_len == 0:
            return None
        return self._crossattn_cache.as_cache_list()

    @torch.no_grad()
    def forward(self, batch: DreamZeroDiTForwardBatch) -> DreamZeroDiTForwardOutput:
        kv_cache = self._input_kv_cache(batch)
        crossattn_cache = self._input_crossattn_cache(batch)
        video, action, updated_kv_cache, updated_crossattn_cache = self.model(
            batch.x,
            batch.timestep,
            batch.context,
            seq_len=batch.seq_len,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start_frame=batch.current_start_frame,
            y=batch.y,
            clip_feature=batch.clip_feature,
            action=batch.action,
            timestep_action=batch.timestep_action,
            state=batch.state,
            embodiment_id=batch.embodiment_id,
            clean_x=batch.clean_x,
            aug_t=batch.aug_t,
            concat_first_frame_latent=batch.concat_first_frame_latent,
            image_context_tokens=batch.image_context_tokens,
            use_crossattn_cache=batch.use_crossattn_cache,
        )

        if batch.update_kv_cache:
            if not any(cache is not None for cache in updated_kv_cache):
                raise RuntimeError("model did not return KV cache to update.")
            self._grow_kv_cache(updated_kv_cache).commit(updated_kv_cache)
        if batch.update_crossattn_cache:
            if any(cache is not None for cache in updated_crossattn_cache):
                self._grow_crossattn_cache(updated_crossattn_cache).commit(
                    updated_crossattn_cache
                )
        return DreamZeroDiTForwardOutput(
            video=video,
            action=action,
            kv_cache=self.kv_cache if batch.update_kv_cache else updated_kv_cache,
            crossattn_cache=(
                self.crossattn_cache
                if batch.update_crossattn_cache
                else updated_crossattn_cache
            ),
        )


__all__ = [
    "DreamZeroLayerKVCache",
    "DreamZeroDiTForwardBatch",
    "DreamZeroDiTForwardOutput",
    "DreamZeroDiTRunner",
]
