"""TorchInductor cache helpers for DreamZero numerical parity."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import torch


OFFICIAL_CLIP_INDUCTOR_RUNTIME = {
    "device_name": "NVIDIA Thor",
    "compute_capability": (11, 0),
    "torch_version": "2.11.0",
    "cuda_version": "13.0",
}
OFFICIAL_CLIP_INDUCTOR_TORCH_VERSIONS = ("2.10.0a0", "2.11.0")
OFFICIAL_CLIP_INDUCTOR_CUDA_VERSIONS = ("13.0", "13.1")
OFFICIAL_CLIP_EMBEDDED_CACHE_ROOT = Path("/tmp/torchinductor_root")

OFFICIAL_CLIP_INDUCTOR_CONFIGS = {
    "bk/5959678433f8b5031f12a364f84d6c48b8c21f089b6bc9e7bd21bf65cdb63f04.best_config": (
        b'{"XBLOCK": 8, "num_warps": 2, "num_stages": 1, '
        b'"configs_hash": "ec7c98dadee3d6e3126c4251be209c7153a1ad4c7909275ea44497e068a40dbb", '
        b'"found_by_coordesc": false, "time_taken_ms": 198, '
        b'"triton_cache_hash": "Y3VLPBHR73K47MTJCPXHILRKEOCBRB3NF33M3BLQG3B3RP5TJ6RA"}'
    ),
    "dt/ad24d4b2497131f065466b72605f573f8c2989f12e2cec26524b0ab6b8bae479.best_config": (
        b'{"XBLOCK": 2, "YBLOCK": 16, "R0_BLOCK": 16, "num_warps": 2, '
        b'"num_stages": 1, '
        b'"configs_hash": "d0b029b54c90fb8edb9ca775291caabac627e3654aae07293a9619b33336adc7", '
        b'"found_by_coordesc": false, "time_taken_ms": 228, '
        b'"triton_cache_hash": "LOD7HAJBAK42JDL67X37Y2CR4SUQYB3UY24ARAIC6COFCIDBNKAQ"}'
    ),
    "np/abd623330d33327f8bd5572e2d406ab99a17fc0e6fcf654b10b3353d5828509f.best_config": (
        b'{"XBLOCK": 1024, "num_warps": 4, "num_stages": 1, '
        b'"configs_hash": "3ca5c3e34d35093f3c9ab2829a9faeebad5e61c4ca13d5ed6053d7b71ce60d5a", '
        b'"found_by_coordesc": false, "time_taken_ms": 96, '
        b'"triton_cache_hash": "EZS6X7WZQ2MTXMVPMEUFFFD6BCVKSMLBLWUXZM52TGA66HOT5QEA"}'
    ),
    "rf/006f93f8634c360b06d872fb310ff85e3bf305c6c11931c9e99dd7c5100242d2.best_config": (
        b'{"XBLOCK": 2, "R0_BLOCK": 2048, "num_warps": 16, '
        b'"num_stages": 1, '
        b'"configs_hash": "ae9be30f9bdf905c664a88230c938247dc015fe8248083f75306723e5aa33083", '
        b'"found_by_coordesc": false, "time_taken_ms": 176, '
        b'"triton_cache_hash": "CTRAXO22YZPEKSFN4ROXOM6H6DBBELYHNROM5FLOLIJK55NXGSUQ"}'
    ),
    "vo/3c566572983417b4cbe3cfbbd8626a8118569b4aa00bb74e5f9d54930709f4e2.best_config": (
        b'{"XBLOCK": 1024, "num_warps": 4, "num_stages": 1, '
        b'"configs_hash": "3ca5c3e34d35093f3c9ab2829a9faeebad5e61c4ca13d5ed6053d7b71ce60d5a", '
        b'"found_by_coordesc": false, "time_taken_ms": 96, '
        b'"triton_cache_hash": "SRBL2GWHFFW4G3PBOFZVAYFM4L7RQYEPTSQMB7JG56JGHAA2W5BQ"}'
    ),
}

OFFICIAL_CLIP_INDUCTOR_CONFIG_SHA256 = {
    "bk/5959678433f8b5031f12a364f84d6c48b8c21f089b6bc9e7bd21bf65cdb63f04.best_config": "18d2aa31b05fb802f5c4f87cf575842ce5a45f945e84b6edd40bba3723d4836e",
    "dt/ad24d4b2497131f065466b72605f573f8c2989f12e2cec26524b0ab6b8bae479.best_config": "27100ce6ed68887fd9581aced8ec409372df3364cf2d9cd7ae84f5dc05911837",
    "np/abd623330d33327f8bd5572e2d406ab99a17fc0e6fcf654b10b3353d5828509f.best_config": "c5e1112f52454bdbc57b798f00c6ea706a945f097ddfedfdae81c176ac820e87",
    "rf/006f93f8634c360b06d872fb310ff85e3bf305c6c11931c9e99dd7c5100242d2.best_config": "e1d4c86d35567c0ef3f255e669ea9f4054e9a8059ce113cfc21c3aafe334c4f5",
    "vo/3c566572983417b4cbe3cfbbd8626a8118569b4aa00bb74e5f9d54930709f4e2.best_config": "45175be240cc622d50f47cacc4f25a847f48ad8017f4e06501e34a589d73577c",
}


def is_official_clip_inductor_runtime(device: torch.device | str) -> bool:
    """Return whether the pinned official CLIP cache is valid on this runtime."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        torch.__version__.split("+", 1)[0] in OFFICIAL_CLIP_INDUCTOR_TORCH_VERSIONS
        and torch.version.cuda in OFFICIAL_CLIP_INDUCTOR_CUDA_VERSIONS
        and torch.cuda.get_device_name(device_index)
        == OFFICIAL_CLIP_INDUCTOR_RUNTIME["device_name"]
        and torch.cuda.get_device_capability(device_index)
        == OFFICIAL_CLIP_INDUCTOR_RUNTIME["compute_capability"]
    )


def seed_official_clip_inductor_configs(
    cache_root: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Atomically install the official Thor CLIP autotune selections."""
    if cache_root is None:
        from torch._inductor.runtime.runtime_utils import cache_dir

        cache_root = cache_dir()
    root = Path(cache_root)
    installed = []
    for relative_path, payload in OFFICIAL_CLIP_INDUCTOR_CONFIGS.items():
        expected = OFFICIAL_CLIP_INDUCTOR_CONFIG_SHA256[relative_path]
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"invalid bundled DreamZero CLIP config {relative_path}: "
                f"expected sha256={expected}, got {actual}"
            )
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if (
            target.exists()
            and hashlib.sha256(target.read_bytes()).hexdigest() == expected
        ):
            installed.append(target)
            continue
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        installed.append(target)
    return tuple(installed)


def seed_official_clip_runtime_configs() -> tuple[Path, ...]:
    """Install configs in active and embedded cache roots used by copied FX graphs."""
    from torch._inductor.runtime.runtime_utils import cache_dir

    roots = (Path(cache_dir()), OFFICIAL_CLIP_EMBEDDED_CACHE_ROOT)
    installed = []
    seen = set()
    for root in roots:
        normalized = root.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        installed.extend(seed_official_clip_inductor_configs(root))
    return tuple(installed)


__all__ = [
    "OFFICIAL_CLIP_INDUCTOR_CONFIGS",
    "OFFICIAL_CLIP_INDUCTOR_CONFIG_SHA256",
    "OFFICIAL_CLIP_INDUCTOR_RUNTIME",
    "OFFICIAL_CLIP_INDUCTOR_CUDA_VERSIONS",
    "OFFICIAL_CLIP_INDUCTOR_TORCH_VERSIONS",
    "OFFICIAL_CLIP_EMBEDDED_CACHE_ROOT",
    "is_official_clip_inductor_runtime",
    "seed_official_clip_inductor_configs",
    "seed_official_clip_runtime_configs",
]
