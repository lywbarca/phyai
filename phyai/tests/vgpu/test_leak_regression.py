"""Regression test for flashinfer green-ctx driver memory growth.

Older flashinfer / CUDA driver stacks leaked driver-level memory on every
``split_device_green_ctx`` call that was not recovered by
``cuStreamDestroy + cuGreenCtxDestroy``, ``empty_cache``, or
``gc.collect``. Newer stacks may have fixed or reduced that growth. This
test keeps the old leak visible without making fixed environments fail.

We only run a handful of iterations (so the test stays fast) and assert
either:
  - growth is observable above a small tolerance (known upstream issue),
    OR
  - growth is below the tolerance (fixed or negligible on this stack).

The assertion is structured as ``xfail``-style: we ``pytest.xfail`` when
large growth is detected and pass when growth is negligible.
"""

from __future__ import annotations

import gc
import subprocess

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="phyai.vgpu requires CUDA",
)


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


def _smi_mem_used_mib(device_idx: int = 0) -> int:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={device_idx}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    return int(out)


def test_flashinfer_split_leaks_or_upstream_fixed():
    if not _flashinfer_available():
        pytest.skip("flashinfer required")

    # nvidia-smi is the only window into driver-level memory; bail if it's
    # missing (e.g. inside a container without the binary).
    try:
        subprocess.check_output(["nvidia-smi", "--version"])
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("nvidia-smi not available")

    # Prime the primary context.
    _ = torch.zeros(1, device="cuda:0")
    torch.cuda.synchronize()

    base_smi = _smi_mem_used_mib(0)

    iters = 5
    from flashinfer.green_ctx import split_device_green_ctx

    for _ in range(iters):
        streams, resources = split_device_green_ctx(
            torch.device("cuda:0"),
            2,
            16,
        )
        del streams, resources
        gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    end_smi = _smi_mem_used_mib(0)
    delta = end_smi - base_smi

    threshold_mib = 32
    if delta >= threshold_mib:
        # Leak still present on this stack. Keep it visible without
        # failing otherwise healthy test runs.
        pytest.xfail(
            f"known flashinfer leak: 5 iter split_device_green_ctx "
            f"caused {delta} MiB driver-side growth. "
            f"vGPU must remain long-lived."
        )

    assert delta < threshold_mib
