"""SYCL fp64 aspect shield for torch.quantile (master patch, shield 3 of 6).

torch.quantile/nanquantile fault on Arc when the SYCL build advertises fp64 it
cannot execute (this is what breaks dynthresh).  The probe decides per install:
if the quantile path is healthy, the shield registers nothing.

Removal plan: drop when the XPU backend's fp64 aspect probe is honest.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

MARKER = "__omnixpu_shield_sycl_quantile__"


def _xpu_has_fp64(orig_quantile) -> bool:
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        return True
    try:
        test = torch.tensor([1.0, 2.0], dtype=torch.float32, device="xpu")
        orig_quantile(test, 0.5)
        return True
    except Exception:
        return False


def apply() -> tuple[bool, str | None]:
    if getattr(torch.quantile, MARKER, False):
        return False, "already installed"

    # Re-run the probe through the *original* quantile so it describes the real
    # hardware capability, not the state left by the fp64 creation shield.
    orig_quantile = torch.quantile
    orig_nanquantile = getattr(torch, "nanquantile", None)

    if _xpu_has_fp64(orig_quantile):
        return False, "XPU fp64 quantile probe passed; nothing to shield"

    def helper(fn, input, q, dim=None, keepdim=False, *args, **kwargs):
        if input.device.type == "xpu":
            orig_device, orig_dtype = input.device, input.dtype
            out_target = kwargs.pop("out", None)
            q_cpu = q.to(device="cpu", dtype=torch.float32) if isinstance(q, torch.Tensor) else q
            input_cpu = input.to(device="cpu", dtype=torch.float32)
            res = fn(input_cpu, q_cpu, dim=dim, keepdim=keepdim, *args, **kwargs)
            res_xpu = res.to(device=orig_device, dtype=orig_dtype)
            if out_target is not None:
                out_target.copy_(res_xpu)
                return out_target
            return res_xpu
        return fn(input, q, dim=dim, keepdim=keepdim, *args, **kwargs)

    def safe_quantile(input, q, dim=None, keepdim=False, *args, **kwargs):
        return helper(orig_quantile, input, q, dim=dim, keepdim=keepdim, *args, **kwargs)

    setattr(safe_quantile, MARKER, True)
    torch.quantile = safe_quantile
    torch.Tensor.quantile = lambda self, *args, **kwargs: safe_quantile(self, *args, **kwargs)

    if orig_nanquantile is not None:
        def safe_nanquantile(input, q, dim=None, keepdim=False, *args, **kwargs):
            return helper(orig_nanquantile, input, q, dim=dim, keepdim=keepdim, *args, **kwargs)

        setattr(safe_nanquantile, MARKER, True)
        torch.nanquantile = safe_nanquantile
        torch.Tensor.nanquantile = lambda self, *args, **kwargs: safe_nanquantile(
            self, *args, **kwargs
        )

    log.info("[OmniXPU shield] sycl_quantile: torch.quantile secured from the SYCL fp64 fault")
    return True, None
