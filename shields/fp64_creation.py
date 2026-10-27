"""FP64 -> FP32 creation and casting shield (master patch, shield 2 of 6).

Intel Arc has no usable fp64 compute path, so every tensor creation and cast
that would produce float64 is redirected to float32.  Contiguity is preserved so
the ESIMD attention contracts keep working.

Removal plan: drop when the SYCL/XPU backend ships a working fp64 path, or when
ComfyUI stops producing fp64 tensors on Intel GPUs.
"""

from __future__ import annotations

import logging

import torch

log = logging.getLogger("ComfyUI-OmniXPU")

MARKER = "__omnixpu_shield_fp64_creation__"
FP64_TYPES = (torch.float64, torch.double)
FACTORY_FUNCS = (
    "zeros", "ones", "empty", "full", "randn", "rand", "arange", "linspace",
    "zeros_like", "ones_like", "empty_like", "full_like",
)


def _ensure_fp32(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.dtype in FP64_TYPES:
        converted = tensor.to(torch.float32)
        return converted.contiguous() if tensor.is_contiguous() else converted
    return tensor


def _narrow(args, kwargs):
    args = tuple(torch.float32 if a in FP64_TYPES else a for a in args)
    if kwargs.get("dtype") in FP64_TYPES:
        kwargs["dtype"] = torch.float32
    return args, kwargs


def apply() -> tuple[bool, str | None]:
    if getattr(torch.Tensor.to, MARKER, False):
        return False, "already installed"

    orig_tensor_to = torch.Tensor.to

    def safe_tensor_to(self, *args, **kwargs):
        args, kwargs = _narrow(args, kwargs)
        return _ensure_fp32(orig_tensor_to(self, *args, **kwargs))

    safe_tensor_to.__wrapped__ = orig_tensor_to
    setattr(safe_tensor_to, MARKER, True)
    torch.Tensor.to = safe_tensor_to

    orig_from_numpy = torch.from_numpy
    orig_tensor = torch.tensor
    orig_as_tensor = torch.as_tensor

    def safe_from_numpy(ndarray):
        return _ensure_fp32(orig_from_numpy(ndarray))

    def safe_tensor(data, *args, **kwargs):
        args, kwargs = _narrow(args, kwargs)
        return _ensure_fp32(orig_tensor(data, *args, **kwargs))

    def safe_as_tensor(data, *args, **kwargs):
        args, kwargs = _narrow(args, kwargs)
        return _ensure_fp32(orig_as_tensor(data, *args, **kwargs))

    for name, fn in (
        ("from_numpy", safe_from_numpy),
        ("tensor", safe_tensor),
        ("as_tensor", safe_as_tensor),
    ):
        setattr(fn, MARKER, True)
        setattr(torch, name, fn)

    try:
        torch.Tensor.double = lambda self, *args, **kwargs: self.float()
    except Exception as exc:  # pragma: no cover - depends on the build
        log.warning("[OmniXPU shield] fp64_creation: Tensor.double not patchable (%s)", exc)

    for func_name in FACTORY_FUNCS:
        if not hasattr(torch, func_name):
            continue
        orig_func = getattr(torch, func_name)
        if getattr(orig_func, MARKER, False) or getattr(orig_func, "_is_xpu_patched", False):
            continue

        def make_patched_factory(fn):
            def patched_factory(*args, **kwargs):
                args, kwargs = _narrow(args, kwargs)
                return _ensure_fp32(fn(*args, **kwargs))

            setattr(patched_factory, MARKER, True)
            patched_factory._is_xpu_patched = True
            return patched_factory

        setattr(torch, func_name, make_patched_factory(orig_func))

    log.info("[OmniXPU shield] fp64_creation: creation points and .to() redirect fp64 to fp32")
    return True, None
