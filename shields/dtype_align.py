"""Linear / conv / layer_norm dtype auto-align (master patch, shield 6 of 6).

Fixes mat1/mat2 BF16-vs-FP32 mismatches at the functional level.  Only fires on
an actual mismatch, so it composes with the INT8 fast paths (they route matched
dtypes through Kitchen untouched).

Removal plan: drop when Kitchen's XPU backends validate dtypes at dispatch.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-OmniXPU")

MARKER = "__omnixpu_shield_dtype_align__"


def apply() -> tuple[bool, str | None]:
    installed = []
    orig_linear = F.linear
    orig_conv1d = F.conv1d
    orig_conv2d = F.conv2d
    orig_layernorm = getattr(F, "layer_norm", None)

    def safe_linear(input, weight, bias=None):
        if input.dtype != weight.dtype and weight.is_floating_point():
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype and weight.is_floating_point():
            bias = bias.to(weight.dtype)
        return orig_linear(input, weight, bias)

    def safe_conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dtype != weight.dtype and weight.is_floating_point():
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype and weight.is_floating_point():
            bias = bias.to(weight.dtype)
        return orig_conv1d(input, weight, bias=bias, stride=stride, padding=padding,
                           dilation=dilation, groups=groups)

    def safe_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if input.dtype != weight.dtype and weight.is_floating_point():
            input = input.to(weight.dtype)
        if bias is not None and bias.dtype != weight.dtype and weight.is_floating_point():
            bias = bias.to(weight.dtype)
        return orig_conv2d(input, weight, bias=bias, stride=stride, padding=padding,
                           dilation=dilation, groups=groups)

    def safe_layernorm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        if weight is not None and input.dtype != weight.dtype and weight.is_floating_point():
            input = input.to(weight.dtype)
        if bias is not None and weight is not None and bias.dtype != weight.dtype \
                and weight.is_floating_point():
            bias = bias.to(weight.dtype)
        return orig_layernorm(input, normalized_shape, weight=weight, bias=bias, eps=eps)

    for name, holder, fn in (
        ("linear", F, safe_linear),
        ("conv1d", F, safe_conv1d),
        ("conv2d", F, safe_conv2d),
        ("layer_norm", F, safe_layernorm),
    ):
        current = getattr(holder, name, None)
        if current is None or getattr(current, MARKER, False):
            continue
        setattr(fn, MARKER, True)
        fn.__wrapped__ = current
        setattr(holder, name, fn)
        installed.append(f"F.{name}")

    if not installed:
        return False, "already installed"

    log.info("[OmniXPU shield] dtype_align: active on %s", ", ".join(installed))
    return True, None
