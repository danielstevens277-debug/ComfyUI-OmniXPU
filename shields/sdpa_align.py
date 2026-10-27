"""SDPA dtype auto-align + optional Inductor Triton fast path (shield 4 of 6).

Master-patch origin.  The Q/K/V and mask auto-align is the part that fixes
"Expected query, key, and value to have the same dtype".  The Triton fast path
is kept as a separate switch: torch.compile works on Windows 11 with Torch 2.14,
but OmniXPU's attention adapter owns the validated CUTE/ESIMD routes, so the
fast path must never shadow an adapter decision.

Removal plan: drop when ComfyUI/Kitchen align Q/K/V dtypes at the call site.
"""

from __future__ import annotations

import logging
import sys

import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-OmniXPU")

MARKER = "__omnixpu_shield_sdpa_align__"
COMPILER_MARKER = "__omnixpu_shield_sdpa_compiler__"

_warned_sdpa_failure = False


def _fastpath_enabled() -> bool:
    pkg = __package__.rsplit(".", 1)[0] if __package__ else "ComfyUI-OmniXPU"
    cfg = sys.modules.get(f"{pkg}.config")
    return bool(getattr(cfg, "shield_sdpa_fastpath", True))


def apply() -> tuple[bool, str | None]:
    global _warned_sdpa_failure

    if getattr(F.scaled_dot_product_attention, MARKER, False):
        return False, "already installed"
    if getattr(F.scaled_dot_product_attention, "_is_omnixpu_hooked", False):
        return False, "SDPA already hooked by another component"

    fastpath = _fastpath_enabled()
    compiled_sdpa = torch.compile(F.scaled_dot_product_attention, backend="inductor", dynamic=True) \
        if fastpath else None

    orig_sdpa = F.scaled_dot_product_attention

    def fast_xpu_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                      scale=None, enable_gqa=False, **kwargs):
        global _warned_sdpa_failure

        # Auto-align mismatched Q, K, V dtypes (e.g. float32 query with bfloat16 key/value)
        if not (query.dtype == key.dtype == value.dtype):
            if key.dtype in (torch.bfloat16, torch.float16):
                target_dtype = key.dtype
            elif value.dtype in (torch.bfloat16, torch.float16):
                target_dtype = value.dtype
            elif query.dtype in (torch.bfloat16, torch.float16):
                target_dtype = query.dtype
            else:
                target_dtype = query.dtype

            if query.dtype != target_dtype:
                query = query.to(target_dtype)
            if key.dtype != target_dtype:
                key = key.to(target_dtype)
            if value.dtype != target_dtype:
                value = value.to(target_dtype)

        # Auto-align attention mask dtype and device
        if attn_mask is not None and isinstance(attn_mask, torch.Tensor):
            if attn_mask.is_floating_point() and attn_mask.dtype != query.dtype:
                attn_mask = attn_mask.to(query.dtype)
            if attn_mask.device != query.device:
                attn_mask = attn_mask.to(query.device)

        # Accelerated Inductor Triton path on XMX
        if (
            compiled_sdpa is not None
            and query.device.type == "xpu"
            and query.dtype in (torch.float16, torch.bfloat16)
            and attn_mask is None
            and not is_causal
            and not enable_gqa
            and dropout_p == 0.0
        ):
            try:
                return compiled_sdpa(query, key, value, scale=scale)
            except Exception as exc:
                if not _warned_sdpa_failure:
                    log.warning("[OmniXPU shield] sdpa_align: Triton SDPA falling back to eager (%s)", exc)
                    _warned_sdpa_failure = True

        try:
            return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                             is_causal=is_causal, scale=scale, enable_gqa=enable_gqa, **kwargs)
        except TypeError:
            return orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                             is_causal=is_causal, scale=scale, **kwargs)

    setattr(fast_xpu_sdpa, MARKER, True)
    setattr(fast_xpu_sdpa, COMPILER_MARKER, fastpath)
    fast_xpu_sdpa.__wrapped__ = orig_sdpa
    F.scaled_dot_product_attention = fast_xpu_sdpa
    torch.nn.functional.scaled_dot_product_attention = fast_xpu_sdpa

    log.info(
        "[OmniXPU shield] sdpa_align: dtype auto-align active (Triton fast path %s)",
        "armed" if fastpath else "off",
    )
    return True, None
