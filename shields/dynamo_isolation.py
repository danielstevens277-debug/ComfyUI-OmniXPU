"""Dynamo isolation (master patch, shield 5 of 6) — runs AFTER the adapters.

Master-patch origin, one deliberate change: the kernel wheel ships its own
Dynamo dispatch boundaries (``omni_xpu_kernel._compile_ops``,
``torch.ops.omni_xpu.*``) so the native kernels stay opaque custom operators
inside a compiled graph.  Forcing ``disallow_in_graph`` on a namespace that
already provides those boundaries reintroduces the graph break that trips
Dynamo's transformer_options resume path (``KeyError: 'total_blocks'`` on
Krea2), so the disallow is only applied when the wheel provides no boundaries.
comfy_kitchen, which has no boundary mechanism, keeps the master patch's
behaviour unchanged.

Removal plan: retire per-namespace once Kitchen registers proper custom-op
boundaries upstream.
"""

from __future__ import annotations

import logging

import torch
import torch._dynamo

log = logging.getLogger("ComfyUI-OmniXPU")

# namespace -> whether the master patch would disallow it unconditionally
DISALLOW_NAMESPACES = ("comfy_kitchen", "omni_xpu", "omni_xpu_kernel")
DISABLE_MODULE_FUNCS = {
    "comfy_kitchen": ("rms_rope_split_half_", "apply_rope1", "_apply_rope1", "int8_linear"),
    "comfy.quant_ops.ck": ("rms_rope_split_half_", "apply_rope1", "_apply_rope1", "int8_linear"),
}
KERNEL_SUBMODULES = ("norm", "sdp", "rotary", "linear_fp8")

DYNAMO_CONFIG = {
    "cache_size_limit": 512,
    "accumulated_cache_size_limit": 2048,
    "guard_nn_modules": False,
    "suppress_errors": True,
    "capture_scalar_outputs": False,
    "specialize_int": True,
}

_installed = False


def _has_compile_boundaries(mod) -> bool:
    return hasattr(mod, "_compile_ops")


def apply() -> tuple[bool, str | None]:
    global _installed
    import sys

    if _installed:
        return False, "already installed"

    actions = []

    for ns in DISALLOW_NAMESPACES:
        mod = getattr(torch.ops, ns, None)
        if mod is None:
            continue
        if _has_compile_boundaries(mod):
            actions.append(f"{ns}: compile boundaries provided by the wheel, left traceable")
            continue
        count = 0
        for op_name in dir(mod):
            if op_name.startswith("_"):
                continue
            try:
                torch._dynamo.disallow_in_graph(getattr(mod, op_name))
                count += 1
            except Exception:
                pass
        if count:
            actions.append(f"{ns}: {count} ops disallowed")

    for mod_path, funcs in DISABLE_MODULE_FUNCS.items():
        mod = sys.modules.get(mod_path)
        if mod is None or _has_compile_boundaries(mod):
            continue
        for fn_name in funcs:
            fn = getattr(mod, fn_name, None)
            if callable(fn) and not getattr(fn, "_omnixpu_disabled", False):
                try:
                    disabled = torch._dynamo.disable(fn)
                    setattr(disabled, "_omnixpu_disabled", True)
                    setattr(mod, fn_name, disabled)
                except Exception:
                    pass

    if _has_compile_boundaries(sys.modules.get("omni_xpu_kernel")):
        actions.append("omni_xpu_kernel submodules: left traceable (wheel boundaries)")
    else:
        kernel = sys.modules.get("omni_xpu_kernel")
        if kernel is not None:
            for submod_name in KERNEL_SUBMODULES:
                sub = getattr(kernel, submod_name, None)
                if sub is None:
                    continue
                for fn in dir(sub):
                    if fn.startswith("_") or not callable(getattr(sub, fn)):
                        continue
                    try:
                        setattr(sub, fn, torch._dynamo.disable(getattr(sub, fn)))
                    except Exception:
                        pass

    for key, value in DYNAMO_CONFIG.items():
        if hasattr(torch._dynamo.config, key):
            setattr(torch._dynamo.config, key, value)

    _installed = True

    log.info(
        "[OmniXPU shield] dynamo_isolation: config applied %s%s",
        DYNAMO_CONFIG,
        ("; " + "; ".join(actions)) if actions else "",
    )
    return True, None