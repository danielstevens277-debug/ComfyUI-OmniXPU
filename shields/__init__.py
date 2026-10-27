"""Global torch-level shields absorbed from the XPU FP64 master patch.

Stage ``math`` runs before the ComfyUI adapters so adapter dispatch sees the
patched creation/cast surface.  Stage ``dynamo`` runs after them, once the
Kitchen/kernel namespaces and their compile boundaries are loaded.  Every shield
exposes ``apply() -> (ok, reason)`` like the adapter components, and each one is
individually switchable through ``config.py``.

Absorbed from the master patch by its author's request ("prevent fp64 from being
used"); the six numbered shields map one-to-one to its sections.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("ComfyUI-OmniXPU")

# (config flag, module, stage)
SHIELDS = (
    ("shield_int8_enforcer", "int8_enforcer", "math"),
    ("shield_fp64_creation", "fp64_creation", "math"),
    ("shield_sycl_quantile", "sycl_quantile", "math"),
    ("shield_sdpa_align", "sdpa_align", "math"),
    ("shield_dtype_align", "dtype_align", "math"),
    ("shield_dynamo_isolation", "dynamo_isolation", "dynamo"),
)


def apply_all_shields(cfg, stage: str) -> list[dict]:
    """Run every shield assigned to ``stage``.  Returns status records."""
    records = []

    for flag, name, shield_stage in SHIELDS:
        if shield_stage != stage:
            continue
        record = {
            "name": f"{name}_shield",
            "kind": "shield",
            "owner": "XPU master patch (absorbed)",
            "module": f"shields/{name}.py",
            "status": "",
            "reason": "",
        }
        if not getattr(cfg, flag, False):
            record["status"] = "skipped"
            record["reason"] = "disabled by env"
            records.append(record)
            continue
        try:
            module = importlib.import_module(f"{__name__}.{name}")
            ok, reason = module.apply()
            record["status"] = "applied" if ok else "skipped"
            record["reason"] = reason or ""
        except Exception as exc:
            record["status"] = "failed"
            record["reason"] = str(exc)
            log.warning("[OmniXPU] %s shield: FAILED (%s)", name, exc)
        else:
            if record["status"] == "skipped":
                log.info("[OmniXPU] %s shield: skipped (%s)", name, record["reason"])
        records.append(record)

    return records
