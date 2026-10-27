"""Unified XPU memory accounting for the merged package.

One owner for ``model_management.get_free_memory``/``get_total_memory``:
the absorbed AVS module (PR #15487 logic + its runtime toggle and routes).
OmniXPU's Windows boundary trim keeps its actual job — the VBAR reclaim at model
boundaries — but no longer installs a second, equivalent wrapper; it defers to
this layer (see ``adapters/dynamic_vram.py``).
"""

from __future__ import annotations

import logging

log = logging.getLogger("ComfyUI-OmniXPU")


def apply() -> list[dict]:
    """Install the unified wrapper.  Returns a status record for diagnostics."""
    from . import avs_vram_fix

    record = {
        "name": "memory_unified_wrapper",
        "kind": "memory",
        "owner": "AVS Intel-XPU-VRAM-Fix (absorbed, GPL-3.0)",
        "module": "memory/avs_vram_fix.py",
        "status": "",
        "reason": "",
    }
    try:
        ok, reason = avs_vram_fix.apply()
        record["status"] = "applied" if ok else "skipped"
        record["reason"] = reason or ""
    except Exception as exc:
        record["status"] = "failed"
        record["reason"] = str(exc)
    if record["status"] == "skipped":
        log.info("[OmniXPU] unified memory wrapper: skipped (%s)", record["reason"])
    return [record]
