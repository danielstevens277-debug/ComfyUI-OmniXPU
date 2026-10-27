"""INT8 pre-quantized safetensors enforcer (master patch, shield 1 of 6).

Keeps an already-quantized checkpoint on its 8-bit memory footprint by clearing
a per-model bf16/fp16 manual cast request.  This overrides ComfyUI's per-model
cast choice, so it stays individually switchable
(OMNIXPU_SHIELD_INT8_ENFORCER=0).

Removal plan: drop when ComfyUI stops applying manual_cast_dtype to
pre-quantized checkpoints on XPU.
"""

from __future__ import annotations

import logging

import torch

import comfy.model_patcher

log = logging.getLogger("ComfyUI-OmniXPU")

MARKER = "__omnixpu_shield_int8_enforcer__"


def apply() -> tuple[bool, str | None]:
    if getattr(comfy.model_patcher.ModelPatcher.__init__, MARKER, False):
        return False, "already installed"

    original_init = comfy.model_patcher.ModelPatcher.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        is_quantized = False
        if hasattr(self, "model") and hasattr(self.model, "parameters"):
            for param in self.model.parameters():
                if param.dtype in (
                    torch.int8,
                    torch.uint8,
                    getattr(torch, "float8_e4m3fn", torch.int8),
                ):
                    is_quantized = True
                    break

        if is_quantized:
            fp_types = (torch.bfloat16, torch.float16)
            if getattr(self, "manual_cast_dtype", None) in fp_types:
                self.manual_cast_dtype = None
            if getattr(getattr(self, "model", None), "manual_cast_dtype", None) in fp_types:
                self.model.manual_cast_dtype = None

    patched_init.__wrapped__ = original_init
    setattr(patched_init, MARKER, True)
    comfy.model_patcher.ModelPatcher.__init__ = patched_init
    log.info("[OmniXPU shield] int8_enforcer: pre-quantized safetensors locked to 8-bit memory")
    return True, None
