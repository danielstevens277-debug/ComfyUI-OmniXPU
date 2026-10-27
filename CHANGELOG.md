# Changelog

## Merge: AVS VRAM Fix + XPU FP64 master patch into ComfyUI-OmniXPU

This node is now the single install for the OmniXPU layer, the Intel XPU VRAM
accounting fix, and the global XPU shields.

Provenance:

- `memory/avs_vram_fix.py`, `web/js/avs_xpu_settings.js`, `LICENSE` —
  absorbed from https://github.com/alvasafin-art/ComfyUI-AVS-Intel-XPU-VRAM-Fix
  (GPL-3.0). Logic preserved; the module is now called from the package init
  instead of being a standalone custom node.
- `shields/` — the six numbered sections of the XPU FP64 master patch, kept as
  separate flag-gated modules. The author requirement "prevent fp64 from being
  used" maps to the fp64 creation/cast shield (default on).
- Everything else — unchanged from the pre-merge OmniXPU tree.

What changed during the merge:

- Single owner for `model_management.get_free_memory`. Previously the AVS
  wrapper and the Windows boundary trim's wrapper computed the same formula, and
  the adapter's wrapper shadowed the AVS one on Windows (the AVS toggle and its
  PR #15487 guard became dead code). The boundary trim now defers to the unified
  wrapper and keeps only its VBAR reclaim job.
- Dynamo isolation no longer forces `disallow_in_graph` on namespaces whose
  kernel wheel ships Dynamo compile boundaries (`omni_xpu_kernel._compile_ops`);
  that preserves the compiled-graph contract and the Krea2 resume path.
  `comfy_kitchen` behaviour is unchanged. torch.compile is expected to work on
  Windows 11 with Torch 2.14.
- Fixed load order documented in `README.md` (math shields before adapters,
  Dynamo isolation after). The master patch cannot run as a prestartup script —
  it imports torch, and the AIMDO takeover needs an unimported runtime.
- The master patch's `_is_omnixpu_hooked` guard on SDPA was dead code (nothing
  set it); the shield keeps the guard and now checks it against real install
  markers. The Triton SDPA fast path is a separate switch
  (`OMNIXPU_SHIELD_SDPA_FASTPATH`).
- The quantile shield re-runs its fp64 probe through the original
  `torch.quantile` so it reports real hardware capability, not state left by the
  fp64 creation shield; it registers nothing when the probe passes.
- `patches.register_record()` added so shields and the memory layer appear in
  the existing OmniXPU Status node.
- AVS compatibility kept: frontend setting, `user/avs_xpu_vram_fix/config.json`,
  and the `/avs-xpu-vram-fix/*` routes are unchanged. Existing installs keep
  their saved toggle.

Added tests: `tests/test_merge_wiring.py` (static merge-contract tests, no GPU
or torch required).
