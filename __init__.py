import logging
import importlib.util
import os
import sys

log = logging.getLogger("ComfyUI-OmniXPU")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Resolve package directory (works regardless of how we're loaded)
_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = "ComfyUI-OmniXPU"

# ── Register root package in sys.modules ─────────────────────────────────────
# ComfyUI loads custom nodes under 'custom_nodes.<folder_name>', but internal
# adapters, memory handlers, and shields resolve submodules under 'ComfyUI-OmniXPU.*'.
# Alias the executing module to _PKG so relative imports (e.g. 'from . import ...')
# and parent lookups resolve properly without searching sys.path.
_self = sys.modules.get(__name__)
if _self is not None:
    if not hasattr(_self, "__path__"):
        _self.__path__ = [_DIR]
    sys.modules[_PKG] = _self
else:
    import types
    _self = types.ModuleType(_PKG)
    _self.__file__ = os.path.abspath(__file__)
    _self.__path__ = [_DIR]
    sys.modules[_PKG] = _self


def _load(rel_path, mod_name):
    """Load a .py file as a named module."""
    fpath = os.path.join(_DIR, *rel_path.split("/"))
    if os.path.isdir(fpath):
        fpath = os.path.join(fpath, "__init__.py")
    elif not fpath.endswith(".py"):
        fpath += ".py"
    search = [os.path.dirname(fpath)] if os.path.basename(fpath) == "__init__.py" else None
    spec = importlib.util.spec_from_file_location(mod_name, fpath, submodule_search_locations=search)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod

    # Bind submodule to parent module if parent exists in sys.modules
    if "." in mod_name:
        parent_name, attr_name = mod_name.rsplit(".", 1)
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], attr_name, mod)

    spec.loader.exec_module(mod)
    return mod


# Directory served as static files by ComfyUI (absorbed from the AVS VRAM fix;
# the frontend settings extension lives here and must stay a direct folder).
WEB_DIRECTORY = "./web/js"

try:
    import torch
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        log.info("[OmniXPU] No Intel XPU detected, skipping all patches")
    else:
        # Load probe and run capability check
        probe = _load("probe.py", f"{_PKG}.probe")
        probe.probe()

        # Load config and patches
        _load("config.py", f"{_PKG}.config")
        patches = _load("patches", f"{_PKG}.patches")

        config_mod = sys.modules[f"{_PKG}.config"]

        # ── Merge load order (fixed, do not reorder casually) ────────────────
        # 1. Global torch shields (math stage).  They rewrite creation/cast
        #    entry points, so every adapter must see the patched surface.
        # 2. Unified memory wrapper, before the adapters: dynamic_vram defers
        #    to it, which keeps the AVS runtime toggle effective on Windows.
        # 3. OmniXPU adapters and fixes (unchanged behaviour).
        # 4. Dynamo isolation shield, after the adapters: it needs the Kitchen
        #    and kernel namespaces loaded to respect their compile boundaries.
        shields = _load("shields", f"{_PKG}.shields")

        for record in shields.apply_all_shields(config_mod.config, "math"):
            patches.register_record(record)

        memory = _load("memory", f"{_PKG}.memory")

        for record in memory.apply():
            patches.register_record(record)

        patches.apply_all_patches(config_mod.config)

        for record in shields.apply_all_shields(config_mod.config, "dynamo"):
            patches.register_record(record)

        # Load diagnostics node
        _load("nodes/__init__.py", f"{_PKG}.nodes")
        diag = _load("nodes/diagnostics.py", f"{_PKG}.nodes.diagnostics")
        NODE_CLASS_MAPPINGS["OmniXPUStatus"] = diag.OmniXPUStatus
        NODE_DISPLAY_NAME_MAPPINGS["OmniXPUStatus"] = "OmniXPU Status"

        sdp_cache = _load("nodes/sdp_cache.py", f"{_PKG}.nodes.sdp_cache")
        NODE_CLASS_MAPPINGS.update(sdp_cache.NODE_CLASS_MAPPINGS)
        NODE_DISPLAY_NAME_MAPPINGS.update(sdp_cache.NODE_DISPLAY_NAME_MAPPINGS)

except Exception as e:
    import traceback
    log.error("[OmniXPU] Initialization failed: %s\n%s", e, traceback.format_exc())

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]