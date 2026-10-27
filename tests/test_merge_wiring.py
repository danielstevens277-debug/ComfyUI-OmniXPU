"""Static wiring tests for the AVS + master-patch merge.

No GPU, no torch, no ComfyUI imports: these check the merge contract only
(flags, load order, delegation, licensing surface).  Run with
`python -m pytest tests/test_merge_wiring.py`.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def _load_config():
    spec = importlib.util.spec_from_file_location("merge_config", ROOT / "config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Config()


SHIELD_FLAGS = (
    "shield_int8_enforcer",
    "shield_fp64_creation",
    "shield_sycl_quantile",
    "shield_sdpa_align",
    "shield_dynamo_isolation",
    "shield_dtype_align",
    "memory_unified_wrapper",
)


def test_all_shields_default_on_and_individually_switchable():
    cfg = _load_config()
    for flag in SHIELD_FLAGS:
        assert getattr(cfg, flag) is True, f"{flag} must default to on"
    env_names = {"OMNIXPU_SHIELDS_ENABLE"}
    for flag in SHIELD_FLAGS:
        if flag == "memory_unified_wrapper":
            env_names.add("OMNIXPU_SHIELD_MEMORY_UNIFIED")
        else:
            env_names.add(f"OMNIXPU_SHIELD_{flag.removeprefix('shield_').upper()}")
    src = _read("config.py")
    for name in sorted(env_names):
        assert name in src, f"missing kill switch {name}"


def test_shield_stage_table_matches_package():
    cfg = _load_config()
    src = _read("shields/__init__.py")
    rows = re.findall(r'\("(shield_\w+)", "(\w+)", "(\w+)"\)', src)
    assert len(rows) == 6
    stages = {module: stage for _flag, module, stage in rows}
    assert stages["dynamo_isolation"] == "dynamo"
    assert all(v == "math" for module, v in stages.items() if module != "dynamo_isolation")
    for flag, module, stage in rows:
        assert getattr(cfg, flag) is True
        assert (ROOT / "shields" / f"{module}.py").is_file()
        expected = cfg.shield_stages.get(module, module)
        assert expected == stage, f"stage table disagrees for {module}"


def test_load_order_is_the_fixed_one():
    src = _read("__init__.py")
    positions = {
        "shields_math": src.index('apply_all_shields(config_mod.config, "math")'),
        "memory": src.index("memory.apply()"),
        "adapters": src.index("patches.apply_all_patches"),
        "shields_dynamo": src.index('apply_all_shields(config_mod.config, "dynamo")'),
    }
    assert positions["shields_math"] < positions["memory"] < positions["adapters"]
    assert positions["adapters"] < positions["shields_dynamo"]


def test_free_memory_has_a_single_owner():
    # The dynamic VRAM adapter must defer to the unified wrapper when present.
    src = _read("adapters/dynamic_vram.py")
    assert "_avs_xpu_vram_fix" in src
    # ...and the unified wrapper must carry the marker the delegate checks.
    assert '_avs_xpu_vram_fix' in _read("memory/avs_vram_fix.py")


def test_frontend_and_routes_preserved_for_compat():
    assert _read("__init__.py").count('WEB_DIRECTORY = "./web/js"') == 1
    js = _read("web/js/avs_xpu_settings.js")
    assert "/avs-xpu-vram-fix/enabled" in js
    py = _read("memory/avs_vram_fix.py")
    assert '"/avs-xpu-vram-fix/status"' in py and '"/avs-xpu-vram-fix/enabled"' in py


def test_license_surface_declared():
    assert (ROOT / "LICENSE").is_file(), "merged tree must carry the GPL-3.0 license"
    assert "GNU GENERAL PUBLIC LICENSE" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_registry_hook_exists():
    assert "def register_record" in _read("patches/__init__.py")
