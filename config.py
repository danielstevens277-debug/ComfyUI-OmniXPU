import os


def _flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) != "0"


class Config:
    """Feature flags controlled via environment variables.

    Generic XPU operators are selected by comfy_kitchen and therefore do not
    have custom-node flags here.  Legacy global workarounds are opt-in.
    """

    def __init__(self):
        master = os.environ.get("OMNIXPU_ENABLE", "1") != "0"
        self.attention = master and os.environ.get("OMNIXPU_ATTENTION", "1") != "0"
        self.sparse_attention = master and os.environ.get("OMNIXPU_SPARSE_ATTENTION", "1") != "0"
        self.rotary = master and os.environ.get("OMNIXPU_ROTARY", "1") != "0"
        self.rms_rope = master and os.environ.get("OMNIXPU_RMS_ROPE", "1") != "0"
        self.norm = master and os.environ.get("OMNIXPU_NORM", "1") != "0"
        self.h3_rms_modulation = (
            master
            # Opt-in: the fused H3 norm+modulation kernel measured only ~0.5%
            # end-to-end on A770 (135.6 s vs 136.3 s over a 6-step ref2va run)
            # while running a different numeric path than the eager chain, so
            # the stock path stays the default.
            and os.environ.get("OMNIXPU_H3_RMS_MODULATION", "0") != "0"
        )
        self.fp8_gemm = master and os.environ.get("OMNIXPU_FP8_GEMM", "1") != "0"
        self.quantized_matmul = (
            master and os.environ.get("OMNIXPU_QUANTIZED_MATMUL", "1") != "0"
        )
        self.int8_ffn = master and os.environ.get("OMNIXPU_INT8_FFN", "1") != "0"
        self.int4_gemm = master and os.environ.get("OMNIXPU_INT4_GEMM", "1") != "0"
        self.lora_memory = (
            master and os.environ.get("OMNIXPU_LORA_MEMORY", "1") != "0"
        )
        self.seedvr_ada_reshape = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_ADA_RESHAPE", "1") != "0"
        )
        self.seedvr_capacity = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_CAPACITY", "1") != "0"
        )
        self.seedvr_cat_pad = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_CAT_PAD", "1") != "0"
        )
        self.seedvr_vae_cpu_stage = (
            master
            and os.environ.get("OMNIXPU_SEEDVR_VAE_CPU_STAGE", "1") != "0"
        )
        self.large_video_preprocess = (
            master
            and os.environ.get("OMNIXPU_LARGE_VIDEO_PREPROCESS", "1") != "0"
        )
        self.dynamic_vram_boundary_trim = (
            master
            and os.environ.get("OMNIXPU_DYNAMIC_VRAM_BOUNDARY_TRIM", "1") != "0"
        )
        self.int8_direct_cast = (
            master and os.environ.get("OMNIXPU_INT8_DIRECT_CAST", "0") != "0"
        )
        self.int8_patch_cache = (
            master and os.environ.get("OMNIXPU_INT8_PATCH_CACHE", "0") != "0"
        )
        self.kitchen_compat = (
            master and os.environ.get("OMNIXPU_KITCHEN_COMPAT", "1") != "0"
        )
        self.int8_native_gate = (
            master and os.environ.get("OMNIXPU_INT8_NATIVE", "1") != "0"
        )
        self.interpolate_fix = (
            master and os.environ.get("OMNIXPU_INTERPOLATE_FIX", "0") != "0"
        )
        self.median_fix = (
            master and os.environ.get("OMNIXPU_MEDIAN_FIX", "0") != "0"
        )
        self.sdp_cache_lifecycle = (
            master
            and os.environ.get("OMNIXPU_SDP_CACHE_AUTOCLEAR", "1")
            .strip().lower()
            not in ("keep", "0", "false", "no", "off")
        )
        self.sampling_prefetch = (
            master
            and os.environ.get("OMNIXPU_H3_SAMPLING_PREFETCH", "1") != "0"
        )
        self.cast_buffer_lifetime = (
            master
            and os.environ.get("OMNIXPU_CAST_BUFFER_LIFETIME", "1") != "0"
        )
        self.host_kv_cache = (
            master
            and os.environ.get("OMNIXPU_HOST_KV_CACHE", "1") != "0"
        )
        self.qwen21_cache_compat = (
            master
            and os.environ.get("OMNIXPU_QWEN21_CACHE_COMPAT", "1") != "0"
        )
        self.minimax_compat = (
            master
            and os.environ.get("OMNIXPU_MINIMAX_COMPAT", "1") != "0"
        )
        self.moss_image_compat = (
            master
            and os.environ.get("OMNIXPU_MOSS_IMAGE_COMPAT", "1") != "0"
        )

        # ── Global shields, absorbed from the XPU FP64 master patch ─────────
        # The shield author requires them on; each stays individually switchable
        # per the contribution boundary (global correctness workarounds must be
        # opt-out-able and carry an upstream removal plan).
        shields = os.environ.get("OMNIXPU_SHIELDS_ENABLE", "1") != "0"
        self.shield_int8_enforcer = shields and _flag("OMNIXPU_SHIELD_INT8_ENFORCER")
        self.shield_fp64_creation = shields and _flag("OMNIXPU_SHIELD_FP64_CREATION")
        self.shield_sycl_quantile = shields and _flag("OMNIXPU_SHIELD_SYCL_QUANTILE")
        self.shield_sdpa_align = shields and _flag("OMNIXPU_SHIELD_SDPA_ALIGN")
        self.shield_sdpa_fastpath = (
            self.shield_sdpa_align and _flag("OMNIXPU_SHIELD_SDPA_FASTPATH")
        )
        self.shield_dynamo_isolation = shields and _flag("OMNIXPU_SHIELD_DYNAMO_ISOLATION")
        self.shield_dtype_align = shields and _flag("OMNIXPU_SHIELD_DTYPE_ALIGN")
        # Memory accounting: one unified wrapper (AVS PR #15487 logic) shared by
        # the old AVS toggle and the OmniXPU Windows boundary trim.
        self.memory_unified_wrapper = shields and _flag("OMNIXPU_SHIELD_MEMORY_UNIFIED")

        # Stage map for shields/__init__.py: "math" runs before the adapters so
        # adapter dispatch sees the patched creation/cast surface; "dynamo" runs
        # after them, once the Kitchen/kernel compile boundaries are loaded.
        self.shield_stages = {
            "int8_enforcer": "math",
            "fp64_creation": "math",
            "sycl_quantile": "math",
            "sdpa_align": "math",
            "dtype_align": "math",
            "dynamo_isolation": "dynamo",
        }


config = Config()
