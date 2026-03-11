"""Runtime hardware resource detection for dynamic configuration."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from wisent.core.utils.config_tools.constants import (
    FP16_GB_PER_BILLION_PARAMS,
    GPU_FRAMEWORK_OVERHEAD_GB,
    MB_PER_GB,
    COMBO_OFFSET,
    TRANSFORMER_PARAM_FACTOR,
    PARAMS_PER_BILLION,
)


@dataclass(frozen=True)
class SystemResources:
    """Detected system hardware resources."""

    cpu_count: int
    total_ram_mb: int
    gpu_mem_mb: int  # 0 if no GPU


@lru_cache(maxsize=1)
def detect_system_resources() -> SystemResources:
    """Detect CPU count, total RAM, and GPU memory."""
    cpu_count = os.cpu_count() or 4
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_ram_mb = (pages * page_size) // (1024 * 1024)
    except (ValueError, OSError, AttributeError):
        total_ram_mb = 8192
    gpu_mem_mb = 0
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_mb = (
                torch.cuda.get_device_properties(0).total_memory
                // (1024 * 1024)
            )
    except Exception:
        pass
    return SystemResources(
        cpu_count=cpu_count,
        total_ram_mb=total_ram_mb,
        gpu_mem_mb=gpu_mem_mb,
    )


# ---------------------------------------------------------------------------
# Docker limits -- scale with available RAM and CPU
# ---------------------------------------------------------------------------

def docker_mem_limit_mb() -> int:
    """25 pct of total RAM, clamped [512, 16384]."""
    res = detect_system_resources()
    return max(512, min(16384, res.total_ram_mb // 4))


def docker_nproc() -> int:
    """32 x cpu_count, clamped [64, 1024]."""
    res = detect_system_resources()
    return max(64, min(1024, 32 * res.cpu_count))


def docker_nofile() -> int:
    """4 x nproc."""
    return 4 * docker_nproc()


def docker_cpu_limit_s() -> int:
    """max(2, cpu_count // 2)."""
    res = detect_system_resources()
    return max(2, res.cpu_count // 2)


def docker_wall_timeout_s() -> int:
    """3 x cpu_limit."""
    return 3 * docker_cpu_limit_s()


def docker_fsize_mb() -> int:
    """Fixed filesystem limit."""
    return 16


def docker_pids_limit() -> int:
    """2 x nproc."""
    return 2 * docker_nproc()


# ---------------------------------------------------------------------------
# Eval limits -- derived from Docker limits
# ---------------------------------------------------------------------------

def eval_mem_limit_mb() -> int:
    """docker_mem_limit_mb() // 4, min 256."""
    return max(256, docker_mem_limit_mb() // 4)


def eval_cpu_limit_s() -> int:
    """Same as docker_cpu_limit_s."""
    return docker_cpu_limit_s()


def eval_time_limit_s() -> int:
    """Same as docker_wall_timeout_s."""
    return docker_wall_timeout_s()


# ---------------------------------------------------------------------------
# Batch sizes -- scale with GPU / CPU memory
# ---------------------------------------------------------------------------

def default_batch_size() -> int:
    """GPU: gpu_mem // 3072, clamped [1, 32]. No GPU: ram // 4096, [1, 8]."""
    res = detect_system_resources()
    if res.gpu_mem_mb > 0:
        return max(1, min(32, res.gpu_mem_mb // 3072))
    return max(1, min(8, res.total_ram_mb // 4096))


def eval_batch_size() -> int:
    """default_batch_size() // 2, min 1."""
    return max(1, default_batch_size() // 2)


def extraction_batch_size() -> int:
    """default_batch_size() // 2, min 1."""
    return max(1, default_batch_size() // 2)


# ---------------------------------------------------------------------------
# Subprocess timeouts -- scale inversely with CPU count
# ---------------------------------------------------------------------------

def subprocess_timeout_s() -> int:
    """max(60, 600 // max(1, cpu_count // 4))."""
    res = detect_system_resources()
    return max(60, 600 // max(1, res.cpu_count // 4))


def subprocess_timeout_long_s() -> int:
    """2 x subprocess_timeout_s, min 300."""
    return max(300, 2 * subprocess_timeout_s())


# ---------------------------------------------------------------------------
# Sandbox-specific limits
# ---------------------------------------------------------------------------

def docker_bigcode_mem_limit_mb() -> int:
    """docker_mem_limit_mb() // 16, clamped [128, 1024]."""
    return max(128, min(1024, docker_mem_limit_mb() // 16))


def docker_code_exec_timeout_s() -> int:
    """5 x docker_wall_timeout_s, min 60."""
    return max(60, 5 * docker_wall_timeout_s())


def code_eval_mem_limit_mb() -> int:
    """Same as eval_mem_limit_mb."""
    return eval_mem_limit_mb()


def docker_sandbox_time_limit_s() -> int:
    """docker_wall_timeout_s // 2, min 5."""
    return max(5, docker_wall_timeout_s() // 2)


def docker_sandbox_cpu_limit_s() -> int:
    """docker_cpu_limit_s // 2, min 2."""
    return max(2, docker_cpu_limit_s() // 2)


def docker_sandbox_mem_limit_mb() -> int:
    """docker_mem_limit_mb() // 8, clamped [256, 2048]."""
    return max(256, min(2048, docker_mem_limit_mb() // 8))


def safe_docker_nproc_default() -> int:
    """docker_nproc() // 2, min 32."""
    return max(32, docker_nproc() // 2)


# ---------------------------------------------------------------------------
# GPU memory estimation -- per-worker and max-parallel-workers
# ---------------------------------------------------------------------------


def _extract_param_billions(model_name: str) -> float:
    """Extract parameter count in billions from model name."""
    match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    if match:
        return float(match.group(COMBO_OFFSET))
    return _estimate_params_from_config(model_name)


def _estimate_params_from_config(model_name: str) -> float:
    """Estimate parameter count in billions from HuggingFace AutoConfig."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    hidden = getattr(cfg, "hidden_size", None)
    layers = getattr(cfg, "num_hidden_layers", None)
    vocab = getattr(cfg, "vocab_size", None)
    if hidden and layers and vocab:
        total_params = (
            TRANSFORMER_PARAM_FACTOR * layers * hidden * hidden
            + vocab * hidden
        )
        return total_params / PARAMS_PER_BILLION
    raise ValueError(f"Cannot estimate parameters for {model_name}")


def estimate_model_memory_mb(model_name: str) -> int:
    """Estimate per-worker GPU memory in MB for a model."""
    params_b = _extract_param_billions(model_name)
    memory_gb = params_b * FP16_GB_PER_BILLION_PARAMS + GPU_FRAMEWORK_OVERHEAD_GB
    return int(memory_gb * MB_PER_GB)


def estimate_max_gpu_workers(model_name: str) -> int:
    """Estimate max parallel workers fitting on the current GPU."""
    res = detect_system_resources()
    if not res.gpu_mem_mb:
        return COMBO_OFFSET
    per_worker_mb = estimate_model_memory_mb(model_name)
    workers = res.gpu_mem_mb // per_worker_mb
    return max(COMBO_OFFSET, workers)


# ---------------------------------------------------------------------------
# DS1000 overrides -- scaled from base limits
# ---------------------------------------------------------------------------

def ds1000_cpu_limit_s() -> int:
    """10 x docker_cpu_limit_s, min 30."""
    return max(30, 10 * docker_cpu_limit_s())


def ds1000_wall_timeout_s() -> int:
    """2 x ds1000_cpu_limit_s."""
    return 2 * ds1000_cpu_limit_s()


def ds1000_nproc() -> int:
    """4 x docker_nproc, clamped [256, 2048]."""
    return max(256, min(2048, 4 * docker_nproc()))
