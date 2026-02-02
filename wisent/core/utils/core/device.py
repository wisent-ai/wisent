"""Centralized torch device and dtype selection helpers."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

import torch

from wisent.core.errors import UnknownTypeError, FileLoadError

DeviceKind = Literal["cuda", "mps", "cpu"]
DtypeKind = Literal["float32", "float16", "bfloat16", "auto"]

# Global dtype setting - can be overridden via set_default_dtype() or WISENT_DTYPE env var
# None = use default (float32), "auto" = use device-optimized, or a specific torch.dtype
_global_dtype_override: Optional[torch.dtype | str] = None


def _mps_available() -> bool:
    return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()


@lru_cache(maxsize=1)
def resolve_default_device() -> DeviceKind:
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def resolve_torch_device() -> torch.device:
    return torch.device(resolve_default_device())


def resolve_device(kind: DeviceKind | None = None) -> torch.device:
    return torch.device(kind or resolve_default_device())


def set_default_dtype(dtype: DtypeKind | torch.dtype | None) -> None:
    """
    Set the global default dtype for all model loading.
    
    Args:
        dtype: One of "float32", "float16", "bfloat16", "auto", a torch.dtype, or None.
               - "auto": Use device-optimized dtypes (bfloat16 for CUDA, float16 for MPS, float32 for CPU)
               - None: Reset to default (float32 everywhere for consistency)
               
    Example:
        >>> set_default_dtype("float32")  # Force float32 everywhere (default)
        >>> set_default_dtype("auto")     # Use device-optimized dtypes for performance
        >>> set_default_dtype(torch.bfloat16)  # Force bfloat16 everywhere
        >>> set_default_dtype(None)       # Reset to default (float32)
    """
    global _global_dtype_override
    
    if dtype is None:
        _global_dtype_override = None
    elif dtype == "auto":
        _global_dtype_override = "auto"  # Special marker for device-optimized mode
    elif isinstance(dtype, torch.dtype):
        _global_dtype_override = dtype
    elif dtype == "float32":
        _global_dtype_override = torch.float32
    elif dtype == "float16":
        _global_dtype_override = torch.float16
    elif dtype == "bfloat16":
        _global_dtype_override = torch.bfloat16
    else:
        raise UnknownTypeError(entity_type="dtype", value=dtype, valid_values=["float32", "float16", "bfloat16", "auto", "torch.dtype"])


def get_default_dtype() -> Optional[torch.dtype]:
    """Get the current global dtype override, or None if using auto."""
    return _global_dtype_override


def _resolve_dtype_from_env() -> Optional[torch.dtype | str]:
    """Check WISENT_DTYPE environment variable."""
    env_dtype = os.environ.get("WISENT_DTYPE", "").lower()
    if env_dtype == "float32":
        return torch.float32
    elif env_dtype == "float16":
        return torch.float16
    elif env_dtype == "bfloat16":
        return torch.bfloat16
    elif env_dtype == "auto":
        return "auto"
    return None


def preferred_dtype(kind: DeviceKind | None = None) -> torch.dtype:
    """
    Return the preferred dtype for model loading.
    
    Default is device-optimized dtype (bfloat16 on CUDA, float16 on MPS, float32 on CPU).
    
    Priority:
    1. Global override set via set_default_dtype()
    2. WISENT_DTYPE environment variable ("float32", "float16", "bfloat16", "auto")
    3. Default: device-optimized (bfloat16 on CUDA, float16 on MPS, float32 on CPU)
    
    Example:
        >>> preferred_dtype()  # bfloat16 on CUDA, float16 on MPS, float32 on CPU
        torch.bfloat16
        >>> set_default_dtype("float32")  # Force float32 everywhere
        >>> preferred_dtype()
        torch.float32
    """
    # Check global override first
    if _global_dtype_override is not None:
        if _global_dtype_override == "auto":
            return device_optimized_dtype(kind)
        return _global_dtype_override
    
    # Check environment variable
    env_dtype = _resolve_dtype_from_env()
    if env_dtype is not None:
        if env_dtype == "auto":
            return device_optimized_dtype(kind)
        return env_dtype
    
    # Default: use device-optimized dtype for best performance
    return device_optimized_dtype(kind)


def device_optimized_dtype(kind: DeviceKind | None = None) -> torch.dtype:
    """
    Return the device-optimized dtype for maximum performance.
    
    Use this when you need performance and don't need cross-device consistency.
    
    - CUDA: bfloat16 (more stable than float16, better for training)
    - MPS: float16 (bfloat16 not well supported on Apple Silicon)
    - CPU: float32
    
    Example:
        >>> device_optimized_dtype()  # On CUDA
        torch.bfloat16
        >>> device_optimized_dtype("mps")
        torch.float16
    """
    chosen = kind or resolve_default_device()
    if chosen == "cuda":
        return torch.bfloat16
    elif chosen == "mps":
        return torch.float16
    else:
        return torch.float32


# ============================================================================
# Steering Vector dtype utilities
# ============================================================================


def steering_vector_dtype() -> torch.dtype:
    """Return the dtype for steering vectors (uses preferred_dtype())."""
    return preferred_dtype()


# Legacy constant for backward compatibility - use steering_vector_dtype() instead
STEERING_VECTOR_DTYPE = torch.float32  # Deprecated: kept for backward compat only


def save_steering_vector(
    path: str,
    vector: torch.Tensor,
    layer: int,
    model_name: str,
    method: str = "caa",
    metadata: dict | None = None,
) -> None:
    """
    Save a steering vector with dtype metadata.
    
    Args:
        path: File path to save to (.pt)
        vector: The steering vector tensor
        layer: Layer index this vector was trained on
        model_name: Name of the model this vector was trained for
        method: Steering method used ("caa", "dac", etc.)
        metadata: Additional metadata to store
    """
    # Store original dtype before conversion
    original_dtype = vector.dtype
    storage_dtype = steering_vector_dtype()
    
    # Store in preferred dtype
    vector_stored = vector.to(dtype=storage_dtype, device="cpu")
    
    save_data = {
        # Primary data
        "steering_vector": vector_stored,
        "layer": layer,
        "model": model_name,
        "method": method,
        # Dtype metadata
        "original_dtype": str(original_dtype),
        "storage_dtype": str(storage_dtype),
        # Legacy keys for backward compatibility
        "vector": vector_stored,
        "layer_index": layer,
    }
    
    if metadata:
        save_data["metadata"] = metadata
    
    torch.save(save_data, path)


def load_steering_vector(
    path: str,
    dtype: torch.dtype | None = None,
    device: str | None = None,
) -> dict:
    """
    Load a steering vector with automatic dtype conversion.
    
    Args:
        path: File path to load from (.pt)
        dtype: Target dtype (None = use current preferred_dtype())
        device: Target device (None = use current default device)
        
    Returns:
        Dictionary with:
        - "steering_vector": The vector tensor (converted to target dtype)
        - "layer": Layer index
        - "model": Model name
        - "method": Steering method
        - "original_dtype": Original dtype when saved
        - "metadata": Any additional metadata
    """
    target_device = device or resolve_default_device()
    data = torch.load(path, map_location=target_device, weights_only=False)
    
    # Get the vector (support both old and new key names)
    vector = data.get("steering_vector") or data.get("vector")
    if vector is None:
        raise FileLoadError(file_path=str(path), reason="No steering vector found")
    
    # Determine target dtype
    target_dtype = dtype or preferred_dtype(target_device)
    
    # Convert to target dtype/device if needed
    vector = vector.to(dtype=target_dtype, device=target_device)
    
    return {
        "steering_vector": vector,
        "layer": data.get("layer") or data.get("layer_index"),
        "model": data.get("model"),
        "method": data.get("method", "unknown"),
        "original_dtype": data.get("original_dtype", "unknown"),
        "metadata": data.get("metadata", {}),
        # Legacy keys
        "vector": vector,
    }


def empty_device_cache(kind: DeviceKind | None = None) -> None:
    chosen = kind or resolve_default_device()
    if chosen == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif chosen == "mps" and _mps_available():
        try:
            torch.mps.empty_cache()  # type: ignore[attr-defined]
        except AttributeError:
            pass


def move_module_to_preferred_device(module: torch.nn.Module) -> torch.nn.Module:
    return module.to(resolve_torch_device())


def ensure_tensor_on_device(tensor: torch.Tensor) -> torch.Tensor:
    target = resolve_torch_device()
    return tensor.to(target) if tensor.device != target else tensor
