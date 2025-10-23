"""
Memory Management Utilities.

Provides functions for aggressive GPU memory cleanup.
"""

from __future__ import annotations

import gc
import time
import torch


def cleanup_gpu_memory(sleep_time: int = 10):
    """
    Aggressively clean up GPU memory.

    Caller should delete their own references first (del model, del collector, etc.)
    then call this function.

    Args:
        sleep_time: Time to sleep after cleanup in seconds (default: 10)

    Example:
        del model
        del collector
        cleanup_gpu_memory(sleep_time=10)
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()
    time.sleep(sleep_time)
