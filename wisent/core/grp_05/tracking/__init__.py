import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

"""
Performance tracking module for wisent.

This module provides comprehensive monitoring capabilities including:
- Memory usage tracking (CPU and GPU)
- Latency/timing analysis
- Performance profiling and optimization insights
"""

from .memory import (
    MemoryTracker,
    MemorySnapshot,
    MemoryStats,
    get_global_tracker as get_global_memory_tracker,
    track_memory,
    get_memory_info,
    format_memory_usage
)

from .latency import (
    LatencyTracker,
    TimingEvent,
    LatencyStats,
    get_global_tracker as get_global_latency_tracker,
    time_function,
    time_operation,
    get_timing_summary,
    format_timing_summary,
    reset_timing,
    Operations
)

__all__ = [
    # Memory tracking
    "MemoryTracker",
    "MemorySnapshot", 
    "MemoryStats",
    "get_global_memory_tracker",
    "track_memory",
    "get_memory_info",
    "format_memory_usage",
    
    # Latency tracking
    "LatencyTracker",
    "TimingEvent",
    "LatencyStats", 
    "get_global_latency_tracker",
    "time_function",
    "time_operation",
    "get_timing_summary",
    "format_timing_summary",
    "reset_timing",
    "Operations"
] 
