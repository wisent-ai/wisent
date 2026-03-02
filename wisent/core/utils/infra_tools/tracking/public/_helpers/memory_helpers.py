"""Standalone utility functions for memory tracking."""

from typing import Dict, Any, Optional, Callable


# Global memory tracker instance
_global_tracker = None


def get_global_tracker():
    """Get or create the global memory tracker instance."""
    from wisent.core.utils.infra_tools.tracking.memory import MemoryTracker

    global _global_tracker
    if _global_tracker is None:
        _global_tracker = MemoryTracker()
    return _global_tracker


def track_memory(operation_name: str):
    """Decorator to track memory usage of a function."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tracker = get_global_tracker()
            with tracker.track_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_memory_info() -> Dict[str, Any]:
    """Get current memory information without tracking."""
    from wisent.core.utils.infra_tools.tracking.memory import MemoryTracker

    tracker = MemoryTracker(auto_cleanup=False)
    return tracker.get_current_usage()


def format_memory_usage(usage: Dict[str, Any]) -> str:
    """Format memory usage dictionary as a readable string."""
    lines = [
        f"CPU Memory: {usage['cpu_memory_mb']:.1f} MB ({usage['cpu_memory_percent']:.1f}%)"
    ]

    if 'gpu_memory_mb' in usage and usage['gpu_memory_mb'] is not None:
        lines.append(f"GPU Memory: {usage['gpu_memory_mb']:.1f} MB")
        if 'gpu_memory_percent' in usage and usage['gpu_memory_percent'] is not None:
            lines[-1] += f" ({usage['gpu_memory_percent']:.1f}%)"

        if 'cached_memory_mb' in usage:
            lines.append(f"GPU Cached: {usage['cached_memory_mb']:.1f} MB")

        if 'allocated_tensors' in usage:
            lines.append(f"GPU Tensors: {usage['allocated_tensors']}")

    return " | ".join(lines)


def format_stats(stats, detailed: bool = False) -> str:
    """Format memory statistics as a readable string."""
    lines = [
        "Memory Usage Statistics:",
        f"  Duration: {stats.duration_seconds:.2f} seconds",
        f"  CPU Memory:",
        f"    Peak: {stats.peak_cpu_mb:.1f} MB",
        f"    Average: {stats.avg_cpu_mb:.1f} MB",
        f"    Minimum: {stats.min_cpu_mb:.1f} MB",
    ]

    if stats.peak_gpu_mb is not None:
        lines.extend([
            f"  GPU Memory:",
            f"    Peak: {stats.peak_gpu_mb:.1f} MB",
            f"    Average: {stats.avg_gpu_mb:.1f} MB",
            f"    Minimum: {stats.min_gpu_mb:.1f} MB",
        ])

    if stats.operations:
        lines.append(f"  Operations: {', '.join(stats.operations)}")

    if detailed and stats.snapshots:
        lines.append(f"  Snapshots: {len(stats.snapshots)} collected")

        # Show peak usage snapshot
        peak_snapshot = max(stats.snapshots, key=lambda s: s.cpu_memory_mb)
        lines.extend([
            f"  Peak Usage Snapshot:",
            f"    Time: {peak_snapshot.timestamp:.2f}",
            f"    CPU: {peak_snapshot.cpu_memory_mb:.1f} MB ({peak_snapshot.cpu_memory_percent:.1f}%)",
        ])

        if peak_snapshot.gpu_memory_mb is not None:
            lines.append(f"    GPU: {peak_snapshot.gpu_memory_mb:.1f} MB")
            if peak_snapshot.allocated_tensors is not None:
                lines.append(f"    Tensors: {peak_snapshot.allocated_tensors}")

    return "\n".join(lines)
