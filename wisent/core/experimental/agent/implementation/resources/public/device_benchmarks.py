"""
Device-specific performance benchmarking for wisent.

This module runs quick performance tests on the current device to measure
actual execution times for different operations, then saves those estimates
for future budget calculations.
"""

import json
import time
import os
import tempfile
import subprocess
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import torch

from wisent.core.utils.config_tools.constants import DEVICE_HASH_PREFIX, JSON_INDENT, SECONDS_PER_DAY, HOURS_PER_DAY
from wisent.core.utils import resolve_default_device
from wisent.core.utils.infra_tools.errors import (
    DeviceBenchmarkError,
    NoBenchmarkDataError,
    InsufficientDataError,
    UnknownTypeError,
)


@dataclass
class DeviceBenchmark:
    """Performance benchmark results for a specific device."""
    device_id: str
    device_type: str  # "cpu", "cuda", "mps", etc.
    model_loading_seconds: float
    benchmark_eval_seconds_per_100_examples: float
    classifier_training_seconds_per_100_samples: float  # Actually measures full classifier creation time (per 100 classifiers)
    data_generation_seconds_per_example: float
    steering_seconds_per_example: float
    benchmark_timestamp: float
    python_version: str
    platform_info: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeviceBenchmark':
        """Create from dictionary loaded from JSON."""
        return cls(**data)
from wisent.core.experimental.agent.resources._device_bench_tests import DeviceBenchTestsMixin1
from wisent.core.experimental.agent.resources._device_bench_tests2 import DeviceBenchTestsMixin2
from wisent.core.experimental.agent.resources._device_bench_runner import DeviceBenchmarkRunnerMixin


class DeviceBenchmarker(DeviceBenchTestsMixin1, DeviceBenchTestsMixin2, DeviceBenchmarkRunnerMixin):
    """Runs performance benchmarks and manages device-specific estimates."""
    
    def __init__(self, benchmarks_file: str):
        self.benchmarks_file = benchmarks_file
        self.cached_benchmark: Optional[DeviceBenchmark] = None
        
    def get_device_id(self) -> str:
        """Generate a unique ID for the current device configuration."""
        import platform
        
        # Create device fingerprint from hardware/software info
        info_parts = [
            platform.machine(),
            platform.processor(),
            platform.system(),
            platform.release(),
            sys.version,
        ]
        
        # Add GPU info if available
        device_kind = resolve_default_device()
        if device_kind == "cuda" and torch.cuda.is_available():
            info_parts.append(f"cuda_{torch.cuda.get_device_name(torch.cuda.current_device())}")
        elif device_kind == "mps":
            info_parts.append("mps")
        
        # Create hash of the combined info
        combined = "|".join(str(part) for part in info_parts)
        device_hash = hashlib.md5(combined.encode()).hexdigest()[:DEVICE_HASH_PREFIX]
        return device_hash
    
    def get_device_type(self) -> str:
        """Detect the device type (cpu, cuda, mps, etc.)."""
        return resolve_default_device()
    
    def load_cached_benchmark(self) -> Optional[DeviceBenchmark]:
        """Load cached benchmark results if they exist and are recent."""
        if not os.path.exists(self.benchmarks_file):
            return None
            
        try:
            with open(self.benchmarks_file, 'r') as f:
                data = json.load(f)
            
            device_id = self.get_device_id()
            if device_id not in data:
                return None
                
            benchmark_data = data[device_id]
            benchmark = DeviceBenchmark.from_dict(benchmark_data)
            
            # Check if benchmark is recent (within 7 days)
            current_time = time.time()
            age_days = (current_time - benchmark.benchmark_timestamp) / SECONDS_PER_DAY
            
            if age_days > 7:
                print(f"   ⚠️ Cached benchmark is {age_days:.1f} days old, will re-run")
                return None
                
            return benchmark
            
        except Exception as e:
            print(f"   ⚠️ Error loading cached benchmark: {e}")
            return None
    
    def save_benchmark(self, benchmark: DeviceBenchmark) -> None:
        """Save benchmark results to JSON file."""
        try:
            # Load existing data
            existing_data = {}
            if os.path.exists(self.benchmarks_file):
                with open(self.benchmarks_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Update with new benchmark
            existing_data[benchmark.device_id] = benchmark.to_dict()
            
            # Save back to file
            with open(self.benchmarks_file, 'w') as f:
                json.dump(existing_data, f, indent=JSON_INDENT)
                
            print(f"   💾 Saved benchmark results to {self.benchmarks_file}")
            
        except Exception as e:
            print(f"   ❌ Error saving benchmark: {e}")
    
def get_device_benchmarker() -> DeviceBenchmarker:
    """Get the global device benchmarker instance."""
    return _device_benchmarker


def ensure_benchmark_exists(force_rerun: bool = False) -> DeviceBenchmark:
    """Ensure device benchmark exists, running it if necessary."""
    return _device_benchmarker.run_full_benchmark(force_rerun=force_rerun)


def estimate_task_time(task_type: str, quantity: int = 1) -> float:
    """
    Convenience function to estimate task time.
    
    Args:
        task_type: Type of task ("model_loading", "benchmark_eval", etc.)
        quantity: Number of items
        
    Returns:
        Estimated time in seconds
    """
    return _device_benchmarker.estimate_task_time(task_type, quantity)


def get_current_device_info() -> Dict[str, str]:
    """Get current device information."""
    benchmarker = get_device_benchmarker()
    return {
        "device_id": benchmarker.get_device_id(),
        "device_type": benchmarker.get_device_type()
    }
