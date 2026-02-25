"""Benchmark orchestration and estimation methods."""
import time
import sys
import hashlib
from typing import Dict, Any, Optional
from wisent.core.errors import DeviceBenchmarkError, NoBenchmarkDataError, UnknownTypeError
from wisent.core.constants import AGENT_BENCH_EVAL_DEFAULT_SECONDS, AGENT_CLASSIFIER_TRAINING_DEFAULT_SECONDS, HASH_DISPLAY_LENGTH


class DeviceBenchmarkRunnerMixin:
    """Mixin providing full benchmark execution and estimation."""

    def run_full_benchmark(self, force_rerun: bool = False) -> DeviceBenchmark:
        """Run complete device benchmark suite."""
        # Check for cached results first
        if not force_rerun:
            cached = self.load_cached_benchmark()
            if cached:
                print(f"   ✅ Using cached benchmark results (device: {cached.device_id[:HASH_DISPLAY_LENGTH]}...)")
                self.cached_benchmark = cached
                return cached
        
        print("🚀 Running device performance benchmark...")
        print("   This will take 1-2 minutes to measure your hardware performance")
        
        import platform
        
        device_id = self.get_device_id()
        device_type = self.get_device_type()
        
        print(f"   🖥️ Device ID: {device_id[:HASH_DISPLAY_LENGTH]}... ({device_type})")
        
        # Run all benchmarks with error handling
        try:
            model_loading = self.run_model_loading_benchmark()
            if model_loading is None:
                print(f"   ❌ Model loading benchmark returned None")
                raise DeviceBenchmarkError(task_name="model_loading")
        except Exception as e:
            print(f"   ❌ Model loading benchmark failed: {e}")
            raise
            
        try:
            benchmark_eval = self.run_benchmark_eval_test()
            if benchmark_eval is None:
                print(f"   ⚠️ Evaluation benchmark returned None, using default value")
                benchmark_eval = AGENT_BENCH_EVAL_DEFAULT_SECONDS
        except Exception as e:
            print(f"   ❌ Evaluation benchmark failed: {e}")
            benchmark_eval = AGENT_BENCH_EVAL_DEFAULT_SECONDS
            
        try:
            classifier_training = self.run_classifier_training_test()
            if classifier_training is None:
                print(f"   ⚠️ Classifier training benchmark returned None, using default value")
                classifier_training = AGENT_CLASSIFIER_TRAINING_DEFAULT_SECONDS
        except Exception as e:
            print(f"   ❌ Classifier training benchmark failed: {e}")
            classifier_training = AGENT_CLASSIFIER_TRAINING_DEFAULT_SECONDS
            
        try:
            steering = self.run_steering_test()
            if steering is None:
                print(f"   ❌ Steering benchmark returned None")
                raise DeviceBenchmarkError(task_name="steering")
        except Exception as e:
            print(f"   ❌ Steering benchmark failed: {e}")
            raise
            
        try:
            data_generation = self.run_data_generation_test()
            if data_generation is None:
                print(f"   ❌ Data generation benchmark returned None")
                raise DeviceBenchmarkError(task_name="data_generation")
        except Exception as e:
            print(f"   ❌ Data generation benchmark failed: {e}")
            raise
        
        # Create benchmark result
        benchmark = DeviceBenchmark(
            device_id=device_id,
            device_type=device_type,
            model_loading_seconds=model_loading,
            benchmark_eval_seconds_per_100_examples=benchmark_eval,
            classifier_training_seconds_per_100_samples=classifier_training,
            data_generation_seconds_per_example=data_generation,
            steering_seconds_per_example=steering,
            benchmark_timestamp=time.time(),
            python_version=sys.version,
            platform_info=platform.platform()
        )
        
        # Save results
        self.save_benchmark(benchmark)
        self.cached_benchmark = benchmark
        
        print("   ✅ Benchmark complete!")
        print(f"      Model loading: {model_loading:.1f}s")
        print(f"      Evaluation: {benchmark_eval:.1f}s per 100 examples")
        print(f"      Classifier creation: {classifier_training:.1f}s per 100 classifiers")
        print(f"      Steering: {steering:.1f}s per example")
        print(f"      Generation: {data_generation:.1f}s per example")
        
        return benchmark
    
    def get_current_benchmark(self, auto_run: bool = True) -> Optional[DeviceBenchmark]:
        """Get current device benchmark, optionally auto-running if needed."""
        if self.cached_benchmark:
            return self.cached_benchmark
            
        cached = self.load_cached_benchmark()
        if cached:
            self.cached_benchmark = cached
            return cached
            
        if auto_run:
            return self.run_full_benchmark()
            
        return None
    
    def estimate_task_time(self, task_type: str, quantity: int = 1) -> float:
        """
        Estimate time for a specific task type and quantity.
        
        Args:
            task_type: Type of task ("model_loading", "benchmark_eval", etc.)
            quantity: Number of items (examples, samples, etc.)
            
        Returns:
            Estimated time in seconds
        """
        benchmark = self.get_current_benchmark()
        if not benchmark:
            raise NoBenchmarkDataError()
        else:
            # Use actual benchmark results
            if task_type == "model_loading":
                return benchmark.model_loading_seconds
            elif task_type == "benchmark_eval":
                base_time = benchmark.benchmark_eval_seconds_per_100_examples
                return (base_time / 100.0) * quantity
            elif task_type == "classifier_training":
                base_time = benchmark.classifier_training_seconds_per_100_samples  # Actually per 100 classifiers now
                return (base_time / 100.0) * quantity
            elif task_type == "steering":
                return benchmark.steering_seconds_per_example * quantity
            elif task_type == "data_generation":
                return benchmark.data_generation_seconds_per_example * quantity
            else:
                raise UnknownTypeError(entity_type="task_type", value=task_type)


# Global benchmarker instance
_device_benchmarker = DeviceBenchmarker()


