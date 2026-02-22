"""Steering and data generation benchmark tests."""
import time
import torch
from typing import Dict, Any, Optional
from wisent.core.utils import resolve_default_device

class DeviceBenchTestsMixin2:
    """Mixin: steering and data generation tests."""

    def run_steering_test(self) -> float:
        """Benchmark steering performance using real CLI functionality."""
        print("   📊 Benchmarking steering performance...")
        
        # Create steering test script using actual CLI
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent.cli import run_task_pipeline
    
    # Run actual steering with real model and minimal examples
    run_task_pipeline(
        task_name="truthfulqa_mc",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        limit=2,  # Minimum examples for timing
        steering_mode=True,
        steering_method="CAA",
        steering_strength=1.0,
        layer="15",
        verbose=False,
        allow_small_dataset=True,
        output_mode="likelihoods"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    # Time per example
    time_per_example = total_time / 2
    print(f"BENCHMARK_RESULT:{time_per_example}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    raise
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name

            result = subprocess.run([
                sys.executable,
                temp_script,
            ], capture_output=True, text=True, timeout=300)

            os.unlink(temp_script)

            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    steering_time = float(line.split(':')[1])
                    print(f"      Steering: {steering_time:.1f}s per example")
                    return steering_time

            print("   ❌ No BENCHMARK_RESULT found in steering output!")
            print(result.stdout)
            return None

        except Exception as e:
            print(f"      Error in steering benchmark: {e}")
            raise DeviceBenchmarkError(task_name="steering", cause=e)
    
    def run_data_generation_test(self) -> float:
        """Benchmark data generation performance using real synthetic generation.""" 
        print("   📊 Benchmarking data generation...")
        
        # Create data generation test script using actual synthetic pair generation
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent.core.model import Model
    from wisent.core.contrastive_pairs.generate_synthetically import SyntheticContrastivePairGenerator
    
    # Load the actual model
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    
    # Create generator and generate actual synthetic pairs
    generator = SyntheticContrastivePairGenerator(model)
    
    # Generate a small set of pairs for timing
    pair_set = generator.generate_contrastive_pair_set(
        trait_description="accuracy and truthfulness",
        num_pairs=1,  # Minimum needed for estimation
        name="benchmark_test"
    )
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate time per generated pair (each pair has 2 responses)
    num_generated_responses = len(pair_set.pairs) * 2
    if num_generated_responses == 0:
        raise InsufficientDataError(reason="No pairs were generated during data generation benchmark")
    
    time_per_example = total_time / num_generated_responses
    print(f"BENCHMARK_RESULT:{time_per_example}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    raise
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=300)  # 5-minute timeout
            
            os.unlink(temp_script)
            
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    generation_time = float(line.split(':')[1])
                    print(f"      Data generation: {generation_time:.1f}s per example")
                    return generation_time
                    
        except Exception as e:
            print(f"      Error in data generation benchmark: {e}")
            raise DeviceBenchmarkError(task_name="data_generation", cause=e)
    
