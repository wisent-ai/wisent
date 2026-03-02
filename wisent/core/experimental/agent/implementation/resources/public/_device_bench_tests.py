"""Model loading, eval, and classifier benchmark tests."""
import time
import os
import tempfile
import subprocess
import sys
import torch
from typing import Dict, Any, Optional
from wisent.core.utils.config_tools.constants import DEFAULT_LAYER, AGENT_BENCH_MIN_PAIRS_TRAINING, BENCH_TEST_SAMPLE_SIZE
from wisent.core.utils.core.hardware import subprocess_timeout_s, subprocess_timeout_long_s
from wisent.core.utils import resolve_default_device

class DeviceBenchTestsMixin1:
    """Mixin: model loading, eval, and classifier training tests."""

    def run_model_loading_benchmark(self) -> float:
        """Benchmark actual model loading time using the real model."""
        print("   📊 Benchmarking model loading...")
        
        # Create actual model loading test script
        test_script = '''
import time
import sys
sys.path.append('.')

start_time = time.time()
try:
    from wisent.core.primitives.models.core.wisent_model import Model
    # Use the actual model that will be used in production
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    end_time = time.time()
    print(f"BENCHMARK_RESULT:{end_time - start_time}")
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    raise
'''
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            
            # Run with 2-minute timeout
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=subprocess_timeout_s())

            # Clean up
            os.unlink(temp_script)

            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('BENCHMARK_RESULT:'):
                    loading_time = float(line.split(':')[1])
                    print(f"      Model loading: {loading_time:.1f}s")
                    return loading_time
                    
        except Exception as e:
            print(f"      Error in model loading benchmark: {e}")
            raise DeviceBenchmarkError(task_name="model_loading", cause=e)
    
    def run_benchmark_eval_test(self) -> float:
        """Benchmark evaluation performance using real CLI functionality."""
        print("   📊 Benchmarking evaluation performance...")
        print("   🔧 DEBUG: Creating evaluation test script...")
        
        # Create evaluation test script using actual CLI
        test_script = '''
import time
import sys
sys.path.append('.')

print("BENCHMARK_DEBUG: Starting evaluation benchmark")
start_time = time.time()
try:
    print("BENCHMARK_DEBUG: Importing CLI...")
    from wisent.cli import run_task_pipeline
    print("BENCHMARK_DEBUG: CLI imported successfully")

    print("BENCHMARK_DEBUG: Running task pipeline...")
    # Run actual evaluation with real model and minimal examples
    run_task_pipeline(
        task_name="truthfulqa_mc",
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        layer="__LAYER__",  # Required parameter
        limit=__BENCH_SAMPLE__,  # Minimum examples for timing
        steering_mode=False,  # No steering for baseline timing
        verbose=False,
        allow_small_dataset=True,
        output_mode="likelihoods"
    )
    print("BENCHMARK_DEBUG: Task pipeline completed")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"BENCHMARK_DEBUG: Total time: {total_time}s for __BENCH_SAMPLE__ examples")
    # Scale to per-100-examples
    time_per_100 = (total_time / __BENCH_SAMPLE__) * 100
    print(f"BENCHMARK_DEBUG: Scaled time per 100: {time_per_100}s")
    print(f"BENCHMARK_RESULT:{time_per_100}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    import traceback
    traceback.print_exc()
    raise
'''
        test_script = test_script.replace("__LAYER__", str(DEFAULT_LAYER))
        test_script = test_script.replace("__BENCH_SAMPLE__", str(BENCH_TEST_SAMPLE_SIZE))
        print("   🔧 DEBUG: Writing test script to temporary file...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            print(f"   🔧 DEBUG: Test script written to {temp_script}")
            
            print("   🔧 DEBUG: Running evaluation subprocess...")
            result = subprocess.run([
                sys.executable, temp_script
            ], capture_output=True, text=True, timeout=subprocess_timeout_s())
            
            print(f"   🔧 DEBUG: Subprocess completed with return code: {result.returncode}")
            print(f"   🔧 DEBUG: Stdout length: {len(result.stdout)} chars")
            print(f"   🔧 DEBUG: Stderr length: {len(result.stderr)} chars")
            
            if result.stderr:
                print(f"   ⚠️ DEBUG: Stderr content:\n{result.stderr}")
            
            os.unlink(temp_script)
            print("   🔧 DEBUG: Temporary script cleaned up")
            
            # Parse result
            print("   🔧 DEBUG: Parsing output for BENCHMARK_RESULT...")
            found_result = False
            for line in result.stdout.split('\n'):
                print(f"   🔍 DEBUG: Output line: {repr(line)}")
                if line.startswith('BENCHMARK_RESULT:'):
                    eval_time = float(line.split(':')[1])
                    print(f"      ✅ Evaluation: {eval_time:.1f}s per 100 examples")
                    found_result = True
                    return eval_time
            
            if not found_result:
                print("   ❌ DEBUG: No BENCHMARK_RESULT found in output!")
                print("   📜 DEBUG: Full stdout:")
                print(result.stdout)
                return None
                    
        except Exception as e:
            print(f"      ❌ Error in evaluation benchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_classifier_training_test(self) -> float:
        """Benchmark ACTUAL classifier training using real synthetic classifier creation."""
        print("   📊 Benchmarking classifier training...")
        print("   🔧 DEBUG: Creating classifier training test script...")
        
        # Create test script that uses real synthetic classifier creation
        test_script = '''
import time
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Optional
try:
    print("BENCHMARK_DEBUG: Importing required modules...")
    from wisent.core.primitives.models.core.wisent_model import Model
    from wisent.core.experimental.agent.diagnose.synthetic_classifier_option import create_classifier_from_trait_description
    from wisent.core.experimental.agent.budget import set_time_budget
    import time
    print("BENCHMARK_DEBUG: All modules imported successfully")
    
    print("BENCHMARK_DEBUG: Starting classifier benchmark")
    
    # Set a budget for the classifier creation
    print("BENCHMARK_DEBUG: Setting time budget...")
    set_time_budget(5.0)  # 5 minutes
    print("BENCHMARK_DEBUG: Set time budget to 5.0 minutes")
    
    # Load the actual model
    print("BENCHMARK_DEBUG: Loading model...")
    model_start = time.time()
    model = Model("meta-llama/Llama-3.1-8B-Instruct")
    model_time = time.time() - model_start
    print(f"BENCHMARK_DEBUG: Model loaded in {model_time}s")
    
    # Create ONE actual classifier using the real synthetic process
    print("BENCHMARK_DEBUG: Creating classifier...")
    classifier_start = time.time()
    classifier = create_classifier_from_trait_description(
        model=model,
        trait_description="accuracy and truthfulness",
        num_pairs=__BENCH_MIN_PAIRS__  # Minimum needed for training
    )
    classifier_time = time.time() - classifier_start
    print(f"BENCHMARK_DEBUG: Classifier created in {classifier_time}s")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"BENCHMARK_DEBUG: Total benchmark time: {total_time}s")
    
    # This is time for ONE complete classifier creation
    # Scale to "per 100 classifiers" for compatibility with existing code
    time_per_100 = total_time * 100
    print(f"BENCHMARK_DEBUG: Scaled time per 100 classifiers: {time_per_100}s")
    print(f"BENCHMARK_RESULT:{time_per_100}")
    
except Exception as e:
    print(f"BENCHMARK_ERROR:{e}")
    import traceback
    traceback.print_exc()
    raise
'''
        
        print("   🔧 DEBUG: Writing classifier test script to temporary file...")
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_script)
                temp_script = f.name
            print(f"   🔧 DEBUG: Classifier test script written to {temp_script}")

            print("   🔧 DEBUG: Running classifier training subprocess (20 min timeout)...")
            result = subprocess.run([
                sys.executable,
                temp_script,
            ], capture_output=True, text=True, timeout=subprocess_timeout_long_s())

            print(f"   🔧 DEBUG: Classifier subprocess completed with return code: {result.returncode}")
            print(f"   🔧 DEBUG: Stdout length: {len(result.stdout)} chars")
            print(f"   🔧 DEBUG: Stderr length: {len(result.stderr)} chars")

            if result.stderr:
                print(f"   ⚠️ DEBUG: Classifier stderr content:\n{result.stderr}")

            os.unlink(temp_script)
            print("   🔧 DEBUG: Classifier temporary script cleaned up")

            # Parse result
            print("   🔧 DEBUG: Parsing classifier output for BENCHMARK_RESULT...")
            for line in result.stdout.split('\n'):
                print(f"   🔍 DEBUG: Classifier output line: {repr(line)}")
                if line.startswith('BENCHMARK_RESULT:'):
                    training_time = float(line.split(':')[1])
                    print(f"      ✅ Classifier training: {training_time:.1f}s per 100 classifiers")
                    return training_time

            print("   ❌ DEBUG: No BENCHMARK_RESULT found in classifier output!")
            print("   📜 DEBUG: Full classifier stdout:")
            print(result.stdout)
            return None

        except Exception as e:
            print(f"      ❌ Error in classifier training benchmark: {e}")
            import traceback
            traceback.print_exc()
            return None
    
