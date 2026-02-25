"""Device benchmarking and estimation utilities."""
import logging
from typing import Dict, List
from wisent.core.errors import NoBenchmarkDataError
from wisent.core.agent.resources._budget_manager import BudgetManager
from wisent.core.agent.resources._budget_functions import get_budget_manager
from wisent.core.constants import AGENT_RESOURCE_BUDGET_MINUTES, DEVICE_HASH_PREFIX, SEPARATOR_WIDTH_COMPACT, SEPARATOR_WIDTH_NARROW, SEPARATOR_WIDTH_MEDIUM, SECONDS_PER_MINUTE
logger = logging.getLogger(__name__)

def estimate_completion_time_minutes(tasks: List[str]) -> float:
    """
    Estimate total completion time for tasks in minutes.
    
    Args:
        tasks: List of task names
        
    Returns:
        Estimated time in minutes
    """
    seconds = _budget_manager.estimate_completion_time(tasks)
    return seconds / SECONDS_PER_MINUTE


def track_task_performance(task_name: str, start_time: float, end_time: float) -> None:
    """
    Track actual task performance to improve future estimates.
    
    Args:
        task_name: Name of the task
        start_time: Start timestamp
        end_time: End timestamp
    """
    _budget_manager.track_task_execution(task_name, start_time, end_time)


def run_device_benchmark(force_rerun: bool = False) -> None:
    """
    Run device performance benchmark and save results.
    
    Args:
        force_rerun: Force re-run even if cached results exist
    """
    from .device_benchmarks import ensure_benchmark_exists
    
    print("🚀 Running device performance benchmark...")
    benchmark = ensure_benchmark_exists(force_rerun=force_rerun)
    
    print("\n✅ Benchmark Results:")
    print("=" * SEPARATOR_WIDTH_MEDIUM)
    print(f"Device ID: {benchmark.device_id[:DEVICE_HASH_PREFIX]}...")
    print(f"Device Type: {benchmark.device_type}")
    print(f"Model Loading: {benchmark.model_loading_seconds:.1f}s")
    print(f"Evaluation: {benchmark.benchmark_eval_seconds_per_100_examples:.1f}s per 100 examples")
    print(f"Classifier Training: {benchmark.classifier_training_seconds_per_100_samples:.1f}s per 100 samples")
    print(f"Steering: {benchmark.steering_seconds_per_example:.1f}s per example")
    print(f"Data Generation: {benchmark.data_generation_seconds_per_example:.1f}s per example")
    print(f"\nResults saved to: device_benchmarks.json")
    
    # Show some example estimates
    print("\n📊 Example Time Estimates:")
    print("-" * SEPARATOR_WIDTH_COMPACT)
    print(f"Loading model: {benchmark.model_loading_seconds:.1f}s")
    print(f"100 eval examples: {benchmark.benchmark_eval_seconds_per_100_examples:.1f}s")
    print(f"Training classifier (200 samples): {(benchmark.classifier_training_seconds_per_100_samples * 2):.1f}s")
    print(f"10 steering examples: {(benchmark.steering_seconds_per_example * 10):.1f}s")


def get_device_info() -> Dict[str, str]:
    """Get current device information."""
    from .device_benchmarks import get_current_device_info
    return get_current_device_info()


def estimate_task_time_direct(task_type: str, quantity: int = 1) -> float:
    """
    Direct estimate of task time using device benchmarks.
    
    Args:
        task_type: Type of task ("model_loading", "benchmark_eval", etc.)
        quantity: Number of items
        
    Returns:
        Estimated time in seconds
    """
    from .device_benchmarks import estimate_task_time
    return estimate_task_time(task_type, quantity)


# CLI functionality for budget management
def main():
    """CLI entry point for budget management and benchmarking."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="wisent budget management and device benchmarking"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run device benchmark')
    benchmark_parser.add_argument(
        '--force', '-f', 
        action='store_true', 
        help='Force re-run benchmark even if cached results exist'
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show device information')
    
    # Estimate command
    estimate_parser = subparsers.add_parser('estimate', help='Estimate task time')
    estimate_parser.add_argument('task_type', help='Type of task')
    estimate_parser.add_argument('quantity', type=int, help='Number of items')
    
    # Budget command
    budget_parser = subparsers.add_parser('budget', help='Calculate budget allocations')
    budget_parser.add_argument('--time-minutes', '-t', type=float, default=AGENT_RESOURCE_BUDGET_MINUTES, help='Time budget in minutes')
    budget_parser.add_argument('--task-type', default='benchmark_evaluation', help='Task type to optimize for')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'benchmark':
            run_device_benchmark(force_rerun=args.force)
            
        elif args.command == 'info':
            print("🖥️ Current Device Information")
            print("=" * SEPARATOR_WIDTH_NARROW)
            device_info = get_device_info()
            for key, value in device_info.items():
                print(f"{key}: {value}")
                
        elif args.command == 'estimate':
            estimated_seconds = estimate_task_time_direct(args.task_type, args.quantity)
            print(f"⏱️ Estimated time for {args.quantity}x {args.task_type}: {estimated_seconds:.1f} seconds ({estimated_seconds/SECONDS_PER_MINUTE:.2f} minutes)")
            
        elif args.command == 'budget':
            max_tasks = calculate_max_tasks_for_time_budget(args.task_type, args.time_minutes)
            
            # Map task types to benchmark types for direct estimation
            benchmark_mapping = {
                "benchmark_evaluation": "benchmark_eval",
                "classifier_training": "classifier_training", 
                "data_generation": "data_generation",
                "steering": "steering",
                "model_loading": "model_loading"
            }
            
            benchmark_type = benchmark_mapping.get(args.task_type, "benchmark_eval")
            
            # Get time per individual task unit
            if benchmark_type in ["benchmark_eval", "classifier_training"]:
                task_time = estimate_task_time_direct(benchmark_type, 100) / 100  # Per unit
            else:
                task_time = estimate_task_time_direct(benchmark_type, 1)
            
            total_time = max_tasks * task_time
            
            print(f"💰 Budget Analysis:")
            print(f"Time budget: {args.time_minutes:.1f} minutes ({args.time_minutes * SECONDS_PER_MINUTE:.0f} seconds)")
            print(f"Task type: {args.task_type} (mapped to {benchmark_type})")
            print(f"Time per task: {task_time:.2f} seconds")
            print(f"Max tasks: {max_tasks}")
            print(f"Total estimated time: {total_time:.1f} seconds ({total_time/SECONDS_PER_MINUTE:.2f} minutes)")
            print(f"Budget utilization: {(total_time / (args.time_minutes * SECONDS_PER_MINUTE)) * 100:.1f}%")
            
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
