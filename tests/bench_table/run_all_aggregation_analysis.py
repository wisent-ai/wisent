#!/usr/bin/env python3
"""
Run aggregation analysis for all benchmarks.

This script sequentially runs aggregation analysis for:
1. GSM8K (math reasoning)
2. BoolQ (question answering)
3. SST2 (sentiment analysis)
4. CB (natural language inference)

Each program saves its plots to its respective directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_script(script_path: Path, benchmark_name: str) -> bool:
    """
    Run a single aggregation analysis script.

    Args:
        script_path: Path to the Python script
        benchmark_name: Name of the benchmark for logging

    Returns:
        True if successful, False if failed
    """
    print("\n" + "=" * 80)
    print(f"RUNNING: {benchmark_name} Aggregation Analysis")
    print(f"Script: {script_path}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    if not script_path.exists():
        print(f"❌ ERROR: Script not found at {script_path}")
        return False

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=script_path.parent,  # Run in script's directory
            check=True,
            capture_output=False,  # Show output in real-time
        )

        print("\n" + "-" * 80)
        print(f"✅ SUCCESS: {benchmark_name} completed")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        return True

    except subprocess.CalledProcessError as e:
        print("\n" + "-" * 80)
        print(f"❌ FAILED: {benchmark_name} failed with exit code {e.returncode}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        return False

    except KeyboardInterrupt:
        print("\n" + "-" * 80)
        print(f"⚠️  INTERRUPTED: {benchmark_name} was interrupted by user")
        print("-" * 80)
        raise

    except Exception as e:
        print("\n" + "-" * 80)
        print(f"❌ ERROR: {benchmark_name} failed with error: {e}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
        return False


def main():
    """Run all aggregation analysis scripts."""

    print("\n" + "#" * 80)
    print("# AGGREGATION ANALYSIS - ALL BENCHMARKS")
    print("#" * 80)
    print(f"# Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 80 + "\n")

    # Get the bench_table directory
    bench_table_dir = Path(__file__).parent

    # Define scripts to run
    scripts = [
        {
            "path": bench_table_dir / "gsm8k" / "aggregations_analysis.py",
            "name": "GSM8K",
        },
        {
            "path": bench_table_dir / "boolq" / "aggregations_analysis.py",
            "name": "BoolQ",
        },
        {
            "path": bench_table_dir / "sst2" / "aggregations_analysis.py",
            "name": "SST2",
        },
        {
            "path": bench_table_dir / "cb" / "aggregations_analysis.py",
            "name": "CB",
        },
    ]

    # Track results
    results = {}
    start_time = datetime.now()

    # Run each script
    for i, script_info in enumerate(scripts, 1):
        print(f"\n[{i}/{len(scripts)}] Processing {script_info['name']}...")

        success = run_script(script_info["path"], script_info["name"])
        results[script_info["name"]] = success

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n\n" + "#" * 80)
    print("# SUMMARY")
    print("#" * 80)
    print(f"# Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Duration: {duration}")
    print("#" * 80)
    print()

    # Print results table
    print("Results:")
    print("-" * 40)
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name:20} {status}")
    print("-" * 40)

    # Overall status
    total = len(results)
    successful = sum(results.values())
    failed = total - successful

    print()
    print(f"Total:      {total}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print()

    # Exit code
    if failed > 0:
        print("⚠️  Some scripts failed. Check the output above for details.")
        sys.exit(1)
    else:
        print("✅ All aggregation analyses completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("❌ INTERRUPTED: Execution interrupted by user (Ctrl+C)")
        print("=" * 80)
        sys.exit(130)
    except Exception as e:
        print("\n\n" + "=" * 80)
        print(f"❌ FATAL ERROR: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
