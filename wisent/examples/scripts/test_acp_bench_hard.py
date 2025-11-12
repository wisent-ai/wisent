#!/usr/bin/env python3
"""Test acp_bench_hard benchmark."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from wisent.examples.scripts.test_one_benchmark import test_benchmark

def main():
    print("="*80)
    print("Testing 'acp_bench_hard' benchmark...")
    print("="*80)
    success = test_benchmark("acp_bench_hard", "mock", output_dir="results")
    print(f"\nacp_bench_hard Result: {'SUCCESS' if success else 'FAILED'}")

if __name__ == "__main__":
    main()
