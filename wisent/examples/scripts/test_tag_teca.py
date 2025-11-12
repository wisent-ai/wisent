#!/usr/bin/env python3
"""Test Tag and teca benchmarks."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from wisent.examples.scripts.test_one_benchmark import test_benchmark

def main():
    model_name = "mock"

    print("="*80)
    print("Testing 'Tag' benchmark...")
    print("="*80)
    success_tag = test_benchmark("Tag", model_name, output_dir="results")
    print(f"\nTag Result: {'SUCCESS' if success_tag else 'FAILED'}")

    print("\n" + "="*80)
    print("Testing 'teca' benchmark...")
    print("="*80)
    success_teca = test_benchmark("teca", model_name, output_dir="results")
    print(f"\nteca Result: {'SUCCESS' if success_teca else 'FAILED'}")

if __name__ == "__main__":
    main()
