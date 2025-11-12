#!/usr/bin/env python3
"""Test advanced_ai_risk benchmark."""

import sys
sys.path.insert(0, '/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source')

from wisent.examples.scripts.test_one_benchmark import test_benchmark

def main():
    print("="*80)
    print("Testing 'advanced_ai_risk' benchmark...")
    print("="*80)
    success = test_benchmark("advanced_ai_risk", "mock", output_dir="results")
    print(f"\nadvanced_ai_risk Result: {'SUCCESS' if success else 'FAILED'}")

if __name__ == "__main__":
    main()
