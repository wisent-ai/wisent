#!/usr/bin/env python3
"""Test the teca benchmark to see if it's the correct match for Tag."""

import sys
from wisent.examples.scripts.test_one_benchmark import test_benchmark

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mock"

    print("Testing 'teca' benchmark...")
    result = test_benchmark("teca", model_name, limit=2)

    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(f"Success: {result['success']}")
    print(f"Error: {result.get('error', 'None')}")

    if result['success'] and result.get('pairs'):
        print(f"\nNumber of pairs: {len(result['pairs'])}")
        print("\nFirst pair:")
        pair = result['pairs'][0]
        print(f"  Prompt: {pair['prompt'][:200]}...")
        print(f"  Positive: {pair['positive_response']}")
        print(f"  Negative: {pair['negative_response']}")

if __name__ == "__main__":
    main()
