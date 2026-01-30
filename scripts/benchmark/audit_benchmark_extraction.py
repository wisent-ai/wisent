#!/usr/bin/env python3
"""
Audit script to definitively check which benchmarks have extraction issues.
For each benchmark with < 500 pairs, checks the actual dataset size.
"""
import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import warnings
warnings.filterwarnings('ignore')

from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import get_extractor, _REGISTRY
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import UnsupportedHuggingFaceBenchmarkError


def get_benchmarks_under_500():
    """Get all benchmarks with < 500 pairs from database."""
    conn = psycopg2.connect('postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:6543/postgres')
    cur = conn.cursor()
    cur.execute('''
        SELECT cps.name, COUNT(cp.id) as pair_count
        FROM "ContrastivePairSet" cps
        LEFT JOIN "ContrastivePair" cp ON cp."setId" = cps.id AND cp."isActive" = true
        GROUP BY cps.id, cps.name
        HAVING COUNT(cp.id) < 500 AND COUNT(cp.id) > 0
        ORDER BY COUNT(cp.id) ASC
    ''')
    results = cur.fetchall()
    conn.close()
    return results


def extract_task_name(benchmark_name):
    """Extract the task name from a benchmark path like 'category/task'."""
    if '/' in benchmark_name:
        return benchmark_name.split('/')[-1]
    return benchmark_name


def test_extractor(task_name):
    """Try to get an extractor and test extraction for a task."""
    try:
        extractor = get_extractor(task_name)
        # Try to extract a few pairs to verify it works
        pairs = extractor.extract_contrastive_pairs(limit=5)
        return {
            'has_extractor': True,
            'works': True,
            'sample_count': len(pairs),
            'error': None
        }
    except UnsupportedHuggingFaceBenchmarkError:
        return {
            'has_extractor': False,
            'works': False,
            'sample_count': 0,
            'error': 'No HF extractor'
        }
    except Exception as e:
        return {
            'has_extractor': True,
            'works': False,
            'sample_count': 0,
            'error': str(e)[:100]
        }


def main():
    print("=" * 80)
    print("BENCHMARK EXTRACTION AUDIT")
    print("Testing all benchmarks with < 500 pairs against actual extractors")
    print("=" * 80)
    print()

    benchmarks = get_benchmarks_under_500()
    print(f"Found {len(benchmarks)} benchmarks with < 500 pairs\n")

    # Categories for results
    broken = []  # Extractor exists but fails
    no_extractor = []  # No HF extractor (uses lm-eval or other)
    working = []  # Extractor works, this is actual dataset size

    for name, db_count in benchmarks:
        task_name = extract_task_name(name)
        result = test_extractor(task_name)

        if not result['has_extractor']:
            no_extractor.append((name, db_count, result['error']))
        elif not result['works']:
            broken.append((name, db_count, result['error']))
        else:
            working.append((name, db_count, result['sample_count']))

    # Print results
    print("\n" + "=" * 80)
    print("🔴 BROKEN EXTRACTORS (extraction fails)")
    print("=" * 80)
    for name, count, error in sorted(broken, key=lambda x: x[1]):
        print(f"  {count:>4} | {name:<45} | {error}")
    print(f"\nTotal broken: {len(broken)}")

    print("\n" + "=" * 80)
    print("🟡 NO HF EXTRACTOR (uses lm-eval or other source)")
    print("=" * 80)
    for name, count, error in sorted(no_extractor, key=lambda x: x[1]):
        print(f"  {count:>4} | {name}")
    print(f"\nTotal without HF extractor: {len(no_extractor)}")

    print("\n" + "=" * 80)
    print("🟢 WORKING EXTRACTORS (need to verify max dataset size)")
    print("=" * 80)
    for name, count, sample in sorted(working, key=lambda x: x[1]):
        status = "✓" if sample > 0 else "?"
        print(f"  {count:>4} | {name:<45} | {status} got {sample} samples")
    print(f"\nTotal working: {len(working)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Broken extractors (need fix):     {len(broken)}")
    print(f"  No HF extractor (check lm-eval):  {len(no_extractor)}")
    print(f"  Working extractors:               {len(working)}")


if __name__ == "__main__":
    main()
