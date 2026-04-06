"""Analyze and clean failing benchmarks grouped by error detail.

Usage:
    python analyze_failures.py          # show failure groups
    python analyze_failures.py clean-stale  # delete stale cache entries
    python analyze_failures.py test-import  # test if wisent imports work
"""
import json
import os
import sys

cache_dir = os.path.join(os.path.dirname(__file__), "hf_cache", "test_results")

mode = sys.argv[1] if len(sys.argv) > 1 else "analyze"

failures = {}
for fname in sorted(os.listdir(cache_dir)):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(cache_dir, fname)) as f:
        data = json.load(f)
    ext = data.get("extraction", {})
    if ext.get("status") != "PASS":
        detail = ext.get("detail", "no detail")
        if len(detail) > 150:
            detail = detail[:150]
        failures[fname[:-5]] = detail

if mode == "clean-stale":
    # Delete all remaining failure cache files (all are stale)
    count = 0
    for name in failures:
        path = os.path.join(cache_dir, f"{name}.json")
        if os.path.exists(path):
            os.remove(path)
            count += 1
    print(f"Deleted {count} stale cache files")
elif mode == "test-import":
    print("Testing wisent import chain...")
    try:
        from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
        print(f"OK: _REGISTRY has {len(_REGISTRY)} entries")
    except Exception:
        import traceback
        traceback.print_exc()
else:
    detail_groups = {}
    for name, detail in failures.items():
        detail_groups.setdefault(detail, []).append(name)

    for detail, names in sorted(detail_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n[{len(names)} benchmarks] {detail}")
        for n in names:
            print(f"  - {n}")

    print(f"\nTotal failing: {len(failures)}")
    print(f"Distinct error groups: {len(detail_groups)}")
