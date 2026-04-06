"""Run test_benchmark for all benchmarks in _REGISTRY and populate HF cache."""

import os
import sys
import json
import time
from pathlib import Path

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ.setdefault("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

sys.path.insert(0, str(Path(__file__).parent))

from wisent.extractors.lm_eval.lm_extractor_registry import _REGISTRY
from wisent.support.examples.scripts.discovery.validation.test_single_benchmark import test_benchmark
from wisent.core.utils.services.benchmarks.registry.benchmark_registry import get_broken_tasks

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

broken = set(t.lower() for t in get_broken_tasks())
all_tasks = sorted(k for k in _REGISTRY.keys() if k not in broken)
total = len(all_tasks)
skipped_broken = len(_REGISTRY) - total
print(f"Running tests for {total} benchmarks from _REGISTRY (skipped {skipped_broken} broken)")
print(f"Model: {MODEL_NAME}\n")

pass_count = 0
fail_count = 0
skip_count = 0
start = time.time()

for i, task_name in enumerate(all_tasks, 1):
    print(f"[{i}/{total}] {task_name}...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = test_benchmark(task_name, skip_cache=False, model_name=MODEL_NAME)
        ext = result.get("extraction", {}).get("status", "SKIP")
        evl = result.get("evaluator", {}).get("status", "SKIP")
        elapsed = time.time() - t0

        if ext != "FAIL" and evl != "FAIL":
            pass_count += 1
            print(f"PASS ({elapsed:.1f}s)")
        else:
            fail_count += 1
            print(f"FAIL ext={ext} evl={evl} ({elapsed:.1f}s)")
    except Exception as e:
        fail_count += 1
        print(f"ERROR: {str(e)[:100]} ({time.time()-t0:.1f}s)")

elapsed_total = time.time() - start
print(f"\n{'='*60}")
print(f"PASS: {pass_count}  FAIL: {fail_count}  TOTAL: {total}")
print(f"Time: {elapsed_total:.0f}s")
