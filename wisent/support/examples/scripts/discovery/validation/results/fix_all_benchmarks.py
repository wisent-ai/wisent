"""Fix all failing benchmarks using Claude Code subagents.

For each failing benchmark in the HF cache:
1. Run the test to get the current error
2. If it now passes (stale cache), update cache and move on
3. If still failing, launch a Claude Code subagent to fix it
4. The agent reads code, diagnoses, applies a fix, and verifies
5. After agent completes, re-test and update cache

Usage:
    python fix_all_benchmarks.py
    python fix_all_benchmarks.py --dry-run
    python fix_all_benchmarks.py --family belebele
    python fix_all_benchmarks.py --parallel 5
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Force unbuffered output
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("Starting fix_all_benchmarks...", flush=True)

os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = Path(__file__).resolve().parent / "hf_cache" / "test_results"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cached_results() -> dict[str, dict]:
    """Load all cached test results from local HF cache."""
    results = {}
    if not CACHE_DIR.exists():
        sys.exit(f"Cache not found: {CACHE_DIR}")
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    files.sort()
    print(f"  Found {len(files)} cache files...", flush=True)
    for i, fname in enumerate(files):
        try:
            with open(CACHE_DIR / fname) as fh:
                results[fname[:-5]] = json.load(fh)  # strip .json
        except (json.JSONDecodeError, OSError):
            continue
        if (i + 1) % 200 == 0:
            print(f"  Loaded {i + 1}/{len(files)}...", flush=True)
    return results


def get_failures(cached: dict[str, dict]) -> list[str]:
    """Return benchmark names where extraction != PASS."""
    return [
        name for name, data in cached.items()
        if data.get("extraction", {}).get("status") != "PASS"
    ]


def update_cache(benchmark: str, result: dict) -> None:
    """Write updated result to local cache."""
    (CACHE_DIR / f"{benchmark}.json").write_text(
        json.dumps(result, indent=2, default=str)
    )


def upload_to_hf(benchmark: str, result: dict) -> None:
    """Upload result to HuggingFace (best-effort)."""
    try:
        from wisent.core.reading.modules.utilities.data.sources.hf.hf_writers import (
            upload_test_results,
        )
        upload_test_results(benchmark, result)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

def run_test(benchmark: str, timeout: int = 300) -> tuple[str, str, str]:
    """Run test_single_benchmark. Returns (ext_status, eval_status, output)."""
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m",
                "wisent.support.examples.scripts.discovery.validation.test_single_benchmark",
                benchmark, "--skip-cache",
            ],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT_ROOT),
        )
        combined = proc.stdout + proc.stderr
        ext = "FAIL"
        evl = "SKIP"
        for line in combined.split("\n"):
            if "extraction=PASS" in line:
                ext = "PASS"
            if "evaluator=PASS" in line:
                evl = "PASS"
            elif "evaluator=SKIP_NO_MODEL" in line:
                evl = "SKIP_NO_MODEL"
        return ext, evl, combined
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "SKIP", "timed out"
    except Exception as e:
        return "ERROR", "SKIP", str(e)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _find_claude() -> str | None:
    return shutil.which("claude")


def build_agent_prompt(benchmark: str, error_output: str) -> str:
    """Build the prompt for the Claude Code subagent."""
    error_excerpt = error_output[-3000:] if len(error_output) > 3000 else error_output

    return f"""Fix the failing benchmark "{benchmark}" in the wisent-open-source project at {PROJECT_ROOT}.

Error output from test_single_benchmark --skip-cache:
```
{error_excerpt}
```

Key files you may need to modify:
- wisent/core/utils/infra_tools/data/loaders/lm_eval/_lm_loader_task_mapping.py (TASK_NAME_MAPPING, GROUP_TASK_EXPANSIONS)
- wisent/core/utils/infra_tools/data/loaders/lm_eval/lm_loader.py
- wisent/extractors/lm_eval/registry/lm_extractor_registry.py
- wisent/extractors/lm_eval/registry/lm_task_pairs_generation.py
- wisent/extractors/lm_eval/registry/lm_task_extractors/ (extractors)
- wisent/extractors/hf/registry/hf_task_extractors/ (HF extractors)
- wisent/support/parameters/lm_eval/broken_in_lm_eval.json (genuinely broken benchmarks)

Instructions:
1. Read the error output carefully
2. Read the relevant code to understand the root cause
3. Apply the minimal fix. Common fixes:
   - Add a TASK_NAME_MAPPING entry if the task name doesn't match lm-eval's name
   - Add to GROUP_TASK_EXPANSIONS if it's a group task needing subtask expansion
   - Fix an extractor if it doesn't handle the data schema correctly
   - Add to broken_in_lm_eval.json ONLY if the benchmark is genuinely broken
     (deprecated dataset script, gated dataset, upstream lm-eval bug)
4. Verify your fix:
   python3 {PROJECT_ROOT}/wisent/support/examples/scripts/discovery/validation/test_single_benchmark.py {benchmark} --skip-cache
5. The test MUST show extraction=PASS (or the benchmark must be in broken_in_lm_eval.json)

Do NOT make assumptions. Read the code first. Make targeted fixes only."""


def fix_benchmark(benchmark: str, error_output: str,
                  agent_timeout: int = 600) -> tuple[str, str, str]:
    """Launch a Claude Code agent to fix one benchmark.

    Returns:
        Tuple of (benchmark_name, outcome, description)
        outcome is one of: "fixed", "marked_broken", "agent_failed"
    """
    claude = _find_claude()
    if not claude:
        return benchmark, "agent_failed", "claude CLI not found"

    prompt = build_agent_prompt(benchmark, error_output)

    try:
        subprocess.run(
            [claude, "--print", "--dangerously-skip-permissions", prompt],
            capture_output=True, text=True, timeout=agent_timeout,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return benchmark, "agent_failed", "agent timed out"
    except Exception as e:
        return benchmark, "agent_failed", str(e)

    # Verify: re-test after agent ran
    ext, evl, _ = run_test(benchmark)
    if ext == "PASS":
        return benchmark, "fixed", "agent fixed it"

    # Check if agent added it to broken list
    try:
        broken = json.loads(
            (PROJECT_ROOT / "wisent" / "support" / "parameters"
             / "lm_eval" / "broken_in_lm_eval.json").read_text()
        )
        if benchmark in broken:
            return benchmark, "marked_broken", "agent marked it broken"
    except Exception:
        pass

    return benchmark, "agent_failed", f"still fails after agent (extraction={ext})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fix all failing benchmarks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only test and report, don't launch agents")
    parser.add_argument("--family", type=str, default=None,
                        help="Only process benchmarks matching this prefix")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout per benchmark test (seconds)")
    parser.add_argument("--agent-timeout", type=int, default=600,
                        help="Timeout per agent fix (seconds)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of agents to run in parallel")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip uploading results to HuggingFace")
    args = parser.parse_args()

    print("Loading cached test results...")
    cached = load_cached_results()
    failures = get_failures(cached)
    print(f"Total cached: {len(cached)} | Failing: {len(failures)}\n")

    if args.family:
        failures = [f for f in failures if f.startswith(args.family)]
        print(f"Filtered to {len(failures)} matching '{args.family}'\n")

    claude = _find_claude()
    if not claude and not args.dry_run:
        sys.exit("ERROR: claude CLI not found. Install Claude Code first.")

    if args.dry_run:
        print(f"Dry run — {len(failures)} benchmarks need fixing:")
        for f in sorted(failures):
            print(f"  - {f}")
        return

    # Launch an agent for every failure. The agent tests, diagnoses, fixes,
    # and verifies. No separate re-test phase — the agent handles everything.
    print(f"Launching agents for {len(failures)} failures "
          f"(parallel={args.parallel})...\n")

    fixed = []
    marked_broken = []
    agent_failed = []

    # Use cached error output as context for the agent (empty string if none)
    failure_data = []
    for benchmark in sorted(failures):
        cached_result = cached.get(benchmark, {})
        # Pass any detail from the cached result as context
        detail = cached_result.get("extraction", {}).get("detail", "")
        failure_data.append((benchmark, detail))

    if args.parallel <= 1:
        for i, (benchmark, context) in enumerate(failure_data, 1):
            print(f"[{i}/{len(failure_data)}] {benchmark}: launching agent...")
            name, outcome, desc = fix_benchmark(
                benchmark, context, agent_timeout=args.agent_timeout
            )
            print(f"  -> {outcome}: {desc}")
            if outcome == "fixed":
                result = {
                    "task": name,
                    "extraction": {"status": "PASS"},
                    "evaluator": {"status": "UNKNOWN"},
                }
                update_cache(name, result)
                if not args.no_upload:
                    upload_to_hf(name, result)
                fixed.append(name)
            elif outcome == "marked_broken":
                marked_broken.append(name)
            else:
                agent_failed.append((name, desc))

            if i % 50 == 0:
                print(f"\n--- Progress: {i}/{len(failure_data)} | "
                      f"Fixed: {len(fixed)} | Broken: {len(marked_broken)} | "
                      f"Failed: {len(agent_failed)} ---\n")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    fix_benchmark, benchmark, context, args.agent_timeout
                ): benchmark
                for benchmark, context in failure_data
            }
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                name, outcome, desc = future.result()
                print(f"[{done_count}/{len(failure_data)}] {name}: {outcome} — {desc}")
                if outcome == "fixed":
                    result = {
                        "task": name,
                        "extraction": {"status": "PASS"},
                        "evaluator": {"status": "UNKNOWN"},
                    }
                    update_cache(name, result)
                    if not args.no_upload:
                        upload_to_hf(name, result)
                    fixed.append(name)
                elif outcome == "marked_broken":
                    marked_broken.append(name)
                else:
                    agent_failed.append((name, desc))

                if done_count % 50 == 0:
                    print(f"\n--- Progress: {done_count}/{len(failure_data)} | "
                          f"Fixed: {len(fixed)} | Broken: {len(marked_broken)} | "
                          f"Failed: {len(agent_failed)} ---\n")

    _print_summary(cached, fixed, marked_broken, agent_failed)


def _print_summary(cached, fixed, marked_broken, agent_failed):
    already_passing = sum(
        1 for d in cached.values()
        if d.get("extraction", {}).get("status") == "PASS"
    )
    total_pass = already_passing + len(fixed)
    total = len(cached)

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Already passing:       {already_passing}")
    print(f"Fixed by agent:        {len(fixed)}")
    print(f"Marked broken:         {len(marked_broken)}")
    print(f"Agent failed:          {len(agent_failed)}")
    print(f"")
    print(f"Overall: {total_pass}/{total} benchmarks pass extraction "
          f"({total_pass / total * 100:.1f}%)")
    print(f"{'='*70}")

    if agent_failed:
        print(f"\nAgent failures ({len(agent_failed)}):")
        for name, desc in sorted(agent_failed):
            print(f"  - {name}: {desc}")


if __name__ == "__main__":
    main()
