"""
Example: Evaluating LiveCodeBench with the coding evaluation framework.

This example shows how to use the LiveCodeBenchProvider with the CodingEvaluator
to run real code execution evaluation on LiveCodeBench problems.

Requirements:
    - Docker must be running
    - Docker image 'coding/sandbox:polyglot-1.0' must be available
    - HuggingFace datasets library must be installed

Usage:
    python -m wisent.benchmarks.coding.examples.ex_livecodebench
"""

from wisent.benchmarks.coding.providers.livecodebench.provider import LiveCodeBenchProvider
from wisent.benchmarks.coding.metrics.evaluator import CodingEvaluator, EvaluatorConfig
from wisent.benchmarks.coding.metrics.passk import estimate_pass_at_k


def dummy_model_fn(task):
    """
    Dummy model function that returns a simple solution.

    In practice, you would replace this with your LLM's code generation.
    """
    # Return a simple correct solution for demonstration
    if "countSeniors" in task.files.get("solution.py", ""):
        # LeetCode problem: count seniors
        return {
            "solution.py": """
from typing import List

class Solution:
    def countSeniors(self, details: List[str]) -> int:
        count = 0
        for detail in details:
            age = int(detail[11:13])
            if age > 60:
                count += 1
        return count
"""
        }
    else:
        # CodeForces problem: return a placeholder
        return {
            "solution.py": """
import sys

def solve():
    lines = sys.stdin.read().strip().split('\\n')
    # TODO: Implement solution
    print("TODO")

if __name__ == "__main__":
    solve()
"""
        }


def main():
    print("LiveCodeBench Evaluation Example")
    print("=" * 80)

    # 1. Create provider with limited problems for demo
    provider = LiveCodeBenchProvider(
        language="python",
        limit=5,  # Just 5 problems for demo
        platform="leetcode",  # LeetCode problems only
    )

    # 2. Create evaluator with configuration
    config = EvaluatorConfig(
        image="coding/sandbox:polyglot-1.0",
        self_repair=False,  # Disable self-repair for this demo
        time_limit_s=8,
        cpu_limit_s=3,
        mem_limit_mb=768,
    )

    evaluator = CodingEvaluator(
        provider=provider,
        model_fn=dummy_model_fn,
        repair_fn=None,
        cfg=config,
    )

    # 3. Run evaluation
    print("\nRunning evaluation...")
    print("-" * 80)

    outcomes = list(evaluator.evaluate())

    # 4. Display results
    print(f"\nResults:")
    print("-" * 80)

    passed = sum(1 for o in outcomes if o.passed)
    total = len(outcomes)

    for outcome in outcomes:
        status_icon = "✓" if outcome.passed else "✗"
        print(f"{status_icon} {outcome.task_id}: {outcome.status} ({outcome.elapsed:.2f}s)")

    print(f"\nPass rate: {passed}/{total} ({100 * passed / total:.1f}%)")

    # 5. Compute pass@k metrics
    results = [{"passed": o.passed} for o in outcomes]

    try:
        pass_at_1 = estimate_pass_at_k(results, k=1)
        print(f"Pass@1: {pass_at_1:.2%}")
    except:
        print("Pass@k metrics require more samples")

    print("\n" + "=" * 80)
    print("Note: This is a demo with a dummy model.")
    print("Replace dummy_model_fn with your LLM to get real results!")


if __name__ == "__main__":
    main()
