"""LiveMathBench evaluator for mathematical olympiad problems.

This evaluator implements two evaluation methods for LiveMathBench:
1. Answer extraction + math_equal comparison (like PolyMath)
2. LLM-as-a-judge evaluation

Also implements G-Pass@k metrics from the LiveMathBench paper (arxiv.org/abs/2412.13147):
- Greedy accuracy: Single-shot accuracy with temperature=0
- Pass@k: Probability that at least one of k samples is correct (G-Pass@k with tau=0)
- G-Pass@k(tau): Probability that at least tau*k of k samples are correct
- mG-Pass@k: Mean G-Pass@k integrated over tau in [0.5, 1.0]

Implementation follows the official GPassK repository:
https://github.com/open-compass/GPassK
"""

import numpy as np
from typing import List
from scipy.stats import hypergeom

from wisent.core.constants import LIVEMATHBENCH_K_VALUES

# Re-export evaluator class and constants from helpers
from wisent.core.evaluators.benchmark_specific.math._helpers.livemathbench_helpers import (
    LiveMathBenchEvaluator,
    LANGUAGE_PROMPTS,
    JUDGE_PROMPTS,
    JUDGE_PROMPT_EN,
    JUDGE_PROMPT_CN,
)


def _compute_g_pass_at_k(n: int, c: int, k: int, m: int) -> float:
    """Internal G-Pass@k computation using hypergeometric survival function.

    Computes the probability that at least m of k randomly selected samples
    are correct, given n total samples with c correct ones.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to select
        m: Minimum number of correct samples required

    Returns:
        Probability in [0, 1]
    """
    if m > min(c, k) or k > n or c < 0 or n <= 0 or m < 0:
        return 0.0
    # hypergeom.sf(m-1, n, c, k) = P(X >= m) where X ~ Hypergeometric(n, c, k)
    return hypergeom.sf(m - 1, n, c, k)


def compute_g_pass_at_k(n: int, c: int, k: int, tau: float) -> float:
    """Compute G-Pass@k with threshold tau.

    G-Pass@k(tau) is the probability that at least ceil(tau*k) of k randomly selected
    samples are correct (with minimum 1).

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to select
        tau: Threshold fraction in [0, 1]

    Returns:
        G-Pass@k(tau) probability in [0, 1]
    """
    m = max(int(np.ceil(k * tau)), 1)
    return _compute_g_pass_at_k(n, c, k, m)


def compute_mg_pass_at_k(n: int, c: int, k: int) -> float:
    """Compute mG-Pass@k (mean G-Pass@k).

    mG-Pass@k = (2/k) * sum(i=ceil(k*0.5)+1 to k) G-Pass@k(i)

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: Number of samples to select

    Returns:
        mG-Pass@k score in [0, 1]
    """
    low = int(np.ceil(k * 0.5))
    high = k

    mg_pass_at_k = 0.0
    for i in range(low + 1, high + 1):
        mg_pass_at_k += _compute_g_pass_at_k(n, c, k, i)
    mg_pass_at_k = 2 * mg_pass_at_k / k

    return mg_pass_at_k


def compute_metrics_for_problem(
    n: int,
    c: int,
    k_values: List[int] = list(LIVEMATHBENCH_K_VALUES),
    tau_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
) -> dict[str, float]:
    """Compute G-Pass@k metrics for a single problem.

    Args:
        n: Total number of samples
        c: Number of correct samples
        k_values: List of k values to compute metrics for
        tau_values: List of tau thresholds for G-Pass@k

    Returns:
        Dictionary with metrics for this problem
    """
    metrics = {}

    for k in k_values:
        if k > n:
            continue

        for tau in tau_values:
            metrics[f"G-Pass@{k}_{tau}"] = compute_g_pass_at_k(n, c, k, tau)

        metrics[f"mG-Pass@{k}"] = compute_mg_pass_at_k(n, c, k)

    return metrics


def compute_all_metrics(
    correct_counts: List[int],
    total_samples: int,
    k_values: List[int] = list(LIVEMATHBENCH_K_VALUES),
    tau_values: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
) -> dict[str, float]:
    """Compute all G-Pass@k metrics aggregated over a set of problems.

    Args:
        correct_counts: List of correct sample counts per problem
        total_samples: Total samples per problem (n)
        k_values: List of k values to compute metrics for
        tau_values: List of tau thresholds for G-Pass@k

    Returns:
        Dictionary with all metrics (averaged across problems)
    """
    num_problems = len(correct_counts)
    if num_problems == 0:
        return {}

    n = total_samples

    # Initialize accumulators
    metric_sums = {}

    for c in correct_counts:
        problem_metrics = compute_metrics_for_problem(n, c, k_values, tau_values)
        for key, value in problem_metrics.items():
            if key not in metric_sums:
                metric_sums[key] = 0.0
            metric_sums[key] += value

    # Average across problems
    metrics = {key: value / num_problems for key, value in metric_sums.items()}

    return metrics
