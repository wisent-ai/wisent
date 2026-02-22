# coding/metrics/passk.py
from __future__ import annotations
from typing import Iterable
import math
from collections import defaultdict
from .core.atoms import SampleOutcome, Metric
from wisent.core.errors import InvalidRangeError

class PassAtK(Metric):
    """
    Exact Pass@k for code generation.
    """

    def __init__(self, k: int = 1):
        if k < 1:
            raise InvalidRangeError(param_name="k", actual=k, min_val=1)
        self.k = k

    def compute(self, outcomes: Iterable[SampleOutcome]) -> float:
        """
        Aggregate counts per task_id

        arguments:
            outcomes: Iterable of SampleOutcome objects

        returns:
            Average Pass@k score across tasks

        intuition:
            For each task, we have n samples, c of which pass.
            We want the probability that at least one of k random picks from these n samples is a passing one.
            This is 1 - (combinations of picking k from the n-c failing ones) / (combinations of picking k from all n).
            We then average this score across all tasks. 
        """
        per_task_counts = defaultdict(lambda: {"n": 0, "c": 0})
        for o in outcomes:
            d = per_task_counts[o.task_id]
            d["n"] += 1
            d["c"] += 1 if o.passed else 0

        if not per_task_counts:
            return 0.0

        scores_sum = 0.0
        task_cnt = 0
        for counts in per_task_counts.values():
            n = counts["n"]
            c = counts["c"]
            if n <= 0:
                continue  

            k = min(self.k, n)
            if c <= 0:
                score = 0.0
            elif k == 0:
                score = 0.0
            elif k == 1:
                score = c / n
            else:
                denom = math.comb(n, k)
                num = math.comb(n - c, k) if k <= (n - c) else 0
                score = 1.0 - (num / denom if denom > 0 else 0.0)

            scores_sum += score
            task_cnt += 1

        return 0.0 if task_cnt == 0 else scores_sum / task_cnt
