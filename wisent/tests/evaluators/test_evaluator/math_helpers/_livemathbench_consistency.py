"""Consistency and metadata tests for LiveMathBench evaluator.

Extracted from test_livemathbench_evaluator.py to keep file under 300 lines.
"""

import pytest
from scipy.stats import hypergeom
from wisent.core.utils.config_tools.constants import ZERO_THRESHOLD
from wisent.core.reading.evaluators.benchmark_specific.livemathbench_evaluator import (
    LiveMathBenchEvaluator,
    compute_g_pass_at_k,
)


class TestLiveMathBenchEvaluatorMetadata:
    """Tests for LiveMathBenchEvaluator metadata."""

    @pytest.fixture
    def evaluator(self):
        return LiveMathBenchEvaluator()

    def test_result_contains_predictions(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{42}",
            expected="42"
        )
        assert "predictions" in result.meta

    def test_result_contains_no_boxed_flag(self, evaluator):
        result = evaluator.evaluate(
            response="The answer is 42",
            expected="42"
        )
        assert "no_boxed" in result.meta
        assert result.meta["no_boxed"] is True

    def test_result_no_boxed_false_when_boxed_present(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{42}",
            expected="42"
        )
        assert result.meta["no_boxed"] is False

    def test_evaluator_name(self, evaluator):
        assert evaluator.name == "livemathbench"
        result = evaluator.evaluate(response="\\boxed{42}", expected="42")
        assert result.method_used == "livemathbench"


class TestGPassAtKConsistencyWithPaper:
    """Tests to verify consistency with the LiveMathBench paper formulas."""

    def test_pass_at_k_is_g_pass_at_k_tau_zero(self):
        """Pass@k should equal G-Pass@k with tau->0 (at least 1 correct)."""
        n, c, k = 48, 24, 16
        g_pass_tau_0 = compute_g_pass_at_k(n, c, k, tau=0.0)
        pass_at_k = hypergeom.sf(0, n, c, k)
        assert abs(g_pass_tau_0 - pass_at_k) < ZERO_THRESHOLD

    def test_g_pass_monotonicity_in_tau(self):
        """G-Pass@k should decrease as tau increases."""
        n, c, k = 48, 24, 16
        tau_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        g_pass_values = [compute_g_pass_at_k(n, c, k, tau) for tau in tau_values]
        for i in range(1, len(g_pass_values)):
            assert g_pass_values[i] <= g_pass_values[i-1] + ZERO_THRESHOLD

    def test_g_pass_monotonicity_in_c(self):
        """G-Pass@k should increase as c increases."""
        n, k, tau = 48, 16, 0.5
        c_values = [0, 12, 24, 36, 48]
        g_pass_values = [compute_g_pass_at_k(n, c, k, tau) for c in c_values]
        for i in range(1, len(g_pass_values)):
            assert g_pass_values[i] >= g_pass_values[i-1] - ZERO_THRESHOLD
