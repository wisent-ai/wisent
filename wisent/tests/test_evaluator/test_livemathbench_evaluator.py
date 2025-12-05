"""Unit tests for LiveMathBenchEvaluator and G-Pass@k metrics."""

import pytest
import numpy as np
from scipy.stats import hypergeom
from wisent.core.evaluators.benchmark_specific.livemathbench_evaluator import (
    LiveMathBenchEvaluator,
    LANGUAGE_PROMPTS,
    _compute_g_pass_at_k,
    compute_g_pass_at_k,
    compute_mg_pass_at_k,
    compute_metrics_for_problem,
    compute_all_metrics,
)


class TestGPassAtKInternal:
    """Tests for internal G-Pass@k computation."""

    def test_compute_g_pass_at_k_basic(self):
        """Test basic G-Pass@k computation using hypergeom."""
        # n=10, c=5, k=4, m=2
        # Should equal hypergeom.sf(1, 10, 5, 4)
        result = _compute_g_pass_at_k(n=10, c=5, k=4, m=2)
        expected = hypergeom.sf(1, 10, 5, 4)
        assert abs(result - expected) < 1e-10

    def test_compute_g_pass_at_k_edge_m_greater_than_c(self):
        """When m > c, probability is 0."""
        result = _compute_g_pass_at_k(n=10, c=3, k=5, m=4)
        assert result == 0.0

    def test_compute_g_pass_at_k_edge_k_greater_than_n(self):
        """When k > n, probability is 0."""
        result = _compute_g_pass_at_k(n=5, c=3, k=10, m=2)
        assert result == 0.0

    def test_compute_g_pass_at_k_edge_c_zero(self):
        """When c = 0, probability is 0 for m > 0."""
        result = _compute_g_pass_at_k(n=10, c=0, k=5, m=1)
        assert result == 0.0

    def test_compute_g_pass_at_k_all_correct(self):
        """When c = n (all correct), G-Pass@k = 1 for any m <= k."""
        result = _compute_g_pass_at_k(n=10, c=10, k=5, m=3)
        assert abs(result - 1.0) < 1e-10


class TestGPassAtKWithTau:
    """Tests for G-Pass@k with tau threshold."""

    def test_g_pass_at_k_tau_zero(self):
        """When τ=0, m = max(ceil(0), 1) = 1, so at least 1 correct."""
        result = compute_g_pass_at_k(n=10, c=5, k=5, tau=0.0)
        expected = _compute_g_pass_at_k(n=10, c=5, k=5, m=1)
        assert abs(result - expected) < 1e-10

    def test_g_pass_at_k_tau_one(self):
        """When τ=1.0, m = k, so all k must be correct."""
        result = compute_g_pass_at_k(n=10, c=5, k=5, tau=1.0)
        expected = _compute_g_pass_at_k(n=10, c=5, k=5, m=5)
        assert abs(result - expected) < 1e-10

    def test_g_pass_at_k_tau_half(self):
        """When τ=0.5, k=4, m = ceil(2) = 2."""
        result = compute_g_pass_at_k(n=10, c=5, k=4, tau=0.5)
        expected = _compute_g_pass_at_k(n=10, c=5, k=4, m=2)
        assert abs(result - expected) < 1e-10

    def test_g_pass_at_k_minimum_one(self):
        """m should always be at least 1."""
        # tau=0.1, k=4 -> ceil(0.4) = 1, but max(1, 1) = 1
        result = compute_g_pass_at_k(n=10, c=5, k=4, tau=0.1)
        expected = _compute_g_pass_at_k(n=10, c=5, k=4, m=1)
        assert abs(result - expected) < 1e-10


class TestMGPassAtK:
    """Tests for mG-Pass@k metric computation."""

    def test_mg_pass_at_k_formula(self):
        """Test mG-Pass@k follows the formula: (2/k) * Σ(i=ceil(k*0.5)+1 to k)."""
        n, c, k = 10, 5, 4
        result = compute_mg_pass_at_k(n, c, k)

        # Manual calculation
        low = int(np.ceil(k * 0.5))  # ceil(2) = 2
        high = k  # 4
        expected = 0.0
        for i in range(low + 1, high + 1):  # i in [3, 4]
            expected += _compute_g_pass_at_k(n, c, k, i)
        expected = 2 * expected / k

        assert abs(result - expected) < 1e-10

    def test_mg_pass_at_k_all_correct(self):
        """When all samples are correct, mG-Pass@k should be high."""
        result = compute_mg_pass_at_k(n=10, c=10, k=4)
        assert result == 1.0

    def test_mg_pass_at_k_bounds(self):
        """mG-Pass@k should be between 0 and 1."""
        result = compute_mg_pass_at_k(n=10, c=5, k=5)
        assert 0.0 <= result <= 1.0


class TestComputeMetricsForProblem:
    """Tests for per-problem metric computation."""

    def test_compute_metrics_for_problem_keys(self):
        """Test that all expected metric keys are present."""
        metrics = compute_metrics_for_problem(
            n=16, c=8,
            k_values=[4, 8],
            tau_values=[0.0, 0.5, 1.0]
        )

        # Check G-Pass@k keys
        assert "G-Pass@4_0.0" in metrics
        assert "G-Pass@4_0.5" in metrics
        assert "G-Pass@4_1.0" in metrics
        assert "G-Pass@8_0.0" in metrics
        assert "G-Pass@8_0.5" in metrics
        assert "G-Pass@8_1.0" in metrics

        # Check mG-Pass@k keys
        assert "mG-Pass@4" in metrics
        assert "mG-Pass@8" in metrics

    def test_compute_metrics_for_problem_skip_large_k(self):
        """k values larger than n should be skipped."""
        metrics = compute_metrics_for_problem(
            n=10, c=5,
            k_values=[4, 16],  # 16 > 10
            tau_values=[0.5]
        )

        assert "G-Pass@4_0.5" in metrics
        assert "G-Pass@16_0.5" not in metrics


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_compute_all_metrics_basic(self):
        """Test basic metric computation across multiple problems."""
        correct_counts = [5, 3, 8, 0, 10]  # 5 problems
        metrics = compute_all_metrics(
            correct_counts,
            total_samples=10,
            k_values=[4, 8],
            tau_values=[0.0, 0.5, 1.0]
        )

        assert "G-Pass@4_0.0" in metrics
        assert "G-Pass@4_0.5" in metrics
        assert "mG-Pass@4" in metrics

    def test_compute_all_metrics_empty(self):
        """Test with empty input."""
        metrics = compute_all_metrics([], total_samples=10)
        assert metrics == {}

    def test_compute_all_metrics_averaging(self):
        """Test that metrics are properly averaged across problems."""
        # Two identical problems
        correct_counts = [5, 5]
        metrics = compute_all_metrics(
            correct_counts,
            total_samples=10,
            k_values=[4],
            tau_values=[0.5]
        )

        # Single problem metric
        single_metric = compute_metrics_for_problem(n=10, c=5, k_values=[4], tau_values=[0.5])

        # Should be the same since both problems are identical
        assert abs(metrics["G-Pass@4_0.5"] - single_metric["G-Pass@4_0.5"]) < 1e-10

    def test_compute_all_metrics_all_perfect(self):
        """Test when all problems have all samples correct."""
        correct_counts = [10, 10, 10]
        metrics = compute_all_metrics(correct_counts, total_samples=10, k_values=[4])

        assert abs(metrics["G-Pass@4_0.0"] - 1.0) < 1e-10
        assert abs(metrics["G-Pass@4_1.0"] - 1.0) < 1e-10
        assert abs(metrics["mG-Pass@4"] - 1.0) < 1e-10


class TestLiveMathBenchEvaluator:
    """Tests for LiveMathBenchEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return LiveMathBenchEvaluator()

    def test_correct_boxed_answer(self, evaluator):
        result = evaluator.evaluate(
            response="The answer is \\boxed{42}",
            expected="42"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_wrong_answer(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{10}",
            expected="42"
        )
        assert result.ground_truth == "UNTRUTHFUL"

    def test_equivalent_fractions(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{0.5}",
            expected="1/2"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_empty_response(self, evaluator):
        result = evaluator.evaluate(
            response="",
            expected="42"
        )
        assert result.ground_truth == "UNKNOWN"

    def test_whitespace_response(self, evaluator):
        result = evaluator.evaluate(
            response="   ",
            expected="42"
        )
        assert result.ground_truth == "UNKNOWN"

    def test_negative_number(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{-3}",
            expected="-3"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_latex_expression(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{\\frac{1}{2}}",
            expected="0.5"
        )
        assert result.ground_truth == "TRUTHFUL"


class TestLiveMathBenchEvaluatorPrompt:
    """Tests for LiveMathBenchEvaluator prompt generation."""

    def test_get_prompt_english(self):
        question = "What is 2+2?"
        prompt = LiveMathBenchEvaluator.get_prompt(question, "en")
        assert question in prompt
        assert "\\boxed{}" in prompt

    def test_get_prompt_chinese(self):
        question = "计算 2+2"
        prompt = LiveMathBenchEvaluator.get_prompt(question, "cn")
        assert question in prompt
        assert "\\boxed{}" in prompt

    def test_get_prompt_default_language(self):
        """Default language should be English."""
        question = "Test question"
        prompt_default = LiveMathBenchEvaluator.get_prompt(question)
        prompt_en = LiveMathBenchEvaluator.get_prompt(question, "en")
        assert prompt_default == prompt_en

    def test_get_prompt_unknown_language_falls_back_to_english(self):
        """Unknown language should fall back to English."""
        question = "Test question"
        prompt = LiveMathBenchEvaluator.get_prompt(question, "xyz")
        assert LANGUAGE_PROMPTS["en"] in prompt


class TestLiveMathBenchEvaluatorMultipleSamples:
    """Tests for multiple sample evaluation."""

    @pytest.fixture
    def evaluator(self):
        return LiveMathBenchEvaluator()

    def test_evaluate_multiple_samples(self, evaluator):
        responses = [
            "\\boxed{42}",  # correct
            "\\boxed{10}",  # wrong
            "\\boxed{42}",  # correct
            "\\boxed{41}",  # wrong
        ]
        results, correct_count = evaluator.evaluate_multiple_samples(
            responses=responses,
            expected="42"
        )

        assert len(results) == 4
        assert correct_count == 2

    def test_evaluate_multiple_samples_all_correct(self, evaluator):
        responses = ["\\boxed{42}"] * 5
        results, correct_count = evaluator.evaluate_multiple_samples(
            responses=responses,
            expected="42"
        )

        assert correct_count == 5

    def test_evaluate_multiple_samples_none_correct(self, evaluator):
        responses = ["\\boxed{10}"] * 5
        results, correct_count = evaluator.evaluate_multiple_samples(
            responses=responses,
            expected="42"
        )

        assert correct_count == 0


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
        """Pass@k should equal G-Pass@k with τ→0 (at least 1 correct)."""
        n, c, k = 48, 24, 16

        # G-Pass@k with tau=0 uses m=max(ceil(0), 1)=1
        g_pass_tau_0 = compute_g_pass_at_k(n, c, k, tau=0.0)

        # This should equal P(X >= 1) = 1 - P(X = 0)
        # Using hypergeom: sf(0, n, c, k) = P(X >= 1)
        pass_at_k = hypergeom.sf(0, n, c, k)

        assert abs(g_pass_tau_0 - pass_at_k) < 1e-10

    def test_g_pass_monotonicity_in_tau(self):
        """G-Pass@k should decrease as tau increases."""
        n, c, k = 48, 24, 16

        tau_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        g_pass_values = [compute_g_pass_at_k(n, c, k, tau) for tau in tau_values]

        # Each subsequent value should be <= previous
        for i in range(1, len(g_pass_values)):
            assert g_pass_values[i] <= g_pass_values[i-1] + 1e-10

    def test_g_pass_monotonicity_in_c(self):
        """G-Pass@k should increase as c increases."""
        n, k, tau = 48, 16, 0.5

        c_values = [0, 12, 24, 36, 48]
        g_pass_values = [compute_g_pass_at_k(n, c, k, tau) for c in c_values]

        # Each subsequent value should be >= previous
        for i in range(1, len(g_pass_values)):
            assert g_pass_values[i] >= g_pass_values[i-1] - 1e-10
