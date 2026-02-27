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

# Re-export tests from consistency module
from wisent.core.constants import ZERO_THRESHOLD
from wisent.tests.test_evaluator._livemathbench_consistency import (
    TestGPassAtKConsistencyWithPaper,
    TestLiveMathBenchEvaluatorMetadata,
)


class TestGPassAtKInternal:
    """Tests for internal G-Pass@k computation."""

    def test_compute_g_pass_at_k_basic(self):
        result = _compute_g_pass_at_k(n=10, c=5, k=4, m=2)
        expected = hypergeom.sf(1, 10, 5, 4)
        assert abs(result - expected) < ZERO_THRESHOLD

    def test_compute_g_pass_at_k_edge_m_greater_than_c(self):
        result = _compute_g_pass_at_k(n=10, c=3, k=5, m=4)
        assert result == 0.0

    def test_compute_g_pass_at_k_edge_k_greater_than_n(self):
        result = _compute_g_pass_at_k(n=5, c=3, k=10, m=2)
        assert result == 0.0

    def test_compute_g_pass_at_k_edge_c_zero(self):
        result = _compute_g_pass_at_k(n=10, c=0, k=5, m=1)
        assert result == 0.0

    def test_compute_g_pass_at_k_all_correct(self):
        result = _compute_g_pass_at_k(n=10, c=10, k=5, m=3)
        assert abs(result - 1.0) < ZERO_THRESHOLD


class TestGPassAtKWithTau:
    """Tests for G-Pass@k with tau threshold."""

    def test_g_pass_at_k_tau_zero(self):
        result = compute_g_pass_at_k(n=10, c=5, k=5, tau=0.0)
        expected = _compute_g_pass_at_k(n=10, c=5, k=5, m=1)
        assert abs(result - expected) < ZERO_THRESHOLD

    def test_g_pass_at_k_tau_one(self):
        result = compute_g_pass_at_k(n=10, c=5, k=5, tau=1.0)
        expected = _compute_g_pass_at_k(n=10, c=5, k=5, m=5)
        assert abs(result - expected) < ZERO_THRESHOLD

    def test_g_pass_at_k_tau_half(self):
        result = compute_g_pass_at_k(n=10, c=5, k=4, tau=0.5)
        expected = _compute_g_pass_at_k(n=10, c=5, k=4, m=2)
        assert abs(result - expected) < ZERO_THRESHOLD

    def test_g_pass_at_k_minimum_one(self):
        result = compute_g_pass_at_k(n=10, c=5, k=4, tau=0.1)
        expected = _compute_g_pass_at_k(n=10, c=5, k=4, m=1)
        assert abs(result - expected) < ZERO_THRESHOLD


class TestMGPassAtK:
    """Tests for mG-Pass@k metric computation."""

    def test_mg_pass_at_k_formula(self):
        n, c, k = 10, 5, 4
        result = compute_mg_pass_at_k(n, c, k)
        low = int(np.ceil(k * 0.5))
        high = k
        expected = 0.0
        for i in range(low + 1, high + 1):
            expected += _compute_g_pass_at_k(n, c, k, i)
        expected = 2 * expected / k
        assert abs(result - expected) < ZERO_THRESHOLD

    def test_mg_pass_at_k_all_correct(self):
        result = compute_mg_pass_at_k(n=10, c=10, k=4)
        assert result == 1.0

    def test_mg_pass_at_k_bounds(self):
        result = compute_mg_pass_at_k(n=10, c=5, k=5)
        assert 0.0 <= result <= 1.0


class TestComputeMetricsForProblem:
    """Tests for per-problem metric computation."""

    def test_compute_metrics_for_problem_keys(self):
        metrics = compute_metrics_for_problem(n=16, c=8, k_values=[4, 8], tau_values=[0.0, 0.5, 1.0])
        assert "G-Pass@4_0.0" in metrics
        assert "G-Pass@4_0.5" in metrics
        assert "G-Pass@4_1.0" in metrics
        assert "G-Pass@8_0.0" in metrics
        assert "mG-Pass@4" in metrics
        assert "mG-Pass@8" in metrics

    def test_compute_metrics_for_problem_skip_large_k(self):
        metrics = compute_metrics_for_problem(n=10, c=5, k_values=[4, 16], tau_values=[0.5])
        assert "G-Pass@4_0.5" in metrics
        assert "G-Pass@16_0.5" not in metrics


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_compute_all_metrics_basic(self):
        correct_counts = [5, 3, 8, 0, 10]
        metrics = compute_all_metrics(correct_counts, total_samples=10, k_values=[4, 8], tau_values=[0.0, 0.5, 1.0])
        assert "G-Pass@4_0.0" in metrics
        assert "mG-Pass@4" in metrics

    def test_compute_all_metrics_empty(self):
        metrics = compute_all_metrics([], total_samples=10)
        assert metrics == {}

    def test_compute_all_metrics_averaging(self):
        correct_counts = [5, 5]
        metrics = compute_all_metrics(correct_counts, total_samples=10, k_values=[4], tau_values=[0.5])
        single_metric = compute_metrics_for_problem(n=10, c=5, k_values=[4], tau_values=[0.5])
        assert abs(metrics["G-Pass@4_0.5"] - single_metric["G-Pass@4_0.5"]) < ZERO_THRESHOLD

    def test_compute_all_metrics_all_perfect(self):
        correct_counts = [10, 10, 10]
        metrics = compute_all_metrics(correct_counts, total_samples=10, k_values=[4])
        assert abs(metrics["G-Pass@4_0.0"] - 1.0) < ZERO_THRESHOLD
        assert abs(metrics["mG-Pass@4"] - 1.0) < ZERO_THRESHOLD


class TestLiveMathBenchEvaluator:
    """Tests for LiveMathBenchEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return LiveMathBenchEvaluator()

    def test_correct_boxed_answer(self, evaluator):
        result = evaluator.evaluate(response="The answer is \\boxed{42}", expected="42")
        assert result.ground_truth == "TRUTHFUL"

    def test_wrong_answer(self, evaluator):
        result = evaluator.evaluate(response="\\boxed{10}", expected="42")
        assert result.ground_truth == "UNTRUTHFUL"

    def test_equivalent_fractions(self, evaluator):
        result = evaluator.evaluate(response="\\boxed{0.5}", expected="1/2")
        assert result.ground_truth == "TRUTHFUL"

    def test_empty_response(self, evaluator):
        result = evaluator.evaluate(response="", expected="42")
        assert result.ground_truth == "UNKNOWN"

    def test_whitespace_response(self, evaluator):
        result = evaluator.evaluate(response="   ", expected="42")
        assert result.ground_truth == "UNKNOWN"

    def test_negative_number(self, evaluator):
        result = evaluator.evaluate(response="\\boxed{-3}", expected="-3")
        assert result.ground_truth == "TRUTHFUL"

    def test_latex_expression(self, evaluator):
        result = evaluator.evaluate(response="\\boxed{\\frac{1}{2}}", expected="0.5")
        assert result.ground_truth == "TRUTHFUL"


class TestLiveMathBenchEvaluatorPrompt:
    """Tests for LiveMathBenchEvaluator prompt generation."""

    def test_get_prompt_english(self):
        prompt = LiveMathBenchEvaluator.get_prompt("What is 2+2?", "en")
        assert "What is 2+2?" in prompt
        assert "\\boxed{}" in prompt

    def test_get_prompt_chinese(self):
        prompt = LiveMathBenchEvaluator.get_prompt("计算 2+2", "cn")
        assert "计算 2+2" in prompt

    def test_get_prompt_default_language(self):
        prompt_default = LiveMathBenchEvaluator.get_prompt("Test question")
        prompt_en = LiveMathBenchEvaluator.get_prompt("Test question", "en")
        assert prompt_default == prompt_en

    def test_get_prompt_unknown_language_falls_back_to_english(self):
        prompt = LiveMathBenchEvaluator.get_prompt("Test question", "xyz")
        assert LANGUAGE_PROMPTS["en"] in prompt


class TestLiveMathBenchEvaluatorMultipleSamples:
    """Tests for multiple sample evaluation."""

    @pytest.fixture
    def evaluator(self):
        return LiveMathBenchEvaluator()

    def test_evaluate_multiple_samples(self, evaluator):
        responses = ["\\boxed{42}", "\\boxed{10}", "\\boxed{42}", "\\boxed{41}"]
        results, correct_count = evaluator.evaluate_multiple_samples(responses=responses, expected="42")
        assert len(results) == 4
        assert correct_count == 2

    def test_evaluate_multiple_samples_all_correct(self, evaluator):
        responses = ["\\boxed{42}"] * 5
        results, correct_count = evaluator.evaluate_multiple_samples(responses=responses, expected="42")
        assert correct_count == 5

    def test_evaluate_multiple_samples_none_correct(self, evaluator):
        responses = ["\\boxed{10}"] * 5
        results, correct_count = evaluator.evaluate_multiple_samples(responses=responses, expected="42")
        assert correct_count == 0
