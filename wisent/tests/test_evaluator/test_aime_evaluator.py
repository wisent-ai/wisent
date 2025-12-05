"""Unit tests for AIMEEvaluator."""

import pytest
from wisent.core.evaluators.benchmark_specific.aime_evaluator import AIMEEvaluator


class TestAIMEEvaluator:
    """Tests for AIMEEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return AIMEEvaluator()

    def test_correct_integer_answer(self, evaluator):
        result = evaluator.evaluate(
            response="The answer is \\boxed{123}",
            expected=123
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_correct_string_expected(self, evaluator):
        """Test that string expected values are handled correctly."""
        result = evaluator.evaluate(
            response="\\boxed{456}",
            expected="456"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_wrong_answer(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{100}",
            expected=200
        )
        assert result.ground_truth == "UNTRUTHFUL"

    def test_no_boxed_in_response(self, evaluator):
        result = evaluator.evaluate(
            response="The answer is 42",
            expected=42
        )
        assert result.ground_truth == "UNKNOWN"

    def test_zero_answer(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{0}",
            expected=0
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_max_aime_answer(self, evaluator):
        """AIME answers are 0-999."""
        result = evaluator.evaluate(
            response="\\boxed{999}",
            expected=999
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_non_integer_model_answer(self, evaluator):
        """Model returns a non-integer - should be UNTRUTHFUL."""
        result = evaluator.evaluate(
            response="\\boxed{3.14}",
            expected=3
        )
        assert result.ground_truth == "UNTRUTHFUL"

    def test_non_numeric_model_answer(self, evaluator):
        """Model returns non-numeric text - should be UNTRUTHFUL."""
        result = evaluator.evaluate(
            response="\\boxed{abc}",
            expected=123
        )
        assert result.ground_truth == "UNTRUTHFUL"

    def test_leading_zeros(self, evaluator):
        """Test that leading zeros are handled correctly."""
        result = evaluator.evaluate(
            response="\\boxed{007}",
            expected=7
        )
        assert result.ground_truth == "TRUTHFUL"


class TestAIMEEvaluatorPrompt:
    """Tests for AIMEEvaluator prompt generation."""

    def test_get_prompt_contains_problem(self):
        problem = "Find the value of x if x^2 = 144"
        prompt = AIMEEvaluator.get_prompt(problem)
        assert problem in prompt

    def test_get_prompt_mentions_aime(self):
        prompt = AIMEEvaluator.get_prompt("Test problem")
        assert "AIME" in prompt

    def test_get_prompt_mentions_0_999(self):
        prompt = AIMEEvaluator.get_prompt("Test problem")
        assert "0" in prompt and "999" in prompt

    def test_get_prompt_mentions_boxed(self):
        prompt = AIMEEvaluator.get_prompt("Test problem")
        assert "\\boxed{}" in prompt


class TestAIMEEvaluatorEdgeCases:
    """Edge case tests for AIMEEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return AIMEEvaluator()

    def test_empty_response(self, evaluator):
        result = evaluator.evaluate(
            response="",
            expected=42
        )
        assert result.ground_truth == "UNKNOWN"

    def test_whitespace_only_response(self, evaluator):
        result = evaluator.evaluate(
            response="   ",
            expected=42
        )
        assert result.ground_truth == "UNKNOWN"

    def test_multiple_boxed_uses_last(self, evaluator):
        result = evaluator.evaluate(
            response="First I tried \\boxed{100} but then \\boxed{42}",
            expected=42
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_type_mismatch_int_vs_string(self, evaluator):
        """This was the original bug - comparing int to string."""
        result = evaluator.evaluate(
            response="\\boxed{42}",
            expected=42  # int from dataset
        )
        assert result.ground_truth == "TRUTHFUL"
