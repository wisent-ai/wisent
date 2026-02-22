"""Unit tests for MathEvaluator."""

import pytest
from wisent.core.evaluators.benchmark_specific.math_evaluator import MathEvaluator
from wisent.core.evaluators.benchmark_specific.math_parsing.extract_boxed import extract_boxed_answer


class TestExtractBoxedAnswer:
    """Tests for extract_boxed_answer utility function."""

    def test_simple_boxed(self):
        text = "The answer is \\boxed{42}"
        assert extract_boxed_answer(text) == "42"

    def test_nested_braces(self):
        text = "\\boxed{\\frac{1}{2}}"
        assert extract_boxed_answer(text) == "\\frac{1}{2}"

    def test_multiple_boxed_returns_last(self):
        text = "First \\boxed{wrong} then \\boxed{correct}"
        assert extract_boxed_answer(text) == "correct"

    def test_no_boxed_returns_none(self):
        text = "No boxed answer here"
        assert extract_boxed_answer(text) is None

    def test_deeply_nested(self):
        text = "\\boxed{\\sqrt{\\frac{a^{2}}{b}}}"
        assert extract_boxed_answer(text) == "\\sqrt{\\frac{a^{2}}{b}}"

    def test_empty_boxed(self):
        text = "\\boxed{}"
        assert extract_boxed_answer(text) == ""

    def test_boxed_with_surrounding_text(self):
        text = "Step 1: Calculate. Step 2: The final answer is \\boxed{123}. Done."
        assert extract_boxed_answer(text) == "123"


class TestMathEvaluator:
    """Tests for MathEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return MathEvaluator()

    def test_exact_match(self, evaluator):
        result = evaluator.evaluate(
            response="The answer is \\boxed{42}",
            expected="\\boxed{42}"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_equivalent_fractions(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{\\frac{1}{2}}",
            expected="\\boxed{0.5}"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_wrong_answer(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{10}",
            expected="\\boxed{42}"
        )
        assert result.ground_truth == "UNTRUTHFUL"

    def test_no_boxed_in_response(self, evaluator):
        result = evaluator.evaluate(
            response="The answer is 42",
            expected="\\boxed{42}"
        )
        assert result.ground_truth == "UNKNOWN"

    def test_raw_expected_without_boxed(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{42}",
            expected="42",
            extract_from_expected=False
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_equivalent_expressions(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{2+2}",
            expected="\\boxed{4}"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_negative_numbers(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{-5}",
            expected="\\boxed{-5}"
        )
        assert result.ground_truth == "TRUTHFUL"

    def test_get_prompt(self):
        prompt = MathEvaluator.get_prompt("What is 2+2?")
        assert "What is 2+2?" in prompt
        assert "\\boxed{}" in prompt


class TestMathEvaluatorEdgeCases:
    """Edge case tests for MathEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return MathEvaluator()

    def test_empty_response(self, evaluator):
        result = evaluator.evaluate(
            response="",
            expected="\\boxed{42}"
        )
        assert result.ground_truth == "UNKNOWN"

    def test_none_expected(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{42}",
            expected=None
        )
        assert result.ground_truth == "UNKNOWN"

    def test_whitespace_in_boxed(self, evaluator):
        result = evaluator.evaluate(
            response="\\boxed{  42  }",
            expected="\\boxed{42}"
        )
        assert result.ground_truth == "TRUTHFUL"
