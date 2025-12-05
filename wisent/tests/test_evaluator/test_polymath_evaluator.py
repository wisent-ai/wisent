"""Unit tests for PolyMathEvaluator."""

import pytest
from wisent.core.evaluators.benchmark_specific.polymath_evaluator import (
    PolyMathEvaluator,
    LANGUAGE_PROMPTS,
)


class TestPolyMathEvaluator:
    """Tests for PolyMathEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return PolyMathEvaluator()

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


class TestPolyMathEvaluatorPrompt:
    """Tests for PolyMathEvaluator prompt generation."""

    def test_get_prompt_english(self):
        question = "What is 2+2?"
        prompt = PolyMathEvaluator.get_prompt(question, "en")
        assert question in prompt
        assert "\\boxed{}" in prompt

    def test_get_prompt_chinese(self):
        question = "计算 2+2"
        prompt = PolyMathEvaluator.get_prompt(question, "zh")
        assert question in prompt
        assert "\\boxed{}" in prompt

    def test_get_prompt_default_language(self):
        """Default language should be English."""
        question = "Test question"
        prompt_default = PolyMathEvaluator.get_prompt(question)
        prompt_en = PolyMathEvaluator.get_prompt(question, "en")
        assert prompt_default == prompt_en

    def test_get_prompt_unknown_language_falls_back_to_english(self):
        """Unknown language should fall back to English."""
        question = "Test question"
        prompt = PolyMathEvaluator.get_prompt(question, "xyz")
        assert LANGUAGE_PROMPTS["en"] in prompt


class TestLanguagePrompts:
    """Tests for language prompt definitions."""

    def test_all_prompts_contain_boxed(self):
        """All language prompts should mention boxed format."""
        for lang, prompt in LANGUAGE_PROMPTS.items():
            assert "\\boxed{}" in prompt, f"Language {lang} missing \\boxed{{}}"

    def test_supported_languages(self):
        """Check that common languages are supported."""
        expected_languages = [
            "en", "zh", "ar", "de", "es", "fr", "ja", "ko", "pt", "ru"
        ]
        for lang in expected_languages:
            assert lang in LANGUAGE_PROMPTS, f"Language {lang} not supported"


class TestPolyMathEvaluatorMetadata:
    """Tests for PolyMathEvaluator metadata."""

    @pytest.fixture
    def evaluator(self):
        return PolyMathEvaluator()

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
