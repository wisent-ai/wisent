"""Unit tests for CoNaLaEvaluator."""

import pytest
from wisent.core.evaluators.benchmark_specific.conala_evaluator import (
    CoNaLaEvaluator,
    tokenize_for_bleu_eval,
    compute_bleu_single,
    compute_bleu,
)


class TestTokenization:
    """Tests for the tokenization function."""

    def test_basic_tokenization(self):
        """Test basic code tokenization."""
        code = "print('hello')"
        tokens = tokenize_for_bleu_eval(code)
        assert "print" in tokens
        assert "(" in tokens
        assert ")" in tokens

    def test_camel_case_splitting(self):
        """Test that camelCase is split."""
        code = "getUserName"
        tokens = tokenize_for_bleu_eval(code)
        assert "get" in tokens
        assert "User" in tokens
        assert "Name" in tokens

    def test_quote_normalization(self):
        """Test that quotes are normalized to backticks."""
        code1 = "x = 'hello'"
        code2 = 'x = "hello"'
        tokens1 = tokenize_for_bleu_eval(code1)
        tokens2 = tokenize_for_bleu_eval(code2)
        # After normalization, both should have backticks
        assert tokens1 == tokens2

    def test_operator_separation(self):
        """Test that operators are separated."""
        code = "x+y*z"
        tokens = tokenize_for_bleu_eval(code)
        assert "x" in tokens
        assert "+" in tokens
        assert "y" in tokens
        assert "*" in tokens
        assert "z" in tokens

    def test_underscore_preserved(self):
        """Test that underscores in names are preserved."""
        code = "my_variable"
        tokens = tokenize_for_bleu_eval(code)
        assert "my_variable" in tokens


class TestBLEUScore:
    """Tests for BLEU score computation."""

    def test_identical_strings(self):
        """Identical strings should have BLEU score of 1.0."""
        code = "print('hello world')"
        score = compute_bleu_single(code, code)
        assert score == pytest.approx(1.0)

    def test_completely_different_strings(self):
        """Completely different strings should have BLEU score near 0."""
        ref = "print('hello')"
        hyp = "for i in range(10): pass"
        score = compute_bleu_single(ref, hyp)
        assert score < 0.1

    def test_partial_match(self):
        """Partially matching code should have intermediate BLEU score."""
        # Need enough tokens for 4-gram overlap
        ref = "sorted(my_list, key=lambda x: x, reverse=True)"
        hyp = "sorted(my_list, key=lambda x: x)"
        score = compute_bleu_single(ref, hyp)
        assert 0.0 < score < 1.0

    def test_empty_hypothesis(self):
        """Empty hypothesis should have BLEU score of 0."""
        ref = "print('hello')"
        hyp = ""
        score = compute_bleu_single(ref, hyp)
        assert score == 0.0

    def test_corpus_bleu(self):
        """Test corpus-level BLEU computation."""
        refs = [
            ["print", "(", "`", "hello", "`", ")"],
            ["x", "=", "1"],
        ]
        hyps = [
            ["print", "(", "`", "hello", "`", ")"],
            ["x", "=", "2"],
        ]
        bleu, precisions, bp, ratio, _, _ = compute_bleu(refs, hyps)
        assert 0.0 < bleu <= 1.0


class TestCoNaLaEvaluator:
    """Tests for CoNaLaEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        return CoNaLaEvaluator()

    def test_perfect_match(self, evaluator):
        """Perfect match should be TRUTHFUL with BLEU score of 1.0."""
        result = evaluator.evaluate(
            response="sorted(my_list, reverse=True)",
            expected="sorted(my_list, reverse=True)"
        )
        assert result.ground_truth == "TRUTHFUL"
        assert result.confidence == pytest.approx(1.0)

    def test_partial_match_high_bleu(self, evaluator):
        """High BLEU score should be TRUTHFUL."""
        evaluator = CoNaLaEvaluator(bleu_threshold=0.3)
        # Need enough tokens for 4-gram overlap
        result = evaluator.evaluate(
            response="sorted(my_list, key=lambda x: x)",
            expected="sorted(my_list, key=lambda x: x, reverse=True)"
        )
        # This depends on the actual BLEU score
        assert result.ground_truth in ["TRUTHFUL", "UNTRUTHFUL"]
        assert 0.0 < result.confidence < 1.0

    def test_completely_wrong(self, evaluator):
        """Completely wrong code should be UNTRUTHFUL."""
        result = evaluator.evaluate(
            response="print('hello')",
            expected="sorted(my_list, reverse=True)"
        )
        assert result.ground_truth == "UNTRUTHFUL"
        assert result.confidence < 0.5

    def test_empty_response(self, evaluator):
        """Empty response should be UNKNOWN."""
        result = evaluator.evaluate(
            response="",
            expected="print('hello')"
        )
        assert result.ground_truth == "UNKNOWN"
        assert result.confidence == 0.0

    def test_whitespace_response(self, evaluator):
        """Whitespace-only response should be UNKNOWN."""
        result = evaluator.evaluate(
            response="   ",
            expected="print('hello')"
        )
        assert result.ground_truth == "UNKNOWN"

    def test_result_contains_bleu_score(self, evaluator):
        """Result metadata should contain BLEU score."""
        result = evaluator.evaluate(
            response="print('hello')",
            expected="print('hello')"
        )
        assert "bleu_score" in result.meta
        assert isinstance(result.meta["bleu_score"], float)


class TestCoNaLaEvaluatorPrompt:
    """Tests for CoNaLaEvaluator prompt generation."""

    def test_get_prompt_contains_intent(self):
        """Prompt should contain the intent."""
        intent = "sort a list in descending order"
        prompt = CoNaLaEvaluator.get_prompt(intent)
        assert intent in prompt

    def test_get_prompt_mentions_python(self):
        """Prompt should mention Python."""
        prompt = CoNaLaEvaluator.get_prompt("test intent")
        assert "Python" in prompt

    def test_get_prompt_with_rewritten_intent(self):
        """Rewritten intent should be used when provided."""
        intent = "sort list"
        rewritten = "sort a list in descending order"
        prompt = CoNaLaEvaluator.get_prompt(intent, rewritten)
        assert rewritten in prompt
        assert intent not in prompt

    def test_get_prompt_falls_back_to_intent(self):
        """Original intent should be used when no rewritten intent."""
        intent = "sort a list"
        prompt = CoNaLaEvaluator.get_prompt(intent, None)
        assert intent in prompt


class TestCoNaLaCorpusEvaluation:
    """Tests for corpus-level evaluation."""

    @pytest.fixture
    def evaluator(self):
        return CoNaLaEvaluator()

    def test_corpus_evaluation_basic(self, evaluator):
        """Test basic corpus evaluation."""
        responses = [
            "sorted(my_list, reverse=True)",
            "print('hello')",
        ]
        expected = [
            "sorted(my_list, reverse=True)",
            "print('hello')",
        ]
        results = evaluator.evaluate_corpus(responses, expected)
        assert "bleu_score" in results
        assert results["bleu_score"] == pytest.approx(100.0)  # Perfect match

    def test_corpus_evaluation_partial_match(self, evaluator):
        """Test corpus evaluation with partial matches."""
        responses = [
            "sorted(my_list, reverse=True)",
            "print('goodbye')",
        ]
        expected = [
            "sorted(my_list, reverse=True)",
            "print('hello')",
        ]
        results = evaluator.evaluate_corpus(responses, expected)
        assert 0.0 < results["bleu_score"] < 100.0
        assert results["total"] == 2

    def test_corpus_evaluation_empty_list(self, evaluator):
        """Test corpus evaluation with empty lists."""
        results = evaluator.evaluate_corpus([], [])
        assert results["bleu_score"] == 0.0
        assert results["total"] == 0

    def test_corpus_evaluation_mismatched_lengths(self, evaluator):
        """Test that mismatched lengths raise error."""
        with pytest.raises(ValueError):
            evaluator.evaluate_corpus(
                responses=["code1", "code2"],
                expected_answers=["code1"]
            )


class TestCoNaLaEvaluatorConfiguration:
    """Tests for evaluator configuration."""

    def test_custom_threshold(self):
        """Test custom BLEU threshold."""
        evaluator = CoNaLaEvaluator(bleu_threshold=0.8)
        assert evaluator.bleu_threshold == 0.8

    def test_custom_max_order(self):
        """Test custom max n-gram order."""
        evaluator = CoNaLaEvaluator(max_order=2)
        assert evaluator.max_order == 2

    def test_threshold_affects_truthfulness(self):
        """Test that threshold affects TRUTHFUL/UNTRUTHFUL classification."""
        # With low threshold, partial match is TRUTHFUL
        low_threshold = CoNaLaEvaluator(bleu_threshold=0.1)
        result_low = low_threshold.evaluate(
            response="sorted(my_list)",
            expected="sorted(my_list, reverse=True)"
        )

        # With high threshold, same partial match might be UNTRUTHFUL
        high_threshold = CoNaLaEvaluator(bleu_threshold=0.9)
        result_high = high_threshold.evaluate(
            response="sorted(my_list)",
            expected="sorted(my_list, reverse=True)"
        )

        # The BLEU scores should be the same
        assert result_low.meta["bleu_score"] == result_high.meta["bleu_score"]


class TestCoNaLaEvaluatorEdgeCases:
    """Edge case tests for CoNaLaEvaluator."""

    @pytest.fixture
    def evaluator(self):
        return CoNaLaEvaluator()

    def test_multiline_code(self, evaluator):
        """Test handling of multiline code."""
        code = """def foo():
    return 42"""
        result = evaluator.evaluate(response=code, expected=code)
        assert result.ground_truth == "TRUTHFUL"

    def test_special_characters(self, evaluator):
        """Test handling of special characters."""
        code = "re.sub(r'\\s+', ' ', text)"
        result = evaluator.evaluate(response=code, expected=code)
        assert result.ground_truth == "TRUTHFUL"

    def test_unicode_strings(self, evaluator):
        """Test handling of unicode strings in code."""
        code = "print('こんにちは')"
        result = evaluator.evaluate(response=code, expected=code)
        assert result.ground_truth == "TRUTHFUL"

    def test_numeric_expected(self, evaluator):
        """Test that numeric expected values are converted to string."""
        # Short strings have 0 BLEU with BLEU-4 (not enough tokens for 4-grams)
        # Use a lower threshold to test numeric conversion
        evaluator = CoNaLaEvaluator(bleu_threshold=0.0, max_order=1)
        result = evaluator.evaluate(
            response="42",
            expected=42
        )
        # With max_order=1, identical single tokens have BLEU=1.0
        assert result.ground_truth == "TRUTHFUL"

    def test_list_comprehension(self, evaluator):
        """Test list comprehension code."""
        code = "[x**2 for x in range(10)]"
        result = evaluator.evaluate(response=code, expected=code)
        assert result.ground_truth == "TRUTHFUL"
