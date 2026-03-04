"""Extracted from test_conala_evaluator.py - TestCoNaLaEvaluatorEdgeCases tail."""


def run_unicode_test(evaluator):
    """Test handling of unicode strings in code.

    Args:
        evaluator: CoNaLaEvaluator instance

    Returns:
        EvaluationResult from evaluating unicode code
    """
    code = "print('\u3053\u3093\u306b\u3061\u306f')"
    result = evaluator.evaluate(response=code, expected=code)
    return result


def run_numeric_expected_test():
    """Test that numeric expected values are converted to string.

    Returns:
        EvaluationResult from evaluating numeric expected values
    """
    from wisent.core.reading.evaluators.benchmark_specific.coding.metrics.conala import (
        CoNaLaEvaluator,
    )
    from wisent.core.utils.config_tools.constants import CONALA_BLEU_THRESHOLD

    evaluator = CoNaLaEvaluator(bleu_threshold=CONALA_BLEU_THRESHOLD)
    result = evaluator.evaluate(
        response="42",
        expected=42
    )
    return result


def run_list_comprehension_test(evaluator):
    """Test list comprehension code.

    Args:
        evaluator: CoNaLaEvaluator instance

    Returns:
        EvaluationResult from evaluating list comprehension code
    """
    code = "[x**2 for x in range(10)]"
    result = evaluator.evaluate(response=code, expected=code)
    return result
