"""LM-Eval-Harness Ground Truth Evaluation package.

This package provides ground truth evaluation using lm-eval-harness tasks.
It supports multiple evaluation methods:
- log-likelihoods: Multiple choice evaluation using log probabilities
- text-generation: Open-ended generation with metric evaluation
- perplexity: Perplexity-based evaluation for language modeling tasks
- code-execution: Code generation with execution-based evaluation
"""

from .core import LMEvalHarnessGroundTruth
from .text_generation import evaluate_text_generation
from .perplexity import calculate_perplexity, evaluate_perplexity
from .code_execution import evaluate_code_execution, evaluate_generic_code_execution

__all__ = [
    'LMEvalHarnessGroundTruth',
    'evaluate_text_generation',
    'calculate_perplexity',
    'evaluate_perplexity',
    'evaluate_code_execution',
    'evaluate_generic_code_execution',
]
