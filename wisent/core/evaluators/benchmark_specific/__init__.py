"""Benchmark-specific evaluators for lm-eval tasks.

This module provides evaluation methods that match lm-eval's native approaches:
- Log likelihood evaluation for multiple-choice tasks
- Generation evaluation for text generation tasks
- Exact match evaluation for precise answer matching
- F1 evaluation for token-level comparison
- Perplexity evaluation for language modeling
- Coding evaluation for code generation tasks
"""

from .log_likelihoods_evaluator import LogLikelihoodsEvaluator
from .generation_evaluator import GenerationEvaluator
from .exact_match_evaluator import ExactMatchEvaluator
from .f1_evaluator import F1Evaluator
from .perplexity_evaluator import PerplexityEvaluator
from .coding.metrics.evaluator import CodingEvaluator
from .conala_evaluator import CoNaLaEvaluator

# MathEvaluator requires math_equivalence which is installed from GitHub
# Make it lazy to avoid import errors when not installed
try:
    from .math_evaluator import MathEvaluator
    _MATH_EVALUATOR_AVAILABLE = True
except ImportError:
    MathEvaluator = None
    _MATH_EVALUATOR_AVAILABLE = False

# Backward compatibility alias
DockerCodeEvaluator = CodingEvaluator

__all__ = [
    'LogLikelihoodsEvaluator',
    'GenerationEvaluator',
    'ExactMatchEvaluator',
    'F1Evaluator',
    'PerplexityEvaluator',
    'CodingEvaluator',
    'DockerCodeEvaluator',  # Backward compatibility
    'MathEvaluator',
    'CoNaLaEvaluator',
]
