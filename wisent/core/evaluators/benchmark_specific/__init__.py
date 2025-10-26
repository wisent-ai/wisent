"""Benchmark-specific evaluators for lm-eval tasks.

This module provides evaluation methods that match lm-eval's native approaches:
- Log likelihood evaluation for multiple-choice tasks
- Generation evaluation for text generation tasks  
- Exact match evaluation for precise answer matching
- F1 evaluation for token-level comparison
- Perplexity evaluation for language modeling
"""

from .log_likelihoods_evaluator import LogLikelihoodsEvaluator
from .generation_evaluator import GenerationEvaluator
from .exact_match_evaluator import ExactMatchEvaluator
from .f1_evaluator import F1Evaluator
from .perplexity_evaluator import PerplexityEvaluator

__all__ = [
    'LogLikelihoodsEvaluator',
    'GenerationEvaluator',
    'ExactMatchEvaluator',
    'F1Evaluator',
    'PerplexityEvaluator',
]
