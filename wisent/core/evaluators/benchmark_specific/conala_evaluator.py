"""CoNaLa evaluator for code generation from natural language.

This evaluator handles the CoNaLa (Code/Natural Language Challenge) benchmark
which evaluates Python code generation from English natural language intents.

Evaluation is done using BLEU score after tokenization, following the official
CoNaLa baseline implementation from:
https://github.com/conala-corpus/conala-baseline/

The tokenization approach is from:
Wang Ling et al., "Latent Predictor Networks for Code Generation" (2016)
"""

import logging
import math
import re
from collections import Counter
from typing import Any

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult
from wisent.core.evaluators.benchmark_specific.utils import extract_boxed_answer

logger = logging.getLogger(__name__)


def tokenize_for_bleu_eval(code: str) -> list[str]:
    """Tokenize code for BLEU evaluation following CoNaLa baseline.

    This tokenizer is from Wang Ling et al., "Latent Predictor Networks
    for Code Generation" (2016).

    Args:
        code: The code string to tokenize

    Returns:
        List of tokens
    """
    # Add spaces around non-alphanumeric characters
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    # Split camelCase (lowercase followed by uppercase)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    # Normalize quotes to backticks
    code = re.sub(r'["\']', '`', code)
    # Collapse whitespace and split
    tokens = code.split()
    return tokens


def _get_ngrams(segment: list[str], max_order: int) -> Counter:
    """Extract n-grams up to max_order from a token list.

    Args:
        segment: List of tokens
        max_order: Maximum n-gram order

    Returns:
        Counter of n-gram frequencies
    """
    ngram_counts = Counter()
    for order in range(1, max_order + 1):
        for i in range(len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(
    references: list[list[str]],
    hypotheses: list[list[str]],
    max_order: int = 4,
    smooth: bool = False,
) -> tuple[float, list[float], float, float, int, int]:
    """Compute corpus-level BLEU score.

    Implementation follows the CoNaLa baseline. CoNaLa has one reference per example.

    Args:
        references: List of reference token lists (one per example).
        hypotheses: List of hypothesis token lists (one per example).
        max_order: Maximum n-gram order to use (default 4).
        smooth: Whether to apply Lin smoothing (default False for CoNaLa).

    Returns:
        Tuple of:
            - BLEU score (0.0 to 1.0)
            - List of n-gram precisions
            - Brevity penalty
            - Length ratio (hypothesis/reference)
            - Hypothesis length
            - Reference length
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    hypothesis_length = 0

    for reference, hypothesis in zip(references, hypotheses):
        reference_length += len(reference)
        hypothesis_length += len(hypothesis)

        # Get n-grams
        ref_ngrams = _get_ngrams(reference, max_order)
        hyp_ngrams = _get_ngrams(hypothesis, max_order)

        # Count matches (clipped to reference count)
        overlap = hyp_ngrams & ref_ngrams
        for ngram, count in overlap.items():
            matches_by_order[len(ngram) - 1] += count

        # Count possible matches
        for order in range(1, max_order + 1):
            possible_matches = len(hypothesis) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    # Compute precisions
    precisions = [0.0] * max_order
    for i in range(max_order):
        if smooth:
            precisions[i] = (matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0)
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = matches_by_order[i] / possible_matches_by_order[i]
            else:
                precisions[i] = 0.0

    # Compute geometric mean of precisions
    if min(precisions) > 0:
        p_log_sum = sum((1.0 / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0.0

    # Compute brevity penalty
    ratio = hypothesis_length / reference_length if reference_length > 0 else 0.0
    if ratio > 1.0:
        bp = 1.0
    elif ratio == 0.0:
        bp = 0.0
    else:
        bp = math.exp(1.0 - 1.0 / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, hypothesis_length, reference_length)


def compute_bleu_single(
    reference: str,
    hypothesis: str,
    max_order: int = 4,
    smooth: bool = False,
) -> float:
    """Compute BLEU score for a single reference-hypothesis pair.

    Args:
        reference: The reference code string
        hypothesis: The generated code string
        max_order: Maximum n-gram order (default 4)
        smooth: Whether to apply smoothing (default False)

    Returns:
        BLEU score (0.0 to 1.0)
    """
    ref_tokens = tokenize_for_bleu_eval(reference)
    hyp_tokens = tokenize_for_bleu_eval(hypothesis)

    bleu, _, _, _, _, _ = compute_bleu(
        references=[ref_tokens],
        hypotheses=[hyp_tokens],
        max_order=max_order,
        smooth=smooth,
    )
    return bleu


class CoNaLaEvaluator(BaseEvaluator):
    """Evaluator for CoNaLa code generation benchmark.

    Designed for CoNaLa benchmarks where:
    - Input is a natural language intent (English)
    - Output is Python code
    - Evaluation uses BLEU score with code-specific tokenization

    The BLEU calculation follows the official CoNaLa baseline:
    https://github.com/conala-corpus/conala-baseline/
    """

    name = "conala"
    description = "CoNaLa evaluator using BLEU score for code generation"

    def __init__(self, bleu_threshold: float = 0.5, max_order: int = 4):
        """Initialize the CoNaLa evaluator.

        Args:
            bleu_threshold: BLEU score threshold for TRUTHFUL classification (default 0.5)
            max_order: Maximum n-gram order for BLEU (default 4)
        """
        self.bleu_threshold = bleu_threshold
        self.max_order = max_order

    @staticmethod
    def get_prompt(
        intent: str,
        rewritten_intent: str | None = None,
        examples: list[tuple[str, str]] | None = None,
    ) -> str:
        """Create instruction prompt for LLM to generate Python code.

        Args:
            intent: The natural language intent from the dataset
            rewritten_intent: Optional rewritten/clarified intent
            examples: Optional list of (intent, snippet) tuples for few-shot

        Returns:
            Formatted prompt string
        """
        nl_intent = rewritten_intent if rewritten_intent else intent

        prompt = "Generate Python code for the following task. Put final answer - python code for the task, in \\boxed{}.\n Here are examples of correct answers:\n" 

        
        # Add few-shot examples if provided
        if examples:
            for ex_intent, ex_snippet in examples:
                prompt += f"\nTask: {ex_intent}\n\\boxed{{{ex_snippet}}}\n"
        


        prompt += f"\nTask: {nl_intent}\n"

        return prompt

    def evaluate(self, response: str, expected: Any, **_kwargs) -> EvalResult:
        """Evaluate model response against expected Python code.

        Args:
            response: Model-generated Python code
            expected: Expected Python code snippet

        Returns:
            EvalResult with BLEU score as confidence
        """
        if not response or not response.strip():
            return EvalResult(
                ground_truth="UNKNOWN",
                method_used=self.name,
                confidence=0.0,
                details="Empty response",
                meta={
                    "response_preview": None,
                    "expected": expected,
                    "bleu_score": 0.0,
                }
            )

        expected_str = str(expected).strip()
        response_str = response.strip()

        # Tokenize for logging
        ref_tokens = tokenize_for_bleu_eval(expected_str)
        hyp_tokens = tokenize_for_bleu_eval(response_str)

        # Compute BLEU score
        bleu_score = compute_bleu_single(
            reference=expected_str,
            hypothesis=response_str,
            max_order=self.max_order,
            smooth=False,
        )

        # Determine truthfulness based on BLEU threshold only
        is_correct = bleu_score >= self.bleu_threshold

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=bleu_score,
            details=f"BLEU: {bleu_score:.4f}",
            meta={
                "bleu_score": bleu_score,
                "expected_tokens": ref_tokens,
                "response_tokens": hyp_tokens,
                "bleu_threshold": self.bleu_threshold,
            }
        )

    def evaluate_corpus(
        self,
        responses: list[str],
        expected_answers: list[str],
    ) -> dict[str, Any]:
        """Evaluate a corpus of responses and compute corpus-level BLEU.

        This is the standard evaluation approach for CoNaLa leaderboard.

        Args:
            responses: List of generated Python code snippets
            expected_answers: List of reference Python code snippets

        Returns:
            Dictionary with corpus-level metrics
        """
        if len(responses) != len(expected_answers):
            raise ValueError("Number of responses must match number of expected answers")

        # Tokenize all pairs
        references = []
        hypotheses = []

        for response, expected in zip(responses, expected_answers):
            ref_tokens = tokenize_for_bleu_eval(str(expected).strip())
            hyp_tokens = tokenize_for_bleu_eval(response.strip() if response else "")

            references.append(ref_tokens)
            hypotheses.append(hyp_tokens)

        # Compute corpus BLEU
        bleu, precisions, bp, ratio, _, _ = compute_bleu(
            references=references,
            hypotheses=hypotheses,
            max_order=self.max_order,
            smooth=False,
        )

        return {
            "bleu_score": bleu * 100,  # Convert to percentage like leaderboard
            "total": len(responses),
            "brevity_penalty": bp,
            "length_ratio": ratio,
            "precisions": precisions,
        }
