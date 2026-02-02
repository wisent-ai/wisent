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
import re
from typing import Any

import evaluate

from wisent.core.evaluators.core.atoms import BaseEvaluator, EvalResult

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
        self.bleu_metric = evaluate.load("bleu")

    @staticmethod
    def get_prompt(
        intent: str,
        rewritten_intent: str | None = None,
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

        prompt = "Generate Python code for the following task. Put final answer, in \\boxed{}."

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

        # Compute BLEU score using HuggingFace evaluate
        result = self.bleu_metric.compute(
            predictions=[response_str],
            references=[[expected_str]],
            tokenizer=tokenize_for_bleu_eval,
            max_order=self.max_order,
            smooth=False,
        )
        bleu_score = result["bleu"]

        # Determine truthfulness based on BLEU threshold only
        is_correct = bleu_score >= self.bleu_threshold

        return EvalResult(
            ground_truth="TRUTHFUL" if is_correct else "UNTRUTHFUL",
            method_used=self.name,
            confidence=bleu_score,
            details=f"BLEU: {bleu_score:.4f}",
            meta={
                "bleu_score": bleu_score,
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

        # Prepare predictions and references for HF BLEU
        predictions = [r.strip() if r else "" for r in responses]
        references = [[str(e).strip()] for e in expected_answers]

        # Check if all predictions are empty (would cause division by zero)
        if all(not p for p in predictions):
            return {
                "bleu_score": 0.0,
                "total": len(responses),
                "brevity_penalty": 0.0,
                "length_ratio": 0.0,
                "precisions": [0.0] * self.max_order,
            }

        # Compute corpus BLEU using HuggingFace evaluate
        result = self.bleu_metric.compute(
            predictions=predictions,
            references=references,
            tokenizer=tokenize_for_bleu_eval,
            max_order=self.max_order,
            smooth=False,
        )

        return {
            "bleu_score": result["bleu"] * 100,  # Convert to percentage like leaderboard
            "total": len(responses),
            "brevity_penalty": result["brevity_penalty"],
            "length_ratio": result["length_ratio"],
            "precisions": result["precisions"],
        }
