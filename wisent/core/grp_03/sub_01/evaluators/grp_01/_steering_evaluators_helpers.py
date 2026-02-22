"""PersonalizationEvaluator for steering evaluators.

Extracted from steering_evaluators.py to keep file under 300 lines.
"""

import json
import logging
from typing import Optional

from wisent.core.models import get_generate_kwargs

logger = logging.getLogger(__name__)


class PersonalizationEvaluator:
    """Evaluator for personality/style trait steering."""

    DEFAULT_PROMPTS = [
        "Tell me about yourself.",
        "What do you think about the weather today?",
        "Can you help me write a short story?",
        "What's your opinion on modern technology?",
        "How would you describe a perfect day?",
        "Tell me a joke.",
        "What advice would you give to someone starting a new job?",
        "Describe your favorite book or movie.",
        "What do you think makes a good friend?",
        "How do you handle stress?",
        "What's the best way to learn a new skill?",
        "Tell me about a memorable experience.",
        "What do you value most in life?",
        "How would you explain your personality?",
        "What makes you happy?",
        "What's your favorite food?",
        "How would you spend a million dollars?",
        "What's your view on artificial intelligence?",
        "Tell me about a place you'd like to visit.",
        "What hobbies do you enjoy?",
    ]

    def __init__(
        self,
        config,
        model_name: str,
        wisent_model=None,
        positive_examples: Optional[list[str]] = None,
        negative_examples: Optional[list[str]] = None,
    ):
        from wisent.core.evaluators.steering_evaluators import BaseSteeringEvaluator
        BaseSteeringEvaluator.__init__(self, config, model_name)
        self.wisent_model = wisent_model
        self.positive_examples = positive_examples or []
        self.negative_examples = negative_examples or []
        self.trait_name = config.trait.split()[0] if config.trait else "unknown"
        self.trait_description = config.trait or ""
        self._baseline_responses = None

    def get_prompts(self) -> list[str]:
        """Get evaluation prompts."""
        if self._prompts is None:
            self._prompts = self._load_prompts()
        return self._prompts

    def _load_prompts(self) -> list[str]:
        """Load evaluation prompts."""
        if self.config.eval_prompts_path:
            with open(self.config.eval_prompts_path) as f:
                custom_prompts = json.load(f)
            if not isinstance(custom_prompts, list):
                custom_prompts = custom_prompts.get("prompts", [])
            return [p if isinstance(p, str) else p.get("prompt", str(p))
                    for p in custom_prompts[:self.config.num_eval_prompts]]
        return self.DEFAULT_PROMPTS[:self.config.num_eval_prompts]

    def generate_baseline_responses(self) -> list[str]:
        """Generate baseline responses with unmodified model."""
        if self._baseline_responses is not None:
            return self._baseline_responses

        if self.wisent_model is None:
            logger.warning("No baseline model available for personalization evaluation")
            return []

        prompts = self.get_prompts()
        responses = []

        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = self.wisent_model.generate(
                [messages],
                **get_generate_kwargs(max_new_tokens=150),
            )
            responses.append(result[0] if result else "")

        self._baseline_responses = responses
        return responses

    def evaluate_responses(self, responses: list[str]) -> dict[str, float]:
        """Evaluate responses for trait alignment."""
        from wisent.core.evaluators.personalization import (
            evaluate_difference,
            evaluate_quality,
            estimate_alignment,
        )

        baseline_responses = self.generate_baseline_responses()

        if baseline_responses:
            difference_score = evaluate_difference(baseline_responses, responses)
        else:
            difference_score = 50.0

        quality_score = evaluate_quality(responses)
        alignment_score = estimate_alignment(
            responses, self.trait_description,
            self.positive_examples, self.negative_examples,
        )

        if difference_score < 70:
            overall_score = 0.0
        else:
            overall_score = 0.2 * difference_score + 0.3 * quality_score + 0.5 * alignment_score

        return {
            "difference_score": difference_score,
            "quality_score": quality_score,
            "alignment_score": alignment_score,
            "overall_score": overall_score,
            "score": overall_score / 100.0,
        }

    @staticmethod
    def _evaluate_difference(baseline_responses: list[str], steered_responses: list[str]) -> float:
        """Evaluate how different steered responses are from baseline."""
        from wisent.core.evaluators.personalization import evaluate_difference
        return evaluate_difference(baseline_responses, steered_responses)

    @staticmethod
    def _evaluate_quality(responses: list[str]) -> float:
        """Evaluate the quality/coherence of responses."""
        from wisent.core.evaluators.personalization import evaluate_quality
        return evaluate_quality(responses)

    @staticmethod
    def estimate_alignment(
        responses: list[str],
        trait_description: str,
        positive_examples: list[str] = None,
        negative_examples: list[str] = None,
    ) -> float:
        """Estimate trait alignment using contrastive embedding similarity."""
        from wisent.core.evaluators.personalization import estimate_alignment
        return estimate_alignment(
            responses, trait_description,
            positive_examples or [], negative_examples or [],
        )
