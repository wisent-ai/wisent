"""
Shared steering evaluators for both optimize-weights and optimize-steering.

This module provides a unified interface for evaluating steering effectiveness
across different evaluation types (refusal, task, personalization, custom).
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.config_tools.constants import (
    DEFAULT_RANDOM_SEED,
)

# Re-export from helpers
from wisent.core.reading.evaluators._steering_evaluators_helpers import PersonalizationEvaluator

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for steering evaluators."""
    evaluator_type: Optional[str] = None  # auto, refusal, task, personalization
    trait: Optional[str] = None
    task: Optional[str] = None
    eval_prompts_path: Optional[str] = None
    eval_topics: Optional[str] = None


class SteeringEvaluatorFactory:
    """Factory for creating steering evaluators."""

    @staticmethod
    def create(
        config: EvaluatorConfig,
        model_name: str,
        wisent_model: Optional[WisentModel] = None,
        positive_examples: Optional[list[str]] = None,
        negative_examples: Optional[list[str]] = None,
        *,
        fast_diversity_seed: int,
        diversity_max_sample_size: int,
        min_sentence_length: int,
        nonsense_min_tokens: int,
        quality_min_response_length: int,
        quality_repetition_ratio_threshold: float,
        quality_bigram_repeat_threshold: int,
        quality_bigram_repeat_penalty: float,
        quality_special_char_ratio_threshold: float,
        quality_special_char_penalty: float,
        quality_char_repeat_count: int,
        quality_char_repeat_penalty: float,
        difference_weight: float,
        quality_weight: float,
        alignment_weight: float,
    ) -> "BaseSteeringEvaluator":
        """Create the appropriate evaluator based on config."""
        evaluator_type = config.evaluator_type

        if evaluator_type is None:
            if config.trait and "refus" in config.trait.lower():
                evaluator_type = "refusal"
            elif config.task:
                evaluator_type = "task"
            elif config.trait:
                evaluator_type = "personalization"
            else:
                evaluator_type = "refusal"

        if evaluator_type == "refusal":
            return RefusalEvaluator(config, model_name)
        elif evaluator_type == "task":
            return TaskEvaluator(config, model_name)
        elif evaluator_type == "personalization":
            return PersonalizationEvaluator(
                config, model_name, wisent_model, positive_examples, negative_examples,
                fast_diversity_seed=fast_diversity_seed,
                diversity_max_sample_size=diversity_max_sample_size,
                min_sentence_length=min_sentence_length,
                nonsense_min_tokens=nonsense_min_tokens,
                quality_min_response_length=quality_min_response_length,
                quality_repetition_ratio_threshold=quality_repetition_ratio_threshold,
                quality_bigram_repeat_threshold=quality_bigram_repeat_threshold,
                quality_bigram_repeat_penalty=quality_bigram_repeat_penalty,
                quality_special_char_ratio_threshold=quality_special_char_ratio_threshold,
                quality_special_char_penalty=quality_special_char_penalty,
                quality_char_repeat_count=quality_char_repeat_count,
                quality_char_repeat_penalty=quality_char_repeat_penalty,
                difference_weight=difference_weight,
                quality_weight=quality_weight,
                alignment_weight=alignment_weight,
            )
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")


class BaseSteeringEvaluator:
    """Base class for steering evaluators."""

    def __init__(self, config: EvaluatorConfig, model_name: str):
        self.config = config
        self.model_name = model_name
        self._prompts = None

    def get_prompts(self) -> list[str]:
        """Get evaluation prompts."""
        if self._prompts is None:
            self._prompts = self._load_prompts()
        return self._prompts

    def _load_prompts(self) -> list[str]:
        """Load prompts from config or defaults. Override in subclass."""
        raise NotImplementedError

    def evaluate_responses(self, responses: list[str]) -> dict[str, float]:
        """Evaluate a list of responses. Override in subclass."""
        raise NotImplementedError

    def evaluate_with_model(self, hf_model, tokenizer) -> dict[str, float]:
        """Generate responses with model and evaluate them."""
        wisent_model = WisentModel(self.model_name, hf_model=hf_model)
        prompts = self.get_prompts()

        responses = []
        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            result = wisent_model.generate(
                [messages],
                **get_generate_kwargs(),
            )
            responses.append(result[0] if result else "")

        return self.evaluate_responses(responses)


class RefusalEvaluator(BaseSteeringEvaluator):
    """Evaluator for refusal/compliance using UncensorBench."""

    def __init__(self, config: EvaluatorConfig, model_name: str):
        super().__init__(config, model_name)
        from uncensorbench import UncensorBench
        self.bench = UncensorBench()
        self._setup_evaluator()

    def _setup_evaluator(self):
        """Setup the semantic or keyword evaluator."""
        from uncensorbench.evaluator import KeywordEvaluator, SemanticEvaluator
        try:
            self.evaluator = SemanticEvaluator()
        except ImportError:
            self.evaluator = KeywordEvaluator()

    def _load_prompts(self) -> list[str]:
        """Load refusal evaluation prompts."""
        if self.config.eval_prompts_path:
            with open(self.config.eval_prompts_path) as f:
                custom_prompts = json.load(f)
            if not isinstance(custom_prompts, list):
                custom_prompts = custom_prompts.get("prompts", [])
            self._prompt_objects = None
            return [p if isinstance(p, str) else p.get("prompt", str(p))
                    for p in custom_prompts[:30]]
        else:
            topics = self.config.eval_topics.split(",") if self.config.eval_topics else None
            prompt_objects = list(self.bench.prompts(topics=topics))[:30]
            self._prompt_objects = prompt_objects
            return [p.prompt for p in prompt_objects]

    def evaluate_responses(self, responses: list[str]) -> dict[str, float]:
        """Evaluate responses for refusal/compliance."""
        prompts = self.get_prompts()
        prompt_objects = getattr(self, '_prompt_objects', None)

        refusal_count = 0
        compliance_scores = []

        for i, response in enumerate(responses):
            if prompt_objects and i < len(prompt_objects):
                eval_result = self.bench.evaluate_response(prompt_objects[i], response, self.evaluator)
                is_refusal = not eval_result.is_uncensored
                compliance_score = eval_result.compliance_score
            else:
                is_refusal = self.bench.detect_refusal(response)
                compliance_score = 0.0 if is_refusal else 1.0

            if is_refusal:
                refusal_count += 1
            compliance_scores.append(compliance_score)

        total = len(responses)
        refusal_rate = refusal_count / total if total > 0 else 0.0
        compliance_rate = 1 - refusal_rate
        avg_compliance = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0.0

        return {
            "refusal_rate": refusal_rate,
            "compliance_rate": compliance_rate,
            "average_compliance_score": avg_compliance,
            "refusal_count": refusal_count,
            "total": total,
            "score": compliance_rate,
        }


class TaskEvaluator(BaseSteeringEvaluator):
    """Evaluator for task-based (lm-eval) benchmarks."""

    def __init__(self, config: EvaluatorConfig, model_name: str):
        super().__init__(config, model_name)
        self._load_task_data()

    def _load_task_data(self):
        """Pre-load task data."""
        from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
        from wisent.core.reading.evaluators.rotator import EvaluatorRotator

        loader = LMEvalDataLoader()
        EvaluatorRotator.discover_evaluators('wisent.core.reading.evaluators.benchmark_specific')

        result = loader._load_one_task(
            task_name=self.config.task,
            split_ratio=0.8, seed=DEFAULT_RANDOM_SEED,
            limit=30,
            training_limit=None,
            testing_limit=30,
        )
        self._test_pairs = result["test_qa_pairs"]
        self._evaluator = EvaluatorRotator(evaluator=None, task_name=self.config.task)

    def _load_prompts(self) -> list[str]:
        """Get prompts from task pairs."""
        return [pair.prompt for pair in self._test_pairs.pairs]

    def get_expected_answers(self) -> list[str]:
        """Get expected answers for evaluation."""
        return [pair.positive_response.model_response for pair in self._test_pairs.pairs]

    def evaluate_responses(self, responses: list[str]) -> dict[str, float]:
        """Evaluate responses against expected answers."""
        expected = self.get_expected_answers()

        correct = 0
        total = len(responses)

        for response, exp in zip(responses, expected):
            is_correct = self._evaluator.is_correct(response, exp)
            if is_correct:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "score": accuracy,
        }
