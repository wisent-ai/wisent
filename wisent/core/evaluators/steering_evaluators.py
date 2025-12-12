"""
Shared steering evaluators for both optimize-weights and optimize-steering.

This module provides a unified interface for evaluating steering effectiveness
across different evaluation types (refusal, task, personalization, custom).
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.inference_config import get_generate_kwargs

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for steering evaluators."""
    evaluator_type: str = "auto"  # auto, refusal, task, personalization
    trait: Optional[str] = None
    task: Optional[str] = None
    eval_prompts_path: Optional[str] = None
    eval_topics: Optional[str] = None
    num_eval_prompts: int = 30


class SteeringEvaluatorFactory:
    """Factory for creating steering evaluators."""
    
    @staticmethod
    def create(
        config: EvaluatorConfig,
        model_name: str,
        wisent_model: Optional[WisentModel] = None,
        positive_examples: Optional[list[str]] = None,
        negative_examples: Optional[list[str]] = None,
    ) -> "BaseSteeringEvaluator":
        """Create the appropriate evaluator based on config.
        
        Args:
            config: Evaluator configuration
            model_name: Model name/path
            wisent_model: Optional WisentModel for baseline generation
            positive_examples: Positive examples from contrastive pairs
            negative_examples: Negative examples from contrastive pairs
            
        Returns:
            Configured evaluator instance
        """
        evaluator_type = config.evaluator_type
        
        # Auto-select evaluator based on trait/task
        if evaluator_type == "auto":
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
                config, model_name, wisent_model, positive_examples, negative_examples
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
                **get_generate_kwargs(max_new_tokens=150),
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
                    for p in custom_prompts[:self.config.num_eval_prompts]]
        else:
            topics = self.config.eval_topics.split(",") if self.config.eval_topics else None
            prompt_objects = list(self.bench.prompts(topics=topics))[:self.config.num_eval_prompts]
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
        from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
        from wisent.core.evaluators.rotator import EvaluatorRotator
        
        loader = LMEvalDataLoader()
        EvaluatorRotator.discover_evaluators('wisent.core.evaluators.benchmark_specific')
        
        result = loader._load_one_task(
            task_name=self.config.task,
            split_ratio=0.8,
            seed=42,
            limit=self.config.num_eval_prompts,
            training_limit=None,
            testing_limit=self.config.num_eval_prompts,
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


class PersonalizationEvaluator(BaseSteeringEvaluator):
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
        config: EvaluatorConfig,
        model_name: str,
        wisent_model: Optional[WisentModel] = None,
        positive_examples: Optional[list[str]] = None,
        negative_examples: Optional[list[str]] = None,
    ):
        super().__init__(config, model_name)
        self.wisent_model = wisent_model
        self.positive_examples = positive_examples or []
        self.negative_examples = negative_examples or []
        self.trait_name = config.trait.split()[0] if config.trait else "unknown"
        self.trait_description = config.trait or ""
        self._baseline_responses = None
    
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
        
        # Evaluate difference from baseline
        if baseline_responses:
            difference_score = evaluate_difference(baseline_responses, responses)
        else:
            difference_score = 50.0  # Default if no baseline
        
        # Evaluate response quality
        quality_score = evaluate_quality(responses)
        
        # Evaluate trait alignment
        alignment_score = estimate_alignment(
            responses,
            self.trait_description,
            self.positive_examples,
            self.negative_examples,
        )
        
        # Compute overall score (weighted average)
        # Difference: 20%, Quality: 30%, Alignment: 50%
        if difference_score < 70:
            overall_score = 0.0  # Steering not effective if responses too similar
        else:
            overall_score = 0.2 * difference_score + 0.3 * quality_score + 0.5 * alignment_score
        
        return {
            "difference_score": difference_score,
            "quality_score": quality_score,
            "alignment_score": alignment_score,
            "overall_score": overall_score,
            "score": overall_score / 100.0,  # Normalize to 0-1
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
            responses,
            trait_description,
            positive_examples or [],
            negative_examples or [],
        )
