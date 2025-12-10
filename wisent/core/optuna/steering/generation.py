"""
Generation helpers for the steering optimization pipeline.
"""

import logging
from typing import Any, Optional

from tqdm import tqdm

from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.task_interface import get_task

from .config import OptimizationConfig


logger = logging.getLogger(__name__)


class GenerationHelper:
    """Helper class for generating baseline and steered predictions."""

    def __init__(
        self,
        wisent_model: WisentModel,
        config: OptimizationConfig,
    ):
        self.wisent_model = wisent_model
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _build_gen_kwargs(self) -> dict[str, Any]:
        """Build generation kwargs from config."""
        gen_kwargs = {}
        if self.config.max_new_tokens:
            gen_kwargs["max_new_tokens"] = self.config.max_new_tokens
        if self.config.do_sample is not None:
            gen_kwargs["do_sample"] = self.config.do_sample
        if self.config.temperature is not None:
            gen_kwargs["temperature"] = self.config.temperature
        return gen_kwargs

    def generate_baseline_batched(self, questions: list[str]) -> list[str]:
        """Generate baseline responses in batches without steering using WisentModel."""
        if not questions:
            return []

        batch_size = self.config.batch_size
        all_responses = []
        gen_kwargs = self._build_gen_kwargs()

        for i in tqdm(range(0, len(questions), batch_size), desc="Generating baseline predictions", leave=False):
            batch_questions = questions[i : i + batch_size]
            batch_responses = self.wisent_model.generate(
                inputs=batch_questions,
                use_steering=False,
                prompt_is_formatted=True,
                **gen_kwargs
            )
            all_responses.extend(batch_responses)

        return all_responses

    def generate_with_steering_batched(
        self, steering_instance: Any, questions: list[str], alpha: float, layer_id: int
    ) -> list[str]:
        """Generate responses with steering applied in batches using WisentModel."""
        if not questions:
            return []

        batch_size = self.config.batch_size
        all_responses = []

        steering_vector = None
        if hasattr(steering_instance, '_trained_vectors') and steering_instance._trained_vectors is not None:
            layer_key = f"layer_{layer_id}"
            if layer_key in steering_instance._trained_vectors:
                steering_vector = steering_instance._trained_vectors[layer_key]
        
        if steering_vector is None:
            self.logger.warning(f"No steering vector found for layer {layer_id}, returning baseline")
            return self.generate_baseline_batched(questions)
        
        layer_name = str(layer_id + 1)
        steering_plan = SteeringPlan.from_raw(
            raw={layer_name: steering_vector},
            scale=alpha,
            normalize=False
        )

        gen_kwargs = self._build_gen_kwargs()

        for i in tqdm(range(0, len(questions), batch_size), desc="Generating steered predictions", leave=False):
            batch_questions = questions[i : i + batch_size]
            batch_responses = self.wisent_model.generate(
                inputs=batch_questions,
                use_steering=True,
                steering_plan=steering_plan,
                prompt_is_formatted=True,
                **gen_kwargs
            )
            all_responses.extend(batch_responses)

        return all_responses

    def generate_predictions_batched(
        self, steering_instance: Any, questions: list[str], alpha: float, layer_id: int
    ) -> list[str]:
        """Generate predictions for a list of questions, with or without steering."""
        if not questions:
            return []
        try:
            if steering_instance is None:
                return self.generate_baseline_batched(questions)
            else:
                return self.generate_with_steering_batched(steering_instance, questions, alpha, layer_id)
        except Exception as e:
            self.logger.warning(f"Batched generation failed: {e}")
            return ["Error"] * len(questions)

    def generate_test_predictions(
        self,
        steering_instance: Any,
        method_name: str,
        layer_id: int,
        alpha: float,
        test_samples: list[dict],
        evaluator_type: str,
        steering_evaluator: Optional[Any] = None,
        trait: Optional[str] = None,
        test_dataset: Optional[str] = None,
    ) -> tuple[list[str], list[str], list[str], list[dict]]:
        """Generate predictions on test data using batched generation."""
        if evaluator_type == "personalization":
            if steering_evaluator is None:
                self.logger.warning("No steering evaluator for personalization test predictions")
                return [], [], [], []
            prompts = steering_evaluator.get_prompts()
            if not prompts:
                return [], [], [], []
            questions = prompts
            ground_truths = [trait] * len(prompts)
            valid_samples = [{"prompt": p, "trait": trait} for p in prompts]
        else:
            questions = []
            ground_truths = []
            valid_samples = []
            task = get_task(test_dataset)
            extractor = task.get_extractor()

            for sample in test_samples:
                qa_pair = extractor.extract_qa_pair(sample, task)
                if not qa_pair:
                    continue
                questions.append(qa_pair["formatted_question"])
                ground_truths.append(qa_pair["correct_answer"])
                valid_samples.append(sample)

        predictions = self.generate_predictions_batched(steering_instance, questions, alpha, layer_id)
        
        for i, (pred, gt) in enumerate(zip(predictions[:3], ground_truths[:3])):
            self.logger.debug(f"Test Sample {i} - Model: ...{pred[-50:] if pred else 'None'}")
            self.logger.debug(f"Test Sample {i} - Ground truth: {gt}")

        return predictions, ground_truths, questions, valid_samples
