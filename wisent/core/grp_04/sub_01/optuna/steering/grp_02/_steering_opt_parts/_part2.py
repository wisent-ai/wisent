"""SteeringTrainer - generic trainer using the centralized steering method registry.

Also contains SteeringMethodTrainer ABC to avoid circular imports.
Split from steering_optimization.py to meet 300-line limit.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from wisent.core.contrastive_pairs.contrastive_pair import ContrastivePair
from wisent.core.contrastive_pairs.contrastive_pair_set import ContrastivePairSet
from wisent.core.models import get_generate_kwargs
from wisent.core.optuna.steering import data_utils
from wisent.core.response import Response
from wisent.core.steering_methods import get_steering_method
from wisent.core.tasks.base.task_interface import get_task
from wisent.core.errors import ModelArchitectureUnknownError

logger = logging.getLogger(__name__)


class SteeringMethodTrainer(ABC):
    """Abstract base class for training different steering methods."""

    @abstractmethod
    def create_method_instance(self, hyperparams: Dict[str, Any], device: str) -> Any:
        """Create an instance of the steering method with given hyperparameters."""

    @abstractmethod
    def train_method(
        self,
        method_instance: Any,
        train_samples: List[Dict],
        layer: int,
        model,
        tokenizer,
        device: str,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Train the steering method on training data."""

    @abstractmethod
    def apply_steering_and_evaluate(
        self,
        method_instance: Any,
        evaluation_samples: List[Dict],
        layer: int,
        strength: float,
        model,
        tokenizer,
        device: str,
        batch_size: int,
        max_length: int,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[List[str], List[str]]:
        """Apply steering and generate predictions for evaluation."""


class SteeringTrainer(SteeringMethodTrainer):
    """Generic trainer that uses the centralized steering method registry."""

    def __init__(self, method_name: str = "caa"):
        self.method_name = method_name

    def create_method_instance(self, hyperparams: Dict[str, Any], device: str):
        """Create steering method instance from registry."""
        return get_steering_method(self.method_name, device=device, **hyperparams)

    def train_method(
        self,
        steering_instance,
        train_samples: List[Dict],
        layer: int,
        model,
        tokenizer,
        device: str,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[bool, Dict[str, Any]]:
        """Train steering method on training data to create steering vectors."""
        try:
            # Extract contrastive pairs from training data using task's extractor
            contrastive_pairs = data_utils.get_task_contrastive_pairs(train_samples, task_name)

            if not contrastive_pairs:
                logger.warning(f"No contrastive pairs extracted from {task_name} training data")
                return False, {"error": "No contrastive pairs"}

            # Convert to ContrastivePairSet format
            pair_set = self._create_pair_set_from_extracted_pairs(contrastive_pairs, layer, model, tokenizer, device)

            # Train steering method
            training_result = steering_instance.train(pair_set, layer)

            success = training_result.get("success", False)
            logger.debug(f"{self.method_name.upper()} training on layer {layer}: {'Success' if success else 'Failed'}")

            return success, training_result

        except Exception as e:
            logger.error(f"{self.method_name.upper()} training failed on layer {layer}: {e}")
            return False, {"error": str(e)}

    def apply_steering_and_evaluate(
        self,
        steering_instance,
        evaluation_samples: List[Dict],
        layer: int,
        strength: float,
        model,
        tokenizer,
        device: str,
        batch_size: int,
        max_length: int,
        task_name: str = "gsm8k",
        max_new_tokens: int = 200,
    ) -> Tuple[List[str], List[str]]:
        """Apply steering and generate predictions using task extractor."""

        predictions = []
        ground_truths = []

        # Get the task and its extractor
        task = get_task(task_name)
        extractor = task.get_extractor()

        # Pre-extract all questions and answers (optimization)
        questions = []
        answers = []

        for sample in evaluation_samples:
            qa_pair = extractor.extract_qa_pair(sample, task)
            if not qa_pair:
                logger.warning(f"Skipping sample - extractor couldn't extract QA pair: {sample.keys()}")
                continue
            questions.append(qa_pair["formatted_question"])
            answers.append(qa_pair["correct_answer"])

        # Process questions with steering in batches (optimized approach)
        ground_truths.extend(answers)

        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-style models
            layer_module = model.model.layers[layer]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT2-style models
            layer_module = model.transformer.h[layer]
        else:
            raise ModelArchitectureUnknownError()

        # Process in batches with steering
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating predictions with steering"):
            batch_questions = questions[i : i + batch_size]

            # First, get actual lengths (before padding) for proper steering
            actual_lengths = []
            for question in batch_questions:
                tokens = tokenizer(question, return_tensors="pt")
                actual_lengths.append(tokens["input_ids"].shape[1])

            # Create batched steering hook that handles variable lengths
            def create_batched_steering_hook(actual_lengths):
                def steering_hook(module, input, output):
                    hidden_states = output[0]  # [batch_size, seq_len, hidden_dim]

                    # Apply steering to each sample's actual last token
                    for j, actual_length in enumerate(actual_lengths):
                        if j < hidden_states.shape[0]:  # Safety check for batch size
                            # Get the actual last token (before padding)
                            last_token = hidden_states[j : j + 1, actual_length - 1 : actual_length, :]
                            steered = steering_instance.apply_steering(last_token, strength=strength)
                            hidden_states[j : j + 1, actual_length - 1 : actual_length, :] = steered

                    return (hidden_states,) + output[1:]

                return steering_hook

            # Register the batched hook
            batched_hook = create_batched_steering_hook(actual_lengths)
            handle = layer_module.register_forward_hook(batched_hook)

            try:
                # Tokenize batch with padding for generation
                inputs = tokenizer(
                    batch_questions, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(device)

                with torch.no_grad():
                    gen_kwargs = get_generate_kwargs(max_new_tokens=max_new_tokens)
                    outputs = model.generate(
                        **inputs,
                        **gen_kwargs,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable cache to avoid cache_position errors
                    )

                # Decode responses for each item in batch
                for j, (output, question) in enumerate(zip(outputs, batch_questions)):
                    response = tokenizer.decode(output, skip_special_tokens=True)
                    prediction = response[len(question) :].strip()
                    predictions.append(prediction)

            finally:
                handle.remove()

        return predictions, ground_truths

    def _create_pair_set_from_extracted_pairs(
        self, extracted_pairs: List[Dict], layer_index: int, model, tokenizer, device: str
    ) -> ContrastivePairSet:
        """Convert extracted pairs to ContrastivePairSet format with proper activation extraction."""
        pair_set = ContrastivePairSet(name="caa_training", task_type="mathematical_reasoning")

        logger.info(f"Creating {len(extracted_pairs)} contrastive pairs for layer {layer_index}")

        for pair_data in tqdm(extracted_pairs, desc="Creating contrastive pairs"):
            # Extract data from GSM8K format
            try:
                question = pair_data["question"]
                correct_answer = pair_data["correct_answer"]
                incorrect_answer = pair_data["incorrect_answer"]

                # Extract activations for correct and incorrect responses
                correct_activations = self._extract_activations_for_text(
                    f"{question} {correct_answer}", layer_index, model, tokenizer, device
                )
                incorrect_activations = self._extract_activations_for_text(
                    f"{question} {incorrect_answer}", layer_index, model, tokenizer, device
                )

                # Create Response objects
                positive_response = Response(text=correct_answer, activations=correct_activations)
                negative_response = Response(text=incorrect_answer, activations=incorrect_activations)

                # Create ContrastivePair
                contrastive_pair = ContrastivePair(
                    prompt=question, positive_response=positive_response, negative_response=negative_response
                )

                pair_set.pairs.append(contrastive_pair)

            except Exception as e:
                logger.warning(f"Failed to create contrastive pair: {e}")
                continue

        logger.info(f"Successfully created ContrastivePairSet with {len(pair_set.pairs)} pairs")
        return pair_set

    def _extract_activations_for_text(self, text: str, layer_index: int, model, tokenizer, device: str) -> torch.Tensor:
        """Extract activations from a specific layer for given text."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)

        activations = []

        def hook(module, input, output):
            # Extract the last token's activations
            hidden_states = output[0]
            last_token_activations = hidden_states[:, -1, :]
            activations.append(last_token_activations.detach().cpu())

        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-style models
            layer_module = model.model.layers[layer_index]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT2-style models
            layer_module = model.transformer.h[layer_index]
        else:
            raise ModelArchitectureUnknownError()

        handle = layer_module.register_forward_hook(hook)

        with torch.no_grad():
            model(**inputs)

        handle.remove()
        return activations[0].squeeze(0)
