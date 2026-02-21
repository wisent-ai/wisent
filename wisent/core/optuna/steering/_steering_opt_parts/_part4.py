"""_SteeringOptimizerEval - baseline collection, activation extraction, classifier scoring.

Split from steering_optimization.py to meet 300-line limit.
"""

import logging
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from wisent.core.activations import ExtractionStrategy
from wisent.core.classifier.classifier import Classifier
from wisent.core.models import get_generate_kwargs
from wisent.core.tasks.base.task_interface import get_task
from wisent.core.errors import (
    ModelArchitectureUnknownError,
    NoActivationDataError,
)

logger = logging.getLogger(__name__)


class _SteeringOptimizerEval:
    """Evaluation mixin: baseline predictions, activation extraction, classifier scoring."""

    def collect_baseline_predictions(
        self,
        evaluation_samples: List[Dict],
        model,
        tokenizer,
        classifier: Classifier,
        device: str,
        batch_size: int,
        max_length: int,
        task_name: str,
        max_new_tokens: int = 200,
    ) -> Tuple[List[str], List[str]]:
        """
        Collect unsteered model predictions for baseline comparison.
        Uses the same evaluation logic as steered evaluation but without steering hooks.

        Returns:
            Tuple of (predictions, ground_truths)
        """
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
                self.logger.warning(f"Skipping sample - extractor couldn't extract QA pair: {sample.keys()}")
                continue
            questions.append(qa_pair["formatted_question"])
            answers.append(qa_pair["correct_answer"])

        # Process questions WITHOUT steering in batches
        ground_truths.extend(answers)

        # Process in batches without steering
        for i in tqdm(range(0, len(questions), batch_size), desc="Generating baseline predictions"):
            batch_questions = questions[i : i + batch_size]

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

        return predictions, ground_truths

    def _extract_activation_for_text(
        self,
        text: str,
        layer_index: int,
        aggregation_strategy: str,
        model,
        tokenizer,
        device: str,
        max_length: int = 512,
    ) -> torch.Tensor:
        """
        Extract activation from text at specified layer with aggregation.

        Returns:
            Aggregated activation tensor
        """
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        activations = []

        def hook(module, input, output):
            # Extract hidden states from the layer
            hidden_states = output[0] if isinstance(output, tuple) else output
            activations.append(hidden_states.detach().cpu())

        # Handle different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # LLaMA-style models
            layer_module = model.model.layers[layer_index]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT2-style models
            layer_module = model.transformer.h[layer_index]
        else:
            raise ModelArchitectureUnknownError()

        # Register hook and run forward pass
        handle = layer_module.register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = model(**inputs)
        finally:
            handle.remove()

        if not activations:
            raise NoActivationDataError()

        # Get the activation tensor [1, seq_len, hidden_dim]
        activation_tensor = activations[0]

        # Apply aggregation strategy
        aggregated = self._apply_aggregation(activation_tensor, aggregation_strategy)

        return aggregated.squeeze(0)  # Return [hidden_dim] tensor

    def _apply_aggregation(self, activation_tensor: torch.Tensor, aggregation_strategy: str) -> torch.Tensor:
        """Apply aggregation strategy to activation tensor."""
        if (
            aggregation_strategy == "mean_pooling"
            or aggregation_strategy == ExtractionStrategy.CHAT_MEAN.value
        ):
            return torch.mean(activation_tensor, dim=1)  # [1, hidden_dim]
        elif (
            aggregation_strategy == "last_token"
            or aggregation_strategy == ExtractionStrategy.CHAT_LAST.value
        ):
            return activation_tensor[:, -1, :]  # [1, hidden_dim]
        elif (
            aggregation_strategy == "first_token"
            or aggregation_strategy == ExtractionStrategy.CHAT_FIRST.value
        ):
            return activation_tensor[:, 0, :]  # [1, hidden_dim]
        elif (
            aggregation_strategy == "max_pooling"
            or aggregation_strategy == ExtractionStrategy.CHAT_MAX_NORM.value
        ):
            return torch.max(activation_tensor, dim=1)[0]  # [1, hidden_dim]
        elif (
            aggregation_strategy == "min_pooling"
            or aggregation_strategy == ExtractionStrategy.CHAT_MEAN.value
        ):
            return torch.min(activation_tensor, dim=1)[0]  # [1, hidden_dim]
        else:
            # Default to mean pooling if unknown
            self.logger.warning(f"Unknown aggregation strategy {aggregation_strategy}, using mean pooling")
            return torch.mean(activation_tensor, dim=1)

    def score_predictions_with_classifier(
        self,
        predictions: List[str],
        model,
        tokenizer,
        device: str,
        max_length: int = 512,
        description: str = "predictions",
    ) -> List[float]:
        """
        Score predictions using the cached classifier.

        This is the core feature that was requested - using the optimized classifier
        to score unsteered vs steered generations.

        Returns:
            List of classifier scores/probabilities for each prediction
        """
        if self._session_classifier is None:
            self.logger.warning("No cached classifier available for scoring")
            return [0.5] * len(predictions)  # Return neutral scores

        if not predictions:
            self.logger.debug("No predictions to score")
            return []

        # Get classifier metadata
        layer = self._session_classifier_metadata.get("layer", 12)
        aggregation = self._session_classifier_metadata.get("aggregation", "mean_pooling")

        self.logger.info(
            f"Scoring {len(predictions)} {description} with cached classifier (layer={layer}, aggregation={aggregation})"
        )

        confidence_scores = []

        # Process predictions in batches for efficiency
        batch_size = 8  # Smaller batch size to avoid OOM
        for i in range(0, len(predictions), batch_size):
            batch_predictions = predictions[i : i + batch_size]
            batch_activations = []

            # Extract activations for each prediction in the batch
            for pred_text in batch_predictions:
                try:
                    # Extract activation for this prediction text
                    activation = self._extract_activation_for_text(
                        text=pred_text,
                        layer_index=layer,
                        aggregation_strategy=aggregation,
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        max_length=max_length,
                    )
                    batch_activations.append(activation)

                except Exception as e:
                    self.logger.debug(f"Failed to extract activation for prediction: {e}")
                    # Use neutral score for failed extractions
                    confidence_scores.append(0.5)
                    continue

            if batch_activations:
                try:
                    # Stack activations into batch tensor
                    batch_tensor = torch.stack(batch_activations)

                    # Convert to numpy for sklearn classifier (float32 required, bfloat16 not supported)
                    batch_numpy = batch_tensor.detach().cpu().float().numpy()

                    # Get prediction probabilities from classifier
                    probabilities = self._session_classifier.predict_proba(batch_numpy)

                    # Extract confidence scores (probability for positive class)
                    # Assuming binary classification with class 1 as positive
                    if probabilities.shape[1] > 1:
                        batch_scores = probabilities[:, 1].tolist()  # Probability of positive class
                    else:
                        batch_scores = probabilities[:, 0].tolist()  # Single class probability

                    confidence_scores.extend(batch_scores)

                except Exception as e:
                    self.logger.warning(f"Failed to score batch of activations: {e}")
                    # Add neutral scores for failed batch
                    confidence_scores.extend([0.5] * len(batch_activations))

        # Ensure we have scores for all predictions
        while len(confidence_scores) < len(predictions):
            confidence_scores.append(0.5)  # Pad with neutral scores if needed

        # Truncate if we have too many scores (shouldn't happen)
        confidence_scores = confidence_scores[: len(predictions)]

        # Log statistics
        avg_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        self.logger.debug(
            f"Generated {len(confidence_scores)} classifier confidence scores for {description} (avg={avg_score:.3f})"
        )

        return confidence_scores
