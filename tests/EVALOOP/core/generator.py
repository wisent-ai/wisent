"""Steering vector generation and response generation."""
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from wisent_guard.cli.data_loaders.data_loader_rotator import DataLoaderRotator
from wisent_guard.cli.steering_methods.steering_rotator import SteeringMethodRotator
from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.core.trainers.steering_trainer import WisentSteeringTrainer
from tests.EVALOOP.core.models import GenerationResult
from tests.EVALOOP.core.config import TraitConfig, EvaluationConfig


class SteeringVectorGenerator:
    """Generates steering vectors from contrastive pairs."""

    def __init__(self, model: WisentModel, method_name: str = "caa"):
        """
        Initialize generator.

        Args:
            model: WisentModel instance
            method_name: Name of steering method (default: "caa")
        """
        self.model = model
        self.method_name = method_name

        # Initialize rotators
        self.data_rotator = DataLoaderRotator()
        self.data_rotator.use("custom")

        self.steer_rotator = SteeringMethodRotator()
        self.steer_rotator.use(method_name)
        self.method = self.steer_rotator._method

    def train_vector(
        self,
        pairs_path: Path,
        layer: str,
        aggregation: str,
        save_dir: Path,
        normalize_layers: bool = True
    ):
        """
        Train steering vector for a specific configuration.

        Args:
            pairs_path: Path to contrastive pairs JSON
            layer: Layer to train on
            aggregation: Aggregation strategy
            save_dir: Directory to save the vector
            normalize_layers: Whether to normalize layer activations

        Returns:
            TrainingResult from the trainer
        """
        # Load contrastive pairs
        data = self.data_rotator.load(path=str(pairs_path))
        training_data = data['train_qa_pairs']

        # Create trainer
        trainer = WisentSteeringTrainer(
            model=self.model,
            pair_set=training_data,
            steering_method=self.method
        )

        # Train
        print(f"Training: Layer={layer}, Aggregation={aggregation}")
        result = trainer.run(
            layers_spec=layer,
            aggregation=aggregation,
            return_full_sequence=False,
            normalize_layers=normalize_layers,
            save_dir=save_dir
        )

        print(f"Saved to: {save_dir}")
        return result

class ResponseGenerator:
    """Generates baseline and steered responses."""

    def __init__(self, model: WisentModel, max_new_tokens: int = 400):
        """
        Initialize response generator.

        Args:
            model: WisentModel instance
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.max_new_tokens = max_new_tokens
        # Track current steering configuration
        self._current_layer = None
        self._current_strength = None
        self._current_aggregation = None

    @staticmethod
    def crop_to_answer(response_text: str) -> str:
        """
        Crops the model response to remove the system prompt and user question,
        returning only the assistant's actual answer.

        The format is:
        system\n\nCutting Knowledge Date: ...\n\nuser\n\n<question>assistant\n\n<answer>

        We want to extract only <answer>.

        Args:
            response_text: The full response text including system prompt

        Returns:
            str: Only the assistant's answer, or original text if pattern not found
        """
        # Look for "assistant\n\n" which marks the start of the actual answer
        marker = "assistant\n\n"

        if marker in response_text:
            # Find the position after "assistant\n\n"
            start_pos = response_text.find(marker) + len(marker)
            return response_text[start_pos:]

        # Fallback: if the marker isn't found, return original text
        return response_text

    def set_steering_configuration(
        self,
        steering_vectors: Dict[str, Any],
        strength: float,
        layer: str,
        aggregation: str
    ):
        """
        Configure the model with steering vectors and tracking metadata.

        Args:
            steering_vectors: Dictionary of steering vectors
            strength: Steering strength to apply
            layer: Layer identifier (metadata)
            aggregation: Aggregation method used (metadata)
        """
        # Apply steering to the model
        self.model.set_steering_from_raw(steering_vectors, scale=strength, normalize=True)

        # Track configuration for metadata
        self._current_layer = layer
        self._current_strength = strength
        self._current_aggregation = aggregation

    def generate_response(self, question: str) -> GenerationResult:
        """
        Generate baseline and steered response for a single question.

        Requires that set_steering_configuration() has been called first.

        Args:
            question: Question string

        Returns:
            GenerationResult object with both baseline and steered responses

        Raises:
            RuntimeError: If steering configuration has not been set
        """
        if self._current_layer is None:
            raise RuntimeError(
                "Steering configuration not set. Call set_steering_configuration() first."
            )

        print(f"  Testing: {question[:50]}...")

        # Generate steered first (while steering is active)
        messages_steered = [[{"role": "user", "content": question}]]
        steered = self.model.generate(
            messages_steered,
            max_new_tokens=self.max_new_tokens,
            use_steering=True
        )[0]

        # Generate baseline (unsteered) - leaves model unhooked after function terminates
        with self.model.detached():
            messages_unsteered = [[{"role": "user", "content": question}]]
            baseline = self.model.generate(
                messages_unsteered,
                max_new_tokens=self.max_new_tokens,
                use_steering=False
            )[0]

        # Crop responses
        baseline_cropped = self.crop_to_answer(baseline)
        steered_cropped = self.crop_to_answer(steered)

        return GenerationResult(
            layer=self._current_layer,
            strength=self._current_strength,
            aggregation_method=self._current_aggregation,
            question=question,
            baseline_response=baseline_cropped,
            steered_response=steered_cropped
        )

    def generate_responses(
        self,
        questions: List[str],
        steering_vectors: Dict[str, Any],
        strength: float,
        layer: str,
        aggregation: str
    ) -> List[GenerationResult]:
        """
        Generate baseline and steered responses for questions.

        Args:
            questions: List of question strings
            steering_vectors: Dictionary of steering vectors
            strength: Steering strength to apply
            layer: Layer identifier
            aggregation: Aggregation method used

        Returns:
            List of GenerationResult objects
        """
        # Set steering configuration
        self.set_steering_configuration(
            steering_vectors=steering_vectors,
            strength=strength,
            layer=layer,
            aggregation=aggregation
        )

        results = []
        for question in questions:
            result = self.generate_response(question)
            results.append(result)

        return results


class GenerationPipeline:
    """Orchestrates the full generation pipeline."""

    def __init__(self, eval_config: EvaluationConfig, config_manager=None):
        """
        Initialize generation pipeline.

        Args:
            eval_config: Evaluation configuration
            config_manager: Optional ConfigManager for path management
        """
        self.config = eval_config
        self.config_manager = config_manager
        self.model = WisentModel(
            model_name=eval_config.model_name,
            layers={},
            device=eval_config.device
        )
        self.vector_generator = SteeringVectorGenerator(self.model)
        self.response_generator = ResponseGenerator(
            self.model,
            max_new_tokens=eval_config.max_new_tokens
        )

    def run_single_configuration(
        self,
        questions: List[str],
        steering_vectors: Dict[str, Any],
        layer: str,
        aggregation: str,
        strength: float
    ) -> List[GenerationResult]:
        """
        Run generation for a single (layer, aggregation, strength) configuration.

        Args:
            questions: List of test questions
            steering_vectors: Pre-trained steering vectors
            layer: Layer identifier
            aggregation: Aggregation strategy
            strength: Steering strength to apply

        Returns:
            List of GenerationResult objects for this configuration
        """
        print(f"\nInference: Layer={layer}, Strength={strength}, Aggregation={aggregation}")

        results = self.response_generator.generate_responses(
            questions=questions,
            steering_vectors=steering_vectors,
            strength=strength,
            layer=layer,
            aggregation=aggregation
        )

        print(f"  Completed {len(questions)} prompts")
        return results

    def run_trait(self, trait_config: TraitConfig) -> List[GenerationResult]:
        """
        Run generation pipeline for a single trait across all configurations.

        Args:
            trait_config: Configuration for the trait

        Returns:
            List of all GenerationResult objects
        """
        print(f"\n{'='*80}")
        print(f"GENERATION PIPELINE: {trait_config.name.upper()}")
        print(f"{'='*80}\n")

        # Load test questions
        all_questions = []
        for path in trait_config.test_questions_paths:
            with open(path) as f:
                questions = [line.strip() for line in f if line.strip()]
                all_questions.extend(questions[:self.config.num_questions])

        all_results = []

        # Iterate through configurations
        for layer in trait_config.layers:
            for aggregation in trait_config.aggregations:
                # Train steering vector once per (layer, aggregation)
                # Use ConfigManager for path if available, otherwise fallback to hardcoded path
                if self.config_manager:
                    save_dir = self.config_manager.get_vector_path(trait_config.name, str(layer), aggregation)
                else:
                    save_dir = Path(f"./tests/EVALOOP/output/{trait_config.name}_vectors/steering_output_layer{layer}_aggregation{aggregation}")

                training_result = self.vector_generator.train_vector(
                    pairs_path=trait_config.contrastive_pairs_path,
                    layer=layer,
                    aggregation=aggregation,
                    save_dir=save_dir,
                    normalize_layers=False
                )

                steering_vectors = training_result.steered_vectors.to_dict()

                # Test with different strengths using the same steering vector
                for strength in trait_config.strengths:
                    results = self.run_single_configuration(
                        questions=all_questions,
                        steering_vectors=steering_vectors,
                        layer=layer,
                        aggregation=aggregation,
                        strength=strength
                    )
                    all_results.extend(results)

        return all_results
