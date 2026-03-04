"""Steering parameter determination and storage mixin for AutonomousAgent."""

from typing import List
from dataclasses import dataclass

from wisent.core.utils.infra_tools.errors import MissingParameterError
from wisent.core.primitives.models import get_generate_kwargs


@dataclass
class SteeringMixinConfig:
    """Required configuration for steering parameter mixin."""
    initial_strength: float
    increment: float
    max_strength: float
    initial_min: float
    initial_max: float
    increment_min: float
    increment_max: float
    max_min: float
    max_max: float
    quality_storage_threshold: float


class SteeringParamsMixin:
    """Mixin providing steering/classifier parameter determination and storage."""

    async def _determine_steering_parameters(
        self, prompt: str, current_quality: float, attempt_number: int
    ) -> "SteeringParams":
        """Use model to determine optimal steering parameters."""
        steering_prompt = f"""
        Determine optimal steering parameters for improving this response:

        Original Prompt: "{prompt}"
        Current Quality Score: {current_quality:.3f} (0=good, 1=bad)
        Attempt Number: {attempt_number}

        Steering Method: CAA (Contrastive Activation Addition)
        - Gentle activation steering, good for general improvements

        Consider:
        - How much improvement is needed? (quality gap: {1.0 - current_quality:.2f})
        - What type of improvement? (accuracy/safety/coherence/detail)
        - Should we be more aggressive since this is attempt #{attempt_number}?
        - Prompt characteristics (technical/casual/creative/safety-sensitive)

        Determine parameters:
        1. Initial Strength (0.1-2.0): How aggressive to start?
        2. Increment (0.1-0.5): How much to increase if this fails?
        3. Maximum Strength (0.5-3.0): Upper limit to prevent over-steering?

        Format response as:
        METHOD: CAA
        INITIAL: [number]
        INCREMENT: [number]
        MAXIMUM: [number]
        REASONING: [one sentence explanation]
        """

        gen_kwargs = get_generate_kwargs()
        result = self.model.generate(steering_prompt, layer_index=self.params.layer, **gen_kwargs)
        response = result[0] if isinstance(result, tuple) else result
        return self._parse_steering_params(response)

    def _parse_steering_params(self, response: str) -> "SteeringParams":
        """Parse model response to extract steering parameters."""
        from ..agent.diagnose.agent_classifier_decision import SteeringParams
        sc = self._steering_mixin_config
        method = "CAA"
        initial = sc.initial_strength
        increment = sc.increment
        maximum = sc.max_strength
        reasoning = "Using configured parameters due to parsing failure"
        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("METHOD:"):
                    method = line.split(":")[1].strip()
                elif line.startswith("INITIAL:"):
                    initial = float(line.split(":")[1].strip())
                elif line.startswith("INCREMENT:"):
                    increment = float(line.split(":")[1].strip())
                elif line.startswith("MAXIMUM:"):
                    maximum = float(line.split(":")[1].strip())
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
        except Exception as e:
            print(f"   Warning: Failed to parse steering parameters: {e}")
            print(f"   Using configured: method={method}, initial={initial}, increment={increment}, max={maximum}")
        if method not in ["CAA"]:
            method = "CAA"
        initial = max(sc.initial_min, min(sc.initial_max, initial))
        increment = max(sc.increment_min, min(sc.increment_max, increment))
        maximum = max(sc.max_min, min(sc.max_max, maximum))
        return SteeringParams(
            steering_method=method, initial_strength=initial, increment=increment,
            maximum_strength=maximum, method_specific_params={}, reasoning=reasoning)

    async def _get_or_determine_classifier_parameters(self, prompt: str, benchmark_names: List[str]) -> "ClassifierParams":
        """Get classifier parameters from memory or determine them fresh."""
        stored_params = self._get_stored_classifier_parameters(prompt)
        if stored_params:
            stored_params.reasoning = f"Retrieved from parameter memory: {stored_params.reasoning}"
            print(f"   Using stored parameters (success rate: {getattr(stored_params, 'success_rate', 0.0):.2%})")
            return stored_params
        print("   No stored parameters found, using model determination...")
        fresh_params = await self._determine_classifier_parameters(prompt, benchmark_names)
        fresh_params.reasoning = f"Model-determined: {fresh_params.reasoning}"
        return fresh_params

    async def _get_or_determine_steering_parameters(self, prompt: str, current_quality: float, attempt_number: int) -> "SteeringParams":
        """Get steering parameters from memory or determine them fresh."""
        stored_params = self._get_stored_steering_parameters(prompt, attempt_number)
        if stored_params:
            adjusted_strength = stored_params.initial_strength + (stored_params.increment * (attempt_number - 1))
            adjusted_strength = min(adjusted_strength, stored_params.maximum_strength)
            stored_params.initial_strength = adjusted_strength
            stored_params.reasoning = f"Retrieved from memory (adjusted): {stored_params.reasoning}"
            print(f"   Using stored steering parameters (success rate: {getattr(stored_params, 'success_rate', 0.0):.2%})")
            return stored_params
        print("   No stored steering parameters found, using model determination...")
        fresh_params = await self._determine_steering_parameters(prompt, current_quality, attempt_number)
        fresh_params.reasoning = f"Model-determined: {fresh_params.reasoning}"
        return fresh_params

    def _get_stored_classifier_parameters(self, prompt: str) -> "ClassifierParams":
        """Retrieve classifier parameters from the parameter file."""
        from ..agent.diagnose.agent_classifier_decision import ClassifierParams
        try:
            classifier_config = self.params._params.get("classifier", {})
            if not classifier_config:
                return None
            required_keys = ["layer", "threshold", "samples", "type", "num_epochs", "batch_size", "learning_rate", "early_stopping_patience", "hidden_dim"]
            missing = [k for k in required_keys if k not in classifier_config]
            if missing:
                raise MissingParameterError(params=missing, context="Classifier configuration must specify all required keys.")
            params = ClassifierParams(
                optimal_layer=classifier_config["layer"], classification_threshold=classifier_config["threshold"],
                training_samples=classifier_config["samples"], classifier_type=classifier_config["type"],
                num_epochs=classifier_config["num_epochs"], batch_size=classifier_config["batch_size"],
                learning_rate=classifier_config["learning_rate"], early_stopping_patience=classifier_config["early_stopping_patience"],
                hidden_dim=classifier_config["hidden_dim"], reasoning="Using parameters from configuration file",
                model_name=self.model_name, aggregation_method=classifier_config.get("aggregation_method"),
                token_aggregation=classifier_config.get("token_aggregation"))
            return params
        except Exception as e:
            print(f"   Warning: Failed to retrieve stored parameters: {e}")
            return None

    def _get_stored_steering_parameters(self, prompt: str, attempt_number: int) -> "SteeringParams":
        """Return None to always use model determination for steering parameters."""
        return None

    def _classify_prompt_type(self, prompt: str) -> str:
        """Classify the prompt into a known type for parameter retrieval."""
        try:
            quality_config = self.params.config.get("quality_control", {})
            prompt_classification = quality_config.get("prompt_classification", {})
            prompt_lower = prompt.lower()
            scores = {}
            for prompt_type, keywords in prompt_classification.items():
                score = sum(1 for keyword in keywords if keyword.lower() in prompt_lower)
                if score > 0:
                    scores[prompt_type] = score
            if scores:
                best_type = max(scores.keys(), key=lambda x: scores[x])
                print(f"   Classified as '{best_type}' (score: {scores[best_type]})")
                return best_type
            return None
        except Exception as e:
            print(f"   Warning: Failed to classify prompt type: {e}")
            return None

    def _store_successful_parameters(self, prompt: str, classifier_params: "ClassifierParams", steering_params: "SteeringParams", final_quality: float):
        """Store successful parameter combinations for future use."""
        try:
            if final_quality < self._steering_mixin_config.quality_storage_threshold:
                return
            prompt_type = self._classify_prompt_type(prompt)
            if not prompt_type:
                print("   Could not classify prompt for storage")
                return
            print(f"   Storing successful parameters for '{prompt_type}' (quality: {final_quality:.3f})")
            print("   Would update parameter file with successful combination")
        except Exception as e:
            print(f"   Warning: Failed to store parameters: {e}")
