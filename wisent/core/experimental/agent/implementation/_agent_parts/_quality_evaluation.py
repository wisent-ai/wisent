"""
Quality evaluation mixin for AutonomousAgent.

Extracted from autonomous_agent.py to comply with the 300-line file limit.
Contains methods for evaluating response quality using classifiers and
model-based threshold determination, plus classifier parameter determination.
"""

from typing import List

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.primitives.model_interface.core.activations.activations import Activations
from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.infra_tools.errors import MissingParameterError


class QualityEvaluationMixin:
    """Mixin providing quality evaluation and classifier parameter methods."""

    async def evaluate_response_quality(
        self, response: str, classifier, classifier_params: "ClassifierParams"
    ) -> "QualityResult":
        """
        Evaluate response quality using classifier + model judgment for threshold determination.

        Args:
            response: The response to evaluate
            classifier: The trained classifier to use
            classifier_params: Parameters used for classifier training

        Returns:
            QualityResult with score and acceptability judgment
        """
        from ..agent.diagnose.agent_classifier_decision import QualityResult

        # Extract activations and get classifier score
        activations_tensor = self.model.extract_activations(response, classifier_params.optimal_layer)
        from ..layer import Layer

        activations = Activations(
            tensor=activations_tensor,
            layer=Layer(index=classifier_params.optimal_layer, type="transformer"),
            aggregation_strategy=ExtractionStrategy.default(),
        )

        features = activations.extract_features_for_classifier()
        features_cpu = features.cpu() if hasattr(features, "cpu") else features

        # Get classifier prediction
        prob_result = classifier.predict_proba([features_cpu.numpy()])
        score = prob_result[0] if isinstance(prob_result, (list, tuple)) else prob_result

        # Use model to determine if this score represents acceptable quality
        acceptable = await self._model_determines_quality_threshold(response, float(score))

        return QualityResult(
            score=float(score),
            acceptable=acceptable,
            reasoning=f"Model judged score {score:.3f} as {'acceptable' if acceptable else 'unacceptable'}",
        )

    async def _model_determines_quality_threshold(self, response: str, classifier_score: float) -> bool:
        """
        Use model to determine if classifier score represents acceptable quality.

        Args:
            response: The response to evaluate
            classifier_score: Score from the classifier (0-1, where 1 typically means more problematic)

        Returns:
            Boolean indicating if quality is acceptable
        """
        threshold_prompt = f"""
        Evaluate if this response quality is acceptable for a user:

        Response: "{response}"

        Quality Classifier Score: {classifier_score:.3f}
        (Note: Score ranges 0-1, where 0 = high quality, 1 = low quality/problematic)

        Consider:
        - Is the response helpful and informative?
        - Are there factual errors or concerning content?
        - Is the response appropriate length and detail?
        - Would a typical user be satisfied with this response?

        Respond ONLY with:
        "ACCEPTABLE" if the response quality is good enough for the user
        "UNACCEPTABLE" if the response needs improvement
        """

        # Generate model judgment (short response for yes/no decision)
        gen_kwargs = get_generate_kwargs()
        result = self.model.generate(threshold_prompt, layer_index=self.params.layer, **gen_kwargs)
        judgment = result[0] if isinstance(result, tuple) else result
        judgment = judgment.strip().upper()

        return "ACCEPTABLE" in judgment

    async def _determine_classifier_parameters(
        self,
        prompt: str,
        benchmark_names: List[str],
        quality_eval_layer_min: int,
        quality_eval_layer_max: int,
        quality_eval_samples_min: int,
        quality_eval_samples_max: int,
    ) -> "ClassifierParams":
        """
        Use model to determine optimal classifier parameters based on prompt analysis.

        Args:
            prompt: The user prompt to analyze
            benchmark_names: Selected benchmarks for training

        Returns:
            ClassifierParams with model-determined parameters
        """

        parameter_prompt = f"""
        Analyze this prompt and determine optimal classifier parameters:

        Prompt: "{prompt}"
        Selected Benchmarks: {benchmark_names}

        Consider:
        - Prompt complexity (simple conversational vs complex technical)
        - Domain type (technical/casual/creative/factual)
        - Expected response length and detail needs
        - Quality requirements and safety considerations

        Determine optimal parameters:
        1. Optimal Layer (8-20): What layer captures the right semantic complexity?
           - Simple prompts: layers 8-12
           - Medium complexity: layers 12-16
           - Complex technical: layers 16-20

        2. Classification Threshold (0.1-0.9): How strict should quality detection be?
           - Lenient (casual conversation): 0.1-0.3
           - Moderate (general use): 0.4-0.6
           - Strict (important/technical): 0.7-0.9

        3. Training Samples (10-50): How many samples needed for good training?
           - Simple patterns: 10-20 samples
           - Medium complexity: 20-35 samples
           - Complex patterns: 35-50 samples

        4. Classifier Type: What classifier works best for this data?
           - logistic: Simple patterns, fast training
           - svm: Medium complexity, robust
           - neural: Complex patterns, more data needed

        Format your response as:
        LAYER: [number]
        THRESHOLD: [number]
        SAMPLES: [number]
        TYPE: [logistic/svm/neural]
        REASONING: [one sentence explanation]
        """

        # Generate model response
        gen_kwargs = get_generate_kwargs()
        result = self.model.generate(parameter_prompt, layer_index=self.params.layer, **gen_kwargs)
        response = result[0] if isinstance(result, tuple) else result

        # Parse the response
        return self._parse_classifier_params(
            response,
            quality_eval_layer_min=quality_eval_layer_min,
            quality_eval_layer_max=quality_eval_layer_max,
            quality_eval_samples_min=quality_eval_samples_min,
            quality_eval_samples_max=quality_eval_samples_max,
        )

    def _parse_classifier_params(
        self,
        response: str,
        quality_eval_layer_min: int,
        quality_eval_layer_max: int,
        quality_eval_samples_min: int,
        quality_eval_samples_max: int,
        default_samples: int = None,
    ) -> "ClassifierParams":
        """Parse model response to extract classifier parameters."""
        from ..agent.diagnose.agent_classifier_decision import ClassifierParams
        if default_samples is None:
            raise MissingParameterError(params=["default_samples"], context="_parse_classifier_params")
        layer = self.params.layer
        threshold = None
        samples = default_samples
        classifier_type = None
        reasoning = ""

        try:
            lines = response.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line.startswith("LAYER:"):
                    layer = int(line.split(":")[1].strip())
                elif line.startswith("THRESHOLD:"):
                    threshold = float(line.split(":")[1].strip())
                elif line.startswith("SAMPLES:"):
                    samples = int(line.split(":")[1].strip())
                elif line.startswith("TYPE:"):
                    classifier_type = line.split(":")[1].strip().lower()
                elif line.startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
        except Exception as e:
            raise MissingParameterError(
                params=["LAYER", "THRESHOLD", "SAMPLES", "TYPE"],
                context=f"Failed to parse classifier parameters from model response: {e}",
            )

        if threshold is None:
            raise MissingParameterError(params=["threshold"], context="Model response must specify THRESHOLD")
        if classifier_type is None:
            raise MissingParameterError(params=["classifier_type"], context="Model response must specify TYPE")

        # Validate ranges
        layer = max(quality_eval_layer_min, min(quality_eval_layer_max, layer))
        threshold = max(0.1, min(0.9, threshold))
        samples = max(quality_eval_samples_min, min(quality_eval_samples_max, samples))
        valid_types = ["logistic", "svm", "neural"]
        if classifier_type not in valid_types:
            raise MissingParameterError(
                params=["classifier_type"],
                context=f"Model must specify TYPE as one of {valid_types}, got: {classifier_type}",
            )

        # Classifier config params must come from config, not silent defaults
        classifier_config = self.params._params.get("classifier", {})
        if "num_epochs" not in classifier_config:
            raise MissingParameterError(params=["num_epochs"], context="classifier config must specify num_epochs")
        if "batch_size" not in classifier_config:
            raise MissingParameterError(params=["batch_size"], context="classifier config must specify batch_size")
        if "learning_rate" not in classifier_config:
            raise MissingParameterError(params=["learning_rate"], context="classifier config must specify learning_rate")
        if "early_stopping_patience" not in classifier_config:
            raise MissingParameterError(params=["early_stopping_patience"], context="classifier config must specify early_stopping_patience")
        if "hidden_dim" not in classifier_config:
            raise MissingParameterError(params=["hidden_dim"], context="classifier config must specify hidden_dim")

        return ClassifierParams(
            optimal_layer=layer,
            classification_threshold=threshold,
            training_samples=samples,
            classifier_type=classifier_type,
            num_epochs=classifier_config["num_epochs"],
            batch_size=classifier_config["batch_size"],
            learning_rate=classifier_config["learning_rate"],
            early_stopping_patience=classifier_config["early_stopping_patience"],
            hidden_dim=classifier_config["hidden_dim"],
            reasoning=reasoning,
            model_name=self.model_name,
        )
