"""
Response diagnostics for autonomous agent analysis.

This module handles:
- Activation-based response quality assessment using trained classifiers
- Issue detection through model activations
- Classifier-based quality scoring
- Decision making for improvements needed
"""

from dataclasses import dataclass
from typing import Any, Dict, List

from wisent.core.activations.core.atoms import ActivationAggregationStrategy
from wisent.core.activations.activations import Activations
from wisent.core.classifier.classifier import Classifier
from wisent.core.layer import Layer
from wisent.core.model import Model


@dataclass
class AnalysisResult:
    """Result of self-analysis."""

    has_issues: bool
    issues_found: List[str]
    confidence: float
    suggestions: List[str]
    quality_score: float


class ResponseDiagnostics:
    """Handles activation-based response analysis and quality assessment for autonomous agents."""

    def __init__(self, model: Model, classifier_configs: List[Dict[str, Any]]):
        """
        Initialize the diagnostics system.

        Args:
            model: The language model to extract activations from
            classifier_configs: List of classifier configurations with paths and layers
                Example: [
                    {"path": "./models/hallucination_classifier.pt", "layer": 15, "issue_type": "hallucination"},
                    {"path": "./models/quality_classifier.pt", "layer": 20, "issue_type": "quality"}
                ]
        """
        if not classifier_configs:
            raise ValueError("classifier_configs is required - no fallback mode available")

        self.model = model

        # Load classifiers
        self.classifiers = []
        for config in classifier_configs:
            try:
                classifier = Classifier()
                classifier.load_model(config["path"])

                self.classifiers.append(
                    {
                        "classifier": classifier,
                        "layer": Layer(index=config["layer"], type="transformer"),
                        "issue_type": config.get("issue_type", "unknown"),
                        "threshold": config.get("threshold", 0.5),
                    }
                )
                print(f"✅ Loaded classifier for {config['issue_type']} at layer {config['layer']}")

            except Exception as e:
                print(f"⚠️ Failed to load classifier {config['path']}: {e}")
                print("   Skipping this classifier and continuing...")
                continue

        if not self.classifiers:
            raise RuntimeError("Failed to load any classifiers - system cannot operate without them")

    async def analyze_response(self, response: str, prompt: str) -> AnalysisResult:
        """Analyze the response using trained classifiers and activation patterns."""
        issues = []
        confidence_scores = []

        # Classifier-based analysis only
        classifier_results = self._analyze_with_classifiers(response)

        for result in classifier_results:
            if result["has_issue"]:
                issues.append(result["issue_type"])
                confidence_scores.append(result["confidence"])

        # Quality assessment using classifiers
        quality_score = self._assess_quality_with_classifiers(response)

        # Overall confidence calculation
        if not confidence_scores:
            # No issues detected, use average confidence from all classifiers
            confidence = (
                sum(result["confidence"] for result in classifier_results) / len(classifier_results)
                if classifier_results
                else 0.5
            )
        else:
            confidence = sum(confidence_scores) / len(confidence_scores)

        # Generate suggestions based on detected issues
        suggestions = self._generate_suggestions(issues)

        result = AnalysisResult(
            has_issues=len(issues) > 0,
            issues_found=issues,
            confidence=confidence,
            suggestions=suggestions,
            quality_score=quality_score,
        )

        return result

    def _analyze_with_classifiers(self, response: str) -> List[Dict[str, Any]]:
        """Analyze response using trained classifiers."""
        results = []

        for classifier_config in self.classifiers:
            # Extract activations for this classifier's layer
            activations_tensor = self.model.extract_activations(response, classifier_config["layer"])

            # Create Activations object
            activations = Activations(
                tensor=activations_tensor,
                layer=classifier_config["layer"],
                aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
            )

            # Get features for classifier
            features = activations.extract_features_for_classifier()

            # Get classifier prediction (ensure CPU tensor for numpy conversion)
            classifier = classifier_config["classifier"]
            features_cpu = features.cpu() if hasattr(features, "cpu") else features

            # Handle different classifier output formats
            pred_result = classifier.predict([features_cpu.numpy()])
            prediction = pred_result[0] if isinstance(pred_result, (list, tuple)) else pred_result

            prob_result = classifier.predict_proba([features_cpu.numpy()])
            probability = prob_result[0] if isinstance(prob_result, (list, tuple)) else prob_result

            # Determine if this indicates an issue
            threshold = classifier_config["threshold"]
            has_issue = float(probability) > threshold
            confidence = abs(float(probability) - 0.5) * 2  # Convert to 0-1 confidence

            results.append(
                {
                    "issue_type": classifier_config["issue_type"],
                    "has_issue": has_issue,
                    "confidence": confidence,
                    "probability": float(probability),
                    "prediction": int(prediction),
                }
            )

        return results

    def _assess_quality_with_classifiers(self, response: str) -> float:
        """Assess response quality using classifiers."""
        quality_scores = []

        for classifier_config in self.classifiers:
            # Extract activations
            activations_tensor = self.model.extract_activations(response, classifier_config["layer"])

            # Create Activations object
            activations = Activations(
                tensor=activations_tensor,
                layer=classifier_config["layer"],
                aggregation_strategy=ActivationAggregationStrategy.LAST_TOKEN,
            )

            # Get features
            features = activations.extract_features_for_classifier()

            # Get classifier probability (ensure CPU tensor for numpy conversion)
            classifier = classifier_config["classifier"]
            features_cpu = features.cpu() if hasattr(features, "cpu") else features

            # Handle different classifier output formats
            prob_result = classifier.predict_proba([features_cpu.numpy()])
            probability = prob_result[0] if isinstance(prob_result, (list, tuple)) else prob_result

            # Convert probability to quality score
            # For most classifiers, low probability (closer to 0) = higher quality
            # This assumes classifiers are trained to detect problems (1 = problematic, 0 = good)
            quality_score = 1.0 - float(probability)
            quality_scores.append(quality_score)

        if not quality_scores:
            raise RuntimeError("No quality scores available - all classifiers failed")

        # Use average quality across all classifiers
        return sum(quality_scores) / len(quality_scores)

    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on detected issues."""
        suggestions = []

        # Map issue types to suggestions
        suggestion_map = {
            "hallucination": "Verify factual accuracy and provide evidence-based responses",
            "quality": "Improve response relevance, completeness, and clarity",
            "harmful": "Revise content to be safe and helpful",
            "bias": "Use more balanced and inclusive language",
            "gibberish": "Ensure response coherence and proper language structure",
            "repetitive": "Reduce repetition and vary language patterns",
            "incoherent": "Improve logical flow and sentence structure",
        }

        for issue in issues:
            if issue in suggestion_map:
                suggestions.append(suggestion_map[issue])
            else:
                suggestions.append(f"Address {issue} issue in the response")

        return suggestions

    def decide_if_improvement_needed(
        self, analysis: AnalysisResult, quality_threshold: float = 0.7, confidence_threshold: float = 0.8
    ) -> bool:
        """
        Decide if improvement is needed based on analysis results.

        Args:
            analysis: Analysis result to evaluate
            quality_threshold: Minimum quality score required
            confidence_threshold: Minimum confidence required for improvement decision

        Returns:
            True if improvement is needed, False otherwise
        """
        # High confidence decisions
        if analysis.confidence >= confidence_threshold:
            if analysis.has_issues:
                return True
            if analysis.quality_score < quality_threshold:
                return True

        # Low confidence - be conservative
        return False

    def add_classifier(self, classifier_path: str, layer_index: int, issue_type: str, threshold: float = 0.5):
        """Add a new classifier to the diagnostic system."""
        classifier = Classifier()
        classifier.load_model(classifier_path)

        self.classifiers.append(
            {
                "classifier": classifier,
                "layer": Layer(index=layer_index, type="transformer"),
                "issue_type": issue_type,
                "threshold": threshold,
            }
        )
        print(f"✅ Added classifier for {issue_type} at layer {layer_index}")

    def get_available_classifiers(self) -> List[Dict[str, Any]]:
        """Get information about loaded classifiers."""
        return [
            {"issue_type": config["issue_type"], "layer": config["layer"].index, "threshold": config["threshold"]}
            for config in self.classifiers
        ]
