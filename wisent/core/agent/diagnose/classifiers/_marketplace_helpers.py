"""Estimation, filtering, and on-demand creation helpers for ClassifierMarketplace."""

from typing import List, Dict, Any
from dataclasses import dataclass
import os
import re
import time
from datetime import datetime

from wisent.core.utils import resolve_default_device


@dataclass
class ClassifierCreationEstimate:
    """Estimate for creating a new classifier."""
    issue_type: str
    estimated_training_time_minutes: float
    estimated_quality_score: float
    training_samples_needed: int
    optimal_layer: int
    confidence: float


class MarketplaceEstimationMixin:
    """Mixin providing estimation, filtering, and on-demand creation for ClassifierMarketplace."""

    def get_creation_estimate(self, issue_type: str) -> ClassifierCreationEstimate:
        """
        Get an estimate for creating a new classifier for the given issue type.

        Args:
            issue_type: The type of issue to create a classifier for

        Returns:
            Estimate including time, quality, and confidence
        """
        available_benchmarks = self._find_available_benchmarks_for_issue(issue_type)

        if available_benchmarks:
            benchmark_count = len(available_benchmarks)
            base = {
                "training_time_minutes": 8.0 + (benchmark_count * 2.0),
                "quality_score": min(0.80, 0.60 + (benchmark_count * 0.05)),
                "samples_needed": min(500, 100 + (benchmark_count * 30)),
                "optimal_layer": self._estimate_optimal_layer_for_issue(issue_type)
            }
            print(f"   Using {benchmark_count} benchmarks for {issue_type}")
        else:
            base = {
                "training_time_minutes": 6.0,
                "quality_score": 0.55,
                "samples_needed": 50,
                "optimal_layer": 14
            }
            print(f"   Using synthetic generation for {issue_type}")

        return self._complete_creation_estimate(base, available_benchmarks, issue_type)

    def _find_available_benchmarks_for_issue(self, issue_type: str) -> List[str]:
        """Find available benchmarks using dynamic semantic analysis."""
        available_tasks = self.model.get_available_tasks()

        relevant = []
        issue_lower = issue_type.lower()

        for task in available_tasks[:1000]:
            task_lower = task.lower()
            similarity_score = self._calculate_task_similarity(issue_lower, task_lower)

            if similarity_score > 0:
                relevant.append((task, similarity_score))
                if len(relevant) >= 30:
                    break

        relevant.sort(key=lambda x: x[1], reverse=True)
        return [task for task, score in relevant[:15]]

    def _calculate_task_similarity(self, issue_type: str, task_name: str) -> float:
        """Calculate similarity between issue type and task name using model decisions."""
        prompt = f"""Rate the similarity between this issue type and evaluation task for training AI safety classifiers.

Issue Type: {issue_type}
Task: {task_name}

Rate similarity from 0.0 to 10.0 (10.0 = highly similar, 0.0 = not similar).
Respond with only the number:"""

        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            score_str = response.strip()

            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))
            return 0.0
        except:
            return 0.0

    def _estimate_optimal_layer_for_issue(self, issue_type: str) -> int:
        """Estimate optimal layer using model analysis of issue complexity."""
        prompt = f"""What transformer layer would be optimal for detecting this AI safety issue?

Issue Type: {issue_type}

Consider:
- Simple issues (formatting, basic patterns) -> early layers (8-12)
- Complex semantic issues (truthfulness, bias) -> middle layers (12-16)
- Abstract conceptual issues (coherence, quality) -> deeper layers (16-20)

Respond with just the layer number (8-20):"""

        try:
            response = self.model.generate(prompt, layer_index=15, max_new_tokens=10, temperature=0.1)
            layer_str = response.strip()

            match = re.search(r'(\d+)', layer_str)
            if match:
                layer = int(match.group(1))
                return max(8, min(20, layer))
            return 14
        except:
            return 14

    def _complete_creation_estimate(self, base: Dict[str, Any], available_benchmarks: List[str], issue_type: str) -> ClassifierCreationEstimate:
        """Complete the creation estimate with hardware adjustments."""
        hardware_multiplier = self._estimate_hardware_speed()
        training_time = base["training_time_minutes"] * hardware_multiplier

        confidence = 0.8 if available_benchmarks else 0.6

        return ClassifierCreationEstimate(
            issue_type=issue_type,
            estimated_training_time_minutes=training_time,
            estimated_quality_score=base["quality_score"],
            training_samples_needed=base["samples_needed"],
            optimal_layer=base["optimal_layer"],
            confidence=confidence
        )

    def _estimate_hardware_speed(self) -> float:
        """Estimate hardware speed multiplier for training time."""
        device_kind = resolve_default_device()
        if device_kind == "cuda":
            return 0.3
        if device_kind == "mps":
            return 0.5
        return 1.0

    def get_marketplace_summary(self) -> str:
        """Get a summary of the classifier marketplace."""
        if not self.available_classifiers:
            self.discover_available_classifiers()

        if not self.available_classifiers:
            return "Classifier Marketplace: No classifiers available"

        summary = f"\nClassifier Marketplace Summary\n"
        summary += f"{'='*50}\n"
        summary += f"Available Classifiers: {len(self.available_classifiers)}\n\n"

        by_issue_type = {}
        for classifier in self.available_classifiers:
            issue_type = classifier.issue_type
            if issue_type not in by_issue_type:
                by_issue_type[issue_type] = []
            by_issue_type[issue_type].append(classifier)

        for issue_type, classifiers in by_issue_type.items():
            best_classifier = max(classifiers, key=lambda x: x.quality_score)
            summary += f"{issue_type.upper()}: {len(classifiers)} available\n"
            summary += f"   Best: {os.path.basename(best_classifier.path)} "
            summary += f"(Quality: {best_classifier.quality_score:.3f}, Layer: {best_classifier.layer})\n"
            summary += f"   Samples: {best_classifier.training_samples}, "
            summary += f"Model: {best_classifier.model_family}\n\n"

        return summary

    def filter_classifiers(self,
                          issue_types: List[str] = None,
                          min_quality: float = 0.0,
                          model_family: str = None,
                          layers: List[int] = None) -> 'List':
        """
        Filter available classifiers by criteria.

        Args:
            issue_types: List of issue types to include
            min_quality: Minimum quality score
            model_family: Required model family
            layers: Allowed layers

        Returns:
            Filtered list of classifier listings
        """
        filtered = self.available_classifiers

        if issue_types:
            filtered = [c for c in filtered if c.issue_type in issue_types]

        if min_quality > 0:
            filtered = [c for c in filtered if c.quality_score >= min_quality]

        if model_family:
            filtered = [c for c in filtered if c.model_family == model_family]

        if layers:
            filtered = [c for c in filtered if c.layer in layers]

        return filtered

    async def create_classifier_on_demand(self,
                                        issue_type: str,
                                        custom_layer: int = None):
        """
        Create a new classifier on demand.

        Args:
            issue_type: Type of issue to create classifier for
            custom_layer: Optional custom layer (otherwise uses optimal)

        Returns:
            Newly created classifier listing
        """
        from .create_classifier import create_classifier_on_demand

        print(f"Creating new classifier for {issue_type}...")

        estimate = self.get_creation_estimate(issue_type)
        layer = custom_layer or estimate.optimal_layer

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/agent_created_{issue_type}_layer{layer}_{timestamp}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        start_time = time.time()
        result = create_classifier_on_demand(
            model=self.model,
            issue_type=issue_type,
            layer=layer,
            save_path=save_path,
            optimize=True
        )
        training_time = time.time() - start_time

        from .classifier_marketplace import ClassifierListing

        listing = ClassifierListing(
            path=result.save_path,
            layer=result.config.layer,
            issue_type=issue_type,
            threshold=result.config.threshold,
            quality_score=result.performance_metrics.get('f1', 0.0),
            training_samples=result.performance_metrics.get('training_samples', 0),
            model_family=self._extract_model_family(self.model.model_name),
            created_at=datetime.now().isoformat(),
            training_time_seconds=training_time,
            metadata=result.performance_metrics
        )

        self.available_classifiers.append(listing)
        self.available_classifiers.sort(key=lambda x: x.quality_score, reverse=True)

        print(f"   Created classifier in {training_time/60:.1f} minutes")
        print(f"   Quality score: {listing.quality_score:.3f}")

        return listing
