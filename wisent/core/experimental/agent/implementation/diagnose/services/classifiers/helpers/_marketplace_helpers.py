"""Estimation, filtering, and on-demand creation helpers for ClassifierMarketplace."""

from typing import List, Dict, Any
from dataclasses import dataclass
import os
import re
import time
from datetime import datetime

from wisent.core.utils import resolve_default_device
from wisent.core.utils.config_tools.constants import SECONDS_PER_MINUTE


@dataclass
class MarketplaceEstimationConfig:
    """Required configuration for marketplace estimation parameters."""
    base_training_time: float
    per_benchmark_time: float
    quality_max: float
    quality_base: float
    quality_increment: float
    max_samples_needed: int
    base_samples_needed: int
    per_benchmark_samples: int
    synthetic_training_time: float
    quality_default: float
    synthetic_samples: int
    default_optimal_layer: int
    max_tasks_to_search: int
    max_relevant_benchmarks: int
    max_relevant_to_return: int
    max_similarity_score: float
    layer_min: int
    layer_max: int
    confidence_with_benchmarks: float
    confidence_without_benchmarks: float
    cuda_speed_multiplier: float
    mps_speed_multiplier: float
    min_quality_default: float


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
        """Get an estimate for creating a new classifier for the given issue type."""
        available_benchmarks = self._find_available_benchmarks_for_issue(issue_type)
        mc = self._mp_config
        if available_benchmarks:
            benchmark_count = len(available_benchmarks)
            base = {
                "training_time_minutes": mc.base_training_time + (benchmark_count * mc.per_benchmark_time),
                "quality_score": min(mc.quality_max, mc.quality_base + (benchmark_count * mc.quality_increment)),
                "samples_needed": min(mc.max_samples_needed, mc.base_samples_needed + (benchmark_count * mc.per_benchmark_samples)),
                "optimal_layer": self._estimate_optimal_layer_for_issue(issue_type)
            }
            print(f"   Using {benchmark_count} benchmarks for {issue_type}")
        else:
            base = {
                "training_time_minutes": mc.synthetic_training_time,
                "quality_score": mc.quality_default,
                "samples_needed": mc.synthetic_samples,
                "optimal_layer": mc.default_optimal_layer
            }
            print(f"   Using synthetic generation for {issue_type}")
        return self._complete_creation_estimate(base, available_benchmarks, issue_type)

    def _find_available_benchmarks_for_issue(self, issue_type: str) -> List[str]:
        """Find available benchmarks using dynamic semantic analysis."""
        available_tasks = self.model.get_available_tasks()
        mc = self._mp_config
        relevant = []
        issue_lower = issue_type.lower()
        for task in available_tasks[:mc.max_tasks_to_search]:
            task_lower = task.lower()
            similarity_score = self._calculate_task_similarity(issue_lower, task_lower)
            if similarity_score > 0:
                relevant.append((task, similarity_score))
                if len(relevant) >= mc.max_relevant_benchmarks:
                    break
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [task for task, score in relevant[:mc.max_relevant_to_return]]

    def _calculate_task_similarity(self, issue_type: str, task_name: str) -> float:
        """Calculate similarity between issue type and task name using model decisions."""
        mc = self._mp_config
        prompt = f"""Rate the similarity between this issue type and evaluation task for training AI safety classifiers.

Issue Type: {issue_type}
Task: {task_name}

Rate similarity from 0.0 to {mc.max_similarity_score} ({mc.max_similarity_score} = highly similar, 0.0 = not similar).
Respond with only the number:"""
        try:
            response = self.model.generate(prompt, layer_index=self.layer)
            score_str = response.strip()
            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(mc.max_similarity_score, max(0.0, score))
            return 0.0
        except:
            return 0.0

    def _estimate_optimal_layer_for_issue(self, issue_type: str) -> int:
        """Estimate optimal layer using model analysis of issue complexity."""
        mc = self._mp_config
        prompt = f"""What transformer layer would be optimal for detecting this AI safety issue?

Issue Type: {issue_type}

Consider:
- Simple issues (formatting, basic patterns) -> early layers ({mc.layer_min}-12)
- Complex semantic issues (truthfulness, bias) -> middle layers (12-16)
- Abstract conceptual issues (coherence, quality) -> deeper layers (16-{mc.layer_max})

Respond with just the layer number ({mc.layer_min}-{mc.layer_max}):"""
        try:
            response = self.model.generate(prompt, layer_index=self.layer)
            layer_str = response.strip()
            match = re.search(r'(\d+)', layer_str)
            if match:
                layer = int(match.group(1))
                return max(mc.layer_min, min(mc.layer_max, layer))
            return mc.default_optimal_layer
        except:
            return mc.default_optimal_layer

    def _complete_creation_estimate(self, base: Dict[str, Any], available_benchmarks: List[str], issue_type: str) -> ClassifierCreationEstimate:
        """Complete the creation estimate with hardware adjustments."""
        mc = self._mp_config
        hardware_multiplier = self._estimate_hardware_speed()
        training_time = base["training_time_minutes"] * hardware_multiplier
        confidence = mc.confidence_with_benchmarks if available_benchmarks else mc.confidence_without_benchmarks
        return ClassifierCreationEstimate(
            issue_type=issue_type, estimated_training_time_minutes=training_time,
            estimated_quality_score=base["quality_score"], training_samples_needed=base["samples_needed"],
            optimal_layer=base["optimal_layer"], confidence=confidence)

    def _estimate_hardware_speed(self) -> float:
        """Estimate hardware speed multiplier for training time."""
        mc = self._mp_config
        device_kind = resolve_default_device()
        if device_kind == "cuda":
            return mc.cuda_speed_multiplier
        if device_kind == "mps":
            return mc.mps_speed_multiplier
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

    def filter_classifiers(self, issue_types: List[str] = None, min_quality: float = None,
                          model_family: str = None, layers: List[int] = None) -> 'List':
        """Filter available classifiers by criteria."""
        if min_quality is None:
            min_quality = self._mp_config.min_quality_default
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

    async def create_classifier_on_demand(self, issue_type: str, custom_layer: int = None):
        """Create a new classifier on demand."""
        from .create_classifier import create_classifier_on_demand
        print(f"Creating new classifier for {issue_type}...")
        estimate = self.get_creation_estimate(issue_type)
        layer = custom_layer or estimate.optimal_layer
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./models/agent_created_{issue_type}_layer{layer}_{timestamp}.pkl"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        start_time = time.time()
        result = create_classifier_on_demand(
            model=self.model, issue_type=issue_type, layer=layer, save_path=save_path, optimize=True, data_oversample_multiplier=self._data_oversample_multiplier)
        training_time = time.time() - start_time
        from .classifier_marketplace import ClassifierListing
        listing = ClassifierListing(
            path=result.save_path, layer=result.config.layer, issue_type=issue_type,
            threshold=result.config.threshold, quality_score=result.performance_metrics.get('f1', 0.0),
            training_samples=result.performance_metrics.get('training_samples', 0),
            model_family=self._extract_model_family(self.model.model_name),
            created_at=datetime.now().isoformat(), training_time_seconds=training_time,
            metadata=result.performance_metrics)
        self.available_classifiers.append(listing)
        self.available_classifiers.sort(key=lambda x: x.quality_score, reverse=True)
        print(f"   Created classifier in {training_time/SECONDS_PER_MINUTE:.1f} minutes")
        print(f"   Quality score: {listing.quality_score:.3f}")
        return listing
