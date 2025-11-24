"""
Geometry analysis for steering vectors.

Analyzes the geometry of positive/negative activation distributions
and steering vectors to understand:
- How well-separated are positive and negative classes?
- How aligned are steering vectors with class means?
- Layer-by-layer coherence and quality metrics

Adapted from Heretic's refusal geometry analysis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["GeometryAnalyzer", "GeometryMetrics", "analyze_steering_geometry"]


@dataclass
class GeometryMetrics:
    """
    Geometry metrics for a single layer.

    Attributes:
        layer_idx: Layer index
        positive_mean_norm: L2 norm of positive activation mean
        negative_mean_norm: L2 norm of negative activation mean
        steering_vector_norm: L2 norm of steering vector
        positive_negative_similarity: Cosine similarity between positive and negative means
        positive_steering_similarity: Cosine similarity between positive mean and steering
        negative_steering_similarity: Cosine similarity between negative mean and steering
        separation_quality: How well-separated the classes are (higher is better)
    """

    layer_idx: int
    positive_mean_norm: float
    negative_mean_norm: float
    steering_vector_norm: float
    positive_negative_similarity: float
    positive_steering_similarity: float
    negative_steering_similarity: float
    separation_quality: float

    def __repr__(self) -> str:
        return (
            f"GeometryMetrics(layer={self.layer_idx}, "
            f"sep_quality={self.separation_quality:.3f}, "
            f"steering_norm={self.steering_vector_norm:.1f})"
        )


class GeometryAnalyzer:
    """
    Analyzes geometry of activation distributions and steering vectors.

    Computes detailed statistics about:
    - Class separation (positive vs negative activations)
    - Steering vector alignment with class means
    - Layer-by-layer coherence

    Adapted from Heretic's print_refusal_geometry functionality.

    Usage:
        analyzer = GeometryAnalyzer()
        metrics = analyzer.analyze(positive_acts, negative_acts, steering_vectors)
        analyzer.print_table(metrics)
    """

    def compute_layer_metrics(
        self,
        positive_activations: Tensor,
        negative_activations: Tensor,
        steering_vector: Tensor,
        layer_idx: int,
    ) -> GeometryMetrics:
        """
        Compute geometry metrics for a single layer.

        Args:
            positive_activations: Activations for positive examples, shape [N, H]
            negative_activations: Activations for negative examples, shape [M, H]
            steering_vector: Steering vector for this layer, shape [H]
            layer_idx: Layer index

        Returns:
            GeometryMetrics for this layer
        """
        # Compute means
        positive_mean = positive_activations.mean(dim=0)  # [H]
        negative_mean = negative_activations.mean(dim=0)  # [H]

        # Compute norms
        positive_norm = positive_mean.norm(p=2).item()
        negative_norm = negative_mean.norm(p=2).item()
        steering_norm = steering_vector.norm(p=2).item()

        # Compute cosine similarities
        pos_neg_sim = F.cosine_similarity(
            positive_mean.unsqueeze(0),
            negative_mean.unsqueeze(0),
        ).item()

        pos_steer_sim = F.cosine_similarity(
            positive_mean.unsqueeze(0),
            steering_vector.unsqueeze(0),
        ).item()

        neg_steer_sim = F.cosine_similarity(
            negative_mean.unsqueeze(0),
            steering_vector.unsqueeze(0),
        ).item()

        # Compute separation quality
        # Good separation: high distance, low similarity between pos/neg
        # Steering should point from negative toward positive
        separation_quality = pos_steer_sim - neg_steer_sim

        return GeometryMetrics(
            layer_idx=layer_idx,
            positive_mean_norm=positive_norm,
            negative_mean_norm=negative_norm,
            steering_vector_norm=steering_norm,
            positive_negative_similarity=pos_neg_sim,
            positive_steering_similarity=pos_steer_sim,
            negative_steering_similarity=neg_steer_sim,
            separation_quality=separation_quality,
        )

    def analyze(
        self,
        positive_activations: dict[int, Tensor],
        negative_activations: dict[int, Tensor],
        steering_vectors: dict[int, Tensor],
    ) -> dict[int, GeometryMetrics]:
        """
        Analyze geometry across all layers.

        Args:
            positive_activations: Dict mapping layer -> positive activations [N, H]
            negative_activations: Dict mapping layer -> negative activations [M, H]
            steering_vectors: Dict mapping layer -> steering vector [H]

        Returns:
            Dictionary mapping layer index to GeometryMetrics
        """
        metrics = {}

        for layer_idx in steering_vectors.keys():
            if (
                layer_idx in positive_activations
                and layer_idx in negative_activations
            ):
                metrics[layer_idx] = self.compute_layer_metrics(
                    positive_activations[layer_idx],
                    negative_activations[layer_idx],
                    steering_vectors[layer_idx],
                    layer_idx,
                )

        return metrics

    def print_table(self, metrics: dict[int, GeometryMetrics]) -> None:
        """
        Print formatted table of geometry metrics.

        Args:
            metrics: Dictionary mapping layer index to GeometryMetrics
        """
        print("\n" + "=" * 120)
        print("STEERING GEOMETRY ANALYSIS")
        print("=" * 120)
        print(
            f"{'Layer':>6} | "
            f"{'|Pos|':>8} {'|Neg|':>8} {'|Steer|':>8} | "
            f"{'S(p,n)':>7} {'S(p,s)':>7} {'S(n,s)':>7} | "
            f"{'SepQual':>8}"
        )
        print("-" * 120)

        for layer_idx in sorted(metrics.keys()):
            m = metrics[layer_idx]
            print(
                f"{layer_idx:>6} | "
                f"{m.positive_mean_norm:>8.2f} {m.negative_mean_norm:>8.2f} {m.steering_vector_norm:>8.2f} | "
                f"{m.positive_negative_similarity:>7.3f} {m.positive_steering_similarity:>7.3f} {m.negative_steering_similarity:>7.3f} | "
                f"{m.separation_quality:>8.3f}"
            )

        print("=" * 120)
        print("\nLegend:")
        print("  |Pos|, |Neg|, |Steer|: L2 norms of positive mean, negative mean, steering vector")
        print("  S(p,n): Cosine similarity between positive and negative means")
        print("  S(p,s): Cosine similarity between positive mean and steering vector")
        print("  S(n,s): Cosine similarity between negative mean and steering vector")
        print("  SepQual: Separation quality = S(p,s) - S(n,s) (higher is better)")
        print("\n")

    def get_summary_statistics(
        self,
        metrics: dict[int, GeometryMetrics],
    ) -> dict[str, float]:
        """
        Compute summary statistics across all layers.

        Args:
            metrics: Dictionary mapping layer index to GeometryMetrics

        Returns:
            Dictionary with summary stats
        """
        if not metrics:
            return {}

        sep_qualities = [m.separation_quality for m in metrics.values()]
        pos_neg_sims = [m.positive_negative_similarity for m in metrics.values()]
        steering_norms = [m.steering_vector_norm for m in metrics.values()]

        return {
            "mean_separation_quality": sum(sep_qualities) / len(sep_qualities),
            "min_separation_quality": min(sep_qualities),
            "max_separation_quality": max(sep_qualities),
            "mean_pos_neg_similarity": sum(pos_neg_sims) / len(pos_neg_sims),
            "mean_steering_norm": sum(steering_norms) / len(steering_norms),
            "std_steering_norm": (
                sum((n - sum(steering_norms) / len(steering_norms)) ** 2 for n in steering_norms)
                / len(steering_norms)
            )
            ** 0.5,
        }

    def identify_best_layers(
        self,
        metrics: dict[int, GeometryMetrics],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Identify layers with best separation quality.

        Args:
            metrics: Dictionary mapping layer index to GeometryMetrics
            top_k: Number of top layers to return

        Returns:
            List of (layer_idx, separation_quality) tuples, sorted by quality
        """
        layer_qualities = [
            (m.layer_idx, m.separation_quality) for m in metrics.values()
        ]
        layer_qualities.sort(key=lambda x: x[1], reverse=True)
        return layer_qualities[:top_k]

    def identify_problematic_layers(
        self,
        metrics: dict[int, GeometryMetrics],
        threshold: float = 0.1,
    ) -> list[tuple[int, str]]:
        """
        Identify layers with potential issues.

        Args:
            metrics: Dictionary mapping layer index to GeometryMetrics
            threshold: Separation quality threshold below which a layer is problematic

        Returns:
            List of (layer_idx, issue_description) tuples
        """
        problems = []

        for m in metrics.values():
            issues = []

            # Low separation quality
            if m.separation_quality < threshold:
                issues.append(f"low separation ({m.separation_quality:.3f})")

            # Steering vector points wrong way
            if m.negative_steering_similarity > m.positive_steering_similarity:
                issues.append("steering points toward negative")

            # High positive-negative similarity (classes not separated)
            if m.positive_negative_similarity > 0.9:
                issues.append(f"high pos-neg similarity ({m.positive_negative_similarity:.3f})")

            # Very low steering norm (weak signal)
            if m.steering_vector_norm < 1.0:
                issues.append(f"weak steering ({m.steering_vector_norm:.2f})")

            if issues:
                problems.append((m.layer_idx, "; ".join(issues)))

        return problems


def analyze_steering_geometry(
    positive_activations: dict[int, Tensor],
    negative_activations: dict[int, Tensor],
    steering_vectors: dict[int, Tensor],
    print_table: bool = True,
) -> dict[int, GeometryMetrics]:
    """
    Convenience function to analyze and optionally print steering geometry.

    Args:
        positive_activations: Dict mapping layer -> positive activations [N, H]
        negative_activations: Dict mapping layer -> negative activations [M, H]
        steering_vectors: Dict mapping layer -> steering vector [H]
        print_table: Whether to print formatted table

    Returns:
        Dictionary mapping layer index to GeometryMetrics

    Example:
        >>> from wisent.core.activation_collection import collect_activations
        >>> pos_acts = collect_activations(model, positive_pairs)
        >>> neg_acts = collect_activations(model, negative_pairs)
        >>> steering = {l: pos_acts[l].mean(0) - neg_acts[l].mean(0) for l in pos_acts}
        >>> metrics = analyze_steering_geometry(pos_acts, neg_acts, steering)
    """
    analyzer = GeometryAnalyzer()
    metrics = analyzer.analyze(positive_activations, negative_activations, steering_vectors)

    if print_table:
        analyzer.print_table(metrics)

        # Print summary
        summary = analyzer.get_summary_statistics(metrics)
        print("Summary Statistics:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")

        # Identify best layers
        print("\nTop 5 Layers (by separation quality):")
        best_layers = analyzer.identify_best_layers(metrics, top_k=5)
        for layer_idx, quality in best_layers:
            print(f"  Layer {layer_idx}: {quality:.3f}")

        # Identify problems
        problems = analyzer.identify_problematic_layers(metrics)
        if problems:
            print("\nProblematic Layers:")
            for layer_idx, issue in problems:
                print(f"  Layer {layer_idx}: {issue}")

    return metrics
