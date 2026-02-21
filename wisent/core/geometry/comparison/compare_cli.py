"""CLI entry point for the compare-steering command."""

from __future__ import annotations

import json
from typing import List, Optional

from wisent.core.steering_methods.steering_object import load_steering_object
from wisent.core.geometry.comparison.cross_trait import compare_steering_objects


def _get_label(obj) -> str:
    """Extract trait label from steering object metadata."""
    if hasattr(obj, 'metadata'):
        m = obj.metadata
        benchmark = getattr(m, 'benchmark', None)
        if benchmark:
            return benchmark
        category = getattr(m, 'category', None)
        if category:
            return category
    return "unknown"


def _parse_layers(layers_str: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated layer string into list of ints."""
    if not layers_str:
        return None
    return [int(x.strip()) for x in layers_str.split(",")]


def _print_similarity_table(result) -> None:
    """Print formatted similarity matrix table."""
    labels = result.labels
    matrix = result.similarity_matrix
    n = len(labels)

    # Header
    max_label = max(len(l) for l in labels) + 2
    header = " " * max_label + "  ".join(f"{l:>10}" for l in labels)
    print(f"\n{'=' * len(header)}")
    print("PAIRWISE COSINE SIMILARITY")
    print(f"{'=' * len(header)}")
    print(header)

    for i in range(n):
        row = f"{labels[i]:<{max_label}}" + "  ".join(
            f"{matrix[i][j]:>10.4f}" for j in range(n)
        )
        print(row)

    # Uniqueness scores
    print(f"\n{'=' * 40}")
    print("UNIQUENESS SCORES (higher = more unique)")
    print(f"{'=' * 40}")
    for label, score in sorted(result.uniqueness_scores.items(),
                                key=lambda x: x[1], reverse=True):
        bar = "#" * int(score * 40)
        print(f"   {label:<20} {score:.4f}  {bar}")

    # Clusters
    print(f"\n{'=' * 40}")
    print("CLUSTER ASSIGNMENTS")
    print(f"{'=' * 40}")
    for i, label in enumerate(labels):
        print(f"   {label:<20} cluster {result.clusters[i]}")

    # Summary
    print(f"\nShared variance explained: {result.shared_variance_explained:.4f}")
    ms = result.most_similar_pair
    md = result.most_different_pair
    print(f"Most similar:  {ms[0]} <-> {ms[1]} (cos={ms[2]:.4f})")
    print(f"Most different: {md[0]} <-> {md[1]} (cos={md[2]:.4f})")


def _result_to_dict(result) -> dict:
    """Convert ComparisonResult to JSON-serializable dict."""
    return {
        "labels": result.labels,
        "similarity_matrix": result.similarity_matrix,
        "per_layer_similarity": {
            str(k): v for k, v in result.per_layer_similarity.items()
        },
        "clusters": result.clusters,
        "uniqueness_scores": result.uniqueness_scores,
        "shared_variance_explained": result.shared_variance_explained,
        "most_similar_pair": {
            "a": result.most_similar_pair[0],
            "b": result.most_similar_pair[1],
            "cosine": result.most_similar_pair[2],
        },
        "most_different_pair": {
            "a": result.most_different_pair[0],
            "b": result.most_different_pair[1],
            "cosine": result.most_different_pair[2],
        },
    }


def execute_compare_steering(args):
    """Execute cross-trait steering comparison.

    Loads steering objects from paths, computes comparison metrics,
    prints summary, and optionally saves full results as JSON.
    """
    object_paths = args.objects
    layers = _parse_layers(getattr(args, 'layers', None))
    output_path = getattr(args, 'output', None)
    output_format = getattr(args, 'format', 'both')

    if len(object_paths) < 2:
        print("Need at least 2 steering objects to compare.")
        return None

    # Load objects
    print(f"Loading {len(object_paths)} steering objects...")
    objects = []
    labels = []
    for path in object_paths:
        obj = load_steering_object(path)
        objects.append(obj)
        labels.append(_get_label(obj))
        print(f"   {path} -> {labels[-1]}")

    # Run comparison
    result = compare_steering_objects(
        objects=objects,
        labels=labels,
        layers=layers,
    )

    # Output
    if output_format in ("table", "both"):
        _print_similarity_table(result)

    if output_path and output_format in ("json", "both"):
        data = _result_to_dict(result)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return result
