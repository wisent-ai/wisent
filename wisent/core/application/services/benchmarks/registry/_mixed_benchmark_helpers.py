"""Helper functions for mixed benchmark sampling."""

import random
import logging
from typing import List, Dict, Any, Optional
from wisent.core.constants import SAMPLES_PER_BENCHMARK

logger = logging.getLogger(__name__)


def sample_benchmarks_by_tag(
    tag: str,
    samples_per_benchmark: int = SAMPLES_PER_BENCHMARK,
    max_benchmarks: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Convenience function to sample from all benchmarks with a specific tag.

    Args:
        tag: Tag to filter benchmarks (e.g., "coding")
        samples_per_benchmark: Number of samples from each benchmark
        max_benchmarks: Maximum number of benchmarks to sample from
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping benchmark names to their samples
    """
    from wisent.core.benchmarks.mixed_benchmark_sampler import MixedBenchmarkSampler
    from wisent.core.benchmarks.cache.managed_cached_benchmarks import get_managed_cache

    sampler = MixedBenchmarkSampler()

    # Get all benchmarks with the tag
    benchmarks = sampler.get_benchmarks_by_tag(tag)

    if max_benchmarks and len(benchmarks) > max_benchmarks:
        if random_seed is not None:
            random.seed(random_seed)
        benchmarks = random.sample(benchmarks, max_benchmarks)

    # Sample from each benchmark
    results = {}
    cache = get_managed_cache()

    for benchmark_name in benchmarks:
        try:
            samples = cache.get_task_samples(
                task_name=benchmark_name,
                limit=samples_per_benchmark,
                force_fresh=False
            )
            results[benchmark_name] = samples
            logger.info(f"Sampled {len(samples)} from {benchmark_name}")

        except Exception as e:
            logger.warning(f"Failed to sample from {benchmark_name}: {e}")
            continue

    return results


def create_mixed_contrastive_pair_set_impl(
    sampler,
    tags: List[str],
    total_samples: int,
    name: Optional[str] = None,
    **kwargs
):
    """
    Create a ContrastivePairSet from mixed benchmark samples.

    Args:
        sampler: MixedBenchmarkSampler instance
        tags: Tags to filter benchmarks
        total_samples: Number of samples to include
        name: Name for the pair set (auto-generated if not provided)
        **kwargs: Additional arguments for sample_mixed_dataset

    Returns:
        ContrastivePairSet ready for training
    """
    from wisent.core.contrastive_pairs import ContrastivePairSet

    # Sample mixed dataset
    train_samples, test_samples = sampler.sample_mixed_dataset(
        tags=tags,
        total_samples=total_samples,
        **kwargs
    )

    # Extract contrastive pairs
    all_samples = train_samples + test_samples
    contrastive_pairs = sampler.extract_contrastive_pairs_from_mixed_samples(all_samples)

    # Create name if not provided
    if name is None:
        name = f"mixed_{'_'.join(tags)}_{total_samples}_samples"

    # Create ContrastivePairSet
    return ContrastivePairSet.from_contrastive_pairs(
        name=name,
        contrastive_pairs=contrastive_pairs,
        task_type="mixed_benchmark"
    )
