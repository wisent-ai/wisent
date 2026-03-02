"""
Managed Cached Benchmarks service for intelligent dataset downloading and caching.

This service controls how much of each benchmark is downloaded based on the limit parameter:
- If limit=5, download only 5 samples
- If limit=3 and we have 5 cached, reuse cached samples
- If limit=10 and we have 5 cached, download 5 more
- Hard errors for unsupported benchmarks, no fallbacks

Uses unified split strategy: all available splits are combined and split 80/20 into train/test.
For cached benchmarks used in evaluation, we use the TEST portion.
"""

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from wisent.core.utils.services.benchmarks import get_extractor
from wisent.core.utils import get_all_docs_from_task, create_deterministic_split
from wisent.core.utils.infra_tools.errors import ExtractorReturnedNoneError, BigCodeTaskRequiresFlagError, TaskNotFoundError
from wisent.core.utils.config_tools.constants import CACHE_CHUNK_SIZE, CACHE_MAX_AGE_DAYS

logger = logging.getLogger(__name__)


class BenchmarkError(Exception):
    """Base exception for benchmark-related errors."""


class UnsupportedBenchmarkError(BenchmarkError):
    """Raised when benchmark has no adapter."""


class SampleNormalizationError(BenchmarkError):
    """Raised when sample normalization fails."""


class InsufficientSamplesError(BenchmarkError):
    """Raised when benchmark doesn't have enough samples."""


class CacheCorruptionError(BenchmarkError):
    """Raised when cache data is corrupted."""


@dataclass
class CacheInfo:
    """Information about cached benchmark data."""

    task_name: str
    samples_count: int
    last_updated: datetime
    cache_version: str
    chunks: List[str]  # List of chunk filenames


@dataclass
class CacheMetadata:
    """Global cache metadata."""

    version: str
    created_at: datetime
    last_cleanup: datetime
    tasks: Dict[str, CacheInfo]


from wisent.core.utils.services.benchmarks.cache._cached_bench_download import CachedBenchDownloadMixin
from wisent.core.utils.services.benchmarks.cache._cached_bench_io import CachedBenchIOMixin

class ManagedCachedBenchmarks(CachedBenchDownloadMixin, CachedBenchIOMixin):
    """
    Service for intelligent benchmark downloading and caching.

    Features:
    - Downloads only what's needed based on limit parameter
    - Reuses cached data when possible
    - Incremental downloads for growing limits
    - Hard errors for unsupported benchmarks
    - Chunk-based storage for efficiency
    """

    CACHE_VERSION = "1.0"
    CHUNK_SIZE = CACHE_CHUNK_SIZE
    MAX_CACHE_AGE_DAYS = CACHE_MAX_AGE_DAYS
    SUPPORTED_BENCHMARKS = None  # Will be initialized in __init__

    def __init__(self, cache_dir: str = "./benchmark_cache"):
        """
        Initialize the managed cache service.

        Args:
            cache_dir: Directory to store cached benchmark data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.metadata_file = self.cache_dir / "metadata.json"
        self._metadata = self._load_metadata()

        # Initialize supported benchmarks including BigCode tasks
        if ManagedCachedBenchmarks.SUPPORTED_BENCHMARKS is None:
            supported = set(EXTRACTORS.keys())
            try:
                from .bigcode_integration import BigCodeTaskLoader

                loader = BigCodeTaskLoader()
                supported.update(loader.TASK_MAPPING.keys())
            except ImportError:
                pass
            ManagedCachedBenchmarks.SUPPORTED_BENCHMARKS = supported

        # Validate all supported benchmarks have extractors
        self._validate_extractor_registry()

    def _validate_extractor_registry(self):
        """Ensure every supported benchmark has a working extractor."""
        for benchmark in self.SUPPORTED_BENCHMARKS:
            try:
                extractor = get_extractor(benchmark)
                if not hasattr(extractor, "extract_qa_pair"):
                    raise AttributeError(f"Extractor for {benchmark} missing extract_qa_pair method")
            except Exception as e:
                raise BenchmarkError(f"Invalid extractor for supported benchmark '{benchmark}': {e}")

    def get_task_samples(self, task_name: str, limit: int, force_fresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get samples for a task, using intelligent caching.

        Args:
            task_name: Name of the benchmark task
            limit: Number of samples needed
            force_fresh: Force fresh download even if cached

        Returns:
            List of normalized sample dictionaries

        Raises:
            UnsupportedBenchmarkError: If task has no extractor
            InsufficientSamplesError: If benchmark doesn't have enough samples
            SampleNormalizationError: If sample extraction fails
        """
        # Hard error for unsupported benchmarks
        if task_name not in self.SUPPORTED_BENCHMARKS:
            raise UnsupportedBenchmarkError(
                f"No extractor found for benchmark '{task_name}'. "
                f"Supported benchmarks: {sorted(self.SUPPORTED_BENCHMARKS)}"
            )

        if limit <= 0:
            return []

        logger.info(f"Getting {limit} samples for task '{task_name}'")

        # Check cache status
        cached_count = self._get_cached_sample_count(task_name)
        logger.info(f"Found {cached_count} cached samples for '{task_name}'")

        if force_fresh:
            logger.info(f"Force fresh download requested for '{task_name}'")
            self._clear_task_cache(task_name)
            cached_count = 0

        # Decision logic
        if cached_count >= limit:
            # Case 1: We have enough - load from cache
            logger.info(f"Loading {limit} samples from cache for '{task_name}'")
            return self._load_cached_samples(task_name, limit)

        if cached_count > 0 and limit <= cached_count * 2:
            # Case 2: We have some, need a bit more - incremental download
            needed = limit - cached_count
            logger.info(f"Incremental download: need {needed} more samples for '{task_name}'")

            new_samples = self._download_samples(task_name, needed, start_offset=cached_count)
            self._append_to_cache(task_name, new_samples)

            return self._load_cached_samples(task_name, limit)

        # Case 3: Major mismatch - fresh download
        logger.info(f"Fresh download: getting {limit} samples for '{task_name}'")
        self._clear_task_cache(task_name)

        new_samples = self._download_samples(task_name, limit, start_offset=0)
        self._save_to_cache(task_name, new_samples)

        return new_samples

def get_managed_cache(cache_dir: str = "./benchmark_cache") -> ManagedCachedBenchmarks:
    """Get the global managed cache instance."""
    global _managed_cache
    if _managed_cache is None:
        _managed_cache = ManagedCachedBenchmarks(cache_dir)
    return _managed_cache
