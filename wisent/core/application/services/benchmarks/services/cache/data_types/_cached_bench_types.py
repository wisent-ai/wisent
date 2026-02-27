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

from wisent.core.benchmarks import get_extractor
from wisent.core.utils import get_all_docs_from_task, create_deterministic_split
from wisent.core.errors import ExtractorReturnedNoneError, BigCodeTaskRequiresFlagError, TaskNotFoundError

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


