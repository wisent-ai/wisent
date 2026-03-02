"""Cache IO, status, and maintenance mixin."""
import json
import logging
import time
from typing import Dict, List, Any
from pathlib import Path
from wisent.core.benchmarks.cache._cached_bench_types import (
    CacheInfo, CacheMetadata, CacheCorruptionError)
from wisent.core.constants import BYTES_PER_MB, JSON_INDENT, DISPLAY_DECIMAL_PRECISION

logger = logging.getLogger(__name__)

class CachedBenchIOMixin:
    """Mixin providing cache IO and maintenance."""

    def _save_to_cache(self, task_name: str, samples: List[Dict[str, Any]]):
        """Save samples to cache in chunks."""
        task_dir = self.cache_dir / task_name
        task_dir.mkdir(exist_ok=True)

        # Save in chunks
        chunks = []
        for i in range(0, len(samples), self.CHUNK_SIZE):
            chunk_samples = samples[i : i + self.CHUNK_SIZE]
            chunk_filename = f"samples_{i + 1}_to_{i + len(chunk_samples)}.json"
            chunk_path = task_dir / chunk_filename

            with open(chunk_path, "w") as f:
                json.dump(chunk_samples, f, indent=JSON_INDENT)

            chunks.append(chunk_filename)

        # Update metadata
        cache_info = CacheInfo(
            task_name=task_name,
            samples_count=len(samples),
            last_updated=datetime.now(),
            cache_version=self.CACHE_VERSION,
            chunks=chunks,
        )

        self._metadata.tasks[task_name] = cache_info
        self._save_metadata()

        logger.info(f"Saved {len(samples)} samples to cache for '{task_name}' in {len(chunks)} chunks")

    def _append_to_cache(self, task_name: str, new_samples: List[Dict[str, Any]]):
        """Append new samples to existing cache."""
        if task_name not in self._metadata.tasks:
            return self._save_to_cache(task_name, new_samples)

        task_dir = self.cache_dir / task_name
        cache_info = self._metadata.tasks[task_name]

        # Load existing samples
        existing_samples = self._load_all_cached_samples(task_name)

        # Combine and re-save in chunks
        all_samples = existing_samples + new_samples
        self._save_to_cache(task_name, all_samples)

    def _load_cached_samples(self, task_name: str, limit: int) -> List[Dict[str, Any]]:
        """Load cached samples up to limit."""
        if task_name not in self._metadata.tasks:
            return []

        cache_info = self._metadata.tasks[task_name]
        task_dir = self.cache_dir / task_name

        samples = []
        samples_loaded = 0

        for chunk_filename in cache_info.chunks:
            if samples_loaded >= limit:
                break

            chunk_path = task_dir / chunk_filename
            if not chunk_path.exists():
                raise CacheCorruptionError(f"Missing chunk file: {chunk_path}")

            try:
                with open(chunk_path) as f:
                    chunk_samples = json.load(f)
            except Exception as e:
                raise CacheCorruptionError(f"Corrupted chunk file {chunk_path}: {e}")

            # Add samples until we reach the limit
            for sample in chunk_samples:
                if samples_loaded >= limit:
                    break
                samples.append(sample)
                samples_loaded += 1

        logger.info(f"Loaded {len(samples)} cached samples for '{task_name}'")
        return samples

    def _load_all_cached_samples(self, task_name: str) -> List[Dict[str, Any]]:
        """Load all cached samples for a task."""
        if task_name not in self._metadata.tasks:
            return []

        cache_info = self._metadata.tasks[task_name]
        return self._load_cached_samples(task_name, cache_info.samples_count)

    def _clear_task_cache(self, task_name: str):
        """Clear cache for a specific task."""
        task_dir = self.cache_dir / task_name

        if task_dir.exists():
            import shutil

            shutil.rmtree(task_dir)

        if task_name in self._metadata.tasks:
            del self._metadata.tasks[task_name]
            self._save_metadata()

        logger.info(f"Cleared cache for task '{task_name}'")

    def _load_metadata(self) -> CacheMetadata:
        """Load cache metadata."""
        if not self.metadata_file.exists():
            return CacheMetadata(
                version=self.CACHE_VERSION, created_at=datetime.now(), last_cleanup=datetime.now(), tasks={}
            )

        try:
            with open(self.metadata_file) as f:
                data = json.load(f)

            # Convert datetime strings back to datetime objects
            tasks = {}
            for task_name, task_data in data.get("tasks", {}).items():
                tasks[task_name] = CacheInfo(
                    task_name=task_data["task_name"],
                    samples_count=task_data["samples_count"],
                    last_updated=datetime.fromisoformat(task_data["last_updated"]),
                    cache_version=task_data["cache_version"],
                    chunks=task_data["chunks"],
                )

            return CacheMetadata(
                version=data.get("version", self.CACHE_VERSION),
                created_at=datetime.fromisoformat(data["created_at"]),
                last_cleanup=datetime.fromisoformat(data["last_cleanup"]),
                tasks=tasks,
            )

        except Exception as e:
            logger.warning(f"Failed to load cache metadata: {e}")
            return CacheMetadata(
                version=self.CACHE_VERSION, created_at=datetime.now(), last_cleanup=datetime.now(), tasks={}
            )

    def _save_metadata(self):
        """Save cache metadata."""
        # Convert to serializable format
        tasks_data = {}
        for task_name, cache_info in self._metadata.tasks.items():
            tasks_data[task_name] = {
                "task_name": cache_info.task_name,
                "samples_count": cache_info.samples_count,
                "last_updated": cache_info.last_updated.isoformat(),
                "cache_version": cache_info.cache_version,
                "chunks": cache_info.chunks,
            }

        data = {
            "version": self._metadata.version,
            "created_at": self._metadata.created_at.isoformat(),
            "last_cleanup": self._metadata.last_cleanup.isoformat(),
            "tasks": tasks_data,
        }

        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

    def cache_status(self) -> Dict[str, Any]:
        """Get comprehensive cache status."""
        total_samples = sum(info.samples_count for info in self._metadata.tasks.values())
        total_size = sum(
            sum(
                (self.cache_dir / task_name / chunk).stat().st_size
                for chunk in info.chunks
                if (self.cache_dir / task_name / chunk).exists()
            )
            for task_name, info in self._metadata.tasks.items()
        )

        return {
            "cache_version": self._metadata.version,
            "total_tasks": len(self._metadata.tasks),
            "total_samples": total_samples,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / BYTES_PER_MB, DISPLAY_DECIMAL_PRECISION),
            "created_at": self._metadata.created_at.isoformat(),
            "last_cleanup": self._metadata.last_cleanup.isoformat(),
            "tasks": {
                task_name: {
                    "samples_count": info.samples_count,
                    "last_updated": info.last_updated.isoformat(),
                    "chunks": len(info.chunks),
                }
                for task_name, info in self._metadata.tasks.items()
            },
        }

    def cleanup_cache(self, max_age_days: int = None):
        """Clean up old cache entries."""
        if max_age_days is None:
            max_age_days = self.MAX_CACHE_AGE_DAYS

        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        tasks_to_remove = []

        for task_name, cache_info in self._metadata.tasks.items():
            if cache_info.last_updated < cutoff_date:
                tasks_to_remove.append(task_name)

        for task_name in tasks_to_remove:
            self._clear_task_cache(task_name)

        self._metadata.last_cleanup = datetime.now()
        self._save_metadata()

        logger.info(f"Cleaned up {len(tasks_to_remove)} old cache entries")
        return len(tasks_to_remove)

    def preload_tasks(self, task_limits: Dict[str, int]):
        """Preload multiple tasks with specified limits."""
        results = {}

        for task_name, limit in task_limits.items():
            try:
                samples = self.get_task_samples(task_name, limit)
                results[task_name] = {"status": "success", "samples_loaded": len(samples), "requested_limit": limit}
                logger.info(f"Preloaded {len(samples)} samples for '{task_name}'")
            except Exception as e:
                results[task_name] = {"status": "error", "error": str(e), "requested_limit": limit}
                logger.error(f"Failed to preload '{task_name}': {e}")

        return results


# Global instance
_managed_cache = None


