"""Sample download and task loading mixin."""
import json
import logging
from typing import Dict, List, Optional, Iterator, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class CachedBenchDownloadMixin:
    """Mixin providing download and task loading."""

    def _get_cached_sample_count(self, task_name: str) -> int:
        """Get number of cached samples for a task."""
        if task_name not in self._metadata.tasks:
            return 0
        return self._metadata.tasks[task_name].samples_count

    def _download_samples(self, task_name: str, limit: int, start_offset: int = 0) -> List[Dict[str, Any]]:
        """
        Download samples from a benchmark task.

        Args:
            task_name: Name of the benchmark
            limit: Number of samples to download
            start_offset: Offset to start downloading from

        Returns:
            List of normalized samples
        """
        logger.info(f"Downloading {limit} samples for '{task_name}' (offset: {start_offset})")

        # Get extractor (hard error if not found)
        extractor = get_extractor(task_name)

        # Load raw task from lm-eval, BigCode, or TaskInterface
        try:
            task = self._load_lm_eval_task(task_name)
        except Exception as e:
            # Check if it's a BigCode task
            from .bigcode_integration import BigCodeTaskLoader

            loader = BigCodeTaskLoader()
            if loader.is_bigcode_task(task_name):
                task = self._load_bigcode_task(task_name)
            # Check if it's a TaskInterface task (like AIME, HLE, etc.)
            elif self._is_taskinterface_task(task_name):
                task = self._load_taskinterface_task(task_name, limit=start_offset + limit)
            else:
                raise BenchmarkError(f"Failed to load task '{task_name}' from lm-eval: {e}")

        # Get sample iterator
        try:
            sample_iterator = self._get_task_sample_iterator(task, start_offset + limit)
        except Exception as e:
            raise BenchmarkError(f"Failed to get samples from task '{task_name}': {e}")

        # Skip to start offset
        for _ in range(start_offset):
            try:
                next(sample_iterator)
            except StopIteration:
                raise InsufficientSamplesError(
                    f"Task '{task_name}' only has {start_offset} samples, cannot skip to offset {start_offset}"
                )

        # Extract samples
        samples = []
        for i in range(limit):
            try:
                raw_sample = next(sample_iterator)
            except StopIteration:
                raise InsufficientSamplesError(
                    f"Task '{task_name}' only has {start_offset + i} samples, but {start_offset + limit} were requested"
                )

            # Extract contrastive pair using extractor (includes both correct and incorrect answers)
            try:
                qa_pair = extractor.extract_contrastive_pair(raw_sample, task)
                if qa_pair is None:
                    raise ExtractorReturnedNoneError(task_name=task_name)
            except Exception as e:
                raise SampleNormalizationError(f"Failed to normalize sample {start_offset + i} from '{task_name}': {e}")

            samples.append(
                {
                    "id": f"sample_{start_offset + i:03d}",
                    "raw_data": raw_sample,
                    "normalized": qa_pair,
                    "extracted_at": datetime.now().isoformat(),
                }
            )

        logger.info(f"Successfully downloaded {len(samples)} samples for '{task_name}'")
        return samples

    def _load_bigcode_task(self, task_name: str):
        """Load task from bigcode-evaluation-harness."""
        from .bigcode_integration import BigCodeTaskLoader

        loader = BigCodeTaskLoader()

        # For APPS, we need to check if HF_ALLOW_CODE_EVAL is set
        if task_name == "apps" and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
            print(f"\n⚠️  Task '{task_name}' requires code evaluation permission.")
            print("This task will execute model-generated code which could be potentially harmful.")
            print("Please review the safety information at: https://arxiv.org/abs/2107.03374")
            response = input("\nDo you want to enable code evaluation? (yes/no): ").strip().lower()

            if response == "yes":
                os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                print("✅ Code evaluation enabled for this session.")
            else:
                raise BenchmarkError(f"Code evaluation permission denied for task '{task_name}'")

        return loader.load_task(task_name)

    def _load_lm_eval_task(self, task_name: str):
        """Load task from lm-eval-harness."""
        try:
            from lm_eval.tasks import get_task_dict

            # First check if it's a BigCode task before trying lm-eval
            from .bigcode_integration import BigCodeTaskLoader

            loader = BigCodeTaskLoader()
            if loader.is_bigcode_task(task_name):
                raise BigCodeTaskRequiresFlagError(task_name=task_name)

            # Check if we need HF_ALLOW_CODE_EVAL for code evaluation tasks
            code_eval_tasks = ["mbpp", "mbpp_plus", "humaneval", "humaneval_plus"]
            if task_name in code_eval_tasks and os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
                print(f"\n⚠️  Task '{task_name}' requires code evaluation permission.")
                print("This task will execute model-generated code which could be potentially harmful.")
                print("Please review the safety information at: https://arxiv.org/abs/2107.03374")
                response = input("\nDo you want to enable code evaluation? (yes/no): ").strip().lower()

                if response == "yes":
                    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
                    print("✅ Code evaluation enabled for this session.")
                else:
                    raise BenchmarkError(f"Code evaluation permission denied for task '{task_name}'")

            task_dict = get_task_dict([task_name])
            if task_name not in task_dict:
                raise TaskNotFoundError(task_name=task_name)

            return task_dict[task_name]
        except ImportError as e:
            raise BenchmarkError("lm-evaluation-harness not available") from e

    def _is_taskinterface_task(self, task_name: str) -> bool:
        """Check if task is a TaskInterface-based task by checking the task registry."""
        from wisent.core.tasks.base.task_interface import list_tasks

        return task_name in list_tasks()

    def _load_taskinterface_task(self, task_name: str, limit: Optional[int] = None):
        """Load TaskInterface task using the central task registry."""
        from wisent.core.tasks.base.task_interface import get_task

        try:
            return get_task(task_name, limit=limit)
        except Exception as e:
            raise BenchmarkError(f"Failed to load TaskInterface task '{task_name}': {e}")

    def _get_task_sample_iterator(self, task, limit: int) -> Iterator[Dict[str, Any]]:
        """
        Get iterator over task samples using unified split strategy.

        Combines ALL available splits and returns only the TEST portion (20%)
        to ensure no data leakage with training (which uses the 80% train portion).
        """
        # Get task name for deterministic splitting
        task_name = getattr(task, 'NAME', getattr(task, 'TASK_NAME', type(task).__name__))

        # Get ALL docs from all splits
        all_docs, split_counts = get_all_docs_from_task(task)

        if not all_docs:
            raise BenchmarkError(f"No document source available for task '{task_name}'")

        # Apply our 80/20 split and get TEST docs only
        _, test_docs = create_deterministic_split(all_docs, task_name)

        logger.info(f"Using {len(test_docs)} test docs from {task_name} "
                   f"(total: {len(all_docs)}, original splits: {split_counts})")

        # Convert to iterator and limit
        for i, doc in enumerate(test_docs):
            if i >= limit:
                break
            yield doc

