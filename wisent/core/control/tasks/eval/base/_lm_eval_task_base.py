"""
LM-Evaluation-Harness task wrapper for task-agnostic architecture.
"""

from typing import Any, Dict, List, Optional

from wisent.core.benchmarks import BenchmarkExtractor
from wisent.core.benchmarks import get_extractor as get_benchmark_extractor
from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor as get_lm_extractor
from wisent.core.tasks.base.task_interface import TaskInterface
from wisent.core.utils import get_test_docs, get_all_docs_from_task, create_deterministic_split


def get_extractor(task_name: str) -> BenchmarkExtractor:
    """Get extractor, trying lm_extractor_registry first, then benchmark_extractors."""
    try:
        return get_lm_extractor(task_name)
    except Exception:
        return get_benchmark_extractor(task_name)


class LMEvalTask(TaskInterface):
    """Wrapper for lm-evaluation-harness tasks."""

    def __init__(self, task_name: str, description: str, categories: List[str]):
        self.task_name = task_name
        self._description = description
        self._categories = categories
        self._extractor = get_extractor(task_name)

    def load_data(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load TEST data from lm-eval using our unified split strategy.

        All available splits are combined and split 80/20 into train/test.
        This method returns the TEST portion for evaluation.
        Training portion is used for contrastive pair generation (see atoms.py).
        """
        try:
            from lm_eval.tasks import get_task_dict

            # Get task directly from lm-eval
            task_dict = get_task_dict([self.task_name])
            if self.task_name not in task_dict:
                print(f"Warning: Task '{self.task_name}' not found in lm-eval")
                return []

            task = task_dict[self.task_name]

            # Get ALL docs from all splits
            all_docs, split_counts = get_all_docs_from_task(task)

            if not all_docs:
                print(f"Warning: No documents found for task '{self.task_name}'")
                return []

            # Apply our 80/20 split and get TEST docs only
            _, test_docs = create_deterministic_split(all_docs, self.task_name)

            # Ensure docs are in dictionary format
            processed_docs = []
            for doc in test_docs:
                if isinstance(doc, dict):
                    processed_docs.append(doc)
                elif isinstance(doc, str):
                    processed_docs.append({"text": doc})
                else:
                    try:
                        processed_docs.append(dict(doc))
                    except:
                        processed_docs.append({"data": str(doc)})

            docs = processed_docs

            # Apply limit if specified
            if limit and len(docs) > limit:
                docs = docs[:limit]

            total = len(all_docs)
            test_count = len(test_docs)
            returned = len(docs)
            print(f"Loaded {returned} test docs from {self.task_name} "
                  f"(total: {total}, test split: {test_count}, original splits: {split_counts})")

            return docs

        except Exception as e:
            print(f"Warning: Could not load lm-eval task '{self.task_name}': {e}")
            return []

    def get_extractor(self) -> BenchmarkExtractor:
        """Get the benchmark extractor for this task."""
        return self._extractor

    def get_name(self) -> str:
        """Get the task name."""
        return self.task_name

    def get_description(self) -> str:
        """Get the task description."""
        return self._description

    def get_categories(self) -> List[str]:
        """Get the task categories."""
        return self._categories


class MBPPTask(LMEvalTask):
    """MBPP task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mbpp",
            description="MBPP: Mostly Basic Python Problems coding benchmark",
            categories=["coding", "reasoning", "python"],
        )


class HumanEvalTask(LMEvalTask):
    """HumanEval task implementation."""

    def __init__(self):
        super().__init__(
            task_name="humaneval",
            description="HumanEval: Human Evaluation of Python coding problems",
            categories=["coding", "reasoning", "python"],
        )


class MBPPPlusTask(LMEvalTask):
    """MBPP Plus task implementation."""

    def __init__(self):
        super().__init__(
            task_name="mbpp_plus",
            description="MBPP Plus: Extended version of MBPP with additional test cases",
            categories=["coding", "reasoning", "python"],
        )


class GSM8KTask(LMEvalTask):
    """GSM8K task implementation."""

    def __init__(self):
        super().__init__(
            task_name="gsm8k",
            description="GSM8K: Grade School Math 8K problems",
            categories=["mathematics", "reasoning", "arithmetic"],
        )

class ArithmeticBaseTask(LMEvalTask):
    """Base class for arithmetic tasks. Uses unified split strategy."""
    pass  # No longer needs special handling - unified split combines all available docs


class Arithmetic1dcTask(ArithmeticBaseTask):
    """Arithmetic 1dc task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_1dc",
            description="Arithmetic 1dc: 1 digit addition arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )


class Arithmetic2daTask(ArithmeticBaseTask):
    """Arithmetic 2da task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_2da",
            description="Arithmetic 2da: 2 digit addition arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic2dmTask(ArithmeticBaseTask):
    """Arithmetic 2dm task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_2dm",
            description="Arithmetic 2dm: 2 digit multiplication arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic2dsTask(ArithmeticBaseTask):
    """Arithmetic 2ds task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_2ds",
            description="Arithmetic 2ds: 2 digit subtraction arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic3daTask(ArithmeticBaseTask):
    """Arithmetic 3da task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_3da",
            description="Arithmetic 3da: 3 digit addition arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic3dsTask(ArithmeticBaseTask):
    """Arithmetic 3ds task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_3ds",
            description="Arithmetic 3ds: 3 digit subtraction arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic4daTask(ArithmeticBaseTask):
    """Arithmetic 4da task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_4da",
            description="Arithmetic 4da: 4 digit addition arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic4dsTask(ArithmeticBaseTask):
    """Arithmetic 4ds task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_4ds",
            description="Arithmetic 4ds: 4 digit subtraction arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic5daTask(ArithmeticBaseTask):
    """Arithmetic 5da task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_5da",
            description="Arithmetic 5da: 5 digit addition arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

class Arithmetic5dsTask(ArithmeticBaseTask):
    """Arithmetic 5ds task implementation"""

    def __init__(self):
        super().__init__(
            task_name="arithmetic_5ds",
            description="Arithmetic 5ds: 5 digit subtraction arithmetic problems",
            categories=["mathematics", "arithmetic"]
        )

