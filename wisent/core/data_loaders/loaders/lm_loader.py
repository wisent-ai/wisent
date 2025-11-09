from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging

from wisent.core.data_loaders.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
)

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = [
    "LMEvalDataLoader",
]

log = logging.getLogger(__name__)

class LMEvalDataLoader(BaseDataLoader):
    """
    Load contrastive pairs from a single lm-evaluation-harness task via `load_lm_eval_task`,
    split into train/test, and return a canonical LoadDataResult.
    """
    name = "lm_eval"
    description = "Load from a single lm-eval task."

    def _load_one_task(
        self,
        task_name: str,
        split_ratio: float,
        seed: int,
        limit: int | None,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> LoadDataResult:
        """
        Load a single lm-eval task by name, convert to contrastive pairs,
        split into train/test, and return a LoadDataResult.
        
        arguments:
            task_name: The name of the lm-eval task to load.
            split_ratio: The fraction of data to use for training (between 0 and 1).
            seed: Random seed for shuffling/splitting.
            limit: Optional limit on total number of pairs to load.
            training_limit: Optional limit on number of training pairs.
            testing_limit: Optional limit on number of testing pairs.
            
        returns:
            A LoadDataResult containing train/test pairs and task info.
            
        raises:
            DataLoaderError if the task cannot be found or if splits are empty.
            ValueError if split_ratio is not in [0.0, 1.0].
            NotImplementedError if load_lm_eval_task is not implemented.
        
        note:
            This loader supports both single tasks and group tasks. For group tasks,
            it loads all subtasks and combines their pairs."""
        loaded = self.load_lm_eval_task(task_name)

        if isinstance(loaded, dict):
            if len(loaded) == 1:
                # Single subtask
                (subname, task_obj), = loaded.items()
                pairs = lm_build_contrastive_pairs(
                    task_name=subname,
                    lm_eval_task=task_obj,
                    limit=limit,
                )
            else:
                # Group task with multiple subtasks - load all and combine
                log.info(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")
                all_pairs = []
                pairs_per_subtask = limit // len(loaded) if limit else None

                for subname, task_obj in loaded.items():
                    try:
                        subtask_pairs = lm_build_contrastive_pairs(
                            task_name=subname,
                            lm_eval_task=task_obj,
                            limit=pairs_per_subtask,
                        )
                        all_pairs.extend(subtask_pairs)
                        log.info(f"Loaded {len(subtask_pairs)} pairs from subtask '{subname}'")
                    except Exception as e:
                        log.warning(f"Failed to load subtask '{subname}': {e}")
                        continue

                if not all_pairs:
                    raise DataLoaderError(f"No pairs could be loaded from any subtask of '{task_name}'")

                pairs = all_pairs
                log.info(f"Combined {len(pairs)} total pairs from {len(loaded)} subtasks")
        else:
            task_obj = loaded
            pairs = lm_build_contrastive_pairs(
                task_name=task_name,
                lm_eval_task=task_obj,
                limit=limit,
            )

        train_pairs, test_pairs = self._split_pairs(
            pairs, split_ratio, seed, training_limit, testing_limit
        )

        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")

        train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
        test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)

        train_set.validate(raise_on_critical=False)
        test_set.validate(raise_on_critical=False)

        return LoadDataResult(
            train_qa_pairs=train_set,
            test_qa_pairs=test_set,
            task_type=task_name,
            lm_task_data=task_obj,
        )

    def load(
        self,
        task: str,  
        split_ratio: float | None = None,
        seed: int = 42,
        limit: int | None = None,
        training_limit: int | None = None,
        testing_limit: int | None = None,
        **_: Any,
    ) -> LoadDataResult:
        """
        Load contrastive pairs from a single lm-eval-harness task, split into train/test sets.
        arguments:
            task:
                The name of the lm-eval task to load (e.g., "winogrande", "hellaswag").
                Must be a single task, not a mixture.
            split_ratio:
                Float in [0.0, 1.0] representing the proportion of data to use for training.
                Defaults to 0.8 if None.
            seed:
                Random seed for shuffling the data before splitting.
            limit:
                Optional maximum number of total pairs to load from the task.
            training_limit:
                Optional maximum number of training pairs to return.
            testing_limit:
                Optional maximum number of testing pairs to return.
            **_:
                Additional keyword arguments (ignored).
        
        returns:
            LoadDataResult with train/test ContrastivePairSets and metadata.
        
        raises:
            DataLoaderError if loading or processing fails.
            ValueError if split_ratio is not in [0.0, 1.0].
            NotImplementedError if load_lm_eval_task is not implemented.
        """
        split = self._effective_split(split_ratio)

        # Single-task path only
        return self._load_one_task(
            task_name=str(task),
            split_ratio=split,
            seed=seed,
            limit=limit,
            training_limit=training_limit,
            testing_limit=testing_limit,
        )

    @staticmethod
    def load_lm_eval_task(task_name: str) -> ConfigurableTask | dict[str, ConfigurableTask]:
        """
        Load a single lm-eval-harness task by name.

        arguments:
            task_name: The name of the lm-eval task to load.

        returns:
            A ConfigurableTask instance or a dict of subtask name to ConfigurableTask.
            For group tasks, flattens nested groups and returns all leaf tasks.

        raises:
            DataLoaderError if the task cannot be found.
        """
        # Map task names to their lm-eval equivalents
        task_name_mapping = {
            "squad2": "squadv2",
            "wikitext103": "wikitext",
            "ptb": "wikitext",
            "penn_treebank": "wikitext",
        }

        # Use mapped name if available, otherwise use original
        lm_eval_task_name = task_name_mapping.get(task_name, task_name)
        if lm_eval_task_name != task_name:
            log.info(f"Mapping task '{task_name}' to lm-eval task '{lm_eval_task_name}'")

        task_manager = LMTaskManager()
        task_manager.initialize_tasks()

        task_dict = get_task_dict([lm_eval_task_name], task_manager=task_manager)

        # Try to get the task directly
        if lm_eval_task_name in task_dict:
            result = task_dict[lm_eval_task_name]
            # If result is a dict with nested groups, flatten it
            if isinstance(result, dict):
                flat_tasks = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        # Nested group - add all subtasks
                        flat_tasks.update(value)
                    else:
                        # Direct task
                        flat_tasks[key] = value
                return flat_tasks if flat_tasks else result
            return result

        # If not found directly, might be the first (and only) key in task_dict
        if len(task_dict) == 1:
            key, value = list(task_dict.items())[0]
            # Check if the key's name matches what we're looking for
            if hasattr(key, 'group') and key.group == lm_eval_task_name:
                if isinstance(value, dict):
                    # Flatten nested groups
                    flat_tasks = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            flat_tasks.update(v)
                        else:
                            flat_tasks[k] = v
                    return flat_tasks if flat_tasks else value
                return value

        # Check if this is a group task where get_task_dict returns subtasks directly
        # This handles both cases:
        # - 'arithmetic' returns {'arithmetic_1dc': task, 'arithmetic_2da': task, ...}
        # - 'hendrycks_ethics' returns {'ethics_cm': task, 'ethics_justice': task, ...}
        # Verify that values are actual Task objects to ensure this is a valid group task
        if task_dict and len(task_dict) > 0:
            from lm_eval.api.task import Task
            # Check if at least one value is a Task object
            if any(isinstance(v, Task) for v in task_dict.values()):
                log.info(f"Task '{lm_eval_task_name}' is a group task with {len(task_dict)} subtasks: {list(task_dict.keys())}")
                return task_dict

        raise DataLoaderError(f"lm-eval task '{lm_eval_task_name}' not found (requested as '{task_name}').")
    
    def _split_pairs(
        self,
        pairs: list[ContrastivePair],
        split_ratio: float,
        seed: int,
        training_limit: int | None,
        testing_limit: int | None,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """
        Split a list of ContrastivePairs into train/test sets.

        arguments:
            pairs: List of ContrastivePair to split.
            split_ratio: Float in [0.0, 1.0] for the training set proportion.
            seed: Random seed for shuffling.
            training_limit: Optional max number of training pairs.
            testing_limit: Optional max number of testing pairs.
        
        returns:
            A tuple of (train_pairs, test_pairs).
        raises:
            ValueError if split_ratio is not in [0.0, 1.0].
        """
        if not pairs:
            return [], []
        from numpy.random import default_rng

        idx = list(range(len(pairs)))
        default_rng(seed).shuffle(idx)
        cut = int(len(pairs) * split_ratio)
        train_idx = set(idx[:cut])

        train_pairs: list[ContrastivePair] = []
        test_pairs: list[ContrastivePair] = []
        for i in idx:
            (train_pairs if i in train_idx else test_pairs).append(pairs[i])

        if training_limit and training_limit > 0:
            train_pairs = train_pairs[:training_limit]
        if testing_limit and testing_limit > 0:
            test_pairs = test_pairs[:testing_limit]

        return train_pairs, test_pairs