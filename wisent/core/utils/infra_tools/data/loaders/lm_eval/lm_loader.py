from __future__ import annotations
from typing import Any, TYPE_CHECKING
import logging
import os

# Configure TensorFlow threading BEFORE any TensorFlow import
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Allow code evaluation for code-related tasks (humaneval, etc.)
os.environ['HF_ALLOW_CODE_EVAL'] = '1'

# Enable trust_remote_code for all datasets
import datasets.config
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

# Patch deprecated 'List' feature type (datasets v3.6.0+)
import datasets.features.features as _features_module
if 'List' not in _features_module._FEATURE_TYPES and 'LargeList' in _features_module._FEATURE_TYPES:
    _features_module._FEATURE_TYPES['List'] = _features_module._FEATURE_TYPES['LargeList']

from wisent.core.utils.infra_tools.data.core.atoms import BaseDataLoader, DataLoaderError, LoadDataResult
from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.set import ContrastivePairSet
from lm_eval.tasks import get_task_dict
from lm_eval.tasks import TaskManager as LMTaskManager
from wisent.extractors.lm_eval.lm_task_pairs_generation import (
    lm_build_contrastive_pairs,
)
from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader_special_cases import get_special_case_handler
from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import (
    TASK_NAME_MAPPING, CASE_SENSITIVE_PREFIXES, GROUP_TASK_EXPANSIONS,
)

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask

__all__ = ["LMEvalDataLoader"]

log = logging.getLogger(__name__)


class LMEvalDataLoader(BaseDataLoader):
    """Load contrastive pairs from a single lm-evaluation-harness task."""
    name = "lm_eval"
    description = "Load from a single lm-eval task."
    _huggingface_only_tasks_cache = None

    @classmethod
    def _get_huggingface_only_tasks(cls):
        """Get the set of HuggingFace-only tasks from central registry."""
        if cls._huggingface_only_tasks_cache is None:
            from wisent.core.utils.services.benchmarks import get_huggingface_only_tasks_set
            cls._huggingface_only_tasks_cache = get_huggingface_only_tasks_set()
        return cls._huggingface_only_tasks_cache

    def _load_one_task(
        self, task_name: str, split_ratio: float, seed: int,
        limit: int | None, training_limit: int | None, testing_limit: int | None,
        *, train_ratio: float,
    ) -> LoadDataResult:
        """Load a single lm-eval task, convert to contrastive pairs, split into train/test."""
        task_name_lower = task_name.lower()
        if task_name_lower in self._get_huggingface_only_tasks():
            log.info(f"Task '{task_name}' is a HuggingFace-only task, loading via HuggingFace extractor")
            pairs = lm_build_contrastive_pairs(task_name=task_name, lm_eval_task=None, limit=limit, train_ratio=train_ratio)
            train_pairs, test_pairs = self._split_pairs(pairs, split_ratio, seed, training_limit, testing_limit)
            if not train_pairs or not test_pairs:
                raise DataLoaderError("One of the splits is empty after splitting.")
            train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
            test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)
            train_set.validate(raise_on_critical=False)
            test_set.validate(raise_on_critical=False)
            return LoadDataResult(
                train_qa_pairs=train_set, test_qa_pairs=test_set,
                task_type=task_name, lm_task_data=None,
            )

        loaded = self.load_lm_eval_task(task_name)

        if isinstance(loaded, dict):
            if len(loaded) == 1:
                (subname, task_obj), = loaded.items()
                pairs = lm_build_contrastive_pairs(task_name=subname, lm_eval_task=task_obj, limit=limit, train_ratio=train_ratio)
            else:
                log.info(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")
                print(f"Task '{task_name}' is a group task with {len(loaded)} subtasks. Loading all subtasks...")
                all_pairs = []
                pairs_per_subtask = limit // len(loaded) if limit else None
                for subname, task_obj in loaded.items():
                    try:
                        subtask_pairs = lm_build_contrastive_pairs(
                            task_name=subname, lm_eval_task=task_obj, limit=pairs_per_subtask, train_ratio=train_ratio,
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
            pairs = lm_build_contrastive_pairs(task_name=task_name, lm_eval_task=task_obj, limit=limit, train_ratio=train_ratio)

        train_pairs, test_pairs = self._split_pairs(pairs, split_ratio, seed, training_limit, testing_limit)
        if not train_pairs or not test_pairs:
            raise DataLoaderError("One of the splits is empty after splitting.")
        train_set = ContrastivePairSet("lm_eval_train", train_pairs, task_type=task_name)
        test_set = ContrastivePairSet("lm_eval_test", test_pairs, task_type=task_name)
        train_set.validate(raise_on_critical=False)
        test_set.validate(raise_on_critical=False)
        return LoadDataResult(
            train_qa_pairs=train_set, test_qa_pairs=test_set,
            task_type=task_name, lm_task_data=task_obj,
        )

    def load(
        self, task: str, split_ratio: float | None = None, seed: int | None = None,
        limit: int | None = None, training_limit: int | None = None,
        testing_limit: int | None = None, *, train_ratio: float, **_: Any,
    ) -> LoadDataResult:
        """Load contrastive pairs from a single lm-eval-harness task, split into train/test sets."""
        if seed is None:
            from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED
            seed = DEFAULT_RANDOM_SEED
        split = self._effective_split(split_ratio)
        return self._load_one_task(
            task_name=str(task), split_ratio=split, seed=seed,
            limit=limit, training_limit=training_limit, testing_limit=testing_limit,
            train_ratio=train_ratio,
        )

    @staticmethod
    def load_lm_eval_task(task_name: str) -> ConfigurableTask | dict[str, ConfigurableTask]:
        """Load a single lm-eval-harness task by name."""
        lm_eval_task_name = TASK_NAME_MAPPING.get(task_name, task_name)
        if lm_eval_task_name != task_name:
            log.info(f"Mapping task '{task_name}' to lm-eval task '{lm_eval_task_name}'")

        # Check for case-sensitive prefixes (including ACVA tasks with camelCase components)
        is_case_sensitive = (
            any(lm_eval_task_name.startswith(prefix) for prefix in CASE_SENSITIVE_PREFIXES) or
            lm_eval_task_name.startswith("arabic_leaderboard_acva_")
        )
        if not is_case_sensitive:
            lm_eval_task_name_normalized = lm_eval_task_name.lower()
            if lm_eval_task_name_normalized != lm_eval_task_name:
                log.info(f"Normalizing task name to lowercase: '{lm_eval_task_name}' -> '{lm_eval_task_name_normalized}'")
                lm_eval_task_name = lm_eval_task_name_normalized

        is_ruler_task = lm_eval_task_name == 'ruler' or lm_eval_task_name.startswith('ruler_') or lm_eval_task_name.startswith('niah_')
        if is_ruler_task:
            task_manager = LMTaskManager(
                verbosity="INFO",
                metadata={"pretrained": "meta-llama/Llama-3.2-1B-Instruct"}
            )
            task_manager.initialize_tasks()
        else:
            task_manager = LMTaskManager()
            task_manager.initialize_tasks()

        if lm_eval_task_name == 'pile' or lm_eval_task_name.startswith('pile_'):
            raise DataLoaderError(
                f"Task '{lm_eval_task_name}' is disabled. "
                f"The Pile benchmark dataset files are hosted on the-eye.eu which is currently unavailable. "
                f"This is an external infrastructure issue and cannot be resolved in Wisent."
            )

        if lm_eval_task_name in GROUP_TASK_EXPANSIONS:
            subtasks = GROUP_TASK_EXPANSIONS[lm_eval_task_name]
            log.info(f"Expanding group task '{lm_eval_task_name}' to {len(subtasks)} subtasks")

            # Special handling for "advanced": try to load parent "advanced_ai_risk" instead
            # of individual subtasks, since lm-eval may not recognize individual subtask names
            if lm_eval_task_name == "advanced":
                log.info("Special case: loading parent 'advanced_ai_risk' instead of individual subtasks")
                parent_dict = get_task_dict(["advanced_ai_risk"], task_manager=task_manager)
                if parent_dict and "advanced_ai_risk" in parent_dict:
                    parent_task = parent_dict["advanced_ai_risk"]
                    # If parent is a dict (group task), flatten and return it
                    if isinstance(parent_task, dict):
                        log.info(f"Parent 'advanced_ai_risk' is a group task with {len(parent_task)} subtasks")
                        return parent_task

            # Standard expansion: try to load all subtasks
            task_dict = get_task_dict(subtasks, task_manager=task_manager)
            return task_dict

        special_handler = get_special_case_handler(lm_eval_task_name)
        if special_handler:
            log.info(f"Using special case handler for task '{lm_eval_task_name}'")
            return special_handler(task_manager)

        task_dict = get_task_dict([lm_eval_task_name], task_manager=task_manager)

        if lm_eval_task_name in task_dict:
            result = task_dict[lm_eval_task_name]
            if isinstance(result, dict):
                flat_tasks = {}
                for key, value in result.items():
                    if isinstance(value, dict):
                        flat_tasks.update(value)
                    else:
                        flat_tasks[key] = value
                return flat_tasks if flat_tasks else result
            return result

        if len(task_dict) == 1:
            key, value = list(task_dict.items())[0]
            if hasattr(key, 'group') and key.group == lm_eval_task_name:
                if isinstance(value, dict):
                    flat_tasks = {}
                    for k, v in value.items():
                        if isinstance(v, dict):
                            flat_tasks.update(v)
                        else:
                            flat_tasks[k] = v
                    return flat_tasks if flat_tasks else value
                return value

        if task_dict and len(task_dict) > 0:
            from lm_eval.api.task import Task
            if any(isinstance(v, Task) for v in task_dict.values()):
                log.info(f"Task '{lm_eval_task_name}' is a group task with {len(task_dict)} subtasks: {list(task_dict.keys())}")
                return task_dict

        raise DataLoaderError(f"lm-eval task '{lm_eval_task_name}' not found (requested as '{task_name}').")

    def _split_pairs(
        self, pairs: list[ContrastivePair], split_ratio: float, seed: int,
        training_limit: int | None, testing_limit: int | None,
    ) -> tuple[list[ContrastivePair], list[ContrastivePair]]:
        """Split a list of ContrastivePairs into train/test sets."""
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
