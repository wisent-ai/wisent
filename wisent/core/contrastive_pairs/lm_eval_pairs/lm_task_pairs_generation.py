from __future__ import annotations

import random
from typing import TYPE_CHECKING

from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair

__all__ = ["build_contrastive_pairs", "lm_build_contrastive_pairs"]
_LOG = setup_logger(__name__)


def _flatten_task_dict(task_dict: dict, prefix: str = "") -> list[tuple[str, "ConfigurableTask"]]:
    """
    Recursively flatten nested group tasks into a list of (name, ConfigurableTask) tuples.
    
    arguments:
        task_dict: Dict of task_name -> ConfigurableTask or nested dict
        prefix: Prefix for nested task names
        
    returns:
        List of (full_task_name, ConfigurableTask) tuples (leaf tasks only)
    """
    from lm_eval.api.task import ConfigurableTask
    
    result = []
    for name, task in task_dict.items():
        full_name = f"{prefix}/{name}" if prefix else name
        if isinstance(task, ConfigurableTask):
            result.append((full_name, task))
        elif isinstance(task, dict):
            # Nested group - recurse
            result.extend(_flatten_task_dict(task, full_name))
    return result


def _add_evaluator_to_pairs(
    pairs: list["ContrastivePair"],
    evaluator_name: str | None,
    task_name: str,
) -> list["ContrastivePair"]:
    """Add evaluator_name and task_name to each pair's metadata."""
    from dataclasses import replace
    
    result = []
    for pair in pairs:
        metadata = dict(pair.metadata) if pair.metadata else {}
        metadata["evaluator_name"] = evaluator_name
        metadata["source_task"] = task_name
        result.append(replace(pair, metadata=metadata))
    return result


def build_contrastive_pairs(
    task_name: str,
    limit: int | None = None,
) -> list["ContrastivePair"]:
    """
    Unified loader for contrastive pairs - handles both HuggingFace and lm-eval tasks.
    
    Automatically:
    - Detects if task is HF or lm-eval
    - Handles group tasks (including nested groups) by sampling from all subtasks
    - Adds evaluator_name to each pair's metadata
    
    arguments:
        task_name:
            Name of the benchmark/task (e.g., "winogrande", "mmlu", "humaneval").
        limit:
            Optional upper bound on the number of pairs to return.
            Values <= 0 are treated as "no limit".
            
    returns:
        A list of ContrastivePair objects, each with metadata containing
        'evaluator_name' and 'source_task'.
    """
    log = bind(_LOG, task=task_name or "unknown")
    log.info("Building contrastive pairs (unified)", extra={"limit": limit})
    
    # Normalize limit
    max_items = None if (limit is None or limit <= 0) else int(limit)
    
    # Get extractor
    extractor = get_extractor(task_name)
    log.info("Using extractor", extra={"extractor": extractor.__class__.__name__})
    
    # Get evaluator_name from extractor
    evaluator_name = getattr(extractor, 'evaluator_name', None)
    
    # HuggingFace extractor - load directly
    if isinstance(extractor, HuggingFaceBenchmarkExtractor):
        log.info("HuggingFace task - loading directly")
        pairs = extractor.extract_contrastive_pairs(limit=max_items)
        return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
    
    # lm-eval extractor - need to load task
    log.info("lm-eval task - loading via LMEvalDataLoader")
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    
    loader = LMEvalDataLoader()
    try:
        task_obj = loader.load_lm_eval_task(task_name)
    except Exception as e:
        log.error(f"Failed to load lm-eval task: {e}")
        raise
    
    # Single task (ConfigurableTask)
    from lm_eval.api.task import ConfigurableTask
    if isinstance(task_obj, ConfigurableTask):
        log.info("Single task")
        pairs = extractor.extract_contrastive_pairs(task_obj, limit=max_items)
        return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
    
    # Group task (dict) - flatten and sample from all subtasks
    if isinstance(task_obj, dict):
        leaf_tasks = _flatten_task_dict(task_obj)
        log.info(f"Group task with {len(leaf_tasks)} leaf subtasks")
        
        if not leaf_tasks:
            log.warning("No leaf tasks found in group")
            return []
        
        # Shuffle to get random sampling across subtasks
        random.shuffle(leaf_tasks)
        
        # Calculate pairs per subtask
        if max_items is None:
            pairs_per_task = None
        else:
            # Distribute limit across subtasks, minimum 1 per task
            pairs_per_task = max(1, max_items // len(leaf_tasks))
        
        all_pairs = []
        for subtask_name, subtask in leaf_tasks:
            try:
                # Get the leaf task name (last part after /)
                leaf_name = subtask_name.split("/")[-1] if "/" in subtask_name else subtask_name
                
                # Try to get extractor for the specific subtask first
                try:
                    subtask_extractor = get_extractor(leaf_name)
                except:
                    # Fall back to parent extractor
                    subtask_extractor = extractor
                
                subtask_evaluator = getattr(subtask_extractor, 'evaluator_name', evaluator_name)
                
                subtask_pairs = subtask_extractor.extract_contrastive_pairs(subtask, limit=pairs_per_task)
                subtask_pairs = _add_evaluator_to_pairs(subtask_pairs, subtask_evaluator, subtask_name)
                all_pairs.extend(subtask_pairs)
                
                # Stop if we have enough
                if max_items is not None and len(all_pairs) >= max_items:
                    break
            except Exception as e:
                log.warning(f"Failed to extract from subtask {subtask_name}: {e}")
                continue
        
        # Shuffle final result and trim to limit
        random.shuffle(all_pairs)
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        
        log.info(f"Extracted {len(all_pairs)} pairs from group task")
        return all_pairs
    
    log.error(f"Unexpected task_obj type: {type(task_obj)}")
    return []


def lm_build_contrastive_pairs(
    task_name: str,
    lm_eval_task: "ConfigurableTask | None",
    limit: int | None = None,
) -> list["ContrastivePair"]:
    """
    Legacy function - resolve the task's extractor and return contrastive pairs.
    
    For new code, prefer using build_contrastive_pairs() which handles
    task loading automatically.

    arguments:
        task_name:
            Name of the lm-eval benchmark/task (e.g., "winogrande").
        lm_eval_task:
            An lm-eval task instance. Can be None for HuggingFace-only tasks
            like livecodebench that don't use lm-eval.
        limit:
            Optional upper bound on the number of pairs to return.
            Values <= 0 are treated as "no limit".

    returns:
        A list of ContrastivePair objects.
    """
    log = bind(_LOG, task=task_name or "unknown")
    log.info("Building contrastive pairs", extra={"limit": limit})

    # 1) Get extractor instance by name (exact or longest-prefix)
    extractor = get_extractor(task_name)

    log.info("Using extractor", extra={"extractor": extractor.__class__.__name__})

    # 2) Normalize limit (<=0 → None)
    max_items = None if (limit is None or limit <= 0) else int(limit)

    log.info("Extracting contrastive pairs", extra={"max_items": max_items})
    
    # Get evaluator_name from extractor
    evaluator_name = getattr(extractor, 'evaluator_name', None)

    # 3) Delegate: extractor loads docs and builds pairs
    # HuggingFace extractors don't need lm_eval_task - they load data directly from HuggingFace
    if isinstance(extractor, HuggingFaceBenchmarkExtractor):
        pairs = extractor.extract_contrastive_pairs(limit=max_items)
    else:
        pairs = extractor.extract_contrastive_pairs(lm_eval_task, limit=max_items)
    
    return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)