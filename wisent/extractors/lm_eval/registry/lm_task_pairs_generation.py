from __future__ import annotations

import random
from typing import TYPE_CHECKING

from wisent.extractors.lm_eval.lm_extractor_registry import get_extractor
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair

__all__ = ["build_contrastive_pairs", "lm_build_contrastive_pairs"]
_LOG = setup_logger(__name__)


def _flatten_task_dict(task_dict: dict, prefix: str = "") -> list[tuple[str, "ConfigurableTask"]]:
    """
    Recursively flatten nested group tasks into a list of (name, ConfigurableTask) tuples.

    Handles both string keys and Task object keys.

    arguments:
        task_dict: Dict of task_name -> ConfigurableTask or nested dict
        prefix: Prefix for nested task names

    returns:
        List of (full_task_name, ConfigurableTask) tuples (leaf tasks only)
    """
    from lm_eval.api.task import ConfigurableTask

    result = []
    for name, task in task_dict.items():
        # Handle both string keys and Task object keys
        if isinstance(name, str):
            task_name = name
        else:
            # Task object as key - try to get its name
            task_name = getattr(name, 'name', None) or getattr(name, 'NAME', None) or str(name)

        full_name = f"{prefix}/{task_name}" if prefix else task_name
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


def _load_subtask_from_parent(task_name: str, loader, log):
    """Try to load a subtask by loading its parent group and finding the subtask within.

    For example, 'aclue_ancient_chinese_culture' -> load 'aclue' group -> find subtask.
    Tries progressively shorter prefixes as the parent name.

    Also handles tasks with variant suffixes (e.g. '_light') by constructing
    suffixed parent candidates.  For example:
      'arabic_leaderboard_acva_arabic_literature_light'
      -> try parent 'arabic_leaderboard_acva_light' (strip the middle subtopic,
         keep the variant suffix) in addition to the plain prefix approach.
    """
    from lm_eval.api.task import ConfigurableTask
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import (
        GROUP_TASK_EXPANSIONS,
    )

    def _normalize_name(name: str) -> str:
        """Normalize name for comparison by converting dashes to underscores."""
        return name.replace("-", "_").lower()

    def _match(leaf_name: str) -> bool:
        """Case-insensitive match between a leaf task name and the requested task name.

        Requires an underscore word-boundary before any suffix match so that a
        short single-word leaf like "flores" does not falsely match
        "african_flores".  Additionally, single-word (no underscore) leaf names
        are only accepted on exact match to prevent spurious hits.
        """
        clean_leaf = leaf_name.split("/")[-1] if "/" in leaf_name else leaf_name
        clean_leaf_normalized = _normalize_name(clean_leaf)
        task_normalized = _normalize_name(task_name)
        if clean_leaf_normalized == task_normalized:
            return True
        # Single-word leaf names only match exactly (handled above).
        if "_" not in clean_leaf_normalized:
            return False
        # leaf ends with task_name (task is a suffix of leaf): require underscore boundary
        if clean_leaf_normalized.endswith(f"_{task_normalized}"):
            return True
        # task_name ends with leaf (leaf is a suffix of task): require underscore boundary
        # Only apply if the leaf itself is multi-word to avoid short-name false positives.
        if task_normalized.endswith(f"_{clean_leaf_normalized}"):
            return True
        return False

    def _try_parent(parent_name: str):
        """Load parent group and search for task_name among its leaf tasks.

        Returns (task_obj, parent_name) tuple on success, (None, None) on failure.
        """
        try:
            parent_obj = loader.load_lm_eval_task(parent_name)
        except Exception:
            return None, None
        if not isinstance(parent_obj, dict):
            return None, None
        for leaf_name, leaf_task in _flatten_task_dict(parent_obj):
            if _match(leaf_name):
                log.info(f"Found subtask '{task_name}' in parent group '{parent_name}'")
                return leaf_task, parent_name
        return None, None

    task_normalized = _normalize_name(task_name)

    # Strategy 0a: task_name is itself a GROUP key in GROUP_TASK_EXPANSIONS.
    # This handles cases where lm-eval registers the group under a different name
    # (e.g. 'afrimgsm-irokobench' instead of 'afrimgsm'), so direct loading of the
    # group fails.  Fall back to loading each known expansion subtask individually
    # and returning a synthetic dict so the caller can iterate over all subtasks.
    for group_key, expansion_subtasks in GROUP_TASK_EXPANSIONS.items():
        if _normalize_name(group_key) == task_normalized:
            log.info(
                f"'{task_name}' is a GROUP_TASK_EXPANSIONS key — "
                f"attempting to load {len(expansion_subtasks)} expansion subtasks individually"
            )
            synthetic_dict: dict = {}
            for subtask_name in expansion_subtasks:
                try:
                    subtask_obj = loader.load_lm_eval_task(subtask_name)
                    if isinstance(subtask_obj, ConfigurableTask):
                        synthetic_dict[subtask_name] = subtask_obj
                    elif isinstance(subtask_obj, dict):
                        synthetic_dict.update(subtask_obj)
                except Exception:
                    pass
            if synthetic_dict:
                log.info(
                    f"Built synthetic group dict with {len(synthetic_dict)} subtasks "
                    f"for '{task_name}'"
                )
                return synthetic_dict, group_key
            break  # found the group key but couldn't load any subtask

    # Strategy 0b: task_name is a subtask listed in GROUP_TASK_EXPANSIONS values.
    # Try loading its parent group and finding the subtask within.
    for parent_name, subtasks in GROUP_TASK_EXPANSIONS.items():
        # Check case-insensitively with dash/underscore normalization
        if any(_normalize_name(s) == task_normalized for s in subtasks):
            log.info(f"Found '{task_name}' in GROUP_TASK_EXPANSIONS under parent '{parent_name}'")
            result, parent = _try_parent(parent_name)
            if result is not None:
                return result, parent

    parts = task_name.split("_")

    # Strategy 1: progressively shorter plain prefixes
    for i in range(len(parts) - 1, 0, -1):
        parent_name = "_".join(parts[:i])
        result, parent = _try_parent(parent_name)
        if result is not None:
            return result, parent

    # Strategy 2: for tasks that end with a known variant suffix (e.g. '_light', '_with_pddl'),
    # also try parents formed by combining the base prefix with that suffix.
    # Example: 'arabic_leaderboard_acva_arabic_literature_light'
    #   suffix = 'light', base parts = ['arabic','leaderboard','acva','arabic','literature']
    #   -> try parents: 'arabic_leaderboard_acva_arabic_literature_light' (already tried above),
    #      'arabic_leaderboard_acva_arabic_light', 'arabic_leaderboard_acva_light', ...
    # Example: 'acp_app_gen_with_pddl'
    #   suffix = 'with_pddl', suffix_parts = ['with', 'pddl'], base_parts = ['acp','app','gen']
    #   -> try parents: 'acp_app_gen_with_pddl' (already tried above),
    #      'acp_app_with_pddl', 'acp_with_pddl', ...
    KNOWN_VARIANT_SUFFIXES = (("light",), ("with", "pddl"))
    for suffix_parts in KNOWN_VARIANT_SUFFIXES:
        suffix_str = "_".join(suffix_parts)
        if task_name.endswith(f"_{suffix_str}"):
            # Check if the task_name ends with the suffix tokens
            num_suffix_parts = len(suffix_parts)
            if len(parts) > num_suffix_parts and parts[-num_suffix_parts:] == list(suffix_parts):
                base_parts = parts[:-num_suffix_parts]  # strip the suffix tokens
                for i in range(len(base_parts) - 1, 0, -1):
                    parent_name = "_".join(base_parts[:i]) + f"_{suffix_str}"
                    result, parent = _try_parent(parent_name)
                    if result is not None:
                        return result, parent

    return None, None


def build_contrastive_pairs(
    task_name: str,
    limit: int | None = None,
    *,
    train_ratio: float,
) -> list["ContrastivePair"]:
    """
    Unified loader for contrastive pairs - handles both HuggingFace and lm-eval tasks.

    Loads from storage first (cache -> HF -> Supabase). If not found in any
    storage, generates via extractors and uploads to HF for future reuse.

    arguments:
        task_name:
            Name of the benchmark/task (e.g., "winogrande", "mmlu", "humaneval").
        limit:
            Optional upper bound on the number of pairs to return.
            Non-positive values are treated as no limit.

    returns:
        A list of ContrastivePair objects, each with metadata containing
        'evaluator_name' and 'source_task'.
    """
    log = bind(_LOG, task=task_name or "unknown")
    log.info("Building contrastive pairs (unified)", extra={"limit": limit})
    from wisent.core.utils.services.benchmarks import validate_benchmark
    validate_benchmark(task_name)
    
    # Normalize limit
    max_items = None if (limit is None or limit <= 0) else int(limit)
    
    # Try loading from storage first (cache -> HF -> Supabase)
    from wisent.extractors.lm_eval.registry.lm_task_pairs_storage import (
        try_load_from_storage, upload_pairs_to_hf,
    )
    stored_pairs = try_load_from_storage(task_name, max_items)
    if stored_pairs:
        extractor = get_extractor(task_name)
        evaluator_name = getattr(extractor, 'evaluator_name', None)
        return _add_evaluator_to_pairs(stored_pairs, evaluator_name, task_name)

    log.info("No stored pairs found, generating from extractors")

    # Get extractor
    extractor = get_extractor(task_name)
    log.info("Using extractor", extra={"extractor": extractor.__class__.__name__})

    # Get evaluator_name from extractor
    evaluator_name = getattr(extractor, 'evaluator_name', None)

    # HuggingFace extractor - load directly
    if isinstance(extractor, HuggingFaceBenchmarkExtractor):
        log.info("HuggingFace task - loading directly")
        pairs = extractor.extract_contrastive_pairs(limit=max_items)
        upload_pairs_to_hf(task_name, pairs)
        return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
    
    # lm-eval extractor - need to load task
    log.info("lm-eval task - loading via LMEvalDataLoader")
    from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
    from wisent.core.utils.infra_tools.data.loaders.lm_eval._lm_loader_task_mapping import (
        GROUP_TASK_EXPANSIONS,
    )

    loader = LMEvalDataLoader()

    # Check if task is a known GROUP with subtasks listed in GROUP_TASK_EXPANSIONS.
    # For such tasks, load subtasks LAZILY (one at a time) rather than all at once to
    # avoid the long initialisation time of get_task_dict with 50+ task names.
    def _normalize(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    task_normalized = _normalize(task_name)
    lazy_subtask_names: list[str] | None = None
    for group_key, expansion_subtasks in GROUP_TASK_EXPANSIONS.items():
        if _normalize(group_key) == task_normalized:
            # Filter out the group key itself (e.g. "advanced_ai_risk" is in its own expansion)
            lazy_subtask_names = [s for s in expansion_subtasks if _normalize(s) != task_normalized]
            log.info(
                f"Known GROUP task '{task_name}' with {len(lazy_subtask_names)} expansion subtasks "
                f"— will load lazily"
            )
            break

    from lm_eval.api.task import ConfigurableTask

    def _to_lm_eval_subtask_name(name: str) -> str:
        """Convert a GROUP_TASK_EXPANSIONS underscore subtask name to its lm-eval dash form.

        For advanced_ai_risk subtasks, lm-eval uses dashes after the variant prefix:
          advanced_ai_risk_fewshot_coordinate_itself  ->  advanced_ai_risk_fewshot-coordinate-itself
          advanced_ai_risk_human_corrigible_less_HHH  ->  advanced_ai_risk_human-corrigible-less-HHH
          advanced_ai_risk_lm_self_awareness_training_nn_architecture
              -> advanced_ai_risk_lm-self-awareness-training-nn-architecture

        Other task names are returned unchanged.
        """
        import re
        # Pattern: advanced_ai_risk_(fewshot|human|lm)_(rest)
        m = re.match(r'^(advanced_ai_risk_(?:fewshot|human|lm))_(.+)$', name)
        if m:
            prefix, rest = m.group(1), m.group(2)
            # Replace underscores with dashes in the suffix, but preserve uppercase HHH
            dash_rest = rest.replace("_", "-")
            return f"{prefix}-{dash_rest}"
        return name

    if lazy_subtask_names is not None and len(lazy_subtask_names) > 0:
        # Lazy group loading: load subtasks one-by-one and stop once we have enough pairs.
        # Convert underscore names (from GROUP_TASK_EXPANSIONS) to lm-eval dash names so
        # each subtask loads in ~5 s directly without triggering a full parent-group reload.
        lm_eval_subtask_names = [_to_lm_eval_subtask_name(s) for s in lazy_subtask_names]
        random.shuffle(lm_eval_subtask_names)
        pairs_per_task = max(1, max_items // len(lm_eval_subtask_names)) if max_items else None

        all_pairs: list["ContrastivePair"] = []
        for subtask_name in lm_eval_subtask_names:
            if max_items is not None and len(all_pairs) >= max_items:
                break
            try:
                subtask_obj = loader.load_lm_eval_task(subtask_name)
            except Exception as _e:
                log.warning(f"Could not load subtask '{subtask_name}': {_e}")
                continue

            # subtask_obj may itself be a ConfigurableTask or a dict
            if isinstance(subtask_obj, ConfigurableTask):
                leaf_pairs_list = [(subtask_name, subtask_obj)]
            elif isinstance(subtask_obj, dict):
                leaf_pairs_list = _flatten_task_dict(subtask_obj)
            else:
                log.warning(f"Unexpected subtask type for '{subtask_name}': {type(subtask_obj)}")
                continue

            for leaf_name_full, leaf_task in leaf_pairs_list:
                if max_items is not None and len(all_pairs) >= max_items:
                    break
                leaf_name = leaf_name_full.split("/")[-1] if "/" in leaf_name_full else leaf_name_full
                try:
                    leaf_extractor = get_extractor(leaf_name)
                except Exception:
                    leaf_extractor = extractor
                leaf_evaluator = getattr(leaf_extractor, 'evaluator_name', evaluator_name)
                try:
                    if isinstance(leaf_extractor, HuggingFaceBenchmarkExtractor):
                        leaf_pairs = leaf_extractor.extract_contrastive_pairs(limit=pairs_per_task)
                    else:
                        leaf_pairs = leaf_extractor.extract_contrastive_pairs(
                            leaf_task, limit=pairs_per_task, train_ratio=train_ratio
                        )
                    leaf_pairs = _add_evaluator_to_pairs(leaf_pairs, leaf_evaluator, leaf_name_full)
                    all_pairs.extend(leaf_pairs)
                except Exception as e:
                    log.warning(f"Failed to extract from subtask '{leaf_name_full}': {e}")

        random.shuffle(all_pairs)
        if max_items is not None:
            all_pairs = all_pairs[:max_items]
        log.info(f"Extracted {len(all_pairs)} pairs from lazy group task")
        upload_pairs_to_hf(task_name, all_pairs)
        return all_pairs

    try:
        task_obj = loader.load_lm_eval_task(task_name)
    except Exception:
        # Subtask not loadable directly — try loading parent group and finding subtask
        task_obj, _ = _load_subtask_from_parent(task_name, loader, log)
        if task_obj is None:
            # Last resort: some extractors (storycloze, multipl_e) can produce pairs
            # without an lm-eval task object. Try the extractor directly.
            try:
                pairs = extractor.extract_contrastive_pairs(None, limit=max_items, train_ratio=train_ratio)
            except TypeError:
                try:
                    pairs = extractor.extract_contrastive_pairs(limit=max_items)
                except Exception:
                    raise
            if pairs:
                upload_pairs_to_hf(task_name, pairs)
                return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
            raise

    # If the loader returned a dict but the caller asked for a specific leaf,
    # narrow down to that leaf so we do not aggregate unrelated subtasks.
    if isinstance(task_obj, dict) and task_name in task_obj:
        task_obj = task_obj[task_name]

    # Single task (ConfigurableTask)
    if isinstance(task_obj, ConfigurableTask):
        log.info("Single task")
        pairs = extractor.extract_contrastive_pairs(task_obj, limit=max_items, train_ratio=train_ratio)
        upload_pairs_to_hf(task_name, pairs)
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
                except Exception:
                    # Fall back to parent extractor
                    subtask_extractor = extractor

                subtask_evaluator = getattr(subtask_extractor, 'evaluator_name', evaluator_name)

                subtask_pairs = subtask_extractor.extract_contrastive_pairs(subtask, limit=pairs_per_task, train_ratio=train_ratio)
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
        upload_pairs_to_hf(task_name, all_pairs)
        return all_pairs

    log.error(f"Unexpected task_obj type: {type(task_obj)}")
    return []


def lm_build_contrastive_pairs(
    task_name: str,
    lm_eval_task: "ConfigurableTask | None",
    limit: int | None = None,
    *,
    train_ratio: float,
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
        pairs = extractor.extract_contrastive_pairs(lm_eval_task, limit=max_items, train_ratio=train_ratio)

    return _add_evaluator_to_pairs(pairs, evaluator_name, task_name)
