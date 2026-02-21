"""Group task manifests for LM Eval benchmarks with multiple subtasks."""

from __future__ import annotations

from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests._group_tasks_part1 import (
    get_part1_mappings,
)
from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests._group_tasks_part1 import *  # noqa: F401,F403
from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests._group_tasks_part2 import (
    get_part2_mappings,
)
from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests._group_tasks_part2 import *  # noqa: F401,F403


def get_all_group_task_mappings() -> dict[str, str]:
    """
    Get all group task to extractor mappings.

    Returns:
        Dictionary mapping task names to extractor module paths.
    """
    all_mappings = {}
    all_mappings.update(get_part1_mappings())
    all_mappings.update(get_part2_mappings())
    return all_mappings
