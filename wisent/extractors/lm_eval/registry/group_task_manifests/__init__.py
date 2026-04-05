"""Group task manifests for LM Eval benchmarks with multiple subtasks."""

from __future__ import annotations

import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

from wisent.extractors.lm_eval.group_task_manifests._group_tasks_a_to_i import (
    get_a_to_i_mappings,
)
from wisent.extractors.lm_eval.group_task_manifests._group_tasks_a_to_i import *  # noqa: F401,F403
from wisent.extractors.lm_eval.group_task_manifests._group_tasks_j_to_z import (
    get_j_to_z_mappings,
)
from wisent.extractors.lm_eval.group_task_manifests._group_tasks_j_to_z import *  # noqa: F401,F403


def get_all_group_task_mappings() -> dict[str, str]:
    """
    Get all group task to extractor mappings.

    Returns:
        Dictionary mapping task names to extractor module paths.
    """
    all_mappings = {}
    all_mappings.update(get_a_to_i_mappings())
    all_mappings.update(get_j_to_z_mappings())
    return all_mappings
