"""Flores group task manifest."""

from __future__ import annotations

from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests.flores_part1 import FLORES_TASKS_PART1
from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests.flores_part2 import FLORES_TASKS_PART2

FLORES_TASKS = {**FLORES_TASKS_PART1, **FLORES_TASKS_PART2}
