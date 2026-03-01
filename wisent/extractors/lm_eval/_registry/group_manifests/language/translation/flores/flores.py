"""Flores group task manifest."""

from __future__ import annotations

from wisent.extractors.lm_eval._registry.group_manifests.flores_part1 import FLORES_TASKS_PART1
from wisent.extractors.lm_eval._registry.group_manifests.flores_part2 import FLORES_TASKS_PART2

FLORES_TASKS = {**FLORES_TASKS_PART1, **FLORES_TASKS_PART2}
