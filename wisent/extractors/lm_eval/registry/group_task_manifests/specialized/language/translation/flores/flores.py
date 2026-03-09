"""Flores group task manifest."""

from __future__ import annotations

from wisent.extractors.lm_eval.group_task_manifests.flores_languages_a_to_e import FLORES_TASKS_A_TO_E
from wisent.extractors.lm_eval.group_task_manifests.flores_languages_f_to_z import FLORES_TASKS_F_TO_Z

FLORES_TASKS = {**FLORES_TASKS_A_TO_E, **FLORES_TASKS_F_TO_Z}
