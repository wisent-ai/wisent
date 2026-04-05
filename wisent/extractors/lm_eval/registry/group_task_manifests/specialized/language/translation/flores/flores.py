"""Flores group task manifest."""

from __future__ import annotations

from .flores_languages_a_to_e import FLORES_TASKS_A_TO_E
from .flores_languages_f_to_z import FLORES_TASKS_F_TO_Z

FLORES_TASKS = {**FLORES_TASKS_A_TO_E, **FLORES_TASKS_F_TO_Z}
