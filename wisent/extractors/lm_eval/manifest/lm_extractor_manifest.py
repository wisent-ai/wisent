"""LM-eval extractor manifest - maps task names to extractor classes."""

from wisent.extractors.lm_eval.registry.group_task_manifests import (
    get_all_group_task_mappings,
)
from wisent.extractors.lm_eval.lm_extractor_manifest_a_to_o import LM_EXTRACTORS_A_TO_O
from wisent.extractors.lm_eval.lm_extractor_manifest_p_to_z import LM_EXTRACTORS_P_TO_Z

__all__ = [
    "EXTRACTORS",
]

# Start with group task mappings
EXTRACTORS: dict[str, str] = get_all_group_task_mappings()

# Add individual task extractors from both parts
EXTRACTORS.update(LM_EXTRACTORS_A_TO_O)
EXTRACTORS.update(LM_EXTRACTORS_P_TO_Z)
