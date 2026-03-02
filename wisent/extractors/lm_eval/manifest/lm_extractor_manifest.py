"""LM-eval extractor manifest - maps task names to extractor classes."""

from wisent.extractors.lm_eval.group_task_manifests import (
    get_all_group_task_mappings,
)
from wisent.extractors.lm_eval.lm_extractor_manifest_part1 import LM_EXTRACTORS_PART1
from wisent.extractors.lm_eval.lm_extractor_manifest_part2 import LM_EXTRACTORS_PART2

__all__ = [
    "EXTRACTORS",
]

# Start with group task mappings
EXTRACTORS: dict[str, str] = get_all_group_task_mappings()

# Add individual task extractors from both parts
EXTRACTORS.update(LM_EXTRACTORS_PART1)
EXTRACTORS.update(LM_EXTRACTORS_PART2)
