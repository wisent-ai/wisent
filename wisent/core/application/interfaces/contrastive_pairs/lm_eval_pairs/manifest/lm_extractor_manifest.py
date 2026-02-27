"""LM-eval extractor manifest - maps task names to extractor classes."""

from wisent.core.contrastive_pairs.lm_eval_pairs.group_task_manifests import (
    get_all_group_task_mappings,
)
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_manifest_part1 import LM_EXTRACTORS_PART1
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_manifest_part2 import LM_EXTRACTORS_PART2

__all__ = [
    "EXTRACTORS",
]

# Start with group task mappings
EXTRACTORS: dict[str, str] = get_all_group_task_mappings()

# Add individual task extractors from both parts
EXTRACTORS.update(LM_EXTRACTORS_PART1)
EXTRACTORS.update(LM_EXTRACTORS_PART2)
