"""HuggingFace extractor manifest - maps benchmark names to extractor classes."""

from wisent.extractors.hf.hf_extractor_manifest_part1 import EXTRACTORS_PART1
from wisent.extractors.hf.hf_extractor_manifest_part2 import EXTRACTORS_PART2

__all__ = [
    "EXTRACTORS",
    "HF_EXTRACTORS",
]

EXTRACTORS: dict[str, str] = {**EXTRACTORS_PART1, **EXTRACTORS_PART2}

# Alias for backwards compatibility
HF_EXTRACTORS = EXTRACTORS
