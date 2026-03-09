"""HuggingFace extractor manifest - maps benchmark names to extractor classes."""

from wisent.extractors.hf.hf_extractor_manifest_math_coding import EXTRACTORS_MATH_CODING
from wisent.extractors.hf.hf_extractor_manifest_safety_multilingual import EXTRACTORS_SAFETY_MULTILINGUAL

__all__ = [
    "EXTRACTORS",
    "HF_EXTRACTORS",
]

EXTRACTORS: dict[str, str] = {**EXTRACTORS_MATH_CODING, **EXTRACTORS_SAFETY_MULTILINGUAL}

# Alias for backwards compatibility
HF_EXTRACTORS = EXTRACTORS
