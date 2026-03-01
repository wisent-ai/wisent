from __future__ import annotations

from wisent.extractors.hf.humaneval import HumanEvalExtractor

__all__ = ["MultiplePyExtractor"]


class MultiplePyExtractor(HumanEvalExtractor):
    """
    Extractor for MultiPL-E Python benchmark.
    
    This is just an alias for HumanEval since MultiPL-E Python
    is the original HumanEval dataset.
    """
    pass
