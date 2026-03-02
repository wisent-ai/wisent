"""Public interface for contrastive pair utilities."""

from wisent.core.pair import ContrastivePair
from wisent.core.set import ContrastivePairSet
from wisent.core.buliders import from_phrase_pairs
from wisent.core.reading.diagnostics import DiagnosticsConfig, DiagnosticsReport, run_all_diagnostics

__all__ = [
    "ContrastivePair",
    "ContrastivePairSet",
    "from_phrase_pairs",
    "DiagnosticsConfig",
    "DiagnosticsReport",
    "run_all_diagnostics",
]