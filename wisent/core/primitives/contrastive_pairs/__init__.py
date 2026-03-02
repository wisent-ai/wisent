"""Contrastive pair types: ContrastivePair, ContrastivePairSet, responses."""
from wisent.core.contrastive_pairs.pair import ContrastivePair
from wisent.core.contrastive_pairs.set import ContrastivePairSet
from wisent.core.contrastive_pairs.io.response import PositiveResponse, NegativeResponse
from wisent.core.contrastive_pairs.buliders import from_phrase_pairs

__all__ = [
    "ContrastivePair",
    "ContrastivePairSet",
    "PositiveResponse",
    "NegativeResponse",
    "from_phrase_pairs",
]
