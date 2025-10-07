from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod, dataclass
from dataclasses import field

if TYPE_CHECKING:
    from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet

@dataclass(frozen=True)
class CleanStepStats:
    total_items: int = 0
    removed_items: int = 0
    modified_items: int = 0

@dataclass(frozen=True)
class CleanerStats:
    step_stats: dict[str, CleanStepStats] = field(default_factory=dict)


class CleanStep(ABC):
    """
    Single step in a cleaning pipeline.

    attributes:
        name:
            Name of the step, used in stats and logging.

    methods:
        stats():
            Return a dict of statistics about the last run of `apply()`.
        apply(items):
            Apply the cleaning step to a list of items. 
    """
    name: str = "step"

    def stats(self) -> CleanStepStats:
        return CleanStepStats()

    @abstractmethod
    def apply(self, items: ContrastivePairSet) -> ContrastivePairSet:
        ...

class Cleaner(ABC):
    """
    Cleaning pipeline composed of multiple `CleanStep`s.
    """
    @abstractmethod
    def clean(
        self, items:ContrastivePairSet
    ) -> tuple[ContrastivePairSet, CleanerStats]:
        ...
