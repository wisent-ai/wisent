from wisent.core.synthetic.cleaners.core.atoms import CleanStep
from wisent.core.synthetic.cleaners.core.atoms import CleanStepStats

from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.synthetic.cleaners.methods.core.atoms import Deduper


__all__ = [
    "DeduperCleaner",
]

class DeduperCleaner(CleanStep):
    """
    Deduplication step; removes duplicate items from the pipeline.

    attributes:
        deduper:
            Deduper instance to use for deduplication.
    """
    name = "deduper_cleaner"

    def __init__(self, deduper: Deduper) -> None:
        self._deduper = deduper
        self._last_stats = 0
        self._last_total = 0
    
    def stats(self) -> CleanStepStats:
        '''
        Return statistics about the last run of 'apply()'.
        
        returns:
            CleanStepStats with total and removed items from the last deduplication run.
        '''
        return CleanStepStats(
            total_items=self._last_total,
            removed_items=self._last_stats,
        )

    def apply(self, items: ContrastivePairSet) -> ContrastivePairSet:
        '''
        Apply the deduplication step to the given ContrastivePairSet.
        
        arguments:
            items:
                ContrastivePairSet to deduplicate.
        
        returns:
            Deduplicated ContrastivePairSet.
        '''
        self._last_total = len(items)
        dedupe_items = self._deduper.dedupe(items)
        self._last_stats = self._last_total - len(dedupe_items)
        return dedupe_items