from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from wisent.core.synthetic.cleaners.core.atoms import CleanStep, Cleaner
from wisent.core.synthetic.cleaners.core.atoms import CleanerStats
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

__all__ = [
    "PairsCleaner",
]

class PairsCleaner(Cleaner):
    """
    Composable cleaner; pass any sequence of CleanStep.
    
    attributes:
        steps:
            Iterable of CleanStep instances to apply in order.
    """

    def __init__(self, steps: Iterable[CleanStep]) -> None:
        self._steps = list(steps)

    def clean(
        self, items: ContrastivePairSet
    ) -> tuple[ContrastivePairSet, CleanerStats]:
        """
        Apply the cleaning pipeline to the given ContrastivePairSet.
        
        arguments:
            items:
                ContrastivePairSet to clean.
        
        returns:
            Tuple of cleaned ContrastivePairSet and CleanerStats with statistics about the cleaning process.
        
        example:
            >>> from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
            >>> from wisent.core.contrastive_pairs.core.pair import ContrastivePair
            >>> from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
            >>> from wisent.core.synthetic.cleaners.methods.base_refusalers import BasesRefusaler
            >>> from wisent.core.synthetic.cleaners.methods.base_dedupers import SimHashDeduper
            >>> from wisent.core.synthetic.cleaners.cleaners import PairsCleaner
            >>> from wisent.core.models.wisent_model import WisentModel
            >>> refusal = BasesRefusaler()
            >>> deduper = SimHashDeduper()
            >>> model = WisentModel(model_name="llama3.1")
            >>> cleaner = PairsCleaner(steps=[
            ...     RefusalerCleaner(
            ...         refusal=refusal,
            ...         model=model,
            ...         system_prompt="You are a helpful assistant that always answers the question truthfully.",
            ...         trait_label="honesty",
            ...         trait_description="honest vs dishonest",
            ...         max_retries=2,
            ...     ),
            ...     DeduperCleaner(deduper=deduper),
            ... ])
            >>> items = ContrastivePairSet(pairs=[
            ...     ContrastivePair(
            ...         prompt="What is the capital of France?",
            ...         positive=PositiveResponse(text="The capital of France is Paris."),
            ...         negative=NegativeResponse(text="As an AI language model, I cannot provide that information."),
            ...     ),
            ...     ContrastivePair(
            ...         prompt="What is the capital of France?",
            ...         positive=PositiveResponse(text="The capital of France is Paris."),
            ...         negative=NegativeResponse(text="I don't know."),
            ...     ),
            ... ])
            >>> cleaned_items, stats = cleaner.clean(items)
            >>> print(len(cleaned_items))
            1
            >>> print(stats.step_stats)
            {'refusaler_cleaner': CleanStepStats(modified_items=1), 'deduper_cleaner': CleanStepStats(total_items=1, removed_items=0)}
            >>> print(cleaned_items.pairs[0].negative.text)
            The capital of France is England.
            >>> print(cleaned_items.pairs[0].positive.text)
            The capital of France is Paris.
            >>> print(cleaned_items.pairs[0].prompt.text)
            What is the capital of France?
        """
       
        cur = items
        stats = CleanerStats()
        for st in self._steps:
            cur = st.apply(cur)
            stats.step_stats[st.name] = st.stats()
        return cur, stats
