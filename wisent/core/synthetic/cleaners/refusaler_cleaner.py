
from wisent.core.synthetic.cleaners.core.atoms import CleanStep
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.synthetic.cleaners.core.atoms import CleanStepStats

from wisent.core.synthetic.cleaners.methods.core.atoms import Refusaler
from wisent.core.models.wisent_model import WisentModel
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse

__all__ = [
    "RefusalerCleaner",
]

class RefusalerCleaner(CleanStep):
    """
    Refusal detection and fixing step.
    """
    name = "refusaler_cleaner"

    def __init__(
        self,
        refusal: Refusaler,
        model: WisentModel,
        system_prompt: str,
        trait_label: str,
        trait_description: str,
        max_retries: int = 2,
    ) -> None:
        self._refusal = refusal
        self._model = model
        self._sys = system_prompt
        self._label = trait_label
        self._desc = trait_description
        self._max_retries = max_retries
        self._retries_used = 0

    def stats(self) -> CleanStepStats:
        '''
        Return statistics about the last run of 'apply()'.
        
        returns:
            CleanStepStats with the number of retries used in the last run.
        '''
        return CleanStepStats(modified_items=self._retries_used)
        
    def apply(self, items: ContrastivePairSet) -> ContrastivePairSet:
        """
        Apply the refusal detection and fixing step to the given ContrastivePairSet.

        arguments:
            items:
                ContrastivePairSet to clean.
        
        returns:
            Cleaned ContrastivePairSet with refusals fixed.

        example:
            >>> from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
            >>> from wisent.core.contrastive_pairs.core.pair import ContrastivePair
            >>> from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
            >>> from wisent.core.synthetic.cleaners.methods.base_refusalers import SimpleRefusaler
            >>> from wisent.core.models.wisent_model import WisentModel
            >>> refusal = SimpleRefusaler()
            >>> model = WisentModel(...)
            >>> cleaner = RefusalerCleaner(
            ...     refusal=refusal,
            ...     model=model,
            ...     system_prompt="You are a helpful assistant.",
            ...     trait_label="honesty",
            ...     trait_description="honest vs dishonest",
            ...     max_retries=2,
            ... )
            >>> items = ContrastivePairSet(
            ...     name="example",
            ...     task_type="test",
            ...     pairs=[
            ...         ContrastivePair(
            ...             prompt="Is the sky blue?",
            ...             positive_response=PositiveResponse(
            ...                 model_response="Yes, the sky is blue.",
            ...                 layers_activations=None,
            ...                 label="harmless"
            ...             ),
            ...             negative_response=NegativeResponse(
            ...                 model_response="I'm sorry, I can't help with that.",
            ...                 layers_activations=None,
            ...                 label="toxic"
            ...             ),
            ...             label="color_question",
            ...             trait_description="hallucinatory"
            ...         )
            ...     ]
            ... )
            >>> cleaned = cleaner.apply(items)
            >>> for cp in cleaned.pairs:
            ...     print(cp)
            ContrastivePair(
                prompt='Is the sky blue?',
                positive_response=PositiveResponse(model_response='Yes, the sky is blue.', layers_activations=None, label='harmless'),
                negative_response=NegativeResponse(model_response='No, the sky is not blue.', layers_activations=None, label='toxic'),
                label='color_question',
                trait_description='hallucinatory'
            )       
            """
        out: ContrastivePairSet = ContrastivePairSet(
            name=items.name,
            task_type=items.task_type,
        )
        retries = 0
        for cp in items.pairs:
            neg = cp.negative_response.model_response
            if self._refusal.looks_like_refusal(neg) and retries < self._max_retries:
                fixed = self._refusal.fix_negative(
                    self._model,
                    prompt=cp.prompt,
                    trait_label=self._label,
                    trait_description=self._desc,
                    system_prompt=self._sys,
                )
                if fixed:
                    neg = fixed
                    retries += 1
            clean_contrastive_pair = ContrastivePair(
                prompt=cp.prompt,
                positive_response=PositiveResponse(model_response=cp.positive_response.model_response),
                negative_response=NegativeResponse(model_response=neg),
                label=cp.label,
                trait_description=cp.trait_description,
            )
            out.pairs.append(clean_contrastive_pair)
        self._retries_used += retries
        return out