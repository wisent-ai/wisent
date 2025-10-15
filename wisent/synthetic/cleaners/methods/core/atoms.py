
from abc import ABC, abstractmethod

from wisent.core.models.wisent_model import WisentModel 
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    
class Refusaler(ABC):
    """
    Refusal detection and fixing step.

    methods:
        looks_like_refusal(text: str) -> bool:
            Detect if the text looks like a refusal. For example, we want to generate negatives that exhibit
            "evil" behavior, but not refusals like "As an AI model, I cannot help with that."
        fix_negative(
            model: WisentModel,
            prompt: str,
            trait_label: str,
            trait_description: str,
            system_prompt: str,
        ) -> str:
            Attempt to fix a refusal negative example by re-prompting the model with the given system prompt. For example,
            we can increase the temperature or change the wording to try to get a non-refusal response.
"""
    @abstractmethod
    def looks_like_refusal(self, text: str) -> bool: ...
    @abstractmethod
    def fix_negative(
        self,
        model: WisentModel,
        prompt: str,
        trait_label: str,
        trait_description: str,
        system_prompt: str,
    ) -> str: ...

class Deduper(ABC):
    """
    Deduplication step; removes duplicate items from the pipeline.
    methods:
        dedupe(items: ContrastivePairSet) -> ContrastivePairSet:
            Remove duplicate items from the given ContrastivePairSet.
    """

    @abstractmethod
    def dedupe(self, items: ContrastivePairSet) -> ContrastivePairSet: ...