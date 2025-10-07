from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

from wisent_guard.synthetic.cleaners.core.atoms import CleanStep, Cleaner

if TYPE_CHECKING:
    from wisent_guard.synthetic.generators.core.atoms import (
        CompletionFn,
        RefusalPolicy,
        Deduper,
    )

class RefusalerCleaner(CleanStep):
    """
    Fix negatives that look like refusals by re-prompting the model.
    
    attributes:
        refusal:
            RefusalPolicy instance to use for detection and fixing.
        completion_fn:
            Function to call the model.
        system_prompt:
            System prompt to use when re-prompting.
        trait_label:
            Trait label to include in the re-prompt. E.g., "helpfulness".
        trait_description:
            Trait description to include in the re-prompt. E.g., "You should be as helpful as possible."
        max_retries:
            Maximum number of re-prompt attempts per negative.
    """
    name = "refusaler_cleaner"

    def __init__(
        self,
        refusal: RefusalPolicy,
        completion_fn: CompletionFn,
        system_prompt: str,
        trait_label: str,
        trait_description: str,
        max_retries: int = 2,
    ) -> None:
        self._refusal = refusal
        self._completion_fn = completion_fn
        self._sys = system_prompt
        self._label = trait_label
        self._desc = trait_description
        self._max_retries = max_retries
        self._retries_used = 0

    def stats(self) -> dict[str, int | float | str]:
        """
        Return stats about the last run.
        
        returns:
            Dict with key "retries" indicating the number of re-prompts used in the last run.
            
        example:
            >>> refusal = DefaultRefusalPolicy()
            >>> step = RefusalFix(
            ...     refusal=refusal,
            ...     completion_fn=dummy_completion_fn,
            ...     system_prompt="You are a helpful assistant.",
            ...     trait_label="helpfulness",
            ...     trait_description="You should be as helpful as possible.",
            ...     max_retries=2,
            ... )
            >>> items = [
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "What is 2+2?",
            ...         "positive": "2+2 is 4.",
            ...         "negative": "As an AI, I don't have opinions.",
            ...     },
            ... ]
            >>> cleaned = step.apply(items)
            >>> stats = step.stats()
            >>> print(stats)
            {'retries': 2}
        """
        return {"retries": self._retries_used}

    def apply(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Apply the cleaning step to the given items.

        arguments:
            items:
                List of dicts with keys "prompt", "positive", "negative".
        
        returns:
            List of dicts with the same keys, but with negatives fixed if they looked like refusals.

        example:
            >>> refusal = DefaultRefusalPolicy()
            >>> step = RefusalFix(
            ...     refusal=refusal,
            ...     completion_fn=dummy_completion_fn,
            ...     system_prompt="You are a helpful assistant.",
            ...     trait_label="helpfulness",
            ...     trait_description="You should be as helpful as possible.",
            ...     max_retries=2,
            ... )
            >>> items = [
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "What is 2+2?",
            ...         "positive": "2+2 is 4.",
            ...         "negative": "As an AI, I don't have opinions.",
            ...     },
            ... ]
            >>> cleaned = step.apply(items)
            >>> for it in cleaned:
            ...     print(it)
            {'prompt': 'Tell me a joke.', 'positive': 'Sure! Why did the chicken cross the road? To get to the other side!', 'negative': 'Why did the scarecrow win an award? Because he was outstanding in his field!'}
            {'prompt': 'What is 2+2?', 'positive': '2+2 is 4.', 'negative': '2+2 is 5.'}
        """
        out: list[dict[str, str]] = []
        retries = 0
        for it in items:
            neg = it["negative"]
            if self._refusal.looks_like_refusal(neg) and retries < self._max_retries:
                fixed = self._refusal.fix_negative(
                    self._completion_fn,
                    prompt=it["prompt"],
                    trait_label=self._label,
                    trait_description=self._desc,
                    system_prompt=self._sys,
                )
                if fixed:
                    neg = fixed
                    retries += 1
            out.append(
                {"prompt": it["prompt"], "positive": it["positive"], "negative": neg}
            )
        self._retries_used += retries
        return out


class Dedupe(CleanStep):
    """
    Deduplication step; removes duplicate items from the pipeline.

    attributes:
        deduper:
            Deduper instance to use for deduplication.
    """
    name = "dedupe"

    def __init__(self, deduper: Deduper) -> None:
        self._deduper = deduper

    def apply(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        """
        Apply the deduplication step to the given items.

        arguments:
            items:
                List of dicts with keys "prompt", "positive", "negative".

        returns:
            List of dicts with the same keys, but with duplicates removed.

        example:
            >>> deduper = SimHashDeduper(threshold_bits=3)
            >>> step = Dedupe(deduper)
            >>> items = [
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "What is 2+2?",
            ...         "positive": "2+2 is 4.",
            ...         "negative": "As an AI, I don't have opinions.",
            ...     },
            ... ]
            >>> cleaned = step.apply(items)
            >>> for it in cleaned:
            ...     print(it)
            {'prompt': 'Tell me a joke.', 'positive': 'Sure! Why did the chicken cross the road? To get to the other side!', 'negative': "I'm sorry, I can't tell jokes."}
            {'prompt': 'What is 2+2?', 'positive': '2+2 is 4.', 'negative': "As an AI, I don't have opinions."}
        """
        return self._deduper.dedupe(items)


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
        self, items: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], dict[str, int | float | str]]:
        """
        Clean the given items using the defined steps.

        arguments:
            items:
                List of dicts with keys "prompt", "positive", "negative".

        returns:
            A tuple containing the cleaned items and a dictionary of statistics.

        example:
            >>> refusal = DefaultRefusalPolicy()
            >>> deduper = SimHashDeduper(threshold_bits=3)
            >>> completion_fn = dummy_completion_fn  # Replace with actual model call
            >>> steps = [
            ...     MetaStrip(refusal),
            ...     RefusalFix(
            ...         refusal=refusal,
            ...         completion_fn=completion_fn,
            ...         system_prompt="You are a helpful assistant.",
            ...         trait_label="helpfulness",
            ...         trait_description="You should be as helpful as possible.",
            ...         max_retries=2,
            ...     ),
            ...     Dedupe(deduper),
            ... ]
            >>> cleaner = PairsCleaner(steps)
            >>> items = [
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "What is 2+2?",
            ...         "positive": "2+2 is 4.",
            ...         "negative": "As an AI, I don't have opinions.",
            ...     },
            ... ]
            >>> cleaned, stats = cleaner.clean(items)
            >>> for it in cleaned:
            ...     print(it)
            {'prompt': 'Tell me a joke.', 'positive': 'Sure! Why did the chicken cross the road? To get to the other side!', 'negative': 'Why did the scarecrow win an award? Because he was outstanding in his field!'}
            {'prompt': 'What is 2+2?', 'positive': '2+2 is 4.', 'negative': '2+2 is 5.'}
            >>> print(stats)
            {'meta_strip.retries': 0, 'refusal_fix.retries': 2, 'dedupe.retries': 0}
        """
        cur = items
        stats: dict[str, int | float | str] = {}
        for st in self._steps:
            cur = st.apply(cur)
            for k, v in (st.stats() or {}).items():
                stats[f"{st.name}.{k}"] = v
        return cur, stats

    @staticmethod
    def default(
        refusal: RefusalPolicy,
        completion_fn: CompletionFn,
        deduper: Deduper,
        roleplay_neg_fix_system_prompt: str,
        trait_label: str,
        trait_description: str,
        max_refusal_retries: int = 2,
    ) -> PairsCleaner:
        """
        Create a default PairsCleaner with the given components.

        arguments:
            refusal:
                RefusalPolicy instance to use for stripping and fixing.
            completion_fn:
                Function to call the model.
            deduper:
                Deduper instance to use for deduplication.
            roleplay_neg_fix_system_prompt:
                System prompt to use when re-prompting.
            trait_label:
                Trait label to include in the re-prompt. E.g., "helpfulness".
            trait_description:
                Trait description to include in the re-prompt. E.g., "You should be as helpful as possible."
            max_refusal_retries:
                Maximum number of re-prompt attempts per negative.

        returns:
            A PairsCleaner instance with the default steps.

        example:
            >>> refusal = DefaultRefusalPolicy()
            >>> deduper = SimHashDeduper(threshold_bits=3)
            >>> completion_fn = dummy_completion_fn  # Replace with actual model call
            >>> cleaner = PairsCleaner.default(
            ...     refusal=refusal,
            ...     completion_fn=completion_fn,
            ...     deduper=deduper,
            ...     roleplay_neg_fix_system_prompt="You are a helpful assistant.",
            ...     trait_label="helpfulness",
            ...     trait_description="You should be as helpful as possible.",
            ...     max_refusal_retries=2,
            ... )
            >>> items = [
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "Tell me a joke.",
            ...         "positive": "Sure! Why did the chicken cross the road? To get to the other side!",
            ...         "negative": "I'm sorry, I can't tell jokes.",
            ...     },
            ...     {
            ...         "prompt": "What is 2+2?",
            ...         "positive": "2+2 is 4.",
            ...         "negative": "As an AI, I don't have opinions.",
            ...     },
            ... ]
            >>> cleaned, stats = cleaner.clean(items)
            >>> for it in cleaned:
            ...     print(it)
            {'prompt': 'Tell me a joke.', 'positive': 'Sure! Why did the chicken cross the road? To get to the other side!', 'negative': 'Why did the scarecrow win an award? Because he was outstanding in his field!'}
            {'prompt': 'What is 2+2?', 'positive': '2+2 is 4.', 'negative': '2+2 is 5.'}
            >>> print(stats)
            {'meta_strip.retries': 0, 'refusal_fix.retries': 2, 'dedupe.retries': 0} 
        """
        return PairsCleaner(
            steps=[
                MetaStrip(refusal),
                RefusalFix(
                    refusal=refusal,
                    completion_fn=completion_fn,
                    system_prompt=roleplay_neg_fix_system_prompt,
                    trait_label=trait_label,
                    trait_description=trait_description,
                    max_retries=max_refusal_retries,
                ),
                Dedupe(deduper),
            ]
        )
