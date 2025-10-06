from __future__ import annotations

import logging


from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair  # type: ignore
from wisent_guard.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse  # type: ignore
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet  # type: ignore

from .core.atoms import (
    ChatMessage,
    CompletionFn,
    GenerationReport,
    DefaultPrompts,
    DefaultRefusalPolicy,
    SimHashDeduper,
    FastDiversity,
    parse_pairs_json,
)
from wisent_guard.synthetic.cleaners.pipeline import PairsCleaner

logger = logging.getLogger(__name__)

def model_adapter(model: object, **gen_kwargs) -> CompletionFn:
    def _call(messages: list[ChatMessage]) -> str:
        out = model.generate([messages], use_steering=False, **gen_kwargs)  # type: ignore[attr-defined]
        return out[0] if out else ""
    return _call


class PairsGenerator:
    """Small, fast contrastive-pairs generator with an extensible cleaning pipeline."""

    def __init__(
        self,
        completion_fn: CompletionFn,
        *,
        prompts: DefaultPrompts | None = None,
        cleaner: PairsCleaner | None = None,
        max_refusal_retries: int = 2,
    ) -> None:
        self.completion_fn = completion_fn
        self.prompts = prompts or DefaultPrompts()
        self.refusal = DefaultRefusalPolicy()
        self.deduper = SimHashDeduper(threshold_bits=3)
        self.diversity = FastDiversity(seed=13)
        self._max_refusal_retries = max_refusal_retries

        # default cleaner stack: meta-strip -> refusal-fix -> dedupe
        self.cleaner = cleaner or PairsCleaner.default(
            refusal=self.refusal,
            completion_fn=self.completion_fn,
            deduper=self.deduper,
            roleplay_neg_fix_system_prompt=self.prompts.get("roleplay_neg_fix"),
            trait_label="honesty",          # placeholder; overridden per call
            trait_description="honest vs dishonest",
            max_refusal_retries=max_refusal_retries,
        )

    def generate(
        self,
        name: str,
        topic: str,
        prescription: str,
        trait_label: str = "honesty",
        trait_description: str = "honest vs dishonest",
        num_pairs: int = 10,
    ) -> tuple[ContrastivePairSet, GenerationReport]:
        # 1) prompt → JSON
        sys = self.prompts.get("generic_pairs")
        usr = self._build_user_prompt(
            topic, prescription, trait_label, trait_description, num_pairs
        )
        raw = self.completion_fn(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
        )

        # 2) parse
        parsed = parse_pairs_json(raw)

        # 3) clean (meta-strip → refusal-fix → dedupe); rebuild default with current trait info
        if isinstance(self.cleaner, PairsCleaner) and len(getattr(self.cleaner, "_steps", [])) == 3:
            self.cleaner = PairsCleaner.default(
                refusal=self.refusal,
                completion_fn=self.completion_fn,
                deduper=self.deduper,
                roleplay_neg_fix_system_prompt=self.prompts.get("roleplay_neg_fix"),
                trait_label=trait_label,
                trait_description=trait_description,
                max_refusal_retries=self._max_refusal_retries,
            )
        cleaned, stats = self.cleaner.clean(parsed)
        retries = int(stats.get("refusal_fix.retries", 0))

        # 4) build domain objects
        cps = ContrastivePairSet(name=name, task_type=trait_label)
        for item in cleaned:
            cps.add(
                ContrastivePair(
                    prompt=item["prompt"],
                    positive_response=PositiveResponse(model_response=item["positive"]),
                    negative_response=NegativeResponse(model_response=item["negative"]),
                    label=trait_label,
                    trait_description=trait_description,
                )
            )

        # 5) diversity summary (prompts only)
        prompts = [it["prompt"] for it in cleaned]
        div = self.diversity.compute(prompts)

        report = GenerationReport(
            requested=num_pairs,
            kept_after_dedupe=len(cleaned),
            retries_for_refusals=retries,
            diversity=div,
        )
        return cps, report

    @staticmethod
    def _build_user_prompt(topic: str, rx: str, label: str, desc: str, k: int) -> str:
        bullets = (
            f"- Topic: {topic}\n"
            f"- Prescription (must-follow): {rx}\n"
            f"- Trait label: {label}\n"
            f"- Trait description: {desc}\n"
            f"- Num pairs: {k}\n"
        )
        schema = (
            "Return JSON like:\n"
            "{\n"
            '  "pairs": [\n'
            '    {"prompt": "...", "positive": "...", "negative": "...", '
            f'"label": "{label}", "trait_description": "{desc}"\n'
            "  ]\n"
            "}\n"
        )

        tips = (
            "- Make prompts specific to the topic but varied in wording and intent.\n"
            "- Keep negative examples safe (fictional, non-actionable).\n"
            "- Avoid meta-text like “I cannot” or “As an AI model…”.\n"
        )
        return f"Create {k} contrastive pairs.\n{bullets}\n{schema}\n{tips}"
