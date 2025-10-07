from __future__ import annotations

import logging


from wisent_guard.core.contrastive_pairs.core.pair import ContrastivePair  
from wisent_guard.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse  
from wisent_guard.core.contrastive_pairs.core.set import ContrastivePairSet 

from wisent_guard.core.models.wisent_model import WisentModel
from wisent_guard.synthetic.db_instructions.core.atoms import DB_Instructions

from wisent_guard.synthetic.generators.core.atoms import GenerationReport

from wisent_guard.synthetic.generators.diversities.core.core import Diversity

from wisent_guard.synthetic.cleaners.pairs_cleaner import PairsCleaner

__all__ = [
    "SyntheticContrastivePairsGenerator",
]

logger = logging.getLogger(__name__)

class SyntheticContrastivePairsGenerator:
    """Small, fast contrastive-pairs generator with an extensible cleaning pipeline."""

    def __init__(
        self,
        model: WisentModel,
        generation_config: dict[str, int | float | str],
        contrastive_set_name: str,
        trait_description: str,
        trait_label: str,
        db_instructions: DB_Instructions,
        cleaner: PairsCleaner,
        diversity: Diversity,
    ) -> None:
        self.model = model
        self.db_instructions = db_instructions
        self.generation_config = generation_config
        self.cleaner = cleaner
        self.diversity = diversity

        self.contrastive_set_name = contrastive_set_name
        self.trait_description = trait_description
        self.trait_label = trait_label


    def generate(
        self,
        num_pairs: int = 10,
    ) -> tuple[ContrastivePairSet, GenerationReport]:
        """
        Generate synthetic contrastive pairs for the given topic and trait.

        arguments:
            num_pairs:
                Number of contrastive pairs to generate (default: 10).
        
        returns:
            Tuple of ContrastivePairSet with the generated pairs and GenerationReport with statistics about the generation
        """
        # 1) generate
        sys = self.db_instructions.get("generic_pairs")
        usr = self._build_user_prompt(
            self.trait_label, self.trait_description, num_pairs
        )
        raw = self.model.generate(
            inputs=[[
                {"role": "system", "content": sys},
                {"role": "user", "content": usr}
            ]],
                 **self.generation_config,
        )

        # 2) parse
        parsed = self.parse_pairs(raw)

        # 3) clean
        cleaned, stats = self.cleaner.clean(parsed)

        retries = stats.step_stats.get("refusaler_cleaner").modified_items 

        # 4) build domain objects
        cps = ContrastivePairSet(name=self.contrastive_set_name, task_type=self.trait_label)
        for item in cleaned.pairs:
            cps.add(
                ContrastivePair(
                    prompt=item.prompt,
                    positive_response=PositiveResponse(model_response=item.positive_response.model_response),
                    negative_response=NegativeResponse(model_response=item.negative_response.model_response),
                    label=item.label or self.trait_label,
                    trait_description=item.trait_description or self.trait_description,
                )
            )
        # 5) diversity summary (prompts only)
        prompts = [it.prompt for it in cleaned.pairs]
        div = self.diversity.compute(prompts)

        report = GenerationReport(
            requested=num_pairs,
            kept_after_dedupe=len(cleaned),
            retries_for_refusals=retries,
            diversity=div,
        )
        return cps, report

    def parse_pairs(self, raw: list[str]) -> ContrastivePairSet:
        """
        Parse raw model outputs into ContrastivePairSet objects.

        arguments:
            raw:
                Raw model output string to parse.
        returns:
            ContrastivePairSet object parsed from the raw string.
        """

        import json

        out: ContrastivePairSet = ContrastivePairSet(
            name=self.contrastive_set_name,
            task_type=self.trait_label,
        )
        for r in raw:
            #TODO: this is very ugly, need to improve robustness
            # r can have instruction, and i want extacrt everything between ```json and ``` (after - You must return answer in valid JSON format only. Don't include any explanations or additional text.assistant)
            # also try to recover like Expecting ',' delimiter
            if "```json" in r:
                r = r.split("```json")[-1]
            if "```" in r:
                r = r.split("```")[0]
            r = r.strip()
            try:
                data = json.loads(r)
            except json.JSONDecodeError:
                # try to recover from common errors
                r = r.replace("'", '"').replace("```", '')
                try:
                    data = json.loads(r)
                except json.JSONDecodeError:
                    continue
            for item in data.get("pairs", []):
                cp = ContrastivePair(
                    prompt=item["prompt"],
                    positive_response=PositiveResponse(model_response=item["positive"]),
                    negative_response=NegativeResponse(model_response=item["negative"]),
                    label=item.get("label", self.trait_label),
                    trait_description=item.get("trait_description", self.trait_description),
                )
                out.add(cp)
        return out

    @staticmethod
    def _build_user_prompt(label: str, desc: str, k: int) -> str:
        bullets = (
            f"- Trait label: {label}\n"
            f"- Trait description: {desc}\n"
            f"- Num pairs: {k}\n"
        )
        schema = (
            "Return JSON like:\n"
            "```json\n"
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
            "- You must return answer in valid JSON format only. Don't include any explanations or additional text.\n"
        )
        return f"Create {k} contrastive pairs.\n{bullets}\n{schema}\n{tips}"
