from __future__ import annotations

import logging


from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

from wisent.core.models.wisent_model import WisentModel
from wisent.core.synthetic.db_instructions.core.atoms import DB_Instructions

from wisent.core.synthetic.generators.core.atoms import GenerationReport

from wisent.core.synthetic.generators.diversities.core.core import Diversity

from wisent.core.synthetic.cleaners.pairs_cleaner import PairsCleaner

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

        refusaler_stats = stats.step_stats.get("refusaler_cleaner")
        retries = refusaler_stats.modified_items if refusaler_stats else 0 

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

        logger.info(f"[PARSE DEBUG] Received {len(raw)} raw outputs to parse")

        for idx, r in enumerate(raw):
            logger.info(f"[PARSE DEBUG] Raw output {idx}:\n{r[:500]}")  # First 500 chars

            original_r = r
            #TODO: this is very ugly, need to improve robustness
            # r can have instruction, and i want extacrt everything between ```json and ``` (after - You must return answer in valid JSON format only. Don't include any explanations or additional text.assistant)
            # also try to recover like Expecting ',' delimiter
            if "```json" in r:
                r = r.split("```json")[-1]
                logger.info(f"[PARSE DEBUG] After json block extraction: {r[:200]}")
            if "```" in r:
                r = r.split("```")[0]
                logger.info(f"[PARSE DEBUG] After backtick removal: {r[:200]}")
            r = r.strip()

            logger.info(f"[PARSE DEBUG] Final cleaned string to parse:\n{r}")

            try:
                data = json.loads(r)
                logger.info(f"[PARSE DEBUG] Successfully parsed JSON: {data}")
            except json.JSONDecodeError as e:
                logger.warning(f"[PARSE DEBUG] JSON decode failed: {e}")
                # try to recover from common errors
                r = r.replace("'", '"').replace("```", '')
                # Fix missing commas between array elements: }\n    { -> },\n    {
                import re
                r = re.sub(r'\}\s*\n\s*\{', '},\n    {', r)
                logger.info(f"[PARSE DEBUG] Attempting recovery with quote replacement and comma fixing: {r[:200]}")
                try:
                    data = json.loads(r)
                    logger.info(f"[PARSE DEBUG] Recovery successful: {data}")
                except json.JSONDecodeError as e2:
                    logger.error(f"[PARSE DEBUG] Recovery failed: {e2}. Skipping this output.")
                    logger.error(f"[PARSE DEBUG] Original raw output was:\n{original_r}")
                    continue

            # Handle both dict with "pairs" key and direct list
            if isinstance(data, list):
                pairs_list = data
                logger.info(f"[PARSE DEBUG] Data is a direct list with {len(pairs_list)} pairs")
            elif isinstance(data, dict):
                pairs_list = data.get("pairs", [])
                logger.info(f"[PARSE DEBUG] Found {len(pairs_list)} pairs in data dict")
            else:
                logger.error(f"[PARSE DEBUG] Unexpected data type: {type(data)}")
                continue

            for item_idx, item in enumerate(pairs_list):
                logger.info(f"[PARSE DEBUG] Processing pair {item_idx}: {item}")
                cp = ContrastivePair(
                    prompt=item["prompt"],
                    positive_response=PositiveResponse(model_response=item["positive"]),
                    negative_response=NegativeResponse(model_response=item["negative"]),
                    label=item.get("label", self.trait_label),
                    trait_description=item.get("trait_description", self.trait_description),
                )
                out.add(cp)
                logger.info(f"[PARSE DEBUG] Successfully added pair {item_idx}")

        logger.info(f"[PARSE DEBUG] Finished parsing. Total pairs collected: {len(out)}")
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
