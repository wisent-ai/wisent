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
        nonsense_mode: str | None = None,
    ) -> None:
        self.model = model
        self.db_instructions = db_instructions
        self.generation_config = generation_config
        self.cleaner = cleaner
        self.diversity = diversity

        self.contrastive_set_name = contrastive_set_name
        self.trait_description = trait_description
        self.trait_label = trait_label
        self.nonsense_mode = nonsense_mode


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
        # Use nonsense-specific instructions if nonsense_mode is set
        if self.nonsense_mode:
            instruction_key = f"nonsense_{self.nonsense_mode}"
            try:
                sys = self.db_instructions.get(instruction_key)
            except KeyError:
                logger.warning(f"Nonsense mode '{self.nonsense_mode}' not found in DB instructions, falling back to generic_pairs")
                sys = self.db_instructions.get("generic_pairs")
        else:
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
        Expects simple text format with ---PAIR--- markers, not JSON.

        arguments:
            raw:
                Raw model output string to parse.
        returns:
            ContrastivePairSet object parsed from the raw string.
        """
        import re

        out: ContrastivePairSet = ContrastivePairSet(
            name=self.contrastive_set_name,
            task_type=self.trait_label,
        )

        logger.info(f"[PARSE DEBUG] Received {len(raw)} raw outputs to parse")

        for idx, r in enumerate(raw):
            logger.info(f"[PARSE DEBUG] Raw output {idx}:\n{r[:500]}")

            # Split by ---PAIR--- markers (flexible with extra dashes)
            pair_blocks = re.split(r'-+PAIR-+', r)

            for block_idx, block in enumerate(pair_blocks):
                if not block.strip():
                    continue

                # Remove ---END--- marker if present (flexible with extra dashes/spaces)
                block = re.sub(r'-+END-+', '', block).strip()

                if not block:
                    continue

                logger.info(f"[PARSE DEBUG] Processing block {block_idx}:\n{block[:200]}")

                # Extract ALL occurrences - model generates ANY labels, not just PROMPT/POSITIVE/NEGATIVE
                # Look for pattern: LABEL1: text1  LABEL2: text2  LABEL3: text3
                # We assume first is prompt, second is positive, third is negative
                lines = block.strip().split('\n')

                prompts = []
                positives = []
                negatives = []

                current_group = []
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('---'):
                        continue

                    # Check if line contains colon (flexible parsing for any LABEL: or LABEL . or LABEL : format)
                    # Model outputs variations like "POSITIVE:", "POSITIVE :", "POSITIVE.", "_PROMPT:", " PROMPT:", etc.
                    if ':' in line or '.' in line:
                        # Find the first colon or period as separator
                        sep_idx = -1
                        for sep in [':', '.']:
                            idx = line.find(sep)
                            if idx > 0:  # Must have at least one char before separator
                                if sep_idx == -1 or idx < sep_idx:
                                    sep_idx = idx

                        if sep_idx > 0:
                            # Extract text after separator
                            text = line[sep_idx + 1:].strip()
                            if text:
                                current_group.append(text)

                                # When we have 3 items, that's a complete pair
                                if len(current_group) == 3:
                                    prompts.append(current_group[0])
                                    positives.append(current_group[1])
                                    negatives.append(current_group[2])
                                    current_group = []

                # All three lists should have same length
                if not (prompts and positives and negatives and len(prompts) == len(positives) == len(negatives)):
                    logger.warning(f"[PARSE DEBUG] Mismatched or missing fields in block {block_idx}")
                    continue

                # Create a pair for each triple
                for prompt, positive, negative in zip(prompts, positives, negatives):
                    prompt = prompt.strip()
                    positive = positive.strip()
                    negative = negative.strip()

                    if not (prompt and positive and negative):
                        logger.warning(f"[PARSE DEBUG] Empty field(s) in triple")
                        continue

                    logger.info(f"[PARSE DEBUG] Extracted - Prompt: {prompt[:50]}, Positive: {positive[:50]}, Negative: {negative[:50]}")

                    cp = ContrastivePair(
                        prompt=prompt,
                        positive_response=PositiveResponse(model_response=positive),
                        negative_response=NegativeResponse(model_response=negative),
                        label=self.trait_label,
                        trait_description=self.trait_description,
                    )
                    out.add(cp)
                    logger.info(f"[PARSE DEBUG] Successfully added pair")

        logger.info(f"[PARSE DEBUG] Finished parsing. Total pairs collected: {len(out)}")
        return out

    @staticmethod
    def _build_user_prompt(label: str, desc: str, k: int) -> str:
        return (
            f"Create {k} contrastive pairs.\n"
            f"- Trait label: {label}\n"
            f"- Trait description: {desc}\n"
            f"\n"
            f"Tips:\n"
            f"- Make prompts specific to the topic but varied in wording and intent.\n"
            f"- Keep negative examples safe (fictional, non-actionable).\n"
            f"- Avoid meta-text like 'I cannot' or 'As an AI model'.\n"
            f"\n"
            f"Generate {k} pairs now."
        )
