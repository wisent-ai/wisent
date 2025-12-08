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

        Generates pairs one at a time:
        1. Generate a prompt/scenario
        2. Generate a positive response (exhibits the trait)
        3. Generate a negative response (does NOT exhibit the trait)

        arguments:
            num_pairs:
                Number of contrastive pairs to generate (default: 10).

        returns:
            Tuple of ContrastivePairSet with the generated pairs and GenerationReport with statistics about the generation
        """
        parsed = ContrastivePairSet(
            name=self.contrastive_set_name,
            task_type=self.trait_label,
        )

        # Generate opposite trait description once
        opposite_instruction = (
            f"What is the OPPOSITE personality trait of: {self.trait_description}?\n\n"
            f"Describe the opposite in one sentence, be specific about what words/style/tone to use."
        )
        opposite_raw = self.model.generate(
            inputs=[[{"role": "user", "content": opposite_instruction}]],
            **self.generation_config,
        )
        opposite_trait = opposite_raw[0].strip() if opposite_raw else "neutral and plain"
        logger.info(f"[GENERATE] Opposite trait: {opposite_trait}")

        # Generate pairs one at a time, retry until we have num_pairs
        max_attempts = num_pairs * 3  # Prevent infinite loops
        attempts = 0

        while len(parsed) < num_pairs and attempts < max_attempts:
            attempts += 1
            logger.info(f"[GENERATE] Generating pair {len(parsed)+1}/{num_pairs} (attempt {attempts})")

            # 1) Generate a prompt/scenario - simple question format
            prompt_instruction = (
                f"Write one short question a user might ask. Example: 'What is your favorite hobby?' "
                f"Just the question, nothing else."
            )
            prompt_raw = self.model.generate(
                inputs=[[{"role": "user", "content": prompt_instruction}]],
                **self.generation_config,
            )
            prompt = prompt_raw[0].strip() if prompt_raw else ""

            if not prompt:
                logger.warning(f"[GENERATE] Failed to generate prompt, retrying...")
                continue

            logger.info(f"[GENERATE] Prompt: {prompt[:100]}")

            # 2) Generate positive response (exhibits the trait)
            positive_instruction = (
                f"Question: {prompt}\n\n"
                f"Answer the question AS IF you have this personality: {self.trait_description}\n\n"
                f"Write 1-2 sentences showing this personality clearly. Just the answer."
            )
            positive_raw = self.model.generate(
                inputs=[[{"role": "user", "content": positive_instruction}]],
                **self.generation_config,
            )
            positive = positive_raw[0].strip() if positive_raw else ""

            if not positive:
                logger.warning(f"[GENERATE] Failed to generate positive, retrying...")
                continue

            logger.info(f"[GENERATE] Positive: {positive[:100]}")

            # 3) Generate negative response - using the opposite trait
            negative_instruction = (
                f"Question: {prompt}\n\n"
                f"Answer the question AS IF you have this personality: {opposite_trait}\n\n"
                f"Write 1-2 sentences showing this personality clearly. Just the answer."
            )
            negative_raw = self.model.generate(
                inputs=[[{"role": "user", "content": negative_instruction}]],
                **self.generation_config,
            )
            negative = negative_raw[0].strip() if negative_raw else ""

            if not negative:
                logger.warning(f"[GENERATE] Failed to generate negative, retrying...")
                continue

            logger.info(f"[GENERATE] Negative: {negative[:100]}")

            # Create the pair
            cp = ContrastivePair(
                prompt=prompt,
                positive_response=PositiveResponse(model_response=positive),
                negative_response=NegativeResponse(model_response=negative),
                label=self.trait_label,
                trait_description=self.trait_description,
            )
            parsed.add(cp)
            logger.info(f"[GENERATE] Successfully added pair {len(parsed)}/{num_pairs}")

        logger.info(f"[GENERATE] Generated {len(parsed)} pairs after {attempts} attempts")

        # Clean (dedupe, refusal check, etc.)
        cleaned, stats = self.cleaner.clean(parsed)

        refusaler_stats = stats.step_stats.get("refusaler_cleaner")
        retries = refusaler_stats.modified_items if refusaler_stats else 0

        # Build final domain objects
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

        # Diversity summary (prompts only)
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
