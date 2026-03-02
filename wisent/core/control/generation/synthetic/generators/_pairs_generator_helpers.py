"""Extracted from pairs_generator.py - parse_pairs tail and _build_user_prompt."""

import logging
from wisent.core.primitives.contrastive_pairs.contrastive_pair import (
    ContrastivePair,
    PositiveResponse,
    NegativeResponse,
)

logger = logging.getLogger(__name__)


def finalize_parse_pair(prompt, positive, negative, trait_label,
                        trait_description, out):
    """Finalize parsing of a single contrastive pair and add to output.

    Creates a ContrastivePair from parsed prompt, positive, and negative
    text, then adds it to the output ContrastivePairSet.

    Args:
        prompt: Parsed prompt text
        positive: Parsed positive response text
        negative: Parsed negative response text
        trait_label: Label for the trait being steered
        trait_description: Description of the trait
        out: ContrastivePairSet to add the pair to
    """
    cp = ContrastivePair(
        prompt=prompt,
        positive_response=PositiveResponse(model_response=positive),
        negative_response=NegativeResponse(model_response=negative),
        label=trait_label,
        trait_description=trait_description,
    )
    out.add(cp)
    logger.info("[PARSE DEBUG] Successfully added pair")


def build_user_prompt(label: str, desc: str, k: int) -> str:
    """Build the user prompt for contrastive pair generation.

    Constructs the instruction prompt that tells the model how many
    pairs to generate and what trait to focus on.

    Args:
        label: Trait label
        desc: Trait description
        k: Number of pairs to generate

    Returns:
        Formatted user prompt string
    """
    return (
        f"Create {k} contrastive pairs.\n"
        f"- Trait label: {label}\n"
        f"- Trait description: {desc}\n"
        f"\n"
        f"Tips:\n"
        f"- Make prompts specific to the topic but varied in wording "
        f"and intent.\n"
        f"- Keep negative examples safe (fictional, non-actionable).\n"
        f"- Avoid meta-text like 'I cannot' or 'As an AI model'.\n"
        f"\n"
        f"Generate {k} pairs now."
    )
