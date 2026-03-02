"""Hallucination generation helpers for HalulensExtractor."""
from __future__ import annotations

import random
import re

from wisent.core.utils.config_tools.constants import (
    YEAR_SHIFT_MIN, YEAR_SHIFT_MAX, YEAR_SHIFT_DEFAULT,
    SMALL_NUM_INCREMENT_MIN, SMALL_NUM_INCREMENT_MAX,
    NUM_PERTURB_SCALE_MIN, NUM_PERTURB_SCALE_MAX,
)


def entity_swap_hallucination(rng: random.Random, answer: str, title: str) -> str:
    """Swap entities with plausible but incorrect alternatives."""
    # Find capitalized words (likely entities)
    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
    if not entities:
        return fabrication_hallucination(rng, answer, title)
    
    # Pick a random entity to swap (not the title itself)
    swappable = [e for e in entities if e.lower() != title.lower()]
    if not swappable:
        return fabrication_hallucination(rng, answer, title)
    
    entity_to_swap = rng.choice(swappable)
    
    # Generate fake replacement
    fake_names = ["Alexander Thompson", "Victoria Institute", "Northern Region", 
                  "Eastern Province", "William Harrison", "Margaret Stewart"]
    replacement = rng.choice(fake_names)
    
    return answer.replace(entity_to_swap, replacement, 1)

def date_shift_hallucination(rng: random.Random, answer: str) -> str:
    """Modify dates and numbers in the answer."""
    # Find years
    def shift_year(match):
        year = int(match.group())
        shift = rng.randint(YEAR_SHIFT_MIN, YEAR_SHIFT_MAX)
        if shift == 0:
            shift = YEAR_SHIFT_DEFAULT
        return str(year + shift)
    
    modified = re.sub(r'\b(1[0-9]{3}|20[0-2][0-9])\b', shift_year, answer)
    
    # Find other numbers
    def shift_number(match):
        num = int(match.group())
        if num < 10:
            return str(num + rng.randint(SMALL_NUM_INCREMENT_MIN, SMALL_NUM_INCREMENT_MAX))
        return str(int(num * rng.uniform(NUM_PERTURB_SCALE_MIN, NUM_PERTURB_SCALE_MAX)))
    
    if modified == answer:
        modified = re.sub(r'\b(\d+)\b', shift_number, answer)
    
    return modified if modified != answer else fabrication_hallucination(rng, answer, "")

def attribute_swap_hallucination(rng: random.Random, answer: str) -> str:
    """Swap attributes or descriptors in the answer."""
    # Common attribute pairs to swap
    swaps = [
        ("first", "last"), ("largest", "smallest"), ("oldest", "newest"),
        ("northern", "southern"), ("eastern", "western"),
        ("major", "minor"), ("primary", "secondary"),
        ("early", "late"), ("ancient", "modern"),
    ]
    
    modified = answer
    for orig, repl in swaps:
        if orig in answer.lower():
            # Case-preserving replacement
            pattern = re.compile(re.escape(orig), re.IGNORECASE)
            modified = pattern.sub(repl, answer, count=1)
            break
        elif repl in answer.lower():
            pattern = re.compile(re.escape(repl), re.IGNORECASE)
            modified = pattern.sub(orig, answer, count=1)
            break
    
    return modified if modified != answer else fabrication_hallucination(rng, answer, "")

def fabrication_hallucination(rng: random.Random, answer: str, title: str) -> str:
    """Add completely fabricated details to the answer."""
    fabrications = [
        f" This was later disputed by scholars at the University of Cambridge.",
        f" However, recent discoveries have challenged this interpretation.",
        f" The original documents were lost in a fire in 1847.",
        f" This claim has been contested by several historians.",
        f" Archaeological evidence from 2019 suggests otherwise.",
        f" According to newly discovered manuscripts, this may be incorrect.",
    ]
    
    # Add fabricated detail to the end
    return answer.rstrip('.') + '.' + rng.choice(fabrications)
