"""
Humanization evaluator with coherence check.

Combines AI detection (Desklib) with coherence evaluation to ensure
outputs are both human-like AND readable/coherent.

Usage:
    from wisent.core.evaluators.custom.examples.humanization_coherent import HumanizationCoherentEvaluator
    
    evaluator = HumanizationCoherentEvaluator()
    result = evaluator("Some text to analyze")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from collections import Counter
import re

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CustomEvaluatorConfig,
)
from wisent.core.evaluators.personalization.coherence import evaluate_quality

__all__ = ["HumanizationCoherentEvaluator", "create_humanization_coherent_evaluator"]

logger = logging.getLogger(__name__)


def _strict_coherence_check(text: str) -> tuple[bool, float, str]:
    """Strict coherence check for humanization.
    
    Returns:
        (passes, score 0-100, reason if failed)
    """
    if not text or len(text.strip()) < 20:
        return False, 0.0, "Too short"
    
    tokens = text.lower().split()
    if len(tokens) < 5:
        return False, 0.0, "Too few words"
    
    # Check 1: Word repetition - same word appearing too often
    # Skip common words that naturally repeat
    common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                   'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                   'should', 'may', 'might', 'must', 'shall', 'can', 'and', 'or', 'but',
                   'if', 'then', 'else', 'when', 'where', 'why', 'how', 'what', 'which',
                   'who', 'whom', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
                   'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our',
                   'their', 'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                   'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                   'between', 'under', 'again', 'further', 'once', 'here', 'there', 'all',
                   'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                   'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just'}
    
    word_counts = Counter(tokens)
    for word, count in word_counts.items():
        if word not in common_words and len(word) > 3:
            # Any content word appearing 2+ times is suspicious for short text
            if count >= 2:
                return False, 20.0, f"Repeated word '{word}' {count} times"
    
    # Check 2: Bigram repetition - any repeated bigram is bad
    if len(tokens) >= 4:
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
        bigram_counts = Counter(bigrams)
        for bigram, count in bigram_counts.items():
            if count >= 2:
                return False, 30.0, f"Repeated phrase '{bigram}'"
    
    # Check 3: Nonsense words (very long or no vowels)
    nonsense_count = 0
    for token in tokens:
        clean = re.sub(r'[^a-z]', '', token)
        if len(clean) > 15:  # Very long word
            nonsense_count += 1
        elif len(clean) > 3 and not re.search(r'[aeiou]', clean):  # No vowels
            nonsense_count += 1
    
    if nonsense_count / len(tokens) > 0.15:
        return False, 25.0, f"Too many nonsense words ({nonsense_count})"
    
    # Check 4: Unique word ratio - too repetitive overall
    unique_tokens = set(tokens)
    unique_ratio = len(unique_tokens) / len(tokens)
    if unique_ratio < 0.5:
        return False, 35.0, f"Low vocabulary diversity ({unique_ratio:.0%})"
    
    # Check 5: Sentence structure - look for subject-verb patterns
    # Simple heuristic: check for common sentence starters
    text_lower = text.lower()
    has_structure = any(text_lower.startswith(s) for s in [
        'the ', 'a ', 'an ', 'i ', 'we ', 'you ', 'it ', 'this ', 'that ',
        'there ', 'here ', 'what ', 'how ', 'why ', 'when ', 'where ',
    ]) or re.match(r'^[A-Z][a-z]+\s', text)  # Starts with capitalized word
    
    if not has_structure:
        # Penalize but don't reject
        pass
    
    # Calculate score based on unique ratio
    score = min(100.0, unique_ratio * 100 + 20)
    
    return True, score, ""


class HumanizationCoherentEvaluator(CustomEvaluator):
    """Combined humanization + coherence evaluator.
    
    Uses Desklib for AI detection and coherence metrics to ensure
    outputs are both human-like AND readable.
    
    Final score = human_score * coherence_weight if coherence passes threshold,
    otherwise 0.
    
    Args:
        coherence_threshold: Minimum coherence score (0-100) to accept output (default: 50)
        coherence_weight: How much coherence affects final score (default: 0.5)
        device: Device for Desklib model
    """
    
    def __init__(
        self,
        coherence_threshold: float = 50.0,
        coherence_weight: float = 0.5,
        device: Optional[str] = None,
    ):
        config = CustomEvaluatorConfig(
            name="humanization_coherent",
            description="Humanization with coherence check (higher = more human-like AND coherent)",
        )
        super().__init__(name="humanization_coherent", description=config.description, config=config)
        
        self.coherence_threshold = coherence_threshold
        self.coherence_weight = coherence_weight
        
        # Load Desklib detector
        from wisent.core.evaluators.custom.examples.desklib_detector import DesklibDetectorEvaluator
        self._desklib = DesklibDetectorEvaluator(device=device)
    
    def evaluate_response(self, response: str, **kwargs) -> Dict[str, Any]:
        """Evaluate response for humanization AND coherence."""
        
        # Use strict coherence check
        passes, coherence_score, reason = _strict_coherence_check(response)
        
        # If coherence fails, return 0
        if not passes:
            return {
                "score": 0.0,
                "human_prob": 0.0,
                "ai_prob": 1.0,
                "coherence_score": coherence_score,
                "rejected_reason": reason,
            }
        
        # Get Desklib score
        desklib_result = self._desklib(response)
        human_prob = desklib_result["human_prob"]
        
        # Combine scores: human_prob * (coherence_factor)
        # coherence_factor scales from coherence_weight to 1.0 based on coherence
        coherence_normalized = coherence_score / 100.0  # 0-1
        coherence_factor = self.coherence_weight + (1.0 - self.coherence_weight) * coherence_normalized
        
        final_score = human_prob * coherence_factor
        
        return {
            "score": final_score,
            "human_prob": human_prob,
            "ai_prob": desklib_result["ai_prob"],
            "coherence_score": coherence_score,
            "coherence_factor": coherence_factor,
        }


def create_humanization_coherent_evaluator(
    coherence_threshold: float = 50.0,
    coherence_weight: float = 0.5,
    device: Optional[str] = None,
    **kwargs
) -> HumanizationCoherentEvaluator:
    """Create a humanization + coherence evaluator.
    
    Args:
        coherence_threshold: Minimum coherence score to accept (default: 50)
        coherence_weight: How much coherence affects score (default: 0.5)
        device: Device for Desklib model
    
    Returns:
        HumanizationCoherentEvaluator instance
    """
    return HumanizationCoherentEvaluator(
        coherence_threshold=coherence_threshold,
        coherence_weight=coherence_weight,
        device=device,
    )


def create_evaluator(**kwargs) -> HumanizationCoherentEvaluator:
    """Factory function for module-based loading."""
    return create_humanization_coherent_evaluator(**kwargs)
