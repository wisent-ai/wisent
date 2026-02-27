"""Text generation evaluation mixin for BigCode evaluator."""
import logging
import re
from typing import Any, Dict, List, Optional

from wisent.core.constants import BLEU_MAX_N_GRAM

logger = logging.getLogger(__name__)


class BigCodeTextEvalMixin:
    """Mixin providing text evaluation for BigCodeEvaluator."""

    def _evaluate_text_generation(self, task: BigCodeTask, generations: List[str]) -> Dict[str, Any]:
        """
        Evaluate text generation tasks (e.g., code-to-text) using BLEU and other metrics.
        
        Args:
            task: BigCodeTask with reference texts
            generations: List of generated texts
            
        Returns:
            Dict with BLEU scores and other metrics
        """
        try:
            import sacrebleu
            has_sacrebleu = True
        except ImportError:
            has_sacrebleu = False
            logger.warning("sacrebleu not installed. Using fallback BLEU implementation.")
        
        results = {
            "bleu_scores": [],
            "avg_bleu": 0.0,
            "exact_match": 0.0,
            "f1_scores": [],
            "avg_f1": 0.0,
        }
        
        samples = task.get_samples()
        total_exact_match = 0
        
        for i, sample in enumerate(samples):
            if i >= len(generations):
                break
            
            generation = generations[i] if isinstance(generations[i], str) else generations[i][0]
            reference = self._extract_reference(sample)
            
            if reference is None:
                continue
            
            # BLEU score
            if has_sacrebleu:
                bleu = self._compute_sacrebleu(generation, reference)
            else:
                bleu = self._compute_simple_bleu(generation, reference)
            results["bleu_scores"].append(bleu)
            
            # Exact match
            if self._normalize_text(generation) == self._normalize_text(reference):
                total_exact_match += 1
            
            # F1 score (token-level)
            f1 = self._compute_token_f1(generation, reference)
            results["f1_scores"].append(f1)
        
        # Compute averages
        if results["bleu_scores"]:
            results["avg_bleu"] = sum(results["bleu_scores"]) / len(results["bleu_scores"])
        if results["f1_scores"]:
            results["avg_f1"] = sum(results["f1_scores"]) / len(results["f1_scores"])
        if samples:
            results["exact_match"] = total_exact_match / min(len(samples), len(generations))
        
        return results
    
    def _extract_reference(self, sample: Dict) -> Optional[str]:
        """Extract reference text from a sample."""
        # Try common field names for reference texts
        for field in ["docstring", "reference", "target", "answer", "comment", "description", "nl"]:
            if field in sample and sample[field]:
                ref = sample[field]
                if isinstance(ref, list):
                    return ref[0] if ref else None
                return str(ref)
        return None
    
    def _compute_sacrebleu(self, hypothesis: str, reference: str) -> float:
        """Compute BLEU score using sacrebleu library."""
        import sacrebleu
        
        # sacrebleu expects references as a list of lists
        bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
        return bleu.score / 100.0  # Normalize to [0, 1]
    
    def _compute_simple_bleu(self, hypothesis: str, reference: str, max_n: int = BLEU_MAX_N_GRAM) -> float:
        """
        Compute simple BLEU score without external dependencies.
        
        Uses smoothed BLEU to handle short sequences.
        """
        from collections import Counter
        import math
        
        def get_ngrams(text: str, n: int) -> Counter:
            tokens = text.lower().split()
            return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
        
        hyp_tokens = hypothesis.lower().split()
        ref_tokens = reference.lower().split()
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, min(max_n + 1, len(hyp_tokens) + 1)):
            hyp_ngrams = get_ngrams(hypothesis, n)
            ref_ngrams = get_ngrams(reference, n)
            
            # Clipped counts
            clipped_count = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
            total_count = sum(hyp_ngrams.values())
            
            # Smoothing: add 1 to avoid zero precision
            precision = (clipped_count + 1) / (total_count + 1)
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        
        # Brevity penalty
        if len(hyp_tokens) >= len(ref_tokens):
            bp = 1.0
        else:
            bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
        
        bleu = bp * math.exp(log_precision)
        return min(1.0, bleu)  # Cap at 1.0
    
    def _compute_token_f1(self, hypothesis: str, reference: str) -> float:
        """Compute token-level F1 score."""
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        common = hyp_tokens & ref_tokens
        
        if not common:
            return 0.0
        
        precision = len(common) / len(hyp_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for exact match comparison."""
        import re
        # Lowercase, remove extra whitespace, remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = ' '.join(text.split())
        return text


# Main interface for BigCode integration
_loader = None
_evaluator = None

