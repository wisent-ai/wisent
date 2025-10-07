from abc import ABC, abstractmethod
from dataclasses import dataclass

__all__ = [
    "Diversity",
    "DiversityScores",
    "GenerationReport",
]

@dataclass(frozen=True)
class DiversityScores:
    """
    Diversity metrics for a list of texts.
    
    attributes:
        unique_unigrams:
            Unique unigrams (ratio of unique unigrams to total unigrams).
        unique_bigrams:
            Unique bigrams (ratio of unique bigrams to total bigrams).
        avg_jaccard_prompt:
            Average Jaccard similarity between all pairs of texts.
        mean_simhash_hamming_prompt:
            Mean Hamming distance between SimHash fingerprints of all pairs of texts.
        min_simhash_hamming_prompt:
            Minimum Hamming distance between SimHash fingerprints of all pairs of texts.
    """
    unique_unigrams: float
    unique_bigrams: float
    avg_jaccard_prompt: float
    mean_simhash_hamming_prompt: float
    min_simhash_hamming_prompt: int

@dataclass(frozen=True)
class GenerationReport:
    """
    Report of a generation+cleaning run.

    attributes:
        requested:
            Number of contrastive pairs requested from the model.
        kept_after_dedupe:
            Number of contrastive pairs kept after deduplication.
        retries_for_refusals:
            Number of retries made to fix refusals in negative examples.
        diversity:
            DiversityScores computed on the final prompts.

    notes:
        LLMs may refuse to generate negative examples that exhibit undesired traits. Increasing
        'max_refusal_retries' in the generator can help mitigate this, but increases cost and latency.
    """
    requested: int
    kept_after_dedupe: int
    retries_for_refusals: int
    diversity: DiversityScores


class Diversity(ABC):
    """
    Diversity metrics computation.
    
    methods:
        compute(texts: list[str]) -> DiversityScores:
            Compute diversity metrics for a list of texts. We can compute Distinct-N, Jaccard similarity,
            and SimHash Hamming distance to get a sense of lexical and structural diversity.
    """
    @abstractmethod
    def compute(self, texts: list[str]) -> DiversityScores: ...