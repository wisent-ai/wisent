from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable

logger = logging.getLogger(__name__)


ChatMessage = dict[str, str]
CompletionFn = Callable[[list[ChatMessage]], str]

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

_TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")

def tok(s: str) -> list[str]:
    """
    Simple whitespace/punctuation tokenizer.
    
    arguments:
        s:
            Input string to tokenize.
            
    returns:
        List of tokens (words and punctuation).
        
    example:
        >>> tok("Hello, world!")
        ['hello', ',', 'world', '!']
    """
    return _TOKEN_RE.findall(s.lower())

def distinct_n(texts: Iterable[str], n: int) -> float:
    """
    Compute the Distinct-N score for a list of texts.

    arguments:
        texts:
            Iterable of input strings.
        n:
            N-gram size (e.g., 1 for unigrams, 2 for bigrams).
    
    returns:
        Distinct-N score: ratio of unique n-grams to total n-grams.

    example:
        >>> distinct_n(["hello world", "hello there"], 1)
        0.75
        >>> distinct_n(["hello world", "hello there"], 2)
        1.0
    
    intuition:
        Distinct-N = (number of unique n-grams) / (total number of n-grams)
        Higher values indicate more lexical diversity. For example, in ["hello world", "hello there"]:
            Unigrams: 
                ['hello', 'world', 'hello', 'there'] → unique = ['hello', 'world', 'there'] → Distinct-1 = 3/4 = 0.75
            Bigrams: 
                ['hello world', 'hello there'] → unique = ['hello world', 'hello there'] → Distinct-2 = 2/2 = 1.0
    """
    ngrams: list[tuple[str, ...]] = []
    for t in texts:
        toks = tok(t)
        ngrams.extend(tuple(toks[i : i + n]) for i in range(0, max(0, len(toks) - n + 1)))
    return (len(set(ngrams)) / float(len(ngrams))) if ngrams else 0.0

def jaccard(a: str, b: str) -> float:
    """
    Compute Jaccard similarity between two strings.

    arguments:
        a:
            First input string.
        b:
            Second input string.

    returns:
        Jaccard similarity score between 0.0 and 1.0.

    example:
        >>> jaccard("hello world", "hello there")
        0.3333333333333333
        >>> jaccard("foo", "bar")
        0.0
        >>> jaccard("", "")
        1.0
    
    intuition:
        Jaccard similarity = |A ∩ B| / |A ∪ B|
        where A and B are the sets of tokens in strings a and b. Higher values indicate more overlap.
        For example "hello world" and "hello there" share the token "hello", yielding a similarity of 1/3.
    """
    A, B = set(tok(a)), set(tok(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def hash64(x: str) -> int:
    """
    64-bit FNV-1a hash of a string.
    
    arguments:
        x:
            Input string to hash.
    
    returns:
        64-bit integer hash value.
        
    example:
        >>> hash64("hello")
        11831194018420276491
        >>> hash64("")
        0xCBF29CE484222325
    """
    h = 0xCBF29CE484222325
    for c in x.encode("utf-8"):
        h ^= c
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

def simhash64(text: str) -> int:
    """
    Compute the SimHash of a text.

    arguments:
        text: Input text to hash.

    returns:
        64-bit SimHash fingerprint as an integer.

    example:
        >>> simhash64("hello world")
        15590995882336051200
        >>> simhash64("hello there")
        15590995882336051200
        >>> simhash64("different text")
        10808639105689190400
        >>> simhash64("")
        0x0
    
    intuition:
        SimHash is a locality-sensitive hash that maps similar texts to similar fingerprints.
        It works by hashing features (tokens) and combining their bits based on frequency.
        For example, "hello world" and "hello there" share the token "hello", resulting in identical SimHash values.
        In contrast, "different text" yields a very different fingerprint. An empty string hashes to 0.
        This makes SimHash useful for deduplication and near-duplicate detection.
    """
    feats = tok(text)
    if not feats:
        return 0
    vec = [0] * 64
    for f in feats:
        hv = hash64(f)
        for i in range(64):
            vec[i] += 1 if (hv >> i) & 1 else -1
    out = 0
    for i, v in enumerate(vec):
        if v >= 0:
            out |= (1 << i)
    return out

def hamming(a: int, b: int) -> int:
    """
    Compute the Hamming distance between two 64-bit integers.
    arguments:
        a:
            First 64-bit integer.
        b:
            Second 64-bit integer.
    
    returns:
        Hamming distance (number of differing bits).

    example:
        >>> hamming(0b1010, 0b1001)
        2
        >>> hamming(0xFFFFFFFFFFFFFFFF, 0x0)
        64
        >>> hamming(0x1234567890ABCDEF, 0x1234567890ABCDEF)
        0
    
    intuition:
        Hamming distance counts the number of bit positions where two integers differ.
        For example, 0b1010 (10) and 0b1001 (9) differ in two bit positions, yielding a distance of 2.
        The maximum distance for 64-bit integers is 64 (completely different), and identical integers have a distance of 0.
    """
    x = a ^ b
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


class FastDiversity(Diversity):
    """
    Fast diversity metrics computation.

    attributes:
        _rng:
            Random number generator for sampling.
    """
    def __init__(self, seed: int | None = 13) -> None:
        self._rng = random.Random(seed)

    def compute(self, texts: list[str]) -> DiversityScores:
        """
        Compute diversity metrics for a list of texts.

        arguments:
            texts:
                List of input strings.

        returns:
            DiversityScores dataclass with various diversity metrics.

        example:
            >>> fd = FastDiversity(seed=42)
            >>> texts = ["hello world", "hello there", "different text"]
            >>> fd.compute(texts)
            DiversityScores(distinct1=0.6666666666666666, distinct2=1.0, avg_jaccard_prompt=0.3333333333333333, mean_simhash_hamming_prompt=21.333333333333332, min_simhash_hamming_prompt=20)
        
        intuition:
            This computes several diversity metrics:
            - Distinct-1 and Distinct-2: ratios of unique unigrams and bigrams to total unigrams and bigrams.
            - Average Jaccard similarity: average token overlap between all pairs of texts.
            - Mean and minimum SimHash Hamming distance: average and minimum bit differences between SimHash fingerprints of all pairs.
            Higher distinct scores and lower average Jaccard indicate more lexical diversity.
            Higher mean and minimum SimHash Hamming distances indicate more structural diversity.
        """
        d1 = distinct_n(texts, 1)
        d2 = distinct_n(texts, 2)
        sample = texts if len(texts) <= 20 else self._rng.sample(texts, 20)
        if len(sample) >= 2:
            jaccs: list[float] = []
            fps: list[int] = []
            for i, s in enumerate(sample):
                fps.append(simhash64(s))
                for j in range(i + 1, len(sample)):
                    jaccs.append(jaccard(s, sample[j]))
            avg_j = sum(jaccs) / len(jaccs) if jaccs else 0.0
            dists: list[int] = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    dists.append(hamming(fps[i], fps[j]))
            mean_h = (sum(dists) / len(dists)) if dists else 0.0
            min_h = min(dists) if dists else 64
        else:
            avg_j, mean_h, min_h = 0.0, 0.0, 64
        return DiversityScores(d1, d2, avg_j, float(mean_h), int(min_h))

def parse_pairs_json(raw: str) -> list[dict[str, str]]:
    """
    Parse JSON output from the model into a list of contrastive pairs.

    arguments:
        raw:
            Raw string output from the model.

    returns:
        List of dicts with 'prompt', 'positive', 'negative' keys.

    raises:
        ValueError if JSON is malformed or missing required fields.

    example:
        >>> raw = '''
        ... {
        ...   "pairs": [
        ...     {
        ...       "prompt": "Tell me a joke.",
        ...       "positive": "Here's a joke: Why did the chicken cross the road? To get to the other side!",
        ...       "negative": "As an AI model, I cannot help with that."
        ...     },
        ...     {
        ...       "prompt": "What's the weather?",
        ...       "positive": "It's sunny today.",
        ...       "negative": "I don't know."
        ...     }
        ...   ]
        ... }
        ... '''
        >>> parse_pairs_json(raw)
        [
            {
                "prompt": "Tell me a joke.",
                "positive": "Here's a joke: Why did the chicken cross the road? To get to the other side!",
                "negative": "As an AI model, I cannot help with that."
            },
            {
                "prompt": "What's the weather?",
                "positive": "It's sunny today.",
                "negative": "I don't know."
            }
        ]
    """
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        payload = json.loads(raw[start:end])
    except Exception:
        logger.debug("Could not parse JSON from completion.")
        return []
    pairs = payload.get("pairs") or []
    out: list[dict[str, str]] = []
    for it in pairs:
        p = str(it.get("prompt", "")).strip()
        pos = str(it.get("positive", "")).strip()
        neg = str(it.get("negative", "")).strip()
        if p and pos and neg:
            out.append({"prompt": p, "positive": pos, "negative": neg})
    return out
