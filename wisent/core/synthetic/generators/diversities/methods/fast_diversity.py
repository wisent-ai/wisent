from __future__ import annotations
from typing import Iterable
import re
import numpy as np
from wisent.core.synthetic.generators.diversities.core.core import Diversity, DiversityScores

__all__ = [
    "FastDiversity",
]

class FastDiversity(Diversity):
    """
    Fast diversity metrics computation.

    attributes:
        _rng:
            Random number generator for sampling.
    """
    _TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")

    def __init__(self, seed: int | None = 13) -> None:
        self._rng = np.random.default_rng(seed)

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
            ... DiversityScores(unique_unigrams=0.6666666666666666, unique_bigrams=1.0, avg_jaccard_prompt=0.3333333333333333, mean_simhash_hamming_prompt=21.333333333333332, min_simhash_hamming_prompt=20)
        
        intuition:
            Higher unique scores and lower average Jaccard indicate more lexical diversity.
            Higher mean and minimum SimHash Hamming distances indicate more structural diversity.
        """
        d1 = self._distinct_n(texts, 1)
        d2 = self._distinct_n(texts, 2)
        sample = texts if len(texts) <= 20 else self._rng.choice(texts, size=20, replace=False).tolist()
        if len(sample) >= 2:
            jaccs: list[float] = []
            fps: list[int] = []
            for i, s in enumerate(sample):
                fps.append(self._simhash64(s))
                for j in range(i + 1, len(sample)):
                    jaccs.append(self._jaccard(s, sample[j]))
            avg_j = sum(jaccs) / len(jaccs) if jaccs else 0.0
            dists: list[int] = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    dists.append(self._hamming(fps[i], fps[j]))
            mean_h = (sum(dists) / len(dists)) if dists else 0.0
            min_h = min(dists) if dists else 64
        else:
            avg_j, mean_h, min_h = 0.0, 0.0, 64
        return DiversityScores(d1, d2, avg_j, float(mean_h), int(min_h))

    def _distinct_n(self, texts: Iterable[str], n: int) -> float:
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
            >>> fd = FastDiversity()
            >>> texts = ["hello world", "hello there"]
            >>> fd._distinct_n(texts, 1)
            0.75
            >>> fd._distinct_n(texts, 2)
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
            toks = self._tok(t)
            ngrams.extend(tuple(toks[i : i + n]) for i in range(0, max(0, len(toks) - n + 1)))
        return (len(set(ngrams)) / float(len(ngrams))) if ngrams else 0.0
    
    def _jaccard(self, a: str, b: str) -> float:
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
            >>> fd = FastDiversity()
            >>> fd._jaccard("hello world", "hello there")
            0.3333333333333333
            >>> fd._jaccard("abc", "xyz")
            0.0
            >>> fd._jaccard("", "")
            1.0
            >>> fd._jaccard("abc", "")
            0.0
        
        intuition:
            Jaccard similarity = |A ∩ B| / |A ∪ B|
            where A and B are the sets of tokens in strings a and b. Higher values indicate more overlap.
            For example "hello world" and "hello there" share the token "hello", yielding a similarity of 1/3.
        """
        A, B = set(self._tok(a)), set(self._tok(b))
        if not A and not B:
            return 1.0
        if not A or not B:
            return 0.0
        inter = len(A & B)
        union = len(A | B)
        return inter / union if union else 0.0
    
    def _tok(self, s: str) -> list[str]:
        """
        Simple whitespace/punctuation tokenizer.
        
        arguments:
            s:
                Input string to tokenize.

        returns:
            List of tokens (words and punctuation).
        """
        return self._TOKEN_RE.findall(s.lower())
    
    def _hamming(self, a: int, b: int) -> int:
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
            >>> fd = FastDiversity()
            >>> fd._hamming(0b1010, 0b1001)
            2
            >>> fd._hamming(0b1111, 0b1111)
            0
            >>> fd._hamming(0b0, 0b1111111111111111)
            64

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

    def _hash64(self, x: str) -> int:
        """
        64-bit FNV-1a hash of a string.
        
        arguments:
            x:
                Input string to hash.
        
        returns:
            64-bit integer hash value.
            
        example:
            >>> fd = FastDiversity()
            >>> fd._hash64("hello")
            11831194018420276491
            >>> fd._hash64("world")
            15195822415430384601
            >>> fd._hash64("")
            14695981039346656037
        """
        h = 0xCBF29CE484222325
        for c in x.encode("utf-8"):
            h ^= c
            h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h

    def _simhash64(self, text: str) -> int:
        """
        Compute the SimHash of a text.

        arguments:
            text: Input text to hash.

        returns:
            64-bit SimHash fingerprint as an integer.

        example:
            >>> fd = FastDiversity()
            >>> fd._simhash64("hello world")
            16204198794447330368
            >>> fd._simhash64("hello there")
            16204198794447330368
            >>> fd._simhash64("different text")
            1080863910568919040
            >>> fd._simhash64("")
            0
        intuition:
            SimHash is a locality-sensitive hash that maps similar texts to similar fingerprints.
            It works by hashing features (tokens) and combining their bits based on frequency.
            For example, "hello world" and "hello there" share the token "hello", resulting in identical SimHash values.
            In contrast, "different text" yields a very different fingerprint. An empty string hashes to 0.
            This makes SimHash useful for deduplication and near-duplicate detection.
        """
        feats = self._tok(text)
        if not feats:
            return 0
        vec = [0] * 64
        for f in feats:
            hv = self._hash64(f)
            for i in range(64):
                vec[i] += 1 if (hv >> i) & 1 else -1
        out = 0
        for i, v in enumerate(vec):
            if v >= 0:
                out |= (1 << i)
        return out