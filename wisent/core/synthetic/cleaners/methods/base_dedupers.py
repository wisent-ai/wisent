import re
import unicodedata
import hashlib
from collections import Counter, defaultdict
from typing import Mapping, Sequence, Callable

from wisent.core.synthetic.cleaners.methods.core.atoms import Deduper
from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
from wisent.core.errors import InvalidValueError, InvalidRangeError

__all__ = [
    "SimHashDeduper",
]

class SimHashDeduper(Deduper):
    """
    Deduplicate items based on near-duplicate similarity of selected fields.
    Uses SimHash + banded LSH for efficient near-duplicate detection.
    """

    def __init__(
        self,
        threshold_bits: int = 3,
        fields_to_hash: Sequence[str] = ("prompt",),
        field_weights: Mapping[str, float] | None = None,
        tokenizer: str = "auto",  # "auto" | "word" | "char"
        word_ngram: int = 3,
        char_ngram: int = 4,
        strip_accents: bool = True,
        stopwords: set[str] | None = None,
        num_bands: int = 8,  # 64 must be divisible by num_bands; band_size = 64/num_bands
        exact_keys: Sequence[str] = ("prompt", "positive", "negative"),
        key_fn: Callable[[Mapping[str, str]], str] | None = None,
    ) -> None:
        if 64 % num_bands != 0:
            raise InvalidValueError(param_name="num_bands", actual=num_bands, expected="divisor of 64 (e.g., 4, 8, 16, 32)")
        if tokenizer not in {"auto", "word", "char"}:
            raise InvalidValueError(param_name="tokenizer", actual=tokenizer, expected="'auto', 'word', or 'char'")
        if word_ngram < 1 or char_ngram < 1:
            raise InvalidRangeError(param_name="n-gram sizes", actual=min(word_ngram, char_ngram), min_val=1)

        self.threshold_bits = threshold_bits
        self.fields_to_hash = tuple(fields_to_hash)
        self.field_weights = dict(field_weights or {})
        self.tokenizer = tokenizer
        self.word_ngram = int(word_ngram)
        self.char_ngram = int(char_ngram)
        self.strip_accents = bool(strip_accents)
        self.stopwords = set(stopwords or self._default_stopwords())
        self.num_bands = int(num_bands)
        self.band_size = 64 // self.num_bands
        self.exact_keys = tuple(exact_keys)
        self.key_fn = key_fn

        # Precompute band masks/shifts
        self._band_masks = [(1 << self.band_size) - 1 for _ in range(self.num_bands)]
        self._band_shifts = [i * self.band_size for i in range(self.num_bands)]

        # Simple CJK detection regex for "auto" tokenizer
        self._re_cjk = re.compile(r"[\u3400-\u9FFF\uF900-\uFAFF\u3040-\u30FF\uAC00-\uD7AF]")


    def dedupe(self, items: ContrastivePairSet) -> ContrastivePairSet:
        """
        Deduplicate items based on near-duplicate similarity of selected fields.

        arguments:
            items: ContrastivePairSet to deduplicate.

        returns:
            deduplicated ContrastivePairSet (first occurrence kept)

        the processing steps are:
          1) Exact dedup by canonical tuple of exact_keys (e.g., prompt+positive+negative).
          2) For each item, compute 64-bit SimHash fingerprint of selected fields.
          3) Use banded LSH to find candidate near-duplicates.
          4) For candidates, compute exact Hamming distance; if within threshold, treat as duplicate.
          5) Keep first item in each near-duplicate cluster; discard others.
        """
        out: ContrastivePairSet = ContrastivePairSet(
            name=items.name,
            task_type=items.task_type,
        )
        out_fps: list[int] = []

        exact_seen: set[tuple[tuple[str, str], ...]] = set()

        buckets: list[defaultdict[int, list[int]]] = [defaultdict(list) for _ in range(self.num_bands)]

        for it in items.pairs:

            it_dict = {
                "prompt": it.prompt,
                "positive": it.positive_response.model_response,
                "negative": it.negative_response.model_response,
            }
            ex_key = self._exact_key(it_dict)
            if ex_key in exact_seen:
                continue

            fp = self._simhash64_for_item(it_dict)

            candidates: set[int] = set()
            for b, shift in enumerate(self._band_shifts):
                band_val = (fp >> shift) & self._band_masks[b]
                if band_val in buckets[b]:
                    candidates.update(buckets[b][band_val])

            if not candidates and out_fps:
                candidates = set(range(len(out_fps)))

            is_dup = any(self._hamming_distance(fp, out_fps[idx]) <= self.threshold_bits for idx in candidates)
            if is_dup:
                continue

            idx = len(out)
            out.add(it)
            out_fps.append(fp)
            exact_seen.add(ex_key)
            for b, shift in enumerate(self._band_shifts):
                band_val = (fp >> shift) & self._band_masks[b]
                buckets[b][band_val].append(idx)

        return out

    def _simhash64_for_item(self, item: Mapping[str, str]) -> int:
        """
        Compute 64-bit SimHash fingerprint for the given item.

        arguments:
            item: mapping of field -> text
        
        returns:
            64-bit integer SimHash fingerprint
        
        example:
            >>> deduper = SimHashDeduper(fields_to_hash=("prompt","positive"), field_weights={"prompt":2.0})
            >>> item = {"prompt":"Tell me a joke.","positive":"Here's a joke.","negative":"I can't help."}
            >>> deduper._simhash64_for_item(item)
            0b101010101010... (64 bits)
        """
        feats: Counter[str] = Counter()
        if self.key_fn:
            text = self.key_fn(item)
            feats.update(self._extract_features(text))
        else:
            for field in self.fields_to_hash:
                text = item.get(field, "") or ""
                w = float(self.field_weights.get(field, 1.0))
                if not text or w == 0.0:
                    continue
                f = self._extract_features(text)
                if w != 1.0:
                    for k, v in f.items():
                        f[k] = v * w
                feats.update(f)
        return self._simhash64(feats)

    def _simhash64(self, features: Mapping[str, float]) -> int:
        """
        Compute 64-bit SimHash fingerprint from weighted features.
        
        arguments:
            features: mapping of feature -> weight (e.g., shingle -> count or tf-idf)
            
        returns:
            64-bit integer SimHash fingerprint
            
        example:
            >>> SimHashDeduper()._simhash64(Counter({'cat': 1, 'sat': 1, 'mat': 1}))
            0b101010101010... (64 bits)
        """
        v = [0.0] * 64
        for feat, weight in features.items():
            h = self._hash64(feat)
            for i in range(64):
                if h & (1 << i):
                    v[i] += weight
                else:
                    v[i] -= weight

        fp = 0
        for i in range(64):
            if v[i] >= 0:
                fp |= (1 << i)
        return fp

    def _extract_features(self, text: str) -> Counter[str]:
        """
        Extract features (shingles) from text based on tokenizer mode.
        
        arguments:
            text: input string
            
        returns:
            Counter of features (shingle -> count)
            
        example:
            >>> SimHashDeduper()._extract_features("The cat sat on the mat.")
            Counter({'cat': 1, 'sat': 1, 'mat': 1})
            >>> SimHashDeduper(tokenizer="char", char_ngram=3)._extract_features("hello")
            Counter({'hel': 1, 'ell': 1, 'llo': 1})
        """
        t = self._normalize(text)
        mode = self._pick_mode(t)

        if mode == "word":
            toks = [tok for tok in re.findall(r"\w+", t) if tok not in self.stopwords]
            if self.word_ngram == 1:
                return Counter(toks)
            shingles = [" ".join(toks[i:i + self.word_ngram]) for i in range(len(toks) - self.word_ngram + 1)]
            return Counter(shingles)

        if self.char_ngram == 1:
            chars = list(t.replace(" ", ""))  
            return Counter(chars)
        s = re.sub(r"\s+", " ", t)
        s = s.replace(" ", "␠")  
        shingles = [s[i:i + self.char_ngram] for i in range(max(0, len(s) - self.char_ngram + 1))]
        return Counter(shingles)

    def _pick_mode(self, text: str) -> str:
        """
        Decide tokenizer mode based on text and config.
        
        arguments:
            text: input string
        
        returns:
            "word" or "char"
        """
        if self.tokenizer == "auto":
            return "char" if self._re_cjk.search(text) else "word"
        return self.tokenizer

    def _normalize(self, text: str) -> str:
        """
        Unicode NFKC normalization, casefold, optional accent strip, URL/email strip, whitespace

        arguments:
            text: input string

        returns:
            normalized string

        example:
            >>> SimHashDeduper()._normalize("Café at https://example.com!")
            'cafe at <URL> !'
            >>> SimHashDeduper(strip_accents=False)._normalize("The cat sat on the mat.")
            'the cat sat on the mat.'

        the processing steps are:
          1) Replace URLs with <URL> token
          2) Replace emails with <EMAIL> token
          3) Unicode NFKC normalization
          4) Casefold (lowercase + some locale-aware folding)
          5) Optional accent strip (NFKD + remove combining marks)
          6) Collapse whitespace to single spaces, trim leading/trailing
        """
        text = re.sub(r"https?://\S+", " <URL> ", text)
        text = re.sub(r"\b\S+@\S+\b", " <EMAIL> ", text)

        text = unicodedata.normalize("NFKC", text).casefold()

        if self.strip_accents:
            text = unicodedata.normalize("NFKD", text)
            text = "".join(ch for ch in text if not unicodedata.combining(ch))

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _hash64(self, s: str) -> int:
        """
        Stable 64-bit hash of a string.
        
        arguments:
            s: input string
            
        returns:
            64-bit integer hash
            
        example:
            >>> SimHashDeduper()._hash64("wisent")
            TODO: actual value" 
        """
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=8)
        return int.from_bytes(h.digest(), "big", signed=False)

    def _hamming_distance(self, a: int, b: int) -> int:
        """
        Compute Hamming distance between two 64-bit integers.

        arguments:
            a, b: 64-bit integers

        returns:
            Hamming distance (number of differing bits)

        intuition:
            XOR the two integers; the number of set bits in the result is the Hamming distance
            For example, let word_1 = "hause" and word_2 = "mause", then
            a = hash64("hause") = 0b110100101011... (64 bits)
            b = hash64("mause") = 0b110100111011... (64 bits)
            a ^ b = 0b000000110000... (64 bits)
            The number of 1s in a ^ b is the Hamming distance, so here it is 2.
        """
        x = a ^ b
        return x.bit_count() if hasattr(int, "bit_count") else bin(x).count("1")

    def _exact_key(self, item: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        kv = [(k, item.get(k, "")) for k in self.exact_keys]
        return tuple(sorted(kv))

    @staticmethod
    def _default_stopwords() -> set[str]:
        return {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
            "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
            "such", "that", "the", "their", "then", "there", "these", "they",
            "this", "to", "was", "will", "with", "i", "you", "he", "she", "we",
        }
