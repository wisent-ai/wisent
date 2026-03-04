"""
Logits processors for ensuring diverse, non-repetitive responses
across conversation turns.
"""

import torch
from typing import List, Set, Dict, Any
from collections import deque
from wisent.core.utils.config_tools.constants import (
    COMBO_OFFSET, PARSER_DEFAULT_LAYER_START,
)


class OpenerPenaltyProcessor:
    """
    Penalize common repetitive openers for the first few generated
    positions only. Prevents the model from always starting responses
    the same way.
    """
    def __init__(
        self, tokenizer, openers: List[str], *,
        penalty: float, cutoff_pos: int,
    ):
        if penalty is None:
            raise ValueError("penalty is required")
        if cutoff_pos is None:
            raise ValueError("cutoff_pos is required")
        self.tok = tokenizer
        self.penalty = penalty
        self.cutoff_pos = cutoff_pos
        self.first_tokens: Set[int] = set()

        for s in openers:
            ids = self.tok.encode(s, add_special_tokens=False)
            if ids:
                self.first_tokens.add(ids[PARSER_DEFAULT_LAYER_START])

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply penalty to opener tokens if within cutoff position."""
        cur_pos = input_ids.shape[COMBO_OFFSET]
        if cur_pos < self.cutoff_pos:
            for t in self.first_tokens:
                scores[:, t] += self.penalty
        return scores


class TriePenaltyProcessor:
    """
    Cross-turn pattern blocker: if the current suffix matches any
    ledger path, penalize the next token that would continue that path.
    """
    def __init__(
        self, tokenizer, ledger_token_seqs: List[List[int]], *,
        penalty: float, max_depth: int,
    ):
        if penalty is None:
            raise ValueError("penalty is required")
        if max_depth is None:
            raise ValueError("max_depth is required")
        self.tok = tokenizer
        self.penalty = penalty
        self.trie: Dict[int, Any] = {}
        self.max_depth = max_depth

        for seq in ledger_token_seqs:
            node = self.trie
            for tid in seq[:self.max_depth]:
                node = node.setdefault(tid, {})
            node.setdefault("_end", True)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply penalty to continuation tokens matching trie paths."""
        ids = input_ids[PARSER_DEFAULT_LAYER_START].tolist()
        depth = min(self.max_depth, len(ids))

        for big_l in range(
            depth, PARSER_DEFAULT_LAYER_START, -COMBO_OFFSET
        ):
            node = self.trie
            suffix = ids[-big_l:]
            ok = True
            for tid in suffix:
                if tid in node:
                    node = node[tid]
                else:
                    ok = False
                    break
            if ok:
                for next_tid in node.keys():
                    if next_tid == "_end":
                        continue
                    scores[:, next_tid] += self.penalty
                break

        return scores


class PhraseLedger:
    """Rolling buffer of recent assistant responses."""
    def __init__(
        self, *, max_items: int, sample_k: int,
        min_clause_length: int, max_clause_length: int,
        window_size: int,
    ):
        if max_items is None:
            raise ValueError("max_items is required")
        if sample_k is None:
            raise ValueError("sample_k is required")
        if min_clause_length is None:
            raise ValueError("min_clause_length is required")
        if max_clause_length is None:
            raise ValueError("max_clause_length is required")
        if window_size is None:
            raise ValueError("window_size is required")
        self.buf = deque(maxlen=max_items)
        self._max_items = max_items
        self._sample_k = sample_k
        self._min_cl = min_clause_length
        self._max_cl = max_clause_length
        self._ws = window_size

    def add(self, text: str):
        """Add a new response to the ledger."""
        self.buf.append(text.lower())

    def sample(self) -> List[str]:
        """Sample distinct clauses from recent responses."""
        out, seen = [], set()
        for t in reversed(self.buf):
            for clause in t.split("."):
                c = clause.strip()
                if self._min_cl <= len(c) <= self._max_cl:
                    if c not in seen:
                        out.append(c)
                        seen.add(c)
                        if len(out) >= self._sample_k:
                            return out
        return out

    def to_token_sequences(self, tokenizer) -> List[List[int]]:
        """Convert ledger phrases to token sequences."""
        seqs = []
        for t in list(self.buf)[:self._max_items]:
            ids = tokenizer.encode(t, add_special_tokens=False)
            rng_end = max(
                PARSER_DEFAULT_LAYER_START,
                len(ids) - self._ws + COMBO_OFFSET,
            )
            for i in range(
                PARSER_DEFAULT_LAYER_START, rng_end, self._ws,
            ):
                seqs.append(ids[i:i + self._ws])
        return seqs


def build_diversity_processors(
    tokenizer, *,
    opener_penalty: float, opener_cutoff_pos: int,
    trie_penalty: float, trie_max_depth: int,
    phrase_ledger: PhraseLedger = None,
) -> list:
    """Build logits processors for ensuring response diversity."""
    processors = []
    bad_openers = [
        "I don't", "But know this", "Then you",
        "Then you've", "Then you have",
        "What are you willing",
    ]
    processors.append(
        OpenerPenaltyProcessor(
            tokenizer, bad_openers,
            penalty=opener_penalty, cutoff_pos=opener_cutoff_pos,
        )
    )
    if phrase_ledger and len(phrase_ledger.buf) > PARSER_DEFAULT_LAYER_START:
        ledger_seqs = phrase_ledger.to_token_sequences(tokenizer)
        if ledger_seqs:
            processors.append(
                TriePenaltyProcessor(
                    tokenizer, ledger_seqs,
                    penalty=trie_penalty, max_depth=trie_max_depth,
                )
            )
    return processors
