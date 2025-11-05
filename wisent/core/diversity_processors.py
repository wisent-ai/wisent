"""
Logits processors for ensuring diverse, non-repetitive responses across conversation turns.
"""

import torch
from typing import List, Set, Dict, Any
from collections import deque


class OpenerPenaltyProcessor:
    """
    Penalize common repetitive openers for the first few generated positions only.
    Prevents the model from always starting responses the same way.
    """
    def __init__(self, tokenizer, openers: List[str], penalty: float = -2.0, cutoff_pos: int = 5):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            openers: List of opener phrases to penalize (e.g. ["I don't", "Then you've"])
            penalty: Logit penalty to apply (negative = discourage)
            cutoff_pos: Only apply penalty for first N token positions
        """
        self.tok = tokenizer
        self.penalty = penalty
        self.cutoff_pos = cutoff_pos
        self.first_tokens: Set[int] = set()

        # Extract first token of each opener phrase
        for s in openers:
            ids = self.tok.encode(s, add_special_tokens=False)
            if ids:
                self.first_tokens.add(ids[0])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply penalty to opener tokens if within cutoff position."""
        cur_pos = input_ids.shape[1]
        if cur_pos < self.cutoff_pos:
            for t in self.first_tokens:
                scores[:, t] += self.penalty
        return scores


class TriePenaltyProcessor:
    """
    Cross-turn pattern blocker: if the current suffix matches any ledger path,
    penalize the next token that would continue that path.

    This prevents the model from repeating multi-token patterns across different
    conversation turns, even when those patterns are paraphrased.
    """
    def __init__(self, tokenizer, ledger_token_seqs: List[List[int]],
                 penalty: float = -1.0, max_depth: int = 6):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            ledger_token_seqs: List of token sequences to penalize (from recent responses)
            penalty: Logit penalty to apply
            max_depth: Maximum token sequence length to track
        """
        self.tok = tokenizer
        self.penalty = penalty
        self.trie: Dict[int, Any] = {}
        self.max_depth = max_depth

        # Build trie from ledger sequences
        for seq in ledger_token_seqs:
            node = self.trie
            for tid in seq[:max_depth]:
                node = node.setdefault(tid, {})
            node.setdefault("_end", True)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply penalty to continuation tokens that match trie paths."""
        # Assume batch size 1 for simplicity (extend if batching)
        ids = input_ids[0].tolist()

        # Walk back up to max_depth and see if a trie path continues
        depth = min(self.max_depth, len(ids))

        # Try all suffix lengths (greedy from longer to shorter)
        for L in range(depth, 0, -1):
            node = self.trie
            suffix = ids[-L:]
            ok = True

            # Try to match suffix in trie
            for tid in suffix:
                if tid in node:
                    node = node[tid]
                else:
                    ok = False
                    break

            if ok:
                # Penalize all next-token options that would continue this path
                for next_tid in node.keys():
                    if next_tid == "_end":
                        continue
                    scores[:, next_tid] += self.penalty
                break

        return scores


class PhraseLedger:
    """
    Rolling buffer of recent assistant responses for cross-turn pattern detection.
    Keeps track of phrases to avoid repeating across conversation.
    """
    def __init__(self, max_items: int = 200):
        """
        Args:
            max_items: Maximum number of phrases to keep in memory
        """
        self.buf = deque(maxlen=max_items)

    def add(self, text: str):
        """Add a new response to the ledger."""
        self.buf.append(text.lower())

    def sample(self, k: int = 5) -> List[str]:
        """
        Sample k distinct clauses from recent responses.
        Returns shorter, meaningful clauses to inject into prompts.
        """
        out, seen = [], set()
        for t in reversed(self.buf):
            for clause in t.split("."):
                c = clause.strip()
                if 12 <= len(c) <= 80 and c not in seen:
                    out.append(c)
                    seen.add(c)
                    if len(out) >= k:
                        return out
        return out

    def to_token_sequences(self, tokenizer, window_size: int = 6) -> List[List[int]]:
        """
        Convert ledger phrases to token sequences for trie processor.

        Args:
            tokenizer: HuggingFace tokenizer
            window_size: Size of sliding window for token sequences

        Returns:
            List of token sequences (each window_size tokens long)
        """
        ledger_token_seqs = []
        for t in list(self.buf)[:200]:  # Use most recent 200
            ids = tokenizer.encode(t, add_special_tokens=False)
            # Sliding windows of window_size tokens
            for i in range(0, max(0, len(ids) - window_size + 1), window_size):
                ledger_token_seqs.append(ids[i:i + window_size])
        return ledger_token_seqs


def build_diversity_processors(tokenizer, phrase_ledger: PhraseLedger = None,
                               bad_openers: List[str] = None) -> list:
    """
    Build a list of logits processors for ensuring response diversity.

    Args:
        tokenizer: HuggingFace tokenizer
        phrase_ledger: Optional PhraseLedger with recent responses
        bad_openers: Optional list of opener phrases to penalize

    Returns:
        List of logits processors to pass to model.generate()
    """
    processors = []

    # Default bad openers if not provided
    if bad_openers is None:
        bad_openers = [
            "I don't", "But know this", "Then you", "Then you've",
            "Then you have", "What are you willing",
        ]

    # Add opener penalty processor
    if bad_openers:
        processors.append(
            OpenerPenaltyProcessor(tokenizer, bad_openers, penalty=-2.0, cutoff_pos=5)
        )

    # Add trie penalty processor if we have a ledger
    if phrase_ledger and len(phrase_ledger.buf) > 0:
        ledger_token_seqs = phrase_ledger.to_token_sequences(tokenizer, window_size=6)
        if ledger_token_seqs:
            processors.append(
                TriePenaltyProcessor(tokenizer, ledger_token_seqs, penalty=-1.0, max_depth=6)
            )

    return processors
