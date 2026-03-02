"""
RawCachedActivations for storing full hidden state sequences.

Stores complete sequences so any extraction strategy in the same text family
can be applied without re-running the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import torch

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy, extract_activation
from .cached_activations import CachedActivations, get_strategy_text_family


@dataclass
class RawPairData:
    """Raw hidden states and metadata for a single contrastive pair."""
    pos_hidden_states: Dict[str, torch.Tensor]
    neg_hidden_states: Dict[str, torch.Tensor]
    pos_answer_text: str
    neg_answer_text: str
    pos_prompt_len: int
    neg_prompt_len: int


@dataclass
class RawCachedActivations:
    """
    Cache full hidden states per (benchmark, text_family).

    Stores complete sequences so any extraction strategy in the same
    text family can be applied without re-running the model.
    """
    benchmark: str
    text_family: str  # "chat", "role_play", "mc", "completion"
    model_name: str
    num_layers: int

    pairs: List[RawPairData] = field(default_factory=list)
    num_pairs: int = 0
    hidden_size: int = 0

    def add_pair(
        self,
        pos_hidden_states: Dict[str, torch.Tensor],
        neg_hidden_states: Dict[str, torch.Tensor],
        pos_answer_text: str,
        neg_answer_text: str,
        pos_prompt_len: int,
        neg_prompt_len: int,
    ) -> None:
        """Add raw hidden states for a contrastive pair."""
        pair_data = RawPairData(
            pos_hidden_states={k: v.clone() for k, v in pos_hidden_states.items()},
            neg_hidden_states={k: v.clone() for k, v in neg_hidden_states.items()},
            pos_answer_text=pos_answer_text,
            neg_answer_text=neg_answer_text,
            pos_prompt_len=pos_prompt_len,
            neg_prompt_len=neg_prompt_len,
        )
        self.pairs.append(pair_data)
        self.num_pairs = len(self.pairs)

        if self.hidden_size == 0 and pos_hidden_states:
            first_tensor = next(iter(pos_hidden_states.values()))
            self.hidden_size = first_tensor.shape[-1]

    def extract_with_strategy(
        self,
        strategy: ExtractionStrategy,
        tokenizer,
        layer: int | str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract activations using a specific strategy.

        Args:
            strategy: Extraction strategy (must be in same text family)
            tokenizer: Tokenizer for computing answer token positions
            layer: Layer to extract from

        Returns:
            Tuple of (pos_activations, neg_activations) each [num_pairs, hidden_size]
        """
        if get_strategy_text_family(strategy) != self.text_family:
            raise ValueError(
                f"Strategy {strategy.value} (family: {get_strategy_text_family(strategy)}) "
                f"incompatible with cached family: {self.text_family}"
            )

        layer_name = str(layer)
        pos_acts = []
        neg_acts = []

        for pair_data in self.pairs:
            pos_hs = pair_data.pos_hidden_states.get(layer_name)
            if pos_hs is not None:
                pos_vec = extract_activation(
                    strategy, pos_hs, pair_data.pos_answer_text,
                    tokenizer, pair_data.pos_prompt_len
                )
                pos_acts.append(pos_vec)

            neg_hs = pair_data.neg_hidden_states.get(layer_name)
            if neg_hs is not None:
                neg_vec = extract_activation(
                    strategy, neg_hs, pair_data.neg_answer_text,
                    tokenizer, pair_data.neg_prompt_len
                )
                neg_acts.append(neg_vec)

        return torch.stack(pos_acts), torch.stack(neg_acts)

    def to_cached_activations(
        self,
        strategy: ExtractionStrategy,
        tokenizer,
    ) -> CachedActivations:
        """Convert to CachedActivations for a specific strategy."""
        cached = CachedActivations(
            benchmark=self.benchmark,
            strategy=strategy,
            model_name=self.model_name,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
        )

        for pair_data in self.pairs:
            pos_dict = {}
            neg_dict = {}

            for layer_name in pair_data.pos_hidden_states.keys():
                pos_hs = pair_data.pos_hidden_states[layer_name]
                neg_hs = pair_data.neg_hidden_states[layer_name]

                pos_dict[layer_name] = extract_activation(
                    strategy, pos_hs, pair_data.pos_answer_text,
                    tokenizer, pair_data.pos_prompt_len
                )
                neg_dict[layer_name] = extract_activation(
                    strategy, neg_hs, pair_data.neg_answer_text,
                    tokenizer, pair_data.neg_prompt_len
                )

            cached.pair_activations.append((pos_dict, neg_dict))

        cached.num_pairs = len(cached.pair_activations)
        return cached

    def get_available_layers(self) -> List[str]:
        """Get list of available layer names."""
        if not self.pairs:
            return []
        return list(self.pairs[0].pos_hidden_states.keys())
