from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING
import torch

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.activations.core.atoms import LayerActivations, LayerName, RawActivationMap
from wisent.core.activations.extraction_strategy import (
    ExtractionStrategy,
    build_extraction_texts,
    extract_activation,
)
from wisent.core.errors import NoHiddenStatesError

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel

__all__ = ["ActivationCollector"]


@dataclass(slots=True)
class ActivationCollector:
    """
    Collect per-layer activations for contrastive pairs.

    Args:
        model: WisentModel instance
        store_device: Device to store collected activations on (default: "cpu" to avoid GPU OOM)
        dtype: Optional torch.dtype to cast activations to

    Example:
        >>> collector = ActivationCollector(model=my_model)
        >>> updated_pair = collector.collect(
        ...     pair,
        ...     strategy=ExtractionStrategy.CHAT_LAST,
        ...     layers=["8", "12"],
        ... )
        >>> pos_acts = updated_pair.positive_response.layers_activations
        >>> pos_acts.summary()
        {'8': {'shape': (2048,), ...}, '12': {...}}
    """

    model: "WisentModel"
    store_device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def collect(
        self,
        pair: ContrastivePair,
        strategy: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
        layers: Sequence[LayerName] | None = None,
        normalize: bool = False,
    ) -> ContrastivePair:
        """
        Collect activations for a contrastive pair.

        Args:
            pair: The contrastive pair to collect activations for
            strategy: Extraction strategy (e.g., CHAT_LAST, CHAT_MEAN, ROLE_PLAY, MC_BALANCED)
            layers: Which layers to collect (e.g., ["8", "12"]), or None for all
            normalize: Whether to L2-normalize activations

        Returns:
            ContrastivePair with activations attached
        """
        pos_text = _resp_text(pair.positive_response)
        neg_text = _resp_text(pair.negative_response)

        needs_other = strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION)
        other_for_pos = neg_text if needs_other else None
        other_for_neg = pos_text if needs_other else None

        pos = self._collect_single(
            pair.prompt, pos_text, strategy, layers, normalize,
            other_response=other_for_pos, is_positive=True
        )
        neg = self._collect_single(
            pair.prompt, neg_text, strategy, layers, normalize,
            other_response=other_for_neg, is_positive=False
        )
        return pair.with_activations(positive=pos, negative=neg)

    def _collect_single(
        self,
        prompt: str,
        response: str,
        strategy: ExtractionStrategy,
        layers: Sequence[LayerName] | None,
        normalize: bool = False,
        other_response: str | None = None,
        is_positive: bool = True,
    ) -> LayerActivations:
        """Collect activations for a single prompt-response pair."""
        self._ensure_eval_mode()
        with torch.inference_mode():
            tok = self.model.tokenizer

            full_text, answer_text, prompt_only = build_extraction_texts(
                strategy, prompt, response, tok,
                other_response=other_response, is_positive=is_positive
            )

            if prompt_only:
                prompt_enc = tok(prompt_only, return_tensors="pt", add_special_tokens=False)
                prompt_len = int(prompt_enc["input_ids"].shape[-1])
            else:
                prompt_len = 0

            full_enc = tok(full_text, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=2048)

            compute_device = getattr(self.model, "compute_device", None) or next(self.model.hf_model.parameters()).device
            full_enc = {k: v.to(compute_device) for k, v in full_enc.items()}

            out = self.model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states

            if not hs:
                raise NoHiddenStatesError()

            n_blocks = len(hs) - 1
            names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
            keep = self._select_indices(layers, n_blocks)

            collected: RawActivationMap = {}
            for idx in keep:
                name = names_by_idx[idx]
                h = hs[idx + 1].squeeze(0)

                value = extract_activation(strategy, h, answer_text, tok, prompt_len)
                value = value.to(self.store_device)

                if self.dtype is not None:
                    value = value.to(self.dtype)

                if normalize:
                    value = self._normalize(value)

                collected[name] = value

            return LayerActivations(collected)

    def collect_batched(
        self,
        texts: list[str],
        strategy: ExtractionStrategy = ExtractionStrategy.CHAT_LAST,
        layers: Sequence[LayerName] | None = None,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Collect activations for multiple texts in batches.

        Args:
            texts: List of text strings to collect activations for
            strategy: Extraction strategy (only CHAT_LAST supported for raw texts)
            layers: Which layers to collect, or None for all
            batch_size: Number of texts to process at once
            show_progress: Whether to print progress

        Returns:
            List of dicts mapping layer name -> activation tensor [H]
        """
        self._ensure_eval_mode()
        results: list[dict[str, torch.Tensor]] = []

        tok = self.model.tokenizer
        compute_device = getattr(self.model, "compute_device", None) or next(self.model.hf_model.parameters()).device

        num_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.inference_mode():
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, len(texts))
                batch_texts = texts[start:end]

                if show_progress and batch_idx % 10 == 0:
                    print(f"Processing batch {batch_idx + 1}/{num_batches}...", end='\r', flush=True)

                encoded = tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=True,
                )
                encoded = {k: v.to(compute_device) for k, v in encoded.items()}

                try:
                    out = self.model.hf_model(**encoded, output_hidden_states=True, use_cache=False)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    out = self.model.hf_model(**encoded, output_hidden_states=True, use_cache=False)

                hs = out.hidden_states
                if not hs:
                    raise NoHiddenStatesError()

                n_blocks = len(hs) - 1
                names_by_idx = [str(i) for i in range(1, n_blocks + 1)]
                keep = self._select_indices(layers, n_blocks)

                attention_mask = encoded.get("attention_mask")

                for i in range(len(batch_texts)):
                    collected: dict[str, torch.Tensor] = {}

                    if attention_mask is not None:
                        seq_len = int(attention_mask[i].sum().item())
                    else:
                        seq_len = hs[0].shape[1]

                    for idx in keep:
                        name = names_by_idx[idx]
                        h = hs[idx + 1][i, :seq_len, :]

                        if strategy == ExtractionStrategy.CHAT_MEAN:
                            value = h.mean(dim=0)
                        elif strategy == ExtractionStrategy.CHAT_FIRST:
                            value = h[0]
                        elif strategy in (ExtractionStrategy.CHAT_LAST, ExtractionStrategy.ROLE_PLAY, 
                                          ExtractionStrategy.MC_BALANCED,
                                          ExtractionStrategy.CHAT_MAX_NORM, ExtractionStrategy.CHAT_WEIGHTED):
                            value = h[-1]
                        else:
                            raise ValueError(f"Unsupported strategy for batched collection: {strategy}")

                        collected[name] = value.to(self.store_device)

                    results.append(collected)

                del out, hs, encoded
                torch.cuda.empty_cache()

        if show_progress:
            print(f"Processed {len(texts)} texts in {num_batches} batches" + " " * 20)

        return results

    def _select_indices(self, layer_names: Sequence[str] | None, n_blocks: int) -> list[int]:
        """Map layer names '1'..'L' -> indices 0..L-1."""
        if not layer_names:
            return list(range(n_blocks))
        out: list[int] = []
        for name in layer_names:
            try:
                i = int(name)
            except ValueError:
                raise KeyError(f"Layer name must be numeric string like '3', got {name!r}")
            if not (1 <= i <= n_blocks):
                raise IndexError(f"Layer '{i}' out of range 1..{n_blocks}")
            out.append(i - 1)
        return sorted(set(out))

    def _normalize(self, x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
        """Safely L2-normalize tensor."""
        if not torch.is_floating_point(x):
            return x
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)
        mask = norm > eps
        safe_norm = torch.where(mask, norm, torch.ones_like(norm))
        y = x / safe_norm
        return torch.where(mask, y, torch.zeros_like(y))

    def _ensure_eval_mode(self) -> None:
        try:
            self.model.hf_model.eval()
        except Exception:
            pass


def _resp_text(resp_obj: object) -> str:
    for attr in ("model_response", "text"):
        if hasattr(resp_obj, attr) and isinstance(getattr(resp_obj, attr), str):
            return getattr(resp_obj, attr)
    return str(resp_obj)
