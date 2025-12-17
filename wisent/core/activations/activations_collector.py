from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING
import torch


from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.activations.core.atoms import LayerActivations, ActivationAggregationStrategy, LayerName, RawActivationMap
from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy
from wisent.core.errors import NoHiddenStatesError, TokenizerMissingMethodError, UnknownTypeError, InvalidDataFormatError

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel

__all__ = ["ActivationCollector"]

@dataclass(slots=True)
class ActivationCollector:
    """
        Collect per-layer activations for (prompt + response) using a chat template.

        arguments:
            model:
                :class: WisentModel
            store_device:
                Device to store collected activations on (default "cpu").
            dtype:
                Optional torch.dtype to cast activations to (e.g., torch.float32).
                If None, keep original dtype.

        detailed explanation:

        Let:
        - L = 4 transformer blocks
        - hidden size H = 256
        - prompt tokenized length T_prompt = 14
        - full sequence (prompt + response) tokenized length T_full = 22

        Step 1: Build templated strings (NOT tokenized yet)
            prompt_text = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
            full_text   = tok.apply_chat_template(
                [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}],
                tokenize=False, add_generation_prompt=False
            )

        Step 2: Tokenize both with identical flags
            prompt_enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            full_enc   = tok(full_text,   return_tensors="pt", add_special_tokens=False)

        Shapes:
            prompt_enc["input_ids"].shape == (1, T_prompt) == (1, 14)
            full_enc["input_ids"].shape   == (1, T_full)   == (1, 22)

        Boundary:
            prompt_len = prompt_enc["input_ids"].shape[-1] == 14
            continuation tokens in the full sequence start at index 14.

        Step 3: Forward pass with hidden states
            out = model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs  = out.hidden_states

        hs is a tuple of length L + 1 (includes embedding layer at index 0):
            len(hs) == 5  -> indices: 0=embeddings, 1..4 = blocks
            Each hs[i].shape == (1, T_full, H) == (1, 22, 256)

        We map layer names "1".."L" to hs[1]..hs[L]:
            "1" -> hs[1], "2" -> hs[2], ..., "4" -> hs[4]

        Step 4: Per-layer extraction
            For a chosen layer i (1-based), get hs[i].squeeze(0) -> shape (T_full, H) == (22, 256)

            If return_full_sequence=True:
                store value with shape (T_full, H) == (22, 256)
            Else (aggregate to a single vector [H]):
                - CONTINUATION_TOKEN / CHOICE_TOKEN: take first continuation token -> cont[0] -> (H,)
                - FIRST_TOKEN:     layer_seq[0]    -> (H,)
                - LAST_TOKEN:      layer_seq[-1]   -> (H,)
                - MEAN_POOLING:    cont.mean(0)    -> (H,)
                - MAX_POOLING:     cont.max(0)[0]  -> (H,)

            where:
                layer_seq = hs[i].squeeze(0)                # (22, 256)
                cont_start = prompt_len = 14
                cont = layer_seq[14:]                       # (22-14=8, 256)

        Step 5: Storage and return
            - We move each stored tensor to 'store_device' (default "cpu") and cast to 'dtype'
            if provided (e.g., float32).
            - Keys are layer names: "1", "2", ..., "L".
            - Results are wrapped into LayerActivations with `activation_aggregation_strategy`
            set to your chosen strategy (or None if keeping full sequences).

        examples:
            Example usage (aggregated vectors per layer)
                >>> collector = ActivationCollector(model=my_wrapper, store_device="cpu", dtype=torch.float32)
                >>> updated_pair = collector.collect_for_pair(
                ...     pair,
                ...     layers=["1", "3"],  # subset (or None for all)
                ...     aggregation=ActivationAggregationStrategy.CONTINUATION_TOKEN,
                ...     return_full_sequence=False,
                ... )
                >>> pos_acts = updated_pair.positive_response.layers_activations
                >>> pos_acts.summary()
                    {
                    '1': {'shape': (256,), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '3': {'shape': (256,), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '_activation_aggregation_strategy': {'strategy': 'continuation_token'}
                    }

            Example usage (full sequences per layer)
                >>> updated_pair = collector.collect_for_pair(
                ...     pair,
                ...     layers=None,  # all layers "1".."L"
                ...     aggregation=ActivationAggregationStrategy.MEAN_POOLING,  # ignored when return_full_sequence=True
                ...     return_full_sequence=True,
                ... )
                >>> neg_acts = updated_pair.negative_response.layers_activations
                >>> # Suppose L=4 and T_full=22, H=256
                >>> neg_acts.summary()
                    {
                    '1': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '2': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '3': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '4': {'shape': (22, 256), 'dtype': 'torch.float32', 'device': 'cpu', 'requires_grad': False},
                    '_activation_aggregation_strategy': {'strategy': None}
                    }
    """

    model: "WisentModel"
    store_device: str | torch.device = "cpu"
    dtype: torch.dtype | None = None

    def collect_for_pair(
        self,
        pair: ContrastivePair,
        layers: Sequence[LayerName] | None = None,
        aggregation: ActivationAggregationStrategy = ActivationAggregationStrategy.CONTINUATION_TOKEN,
        return_full_sequence: bool = False,
        normalize_layers: bool = False,
        prompt_strategy: PromptConstructionStrategy = PromptConstructionStrategy.CHAT_TEMPLATE,
    ) -> ContrastivePair:
        pos_text = _resp_text(pair.positive_response)
        neg_text = _resp_text(pair.negative_response)
        
        # For multiple choice, we need both responses to construct the prompt
        other_response = neg_text if prompt_strategy == PromptConstructionStrategy.MULTIPLE_CHOICE else None
        pos = self._collect_for_texts(pair.prompt, pos_text,
                                      layers, aggregation, return_full_sequence, normalize_layers, prompt_strategy,
                                      other_response=other_response, is_positive=True)
        
        other_response = pos_text if prompt_strategy == PromptConstructionStrategy.MULTIPLE_CHOICE else None
        neg = self._collect_for_texts(pair.prompt, neg_text,
                                      layers, aggregation, return_full_sequence, normalize_layers, prompt_strategy,
                                      other_response=other_response, is_positive=False)
        return pair.with_activations(positive=pos, negative=neg)

    def _collect_for_texts(
        self,
        prompt: str,
        response: str,
        layers: Sequence[LayerName] | None,
        aggregation: ActivationAggregationStrategy,
        return_full_sequence: bool,
        normalize_layers: bool = False,
        prompt_strategy: PromptConstructionStrategy = PromptConstructionStrategy.CHAT_TEMPLATE,
        other_response: str | None = None,
        is_positive: bool = True,
    ) -> LayerActivations:

        self._ensure_eval_mode()
        with torch.inference_mode():
            tok = self.model.tokenizer # type: ignore[union-attr]

            # 1) Build prompts based on strategy
            prompt_text, full_text = self._build_prompts_for_strategy(
                prompt, response, prompt_strategy, tok,
                other_response=other_response, is_positive=is_positive
            )

            # 2) Tokenize both with identical flags
            prompt_enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            full_enc   = tok(full_text,   return_tensors="pt", add_special_tokens=False)

            # 3) Boundary from prompt-only tokens (CPU is fine)
            prompt_len = int(prompt_enc["input_ids"].shape[-1])

            # 4) Move only the batch that goes into the model
            compute_device = getattr(self.model, "compute_device", None) or next(self.model.hf_model.parameters()).device
            full_enc = {k: v.to(compute_device) for k, v in full_enc.items()}

            # 5) Forward on the full sequence to get hidden states
            out = self.model.hf_model(**full_enc, output_hidden_states=True, use_cache=False)
            hs: tuple[torch.Tensor, ...] = out.hidden_states  # hs[0]=emb, hs[1:]=layers

            if not hs:
                raise NoHiddenStatesError()

            n_blocks = len(hs) - 1
            names_by_idx = [str(i) for i in range(1, n_blocks + 1)]

            keep = self._select_indices(layers, n_blocks)
            collected: RawActivationMap = {}

            for idx in keep:
                name = names_by_idx[idx]
                h = hs[idx + 1].squeeze(0)  # [1, T, H] -> [T, H]
                if return_full_sequence:
                    value = h
                else:
                    value = self._aggregate(h, aggregation, prompt_len)
                value = value.to(self.store_device)
                if self.dtype is not None:
                    value = value.to(self.dtype)
                
                if normalize_layers:
                    value = self._normalization(value)

                collected[name] = value

            return LayerActivations(
                collected,
                activation_aggregation_strategy=None if return_full_sequence else aggregation,
            )

    def _build_prompts_for_strategy(
        self,
        prompt: str,
        response: str,
        strategy: PromptConstructionStrategy,
        tokenizer,
        other_response: str | None = None,
        is_positive: bool = True,
    ) -> tuple[str, str]:
        """
        Build prompt_text and full_text based on the chosen prompt construction strategy.

        Args:
            prompt: The user prompt/question
            response: The response to collect activations for
            strategy: The prompt construction strategy
            tokenizer: The tokenizer
            other_response: For multiple_choice, the other response option
            is_positive: For multiple_choice, whether 'response' is the positive option

        Returns:
            (prompt_text, full_text): Tuple of prompt-only text and prompt+response text
        """
        if strategy == PromptConstructionStrategy.CHAT_TEMPLATE:
            # Use model's built-in chat template
            if not hasattr(tokenizer, "apply_chat_template"):
                raise TokenizerMissingMethodError("apply_chat_template")
            try:
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                full_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": response}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except ValueError as e:
                if "chat_template is not set" in str(e):
                    # Fallback to direct completion for models without chat templates
                    prompt_text = prompt
                    full_text = f"{prompt} {response}"
                else:
                    raise

        elif strategy == PromptConstructionStrategy.DIRECT_COMPLETION:
            # Raw text without any chat formatting
            prompt_text = prompt
            full_text = f"{prompt} {response}"

        elif strategy == PromptConstructionStrategy.INSTRUCTION_FOLLOWING:
            # Use chat template with instruction-style user message
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    instruction_prompt = f"Follow this instruction: {prompt}"
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": instruction_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    full_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": instruction_prompt},
                         {"role": "assistant", "content": response}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except ValueError:
                    prompt_text = f"Follow this instruction: {prompt}"
                    full_text = f"{prompt_text} {response}"
            else:
                prompt_text = f"Follow this instruction: {prompt}"
                full_text = f"{prompt_text} {response}"

        elif strategy == PromptConstructionStrategy.MULTIPLE_CHOICE:
            # Multiple choice: show both responses as A/B, answer with just "A" or "B"
            # IMPORTANT: We randomize whether positive goes in A or B (based on prompt hash)
            # This ensures balanced A/B across the dataset, so the A vs B bias cancels out
            # and only the semantic signal remains (like in the CAA paper)
            if other_response is None:
                raise ValueError("MULTIPLE_CHOICE strategy requires other_response to be provided")
            
            # Deterministic "random" based on prompt - same for both pos and neg of a pair
            pos_goes_in_b = hash(prompt) % 2 == 0
            
            if is_positive:
                # response is positive, other_response is negative
                if pos_goes_in_b:
                    option_a = other_response  # negative
                    option_b = response        # positive
                    answer = "B"
                else:
                    option_a = response        # positive
                    option_b = other_response  # negative
                    answer = "A"
            else:
                # response is negative, other_response is positive
                if pos_goes_in_b:
                    option_a = response        # negative
                    option_b = other_response  # positive
                    answer = "A"
                else:
                    option_a = other_response  # positive
                    option_b = response        # negative
                    answer = "B"
            
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    mc_prompt = f"Which is better for the question '{prompt}'?\nA. {option_a}\nB. {option_b}\nAnswer with A or B:"
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": mc_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    full_text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": mc_prompt},
                         {"role": "assistant", "content": answer}],
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except ValueError:
                    prompt_text = f"Which is better for '{prompt}'?\nA. {option_a}\nB. {option_b}\nAnswer:"
                    full_text = f"{prompt_text} {answer}"
            else:
                prompt_text = f"Which is better for '{prompt}'?\nA. {option_a}\nB. {option_b}\nAnswer:"
                full_text = f"{prompt_text} {answer}"

        elif strategy == PromptConstructionStrategy.ROLE_PLAYING:
            # Use chat template with role-playing system message
            # The response goes in the system message, assistant says a random general token
            # This way the difference is purely in the system persona, not the output
            # Using a deterministic token based on prompt hash so pos/neg pairs get same token
            random_tokens = ["I", "Well", "The", "Sure", "Let", "That", "It", "This", "My", "To"]
            random_token = random_tokens[hash(prompt) % len(random_tokens)]
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    role_prompt = f"Respond to: {prompt}"
                    system_msg = f"You are a person who would naturally respond with sentiments like: {response}"
                    prompt_text = tokenizer.apply_chat_template(
                        [{"role": "system", "content": system_msg},
                         {"role": "user", "content": role_prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    # Append random token directly without end token (like repeng)
                    # This captures activations mid-generation, not at end
                    full_text = f"{prompt_text}{random_token}"
                except ValueError:
                    prompt_text = f"As someone who feels {response}, respond to: {prompt}"
                    full_text = f"{prompt_text} {random_token}"
            else:
                prompt_text = f"As someone who feels {response}, respond to: {prompt}"
                full_text = f"{prompt_text} {random_token}"

        else:
            raise UnknownTypeError(entity_type="prompt_construction_strategy", value=str(strategy))

        return prompt_text, full_text

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

    def _aggregate(
        self,
        layer_seq: torch.Tensor,  # [T, H]
        aggregation: ActivationAggregationStrategy,
        prompt_len: int,
    ) -> torch.Tensor:          # [H]
        if layer_seq.ndim != 2:
            raise InvalidDataFormatError(reason=f"Expected [seq_len, hidden_dim], got {tuple(layer_seq.shape)}")

        # continuation = tokens after the prompt boundary
        cont_start = min(max(prompt_len, 0), layer_seq.shape[0] - 1)
        cont = layer_seq[cont_start:] if cont_start < layer_seq.shape[0] else layer_seq[-1:].contiguous()
        if cont.numel() == 0:
            cont = layer_seq[-1:].contiguous()

        s = aggregation
        
        if s in (ActivationAggregationStrategy.CONTINUATION_TOKEN):
            return cont[0]
        
        elif s in (ActivationAggregationStrategy.CHOICE_TOKEN):
            choice_idx = prompt_len + 1
            if choice_idx < layer_seq.shape[0]:
                return layer_seq[choice_idx]
            else:
                return layer_seq[-1]
        elif s is ActivationAggregationStrategy.FIRST_TOKEN:
            return layer_seq[0]
        elif s is ActivationAggregationStrategy.LAST_TOKEN:
            return layer_seq[-1]
        elif s is ActivationAggregationStrategy.MEAN_POOLING:
            return cont.mean(dim=0)
        elif s is ActivationAggregationStrategy.MAX_POOLING:
            return cont.max(dim=0).values
        elif s is ActivationAggregationStrategy.MIN_POOLING:
            return cont.min(dim=0).values
        else:
            return cont[0]

    def _normalization(
        self,
        x: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """
        Safely L2-normalize 'x' along 'dim'.

        arguments:
            x:
                Tensor of the shape [..., H] or [T, H]
            dim:
                Dimension along which to normalize (default -1, the last dimension).
            eps:
                Small value to avoid division by zero (default 1e-12).

        returns:
            L2-normalized tensor of the same shape as 'x'.
        """
        if not torch.is_floating_point(x):
            return x
        
        norm = torch.linalg.vector_norm(x, ord=2, dim=dim, keepdim=True)

        mask = norm > eps

        safe_norm = torch.where(mask, norm, torch.ones_like(norm))
        y = x / safe_norm
        y = torch.where(mask, y, torch.zeros_like(y))

        return y

    def _ensure_eval_mode(self) -> None:
        try:
            self.model.hf_model.eval()
        except Exception:
            pass

    def collect_batched(
        self,
        texts: list[str],
        layers: Sequence[LayerName] | None = None,
        aggregation: ActivationAggregationStrategy = ActivationAggregationStrategy.LAST_TOKEN,
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Collect activations for multiple texts in batches.
        
        Args:
            texts: List of text strings to collect activations for
            layers: Which layers to collect (e.g., ["8"]), or None for all
            aggregation: How to aggregate sequence to single vector
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
                    print(f"      Processing batch {batch_idx + 1}/{num_batches} ({start}/{len(texts)} texts)...", end='\r', flush=True)
                
                # Tokenize batch with padding
                encoded = tok(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=True,
                )
                encoded = {k: v.to(compute_device) for k, v in encoded.items()}
                
                # Forward pass
                try:
                    out = self.model.hf_model(**encoded, output_hidden_states=True, use_cache=False)
                except torch.cuda.OutOfMemoryError:
                    # Try to recover by clearing cache and retrying
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
                
                # Get attention mask to find actual sequence lengths
                attention_mask = encoded.get("attention_mask")
                
                # Process each item in batch
                for i in range(len(batch_texts)):
                    collected: dict[str, torch.Tensor] = {}
                    
                    # Find actual sequence length for this item
                    if attention_mask is not None:
                        seq_len = int(attention_mask[i].sum().item())
                    else:
                        seq_len = hs[0].shape[1]
                    
                    for idx in keep:
                        name = names_by_idx[idx]
                        h = hs[idx + 1][i, :seq_len, :]  # [seq_len, H]
                        
                        # Aggregate to single vector
                        if aggregation == ActivationAggregationStrategy.LAST_TOKEN:
                            value = h[-1]
                        elif aggregation == ActivationAggregationStrategy.FIRST_TOKEN:
                            value = h[0]
                        elif aggregation == ActivationAggregationStrategy.MEAN_POOLING:
                            value = h.mean(dim=0)
                        else:
                            value = h[-1]  # Default to last token
                        
                        collected[name] = value.to(self.store_device)
                    
                    results.append(collected)
                
                # Clear GPU memory after each batch
                del out, hs, encoded
                torch.cuda.empty_cache()
        
        if show_progress:
            print(f"      Processed {len(texts)} texts in {num_batches} batches" + " " * 20)
        
        return results


def _resp_text(resp_obj: object) -> str:
    for attr in ("model_response", "text"):
        if hasattr(resp_obj, attr) and isinstance(getattr(resp_obj, attr), str):
            return getattr(resp_obj, attr)
    return str(resp_obj)

if __name__ == "__main__":
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse

    model = WisentModel(model_name="/home/gg/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6")
    collector = ActivationCollector(model=model, store_device="cpu")

    pair = ContrastivePair(
        prompt="The capital of France is",
        positive_response=PositiveResponse(" Paris."),
        negative_response=NegativeResponse(" London."),
    )

    updated = collector.collect_for_pair(
        pair,
        layers=["1", "3"],
        aggregation=ActivationAggregationStrategy.CONTINUATION_TOKEN,
        return_full_sequence=False,
    )

    print(updated)

