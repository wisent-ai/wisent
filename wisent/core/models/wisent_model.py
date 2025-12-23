from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterable

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextIteratorStreamer
)



from wisent.core.models.core.atoms import SteeringPlan, SteeringVector, HookHandleGroup, GenerationStats, TopLogits
from wisent.core.activations.core.atoms import RawActivationMap

from wisent.core.prompts.core.atom import ChatMessage
from wisent.core.utils.device import resolve_default_device, resolve_torch_device, preferred_dtype
from wisent.core.contrastive_pairs.diagnostics import run_control_steering_diagnostics
from wisent.core.errors import (
    ChatTemplateNotAvailableError,
    DecoderLayersNotFoundError,
    HiddenSizeNotFoundError,
    TokenizerMissingMethodError,
    ControlVectorDiagnosticsError,
    LayerNotFoundError,
    InsufficientDataError,
)

import threading

__all__ = ["WisentModel"]


logger = logging.getLogger(__name__)

class WisentModel:
    """
    Wrapper around a causal LM (HF transformers) with steering capabilities.

    atributes:
        model_name:
            HF repo id or local path (e.g., 'meta-llama/Llama-3-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct).
        device:
            'cuda', 'cuda:0', 'cpu', etc. If None, leave to HF defaults/accelerate.
        hf_model:
            the loaded PreTrainedModel instance.    
        tokenizer:
            the loaded PreTrainedTokenizerBase instance.
        hidden_size:
            model hidden size (last dim of residual stream).
        num_layers:
            number of decoder blocks we can hook.
        _steering_plan:
            current SteeringPlan (can be empty).
        _hook_group:
            manages active hooks for clean detach.
    """
    def __init__(
            self,
            model_name: str,
            steering_layers: list[RawActivationMap] | RawActivationMap | None = None,
            steering_weights: list[float] | None = None,
            layers_description: list[str] | None = None,
            device: str | None = None,
            hf_model: AutoModelForCausalLM | None = None
        ):
        """
        Initialize the wrapper (model + tokenizer + default steering plan).

        arguments:
            model_name:
                HF repo id or local path (e.g., 'meta-llama/Llama-3-8B-Instruct', 'Qwen/Qwen2.5-7B-Instruct').
            steering_layers:
                list of RawActivationMap or single RawActivationMap of steering vectors (layer_name -> tensor), optional (can be {}).
                We can have for example steering vectors obtained during training on a specific trait (e.g., toxicity and evilness).
                So, by passing multiple steering vectors, we can combine them at inference time. If we don't pass any weights,
                they will be equally weighted.
            steering_weights:
                list of weights for each steering vector, optional (can be None). If None, all vectors are equally weighted.
            device:
                'cuda', 'cuda:0', 'cpu', etc. If None, leave to HF defaults/accelerate.
            hf_model:
                optional preloaded model (skips from_pretrained if provided).       
        """
        self.model_name = model_name
        self.device = resolve_default_device() if device is None or device == "auto" else device

        # Determine appropriate dtype and settings for the device
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",  # Always use eager attention - no flash-attn dependency
        }

        # Use centralized preferred_dtype for consistency across codebase
        load_kwargs["torch_dtype"] = preferred_dtype(self.device)
        if self.device == "mps":
            load_kwargs["device_map"] = "mps"
        elif self.device == "cuda" or self.device == "auto":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        self.hf_model: PreTrainedModel = hf_model or AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )

        device_map_used = load_kwargs.get("device_map")

        # Only move to device if device_map wasn't used
        if device_map_used is None:
            self.hf_model.to(self.device)

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )

        if not self._is_chat_tokenizer():
            raise TokenizerMissingMethodError("apply_chat_template")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.hf_model.generation_config, "pad_token_id", None) is None:
            self.hf_model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self._steering_plan: SteeringPlan = SteeringPlan.from_raw(
            raw=steering_layers,
            weights=steering_weights,
            layers_description=layers_description,
            )
        self._hook_group = HookHandleGroup()

        self._layers, self._hidden_size = self._resolve_decoder_layers_and_hidden()


    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def _resolve_decoder_layers_and_hidden(self) -> tuple[list[nn.Module], int]:
        m = self.hf_model
        hidden_size = getattr(m.config, "hidden_size", None) or getattr(m.config, "n_embd", None)
        layers: list[nn.Module] = []

        candidates = [
            "layers",
            "model.layers",
            "model.decoder.layers",
            "transformer.h",
            "base_model.model.layers",
            "blocks", "model.blocks",
            "gpt_neox.layers",  # Pythia models
        ]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    if attr:
                        obj = getattr(obj, attr)
                if (isinstance(obj, nn.ModuleList) or
                    (isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], nn.Module))):
                    layers = list(obj)
                    break
            except AttributeError:
                continue

        if not layers:
            raise DecoderLayersNotFoundError()

        if hidden_size is None:
            for p in m.parameters():
                if p.ndim >= 2:
                    hidden_size = int(p.shape[-1]); break
        if hidden_size is None:
            raise HiddenSizeNotFoundError()

        return layers, int(hidden_size)

    def _is_chat_tokenizer(self) -> bool:
        return hasattr(self.tokenizer, "apply_chat_template") and callable(getattr(self.tokenizer, "apply_chat_template"))

    def apply_steering(self, plan: SteeringPlan | None = None) -> None:
        """
        Register forward hooks to add steering vectors *after* the selected decoder blocks.
        If plan is None, use the internal plan set at init or via set_steering_from_raw().
        Multiple vectors per layer are summed inside the hook.

        arguments:
            plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.

                SteeringPlan maps layer names (str) to list of SteeringVector, each with its own scale.
                Like this:
                    plan = SteeringPlan.from_raw({"6": torch.randn(wm.hidden_size)}, scale=0.7)

        example:
            >>> wm.apply_steering()   # uses current internal plan
            >>> # ... generate ...
            >>> wm.detach()           # back to vanilla
        """
        p = plan or self._steering_plan
        if p.is_empty():
            return

        p.validate_hidden_size(hidden_size=self._hidden_size)
        self.detach() 

        name_to_index = {str(i + 1): i for i in range(len(self._layers))}

        for lname, vec in p.layers.items():
            if lname not in name_to_index:
                continue
            idx = name_to_index[lname]
            layer = self._layers[idx]

            def _hook_factory(v: SteeringVector):
                def _hook(_mod: nn.Module, _inp: tuple, out: torch.Tensor | tuple) -> torch.Tensor | tuple:
                    if isinstance(out, tuple):
                        hs = out[0]
                        delta = torch.zeros_like(hs)
                        delta = delta + v.materialize(hs)
                        return (hs + delta,) + out[1:]
                    else:
                        hs = out
                        delta = torch.zeros_like(hs)
                        delta = delta + v.materialize(hs)
                        return hs + delta
                return _hook

            handle = layer.register_forward_hook(_hook_factory(vec))
            self._hook_group.add(handle)

    def detach(self) -> None:
        """
        Remove all registered steering hooks; model returns to unsteered behavior.
        """
        self._hook_group.remove_all()

    @contextmanager
    def detached(self):
        """
        Context manager that guarantees a vanilla (unsteered) model inside the block.

        example:
            >>> with wm.detached():
            ...     txt = wm.generate([[{"role": "user", "content": "Plain run"}]], use_steering=False)[0]
        """
        self.detach()
        try:
            yield
        finally:
            self.detach()

    def _encode_one(
        self,
        message: list[ChatMessage],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Encode a single input in chat format.

        arguments:
            messages:
                list of {'role': str, 'content': str} dicts (chat messages).
            add_generation_prompt:
                If True, append the model's generation prompt at the end.
            enable_thinking:
                If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen).

        returns:
            dict with 'input_ids' and 'attention_mask' tensors.

        example:
            >>> msgs = [
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]
            >>> wm._encode_one(msgs, add_generation_prompt=True)
            {"input_ids": tensor([[...]]), "attention_mask": tensor([[...]])}
        """

        try:
            ids = self.tokenizer.apply_chat_template(
                message, tokenize=True, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking, return_tensors="pt"
            )[0]
        except ValueError as e:
            # No fallback - raise error if chat template is not available
            raise ChatTemplateNotAvailableError(cause=e)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }

    def _batch_encode(
        self,
        inputs: list[list[ChatMessage]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Batch-encode a list of chat messages.

        arguments:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts).
            add_generation_prompt:
                If True, append the model's generation prompt at the end of each.
            enable_thinking:
                If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen).

        returns:
            dict with batched 'input_ids' and 'attention_mask' tensors.

        example:
            >>> msgs1 = [
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]
            >>> msgs2 = [
            ...   {"role":"user","content":"Write a haiku about rain."}
            ... ]
            >>> wm._batch_encode([msgs1, msgs2], add_generation_prompt=True)
            {"input_ids": tensor([[...],[...]]), "attention_mask": tensor([[...],[...]])}
        """

        singles = []
        for item in inputs:
            singles.append(self._encode_one(item, add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking))

        batch = self.tokenizer.pad(singles, padding=True, return_tensors="pt")

        batch = {k: v.to(self.device) for k, v in batch.items()}

        return batch

    def _extract_assistant_response(self, text: str) -> str:
        """
        Extract only the assistant's response from a decoded chat template output.

        When using chat templates, tokenizer.batch_decode returns the full formatted
        text including system prompt, user prompt, and assistant response. This method
        extracts only the assistant's response portion.

        arguments:
            text:
                Full decoded text from tokenizer.batch_decode

        returns:
            Extracted assistant response text

        example:
            >>> full_text = "system\\n\\nCutting Knowledge Date: December 2023\\nToday Date: 02 Nov 2025\\n\\nuser\\n\\nHello\\nassistant\\n\\nHi there!"
            >>> model._extract_assistant_response(full_text)
            "Hi there!"
        """
        import re

        # Look for the assistant marker in the decoded text
        if "assistant" in text:
            # Split on "assistant" and take everything after it
            response = text.split("assistant", 1)[1]
            # Strip leading/trailing whitespace and newlines
            response = response.strip()
        else:
            response = text

        # Remove empty thinking blocks that Qwen adds when enable_thinking=False
        # Pattern matches <think>\n\n</think>\n\n or variations with different whitespace
        response = re.sub(r'^<think>\s*</think>\s*', '', response)

        # Also remove thinking blocks with content (full thinking output)
        # This handles cases where thinking mode was enabled
        response = re.sub(r'^<think>.*?</think>\s*', '', response, flags=re.DOTALL)

        return response.strip()

    @torch.inference_mode()
    def generate(
        self,
        inputs: list[list[ChatMessage]] | str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.12,
        no_repeat_ngram_size: int = 4,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        use_steering: bool = False,
        steering_plan: SteeringPlan | None = None,
        enable_thinking: bool = False,
        prompt_is_formatted: bool = False,
        ensure_varied_responses: bool = False,
        phrase_ledger: Any = None,
        **gen_kwargs: Any,
    ) -> list[str]:
        """
        Batched text generation with optional steering.

        attributes:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts) OR pre-formatted string.
            max_new_tokens:
                max tokens to generate (beyond the prompt).
            temperature:
                sampling temperature (0 = greedy, 1 = default sampling).
            top_p:
                nucleus sampling probability (1.0 = no nucleus).
            do_sample:
                if False, uses greedy decoding (top_k=1).
            num_return_sequences:
                number of completions to generate per input.
            use_steering:
                if True, apply the current steering plan (if any).
            steering_plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.
            enable_thinking:
                If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen).
            prompt_is_formatted:
                If True, inputs is a pre-formatted string with chat template already applied.
            **gen_kwargs:
                additional kwargs passed to 'model.generate()'.

        returns:
            list of generated strings (length = len(inputs) * num_return_sequences).

        generation flow:
            notation:
                - Let B be batch size, T_in the (padded) input length, H the hidden size.
                - Decoder has L layers; we index user-facing layers as strings "1".. "L" (layer 1 is the first decoder block).
                - Steering plan maps layer names to one or more steering vectors with scales:
                '{"6": [SteeringVector(v6, scale=0.7)], "12": [SteeringVector(v12a, 1.0), SteeringVector(v12b, 0.4)]}'

        preparation:
            Given chat messages:
                msgs = [
                    {"role":"system","content":"Be concise."},
                    {"role":"user","content":"Two bullet points about koalas."}
                ]

            Encoding produces:
                - If chat template is available, 'apply_chat_template(..., tokenize=True)' yields `input_ids` of shape '[T1]'.
                - After 'tokenizer.pad([...])', the batch tensors have shapes:
                    - 'input_ids:  [B, T_in]'
                    - 'attention_mask: [B, T_in]'
          where 'T_in = T1' and 'B = 2' in this example.

        without steering:
            >>> wm = WisentModel("meta-llama/Meta-Llama-3-8B-Instruct", layers={}, device="cuda")
            >>> out_plain = wm.generate([msgs], max_new_tokens=32, use_steering=False)
            # out_plain: list[str] length B (or B * num_return_sequences)

            >>> for i, msg in enumerate(msgs):
            ...     print(f"User {i+1}: {msg['content']}")
            ...     print(f"Assistant {i+1}: {out_plain[i]}")

            internally during generation step 't = 0..T_out-1':
                - Each decoder block 'i' outputs a residual stream tensor of shape '[B, T_in + t, H]'.
                - No modification is applied; the model returns logits → token → appended to sequence.

        with steering (add AFTER layer i):
            # Build steering vectors of shape [H] for chosen layers; scales are per-vector.
            >>> plan = SteeringPlan.from_raw({
            ...     "6":  torch.randn(wm.hidden_size),   # will be normalized/broadcast if needed
            ...     "12": torch.randn(wm.hidden_size),
            ... }, scale=0.7, normalize=True)

           # Set once and use
            >>> wm.set_steering_from_raw({"6": plan.layers["6"][0].vector, "12": plan.layers["12"][0].vector},
                                 scale=0.7, normalize=True)


            What the hook 'sees' at a steered layer 'i' on step 't':
                - The layer's output (residual stream) 'h_i' has shape '[B, T_in + t, H]'.
                - Your steering vector 'v_i' is materialized to '[1, 1, H]' (or '[B,T,H]' if you passed that) and cast to the same dtype/device.
                - The hook returns 'h_i' = h_i + α_i * v_i' (if multiple vectors are configured for the same layer, it sums them).
                - This addition is cheap: one broadcasted add per steered layer, per step.

        
        shapes recap at generation step t (same for chat or plain strings):
        - Decoder block output:                '[B, T_in + t, H]'
        - Materialized steering vector:        '[1, 1, H]' (broadcast to '[B, T_in + t, H]')
        - Residual after steering (per layer): '[B, T_in + t, H]'

        example (one batch):
            >>> msgs = [
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]
            >>> wm.apply_steering()  # or pass use_steering=True below
            >>> out = wm.generate([msgs], max_new_tokens=32, use_steering=True)
            >>> for i, msg in enumerate(msgs):
            ...     print(f"User {i+1}: {msg['content']}")
            ...     print(f"Assistant {i+1}: {out[i]}")
        """
        if use_steering:
            self.apply_steering(steering_plan)

        if prompt_is_formatted and isinstance(inputs, str):
            # Direct tokenization of pre-formatted prompt
            tokenizer_output = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=False,  # Single prompt, no padding needed
                truncation=True,  # Avoid errors on long inputs
                max_length=self.tokenizer.model_max_length,  # Use model's actual limit
            )
            # Move tensors to the correct device (same as _batch_encode does)
            batch = {
                "input_ids": tokenizer_output["input_ids"].to(self.device),
                "attention_mask": tokenizer_output["attention_mask"].to(self.device)
            }
        else:
            # Current behavior: apply chat template
            batch = self._batch_encode(inputs, add_generation_prompt=True, enable_thinking=enable_thinking)

        # Build generation kwargs
        generation_kwargs = dict(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=False,
            **gen_kwargs,
        )

        # Add diversity processors if requested
        if ensure_varied_responses and phrase_ledger:
            from wisent.core.diversity_processors import build_diversity_processors
            logits_processors = build_diversity_processors(self.tokenizer, phrase_ledger)
            if logits_processors:
                generation_kwargs['logits_processor'] = logits_processors

        gen_out = self.hf_model.generate(**generation_kwargs)

        if use_steering:
            self.detach()

        seqs = gen_out.sequences  # [B * num_return_sequences, T_total]
        texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

        # Extract only assistant responses when using chat templates
        if not prompt_is_formatted and not isinstance(inputs, str):
            texts = [self._extract_assistant_response(text) for text in texts]

        return texts

    @torch.inference_mode()
    def generate_with_stats(
        self,
        inputs: list[list[ChatMessage]],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.12,
        no_repeat_ngram_size: int = 4,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        collect_topk: int = 5,
        use_steering: bool = False,
        steering_plan: SteeringPlan | None = None,
        enable_thinking: bool = False,
        ensure_varied_responses: bool = False,
        phrase_ledger: Any = None,
        **gen_kwargs: Any,
    ) -> tuple[list[str], list[GenerationStats]]:
        """
        Generate with efficient per-token stats (logits / probs), compatible with steering.
        Implementation detail: uses `output_scores=True` + `return_dict_in_generate=True` (HF standard).  :contentReference[oaicite:11]{index=11}

        attributes:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts).
            max_new_tokens:
                max tokens to generate (beyond the prompt).
            temperature:
                sampling temperature (0 = greedy, 1 = default sampling).
            top_p:
                nucleus sampling probability (0 = no filtering, 1 = full filtering).
            do_sample:
                if False, uses greedy decoding (top_k=1).
            num_return_sequences:
                number of completions to generate per input.
            collect_topk:
                if > 0, collect top-k logits/probs per step for analysis/visualization.
            use_steering:
                if True, apply the current steering plan (if any).
            steering_plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.
            enable_thinking:
                If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen).
            **gen_kwargs:
                additional kwargs passed to 'model.generate()'.

        returns:
            - list of generated strings (length = len(inputs) * num_return_sequences).
            - list of GenerationStats (length = len(inputs) * num_return_sequences).
              Each GenerationStats has:
                tokens:
                    list of generated token ids (length = actual generated tokens).
                per_step:
                     if collect_topk > 0, list of TopLogits (length = actual generated tokens).
                    Each TopLogits has:
                        token_id:
                            the generated token id at that step.
                        logit:
                            the raw logit for that token.
                        prob:
                            the softmax probability for that token.
                        topk_ids:
                            if collect_topk > 0, list of top-k token ids at that step.
                        topk_probs:
                            if collect_topk > 0, list of top-k probabilities at that step.

        example:
            >>> msgs = [[
            ...   {"role":"system","content":"Be concise."},
            ...   {"role":"user","content":"Two bullet points about koalas."}
            ... ]]
            >>> wm = WisentModel("meta-llama/Meta-Llama-3-8B-Instruct", layers={}, device="cuda")
            >>> wm.set_steering_from_raw({"6": torch.randn(wm.hidden_size), "12": torch.randn(wm.hidden_size)}, scale=0.7, normalize=True)
            >>> texts, stats = wm.generate_with_stats(
            ...   msgs,
            ...   max_new_tokens=48, collect_topk=5, use_steering=True
            ... )
            >>> stats[0].per_step[0].prob  # probability of the first generated token
        """
        if use_steering:
            self.apply_steering(steering_plan)

        batch = self._batch_encode(inputs, add_generation_prompt=True, enable_thinking=enable_thinking)

        # Build generation kwargs
        generation_kwargs = dict(
            **batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_scores=True,
            **gen_kwargs,
        )

        # Add diversity processors if requested
        if ensure_varied_responses and phrase_ledger:
            from wisent.core.diversity_processors import build_diversity_processors
            logits_processors = build_diversity_processors(self.tokenizer, phrase_ledger)
            if logits_processors:
                generation_kwargs['logits_processor'] = logits_processors

        out = self.hf_model.generate(**generation_kwargs)

        if use_steering:
            self.detach()

        texts = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)

        # Extract only assistant responses when using chat templates
        texts = [self._extract_assistant_response(text) for text in texts]

        scores: list[torch.Tensor] = list(out.scores or [])
        stats: list[GenerationStats] = []

        if scores:
            stacked = torch.stack(scores, dim=0)             # [steps, B*num_ret, V]
            steps = stacked.size(0)
            gen_token_ids = out.sequences[:, -steps:]        # [B*num_ret, steps]

            logprobs = torch.log_softmax(stacked.float(), dim=-1)  # [steps, B, V]
            B = logprobs.size(1)
            V = logprobs.size(2)

            for b in range(B):
                toks = gen_token_ids[b].tolist()
                per_step: list[TopLogits] = []
                for t, tok_id in enumerate(toks):
                    lp_row = logprobs[t, b]                        # [V]
                    logit = scores[t][b, tok_id].item()
                    prob = float(lp_row[tok_id].exp().item())
                    if collect_topk > 0:
                        topk_vals, topk_ids = lp_row.topk(min(collect_topk, V))
                        per_step.append(TopLogits(
                            token_id=int(tok_id),
                            logit=float(logit),
                            prob=float(prob),
                            topk_ids=topk_ids.tolist(),
                            topk_probs=topk_vals.exp().tolist(),
                        ))
                    else:
                        per_step.append(TopLogits(
                            token_id=int(tok_id),
                            logit=float(logit),
                            prob=float(prob),
                        ))
                stats.append(GenerationStats(tokens=toks, per_step=per_step))
        else:
            for _ in range(out.sequences.size(0)):
                stats.append(GenerationStats(tokens=[], per_step=None))

        return texts, stats
    
    @torch.inference_mode()
    def generate_stream(
        self,
        inputs: list[list[ChatMessage]] | str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.12,
        no_repeat_ngram_size: int = 4,
        do_sample: bool = True,
        use_steering: bool = False,
        steering_plan: SteeringPlan | None = None,
        skip_prompt: bool = True,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        prompt_is_formatted: bool = False,
        ensure_varied_responses: bool = False,
        phrase_ledger: Any = None,
        **gen_kwargs: Any,
    ) -> Iterable[str]:
        """
        Streamed text generation with optional steering.
        Uses the TextIteratorStreamer from transformers.

        attributes:
            inputs:
                list of chat messages (each a list of {'role','content'} dicts) OR pre-formatted string.
                Currently only one conversation is supported.
            max_new_tokens:
                max tokens to generate (beyond the prompt).
            temperature:
                sampling temperature (0 = greedy, 1 = default sampling).
            top_p:
                nucleus sampling probability (1.0 = no nucleus).
            do_sample:
                if False, uses greedy decoding (top_k=1).
            use_steering:
                if True, apply the current steering plan (if any).
            steering_plan:
                optional SteeringPlan to use for this call only (overrides internal plan).
                If None, uses the internal plan.
            skip_prompt:
                if True, the yielded text excludes the input prompt.
            skip_special_tokens:
                if True, special tokens are removed from the yielded text.
            enable_thinking:
                If False, disable thinking/reasoning mode (prevents <think> tags for supported models like Qwen).
            prompt_is_formatted:
                If True, inputs is a pre-formatted string with chat template already applied.
            **gen_kwargs:
                additional kwargs passed to 'model.generate()'.

        yields:
            generated text chunks (str), as they become available.
        """

        if use_steering:
            self.apply_steering(steering_plan)

        if prompt_is_formatted and isinstance(inputs, str):
            # Direct tokenization of pre-formatted prompt
            tokenizer_output = self.tokenizer(
                inputs,
                return_tensors="pt",
                padding=False,  # Single prompt, no padding needed
                truncation=True,  # Avoid errors on long inputs
                max_length=self.tokenizer.model_max_length,  # Use model's actual limit
            )
            # Move tensors to the correct device (same as _batch_encode does)
            batch = {
                "input_ids": tokenizer_output["input_ids"].to(self.device),
                "attention_mask": tokenizer_output["attention_mask"].to(self.device)
            }
        else:
            # Current behavior: apply chat template
            if not isinstance(inputs, list) or len(inputs) != 1:
                raise InsufficientDataError(
                    reason=f"generate_stream currently supports exactly one conversation at a time (got {type(inputs)} with {len(inputs) if isinstance(inputs, list) else 'N/A'} items)"
                )
            batch = self._batch_encode(inputs, add_generation_prompt=True, enable_thinking=enable_thinking)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens,
        )

        generation_kwargs = dict(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            return_dict_in_generate=False,
            output_scores=False,
            streamer=streamer,
            **gen_kwargs,
        )

        # Add diversity processors if requested
        if ensure_varied_responses and phrase_ledger:
            from wisent.core.diversity_processors import build_diversity_processors
            logits_processors = build_diversity_processors(self.tokenizer, phrase_ledger)
            if logits_processors:
                generation_kwargs['logits_processor'] = logits_processors

        worker = threading.Thread(
            target=self.hf_model.generate,
            kwargs=generation_kwargs,
            daemon=True,
        )
        worker.start()

        try:
            for new_text in streamer:
                if new_text:
                    yield new_text
        finally:
            if use_steering:
                self.detach()
            worker.join(timeout=0.0)
    

    def set_steering_from_raw(self, raw: list[RawActivationMap] | RawActivationMap | None, layers_description: list[str] | None = None, steering_weights: list[float] | None = None, scale: float = 1.0, normalize: bool = False) -> None:
        """
        Replace the internal steering plan using a RawActivationMap (layer_name -> tensor).
        If raw is None or empty, clears any existing steering. If we
        """
        if not raw:
            self._steering_plan = SteeringPlan()
            return

        # TODO: this should be outside
        reports = run_control_steering_diagnostics(raw)
        for report in reports:
            for issue in report.issues:
                log_method = logger.error if issue.severity == "critical" else logger.warning
                log_method(
                    "[control_vector diagnostics] %s (details=%s)",
                    issue.message,
                issue.details,
            )

        if any(report.has_critical_issues for report in reports):
            raise ControlVectorDiagnosticsError()

        self._steering_plan = SteeringPlan.from_raw(
            raw=raw,
            layers_description=layers_description,
            weights=steering_weights,
            scale=scale,
            normalize=normalize,
            expected_hidden_size=self._hidden_size
            )

    def clear_steering(self) -> None:
        """
        Remove any existing steering configuration and active hooks.
        After calling this, generation is vanilla.
        """
        self._steering_plan = SteeringPlan()
        self.detach()