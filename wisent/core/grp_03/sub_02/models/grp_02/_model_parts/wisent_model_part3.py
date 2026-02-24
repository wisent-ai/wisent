"""WisentModel methods: generate, set_steering_from_raw, clear_steering."""
from __future__ import annotations

import logging
from typing import Any

import torch

from wisent.core.models.core.atoms import SteeringPlan
from wisent.core.activations.core.atoms import RawActivationMap
from wisent.core.prompts.core.atom import ChatMessage
from wisent.core.contrastive_pairs.diagnostics import run_control_steering_diagnostics
from wisent.core.errors import ControlVectorDiagnosticsError
from wisent.core.constants import (
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_INFERENCE_TEMPERATURE,
    DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE, DEFAULT_BASE_STRENGTH,
)

logger = logging.getLogger(__name__)


@torch.inference_mode()
def _generate(
    self,
    inputs: list[list[ChatMessage]] | str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_INFERENCE_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    use_steering: bool = False,
    steering_plan: SteeringPlan | None = None,
    steering_object: "BaseSteeringObject | None" = None,
    steering_strength: float = DEFAULT_BASE_STRENGTH,
    steering_strategy: str = "constant",
    steering_strategy_config: dict | None = None,
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
            optional SteeringPlan to use for this call only.
        steering_object:
            optional SteeringObject for method-specific steering.
        steering_strength:
            Base strength multiplier when using steering_object (default: 1.0).
        enable_thinking:
            If False, disable thinking/reasoning mode.
        prompt_is_formatted:
            If True, inputs is a pre-formatted string with chat template already applied.
        **gen_kwargs:
            additional kwargs passed to 'model.generate()'.

    returns:
        list of generated strings (length = len(inputs) * num_return_sequences).
    """
    if steering_object is not None:
        self.apply_steering_object(
            steering_object,
            base_strength=steering_strength,
            steering_strategy=steering_strategy,
            steering_strategy_config=steering_strategy_config,
            max_new_tokens=max_new_tokens,
        )
    elif use_steering:
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
        from wisent.core.tasks.base.diversity_processors import build_diversity_processors
        logits_processors = build_diversity_processors(self.tokenizer, phrase_ledger)
        if logits_processors:
            generation_kwargs['logits_processor'] = logits_processors

    gen_out = self.hf_model.generate(**generation_kwargs)

    if use_steering or steering_object is not None:
        self.detach()

    seqs = gen_out.sequences  # [B * num_return_sequences, T_total]
    texts = self.tokenizer.batch_decode(seqs, skip_special_tokens=True)

    # Extract only assistant responses when using chat templates
    if not prompt_is_formatted and not isinstance(inputs, str):
        texts = [self._extract_assistant_response(text) for text in texts]

    return texts


def _set_steering_from_raw(
    self,
    raw: list[RawActivationMap] | RawActivationMap | None,
    layers_description: list[str] | None = None,
    steering_weights: list[float] | None = None,
    scale: float = DEFAULT_BASE_STRENGTH,
    normalize: bool = False,
) -> None:
    """
    Replace the internal steering plan using a RawActivationMap (layer_name -> tensor).
    If raw is None or empty, clears any existing steering.
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


def _clear_steering(self) -> None:
    """
    Remove any existing steering configuration and active hooks.
    After calling this, generation is vanilla.
    """
    self._steering_plan = SteeringPlan()
    self.detach()
