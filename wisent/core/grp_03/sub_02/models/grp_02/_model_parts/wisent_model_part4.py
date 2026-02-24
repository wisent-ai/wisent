"""WisentModel method: generate_with_stats."""
from __future__ import annotations

from typing import Any

import torch

from wisent.core.models.core.atoms import SteeringPlan, GenerationStats, TopLogits
from wisent.core.prompts.core.atom import ChatMessage
from wisent.core.constants import (
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_INFERENCE_TEMPERATURE,
    DEFAULT_TOP_P, DEFAULT_TOP_K, DEFAULT_REPETITION_PENALTY,
    DEFAULT_NO_REPEAT_NGRAM_SIZE, DEFAULT_BASE_STRENGTH, MODEL_COLLECT_TOPK,
)


@torch.inference_mode()
def _generate_with_stats(
    self,
    inputs: list[list[ChatMessage]],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_INFERENCE_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    collect_topk: int = MODEL_COLLECT_TOPK,
    use_steering: bool = False,
    steering_plan: SteeringPlan | None = None,
    steering_object: "BaseSteeringObject | None" = None,
    steering_strength: float = DEFAULT_BASE_STRENGTH,
    enable_thinking: bool = False,
    ensure_varied_responses: bool = False,
    phrase_ledger: Any = None,
    **gen_kwargs: Any,
) -> tuple[list[str], list[GenerationStats]]:
    """
    Generate with efficient per-token stats (logits / probs), compatible with steering.

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
            optional SteeringPlan to use for this call only.
        enable_thinking:
            If False, disable thinking/reasoning mode.
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
    """
    if steering_object is not None:
        self.apply_steering_object(steering_object, base_strength=steering_strength)
    elif use_steering:
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
        from wisent.core.tasks.base.diversity_processors import build_diversity_processors
        logits_processors = build_diversity_processors(self.tokenizer, phrase_ledger)
        if logits_processors:
            generation_kwargs['logits_processor'] = logits_processors

    out = self.hf_model.generate(**generation_kwargs)

    if use_steering or steering_object is not None:
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
