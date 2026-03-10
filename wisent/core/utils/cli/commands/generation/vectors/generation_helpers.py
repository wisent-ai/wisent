"""Batched and sequential generation helpers for generate_responses."""

import time
from datetime import datetime, timezone

from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy, extract_activation
from wisent.core.control.steering_methods.configs.optimal import get_optimal_extraction_strategy
from wisent.core.utils.config_tools.constants import (
    GENERATION_BATCH_SIZE, DISPLAY_TRUNCATION_COMPACT, INDEX_FIRST, COMBO_OFFSET, COMBO_BASE,
)


def _build_result_entry(idx, pair, generated_text):
    """Build a result dict for one prompt/response pair."""
    entry = {
        "question_id": idx,
        "prompt": pair.prompt,
        "generated_response": generated_text,
        "positive_reference": pair.positive_response.model_response,
        "negative_reference": pair.negative_response.model_response,
    }
    if pair.metadata and pair.metadata.get('correct_answers'):
        entry['correct_answers'] = pair.metadata['correct_answers']
    else:
        entry['correct_answers'] = [pair.positive_response.model_response]
    if pair.metadata and pair.metadata.get('incorrect_answers'):
        entry['incorrect_answers'] = pair.metadata['incorrect_answers']
    else:
        entry['incorrect_answers'] = [pair.negative_response.model_response]
    return entry


def _build_error_entry(idx, pair, error):
    """Build an error result dict for one prompt that failed."""
    entry = _build_result_entry(idx, pair, None)
    entry["error"] = str(error)
    return entry


def _generate_one(pair, model, gen_kwargs, args, steering_object, steering_strategy):
    """Generate a single response. Returns the generated text string."""
    messages = [{"role": "user", "content": pair.prompt}]
    responses = model.generate(
        inputs=[messages],
        **gen_kwargs,
        use_steering=args.use_steering,
        steering_object=steering_object,
        steering_strength=args.steering_strength,
        steering_strategy=steering_strategy,
    )
    return responses[INDEX_FIRST] if responses else ""


def generate_batched(pairs, model, gen_kwargs, args, steering_object, steering_strategy):
    """Generate responses in batches of GENERATION_BATCH_SIZE."""
    results = []
    total = len(pairs)
    batch_num = INDEX_FIRST
    n_batches = (total + GENERATION_BATCH_SIZE - COMBO_OFFSET) // GENERATION_BATCH_SIZE
    gen_start = time.monotonic()
    print(
        f"[generate_batched] {datetime.now(timezone.utc).isoformat()} "
        f"start: {total} prompts, {n_batches} batches, "
        f"max_new_tokens={gen_kwargs.get('max_new_tokens')}, "
        f"strength={getattr(args, 'steering_strength', None)}, "
        f"strategy={steering_strategy}",
        flush=True,
    )
    for batch_start in range(INDEX_FIRST, total, GENERATION_BATCH_SIZE):
        batch_pairs = pairs[batch_start:batch_start + GENERATION_BATCH_SIZE]
        first_idx = batch_start + COMBO_OFFSET
        last_idx = batch_start + len(batch_pairs)
        batch_num += COMBO_OFFSET
        batch_t0 = time.monotonic()
        print(
            f"[generate_batched] {datetime.now(timezone.utc).isoformat()} "
            f"batch {batch_num}/{n_batches} start (questions {first_idx}-{last_idx})",
            flush=True,
        )
        try:
            batch_messages = [[{"role": "user", "content": p.prompt}] for p in batch_pairs]
            responses = model.generate(
                inputs=batch_messages,
                **gen_kwargs,
                use_steering=args.use_steering,
                steering_object=steering_object,
                steering_strength=args.steering_strength,
                steering_strategy=steering_strategy,
            )
            batch_elapsed = time.monotonic() - batch_t0
            token_counts = [len(r) for r in responses]
            print(
                f"[generate_batched] {datetime.now(timezone.utc).isoformat()} "
                f"batch {batch_num}/{n_batches} done in {batch_elapsed:.1f}s, "
                f"response_chars={token_counts}",
                flush=True,
            )
            for i, pair in enumerate(batch_pairs):
                idx = batch_start + i + COMBO_OFFSET
                text = responses[i] if i < len(responses) else ""
                if args.verbose:
                    print(f"   Q{idx}: {text[:DISPLAY_TRUNCATION_COMPACT]}...")
                results.append(_build_result_entry(idx, pair, text))
        except Exception as batch_err:
            print(f"   Batch failed ({batch_err}), falling back to sequential")
            for i, pair in enumerate(batch_pairs):
                idx = batch_start + i + COMBO_OFFSET
                try:
                    text = _generate_one(pair, model, gen_kwargs, args, steering_object, steering_strategy)
                    results.append(_build_result_entry(idx, pair, text))
                except Exception as e:
                    print(f"   Error generating response for question {idx}: {e}")
                    results.append(_build_error_entry(idx, pair, e))
    total_elapsed = time.monotonic() - gen_start
    print(
        f"[generate_batched] {datetime.now(timezone.utc).isoformat()} "
        f"all batches done in {total_elapsed:.1f}s, {len(results)} results",
        flush=True,
    )
    return results


def generate_sequential(pairs, model, gen_kwargs, args, steering_object, steering_strategy, do_extract):
    """Generate responses one at a time (needed when extracting activations)."""
    results = []
    for idx, pair in enumerate(pairs, COMBO_OFFSET):
        if args.verbose:
            print(f"Question {idx}/{len(pairs)}: {pair.prompt[:DISPLAY_TRUNCATION_COMPACT]}...")
        try:
            text = _generate_one(pair, model, gen_kwargs, args, steering_object, steering_strategy)
            entry = _build_result_entry(idx, pair, text)
            if do_extract and text:
                messages = [{"role": "user", "content": pair.prompt}]
                extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', get_optimal_extraction_strategy()))
                layers = getattr(args, 'layers', None)
                if layers:
                    layer_list = [f"layer.{l.strip()}" for l in layers.split(',')]
                else:
                    layer_list = [f"layer.{model.num_layers // COMBO_BASE}"]
                formatted = model.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                full_response = formatted + text
                layer_acts = model.adapter.extract_activations(full_response, layers=layer_list)
                prompt_len = len(model.tokenizer(formatted, add_special_tokens=False)["input_ids"])
                activations_dict = {}
                for layer_name, act in layer_acts.items():
                    if act is not None:
                        extracted = extract_activation(
                            extraction_strategy, act[INDEX_FIRST], text, model.tokenizer, prompt_len,
                        )
                        activations_dict[layer_name] = extracted.cpu().tolist()
                entry["activations"] = activations_dict
                entry["extraction_strategy"] = extraction_strategy.value
            results.append(entry)
        except Exception as e:
            print(f"   Error generating response for question {idx}: {e}")
            results.append(_build_error_entry(idx, pair, e))
    return results
