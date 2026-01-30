"""CLI command for steering effect visualization.

Uses database activations if available, regenerates if not.
"""

import os
os.environ["NUMBA_NUM_THREADS"] = "1"

import json
import base64
import tempfile
from pathlib import Path
from argparse import Namespace


def execute_steering_viz(args):
    """Execute the steering-viz command."""
    import torch
    from wisent.core.geometry.steering import (
        create_steering_effect_figure, create_interactive_steering_figure,
        train_classifier_and_predict,
    )

    print(f"\n{'='*60}\nSTEERING EFFECT VISUALIZATION\n{'='*60}")

    # Step 1: Try to load reference activations from database
    print(f"\n[Step 1/4] Loading reference activations...")
    pos_ref, neg_ref = _load_or_generate_reference_activations(args)
    print(f"  Loaded {len(pos_ref)} positive, {len(neg_ref)} negative reference activations")

    # Step 2: Compute steering vector
    print(f"\n[Step 2/4] Computing steering vector...")
    steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
    steering_vector = steering_vector / steering_vector.norm()
    print(f"  Steering vector norm: {steering_vector.norm().item():.4f}")

    # Step 3: Generate steered/unsteered responses and extract activations
    print(f"\n[Step 3/4] Generating responses and extracting activations...")
    base_data, steered_data, base_activations, steered_activations = _generate_and_extract(
        args, steering_vector
    )
    base_evaluations = [r.get('evaluation', 'UNKNOWN') for r in base_data]
    steered_evaluations = [r.get('evaluation', 'UNKNOWN') for r in steered_data]
    print(f"  Generated {len(base_data)} response pairs")

    # Step 4: Train classifier and visualize
    print(f"\n[Step 4/4] Creating visualization...")
    base_space_probs, steered_space_probs, train_report = train_classifier_and_predict(
        pos_ref, neg_ref, base_activations, steered_activations,
        classifier_type=getattr(args, 'space_classifier', 'mlp')
    )

    viz_args = dict(
        pos_activations=pos_ref, neg_activations=neg_ref,
        base_activations=base_activations, steered_activations=steered_activations,
        title=f"Steering Effect: {args.task} (layer {args.layer}, strength {args.strength})",
        base_evaluations=base_evaluations, steered_evaluations=steered_evaluations,
        base_space_probs=base_space_probs, steered_space_probs=steered_space_probs,
    )

    multipanel = getattr(args, 'multipanel', False)
    interactive = getattr(args, 'interactive', False)

    if interactive:
        # Add response texts for hover information
        viz_args['prompts'] = [r.get('prompt', '') for r in base_data]
        viz_args['base_responses'] = [r.get('response', '') for r in base_data]
        viz_args['steered_responses'] = [r.get('response', '') for r in steered_data]
        viz_html = create_interactive_steering_figure(**viz_args)
        output_path = Path(args.output).with_suffix('.html')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(viz_html)
        print(f"\nInteractive visualization saved to: {output_path}")
    elif multipanel:
        from wisent.core.geometry.steering import create_steering_multipanel_figure
        extraction_strategy = getattr(args, 'extraction_strategy', 'chat_last')
        viz_b64 = create_steering_multipanel_figure(**viz_args, extraction_strategy=extraction_strategy)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(viz_b64))
        print(f"\nVisualization saved to: {output_path}")
    else:
        viz_b64 = create_steering_effect_figure(**viz_args)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(viz_b64))
        print(f"\nVisualization saved to: {output_path}")

    # Save summary
    _save_summary(output_path, args, base_evaluations, steered_evaluations,
                  base_space_probs, steered_space_probs, train_report, base_data, steered_data)

    print(f"\n{'='*60}\nSTEERING VISUALIZATION COMPLETE\n{'='*60}")
    return {"output": str(output_path)}


def _load_or_generate_reference_activations(args):
    """Load activations from database if available, otherwise generate."""
    from wisent.core.geometry.repscan_with_concepts import load_activations_from_database

    try:
        pos_ref, neg_ref = load_activations_from_database(
            model_name=args.model, task_name=args.task, layer=args.layer,
            prompt_format=getattr(args, 'prompt_format', 'chat'),
            extraction_strategy=getattr(args, 'extraction_strategy', 'chat_last'),
            limit=getattr(args, 'limit', 100),
            database_url=getattr(args, 'database_url', None),
        )
        print(f"  Found activations in database")
        return pos_ref, neg_ref
    except Exception as e:
        print(f"  Not in database ({e}), generating...")
        return _generate_reference_activations(args)


def _generate_reference_activations(args):
    """Generate reference activations using CLI commands."""
    import torch
    import tempfile
    from pathlib import Path
    from wisent.core.cli.analysis.geometry.get_activations import execute_get_activations
    from wisent.core.geometry.repscan_with_concepts import load_pair_texts_from_database

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        pairs_path = tmpdir / "pairs.json"
        pair_texts = load_pair_texts_from_database(
            task_name=args.task, limit=getattr(args, 'limit', 100),
            database_url=getattr(args, 'database_url', None)
        )
        pairs_list = [{"prompt": p.get("prompt", ""),
                       "positive_response": {"model_response": p.get("positive", "")},
                       "negative_response": {"model_response": p.get("negative", "")}}
                      for p in pair_texts.values()]
        with open(pairs_path, 'w') as f:
            json.dump({"task_name": args.task, "pairs": pairs_list}, f)

        enriched_path = tmpdir / "enriched_pairs.json"
        execute_get_activations(Namespace(
            pairs_file=str(pairs_path), model=args.model, output=str(enriched_path),
            device=getattr(args, 'device', None),
            layers=str(args.layer), extraction_strategy=getattr(args, 'extraction_strategy', 'chat_last'),
            verbose=False, timing=False, limit=None, raw=False,
        ))

        with open(enriched_path) as f:
            data = json.load(f)

        pos_acts, neg_acts = [], []
        layer_key = str(args.layer)
        for pair in data.get('pairs', []):
            pos_layer = pair.get('positive_response', {}).get('layers_activations', {}).get(layer_key)
            neg_layer = pair.get('negative_response', {}).get('layers_activations', {}).get(layer_key)
            if pos_layer and neg_layer:
                pos_acts.append(torch.tensor(pos_layer))
                neg_acts.append(torch.tensor(neg_layer))

        return torch.stack(pos_acts), torch.stack(neg_acts)


def _generate_and_extract(args, steering_vector):
    """Generate steered/unsteered responses and extract activations."""
    import torch
    from wisent.core.wisent import Wisent
    from wisent.core.adapters.base import SteeringConfig
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.activations.extraction_strategy import ExtractionStrategy, extract_activation
    from wisent.core.evaluators.rotator import EvaluatorRotator
    from wisent.core.geometry.repscan_with_concepts import load_pair_texts_from_database

    wisent = Wisent.for_text(args.model)
    adapter = wisent.adapter
    layer_name = f"layer.{args.layer}"

    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.oracles")
    EvaluatorRotator.discover_evaluators("wisent.core.evaluators.benchmark_specific")
    evaluator = EvaluatorRotator(evaluator=None, task_name=args.task).current

    # Map old names to new enum values
    strategy_str = getattr(args, 'extraction_strategy', 'chat_last')
    strategy_map = {"last_token": "chat_last", "first_token": "chat_first", "mean": "chat_mean"}
    strategy_str = strategy_map.get(strategy_str, strategy_str)
    extraction_strategy = ExtractionStrategy(strategy_str)
    max_new_tokens = getattr(args, 'max_new_tokens', 100)

    steering_vectors = LayerActivations({layer_name: steering_vector})
    config = SteeringConfig(scale={layer_name: args.strength})

    pair_texts = load_pair_texts_from_database(
        task_name=args.task, limit=getattr(args, 'limit', 100),
        database_url=getattr(args, 'database_url', None)
    )

    base_data, steered_data = [], []
    base_acts, steered_acts = [], []

    for i, (pair_id, pair) in enumerate(pair_texts.items()):
        prompt = pair.get("prompt", "")
        pos_ref_text = pair.get("positive", "")
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = adapter.apply_chat_template(messages, add_generation_prompt=True)

        base_full = adapter._generate_unsteered(formatted_prompt, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=True)
        steered_full = adapter.forward_with_steering(formatted_prompt, steering_vectors=steering_vectors, config=config)

        base_response = base_full[len(formatted_prompt):].strip() if base_full.startswith(formatted_prompt) else base_full
        steered_response = steered_full[len(formatted_prompt):].strip() if steered_full.startswith(formatted_prompt) else steered_full

        base_layer_acts = adapter.extract_activations(base_full, layers=[layer_name])
        steered_layer_acts = adapter.extract_activations(steered_full, layers=[layer_name])

        prompt_len = len(adapter.tokenizer(formatted_prompt, add_special_tokens=False)["input_ids"])
        base_act = base_layer_acts.get(layer_name)
        steered_act = steered_layer_acts.get(layer_name)

        if base_act is not None and steered_act is not None:
            base_extracted = extract_activation(extraction_strategy, base_act[0], base_response, adapter.tokenizer, prompt_len)
            steered_extracted = extract_activation(extraction_strategy, steered_act[0], steered_response, adapter.tokenizer, prompt_len)
            base_acts.append(base_extracted.cpu())
            steered_acts.append(steered_extracted.cpu())

        base_eval = evaluator.evaluate(base_response, pos_ref_text).ground_truth
        steered_eval = evaluator.evaluate(steered_response, pos_ref_text).ground_truth

        base_data.append({"prompt": prompt, "response": base_response, "evaluation": base_eval})
        steered_data.append({"prompt": prompt, "response": steered_response, "evaluation": steered_eval})

        if i % 10 == 0:
            print(f"  Processed {i+1}/{len(pair_texts)}...")

    return base_data, steered_data, torch.stack(base_acts), torch.stack(steered_acts)


def _save_summary(output_path, args, base_evals, steered_evals, base_probs, steered_probs, train_report, base_data, steered_data):
    """Save JSON summary."""
    import numpy as np
    base_truthful = sum(1 for e in base_evals if e == "TRUTHFUL")
    steered_truthful = sum(1 for e in steered_evals if e == "TRUTHFUL")
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump({"model": args.model, "task": args.task, "layer": args.layer, "strength": args.strength,
                   "text_evaluation": {"base_truthful": base_truthful, "steered_truthful": steered_truthful, "total": len(base_evals)},
                   "activation_space": {"classifier_accuracy": train_report.final.accuracy,
                       "base_in_truthful": sum(1 for p in base_probs if p >= 0.5),
                       "steered_in_truthful": sum(1 for p in steered_probs if p >= 0.5),
                       "base_mean_prob": float(np.mean(base_probs)), "steered_mean_prob": float(np.mean(steered_probs))},
                   "responses": [{"prompt": b["prompt"], "base": b["response"], "steered": s["response"],
                                  "base_eval": b["evaluation"], "steered_eval": s["evaluation"]}
                                 for b, s in zip(base_data, steered_data)]}, f, indent=2)
    print(f"Summary saved to: {json_path}")
