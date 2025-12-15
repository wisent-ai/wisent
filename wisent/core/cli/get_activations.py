"""Get activations command execution logic."""

import sys
import json
import os
import time


def execute_get_activations(args):
    """Execute the get-activations command - load pairs and collect activations."""
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    print(f"\n🎨 Collecting activations from contrastive pairs")
    print(f"   Input file: {args.pairs_file}")
    print(f"   Model: {args.model}")

    start_time = time.time() if args.timing else None

    try:
        # 1. Load pairs from JSON
        print(f"\n📂 Loading contrastive pairs...")
        if not os.path.exists(args.pairs_file):
            raise FileNotFoundError(f"Pairs file not found: {args.pairs_file}")

        with open(args.pairs_file, 'r') as f:
            data = json.load(f)

        # Handle both formats: dict with 'pairs' key or direct list
        if isinstance(data, dict):
            pairs_list = data.get('pairs', [])
            task_name = data.get('task_name', 'unknown')
            trait_label = data.get('trait_label', task_name)
        else:
            pairs_list = data
            task_name = 'unknown'
            trait_label = 'unknown'

        # Apply limit if specified
        if hasattr(args, 'limit') and args.limit:
            pairs_list = pairs_list[:args.limit]

        print(f"   ✓ Loaded {len(pairs_list)} pairs")

        # 2. Load model
        print(f"\n🤖 Loading model '{args.model}'...")
        model = WisentModel(args.model, device=args.device)
        print(f"   ✓ Model loaded with {model.num_layers} layers")

        # 3. Determine layers to collect
        if args.layers is None:
            # Default: use ALL layers (0-indexed)
            layers = list(range(model.num_layers))
        elif args.layers.lower() == 'all':
            # Use all layers (0-indexed)
            layers = list(range(model.num_layers))
        else:
            layers = [int(l.strip()) for l in args.layers.split(',')]

        # Convert to strings for API
        layer_strs = [str(l) for l in layers]

        print(f"\n🎯 Collecting activations from {len(layers)} layer(s): {layers}")

        # 4. Set up aggregation strategy
        aggregation_map = {
            'average': 'MEAN_POOLING',
            'final': 'LAST_TOKEN',
            'first': 'FIRST_TOKEN',
            'max': 'MAX_POOLING',
            'min': 'MAX_POOLING',
        }
        aggregation_key = aggregation_map.get(args.token_aggregation.lower(), 'MEAN_POOLING')
        aggregation_strategy = ActivationAggregationStrategy[aggregation_key]

        # 5. Map prompt strategy string to enum
        prompt_strategy_map = {
            'chat_template': PromptConstructionStrategy.CHAT_TEMPLATE,
            'direct_completion': PromptConstructionStrategy.DIRECT_COMPLETION,
            'instruction_following': PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
            'multiple_choice': PromptConstructionStrategy.MULTIPLE_CHOICE,
            'role_playing': PromptConstructionStrategy.ROLE_PLAYING,
        }
        prompt_strategy = prompt_strategy_map.get(args.prompt_strategy.lower(), PromptConstructionStrategy.CHAT_TEMPLATE)

        print(f"   Token aggregation: {args.token_aggregation} ({aggregation_key})")
        print(f"   Prompt strategy: {args.prompt_strategy}")

        # 5. Create pair set and reconstruct pairs
        pair_set = ContrastivePairSet(name=task_name, task_type=trait_label)

        for pair_data in pairs_list:
            pair = ContrastivePair(
                prompt=pair_data['prompt'],
                positive_response=PositiveResponse(
                    model_response=pair_data['positive_response']['model_response']
                ),
                negative_response=NegativeResponse(
                    model_response=pair_data['negative_response']['model_response']
                ),
                label=pair_data.get('label', trait_label),
                trait_description=pair_data.get('trait_description', ''),
            )
            pair_set.add(pair)

        # 6. Collect activations
        print(f"\n⚡ Collecting activations...")
        collector = ActivationCollector(model=model, store_device="cpu")

        enriched_pairs = []
        for i, pair in enumerate(pair_set.pairs):
            if args.verbose:
                print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...")

            # Collect activations for all requested layers at once
            updated_pair = collector.collect_for_pair(
                pair,
                layers=layer_strs,
                aggregation=aggregation_strategy,
                return_full_sequence=False,
                normalize_layers=False,
                prompt_strategy=prompt_strategy
            )

            enriched_pairs.append(updated_pair)

        print(f"   ✓ Collected activations for {len(enriched_pairs)} pairs")

        # 7. Convert to JSON format
        print(f"\n💾 Saving enriched pairs to '{args.output}'...")
        output_data = {
            'task_name': task_name,
            'trait_label': trait_label,
            'model': args.model,
            'layers': layers,
            'token_aggregation': args.token_aggregation,
            'num_pairs': len(enriched_pairs),
            'pairs': []
        }

        for pair in enriched_pairs:
            pair_dict = {
                'prompt': pair.prompt,
                'positive_response': {
                    'model_response': pair.positive_response.model_response,
                    'layers_activations': {}
                },
                'negative_response': {
                    'model_response': pair.negative_response.model_response,
                    'layers_activations': {}
                },
                'label': pair.label,
                'trait_description': pair.trait_description,
            }

            # Convert activations to lists for JSON serialization
            if pair.positive_response.layers_activations:
                for layer_str, act in pair.positive_response.layers_activations.items():
                    if act is not None:
                        pair_dict['positive_response']['layers_activations'][layer_str] = act.cpu().tolist()

            if pair.negative_response.layers_activations:
                for layer_str, act in pair.negative_response.layers_activations.items():
                    if act is not None:
                        pair_dict['negative_response']['layers_activations'][layer_str] = act.cpu().tolist()

            output_data['pairs'].append(pair_dict)

        # 8. Save to file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"   ✓ Saved enriched pairs to: {args.output}")

        if args.timing:
            elapsed = time.time() - start_time
            print(f"   ⏱️  Total time: {elapsed:.2f}s")

        print(f"\n✅ Activation collection completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
