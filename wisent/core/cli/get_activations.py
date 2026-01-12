"""Get activations command execution logic."""

import sys
import json
import os
import time


def execute_get_activations(args):
    """Execute the get-activations command - load pairs and collect activations."""
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.extraction_strategy import ExtractionStrategy
    
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    raw_mode = getattr(args, 'raw', False)
    mode_str = "raw hidden states" if raw_mode else "activations"
    
    print(f"\n🎨 Collecting {mode_str} from contrastive pairs")
    print(f"   Input file: {args.pairs_file}")
    print(f"   Model: {args.model}")
    if raw_mode:
        print(f"   Mode: RAW (full sequences, reusable for multiple strategies)")

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

        # 3. Determine layers to collect (1-indexed for API)
        if args.layers is None:
            # Default: use ALL layers (1-indexed: 1..num_layers)
            layers = list(range(1, model.num_layers + 1))
        elif args.layers.lower() == 'all':
            # Use all layers (1-indexed: 1..num_layers)
            layers = list(range(1, model.num_layers + 1))
        else:
            layers = [int(l.strip()) for l in args.layers.split(',')]

        # Convert to strings for API (1-indexed)
        layer_strs = [str(l) for l in layers]

        print(f"\n🎯 Collecting activations from {len(layers)} layer(s): {layers}")

        # 4. Get extraction strategy from args
        extraction_strategy = ExtractionStrategy(getattr(args, 'extraction_strategy', 'chat_last'))
        print(f"   Extraction strategy: {extraction_strategy.value}")

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
        collector = ActivationCollector(model=model)

        if raw_mode:
            # RAW MODE: Collect full hidden states [seq_len, hidden_size]
            print(f"\n⚡ Collecting RAW hidden states...")
            from wisent.core.activations.activation_cache import get_strategy_text_family
            text_family = get_strategy_text_family(extraction_strategy)
            
            raw_pairs_data = []
            for i, pair in enumerate(pair_set.pairs):
                if args.verbose or (i + 1) % 10 == 0:
                    print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...", end='\r', flush=True)

                # Collect RAW hidden states (full sequences)
                raw_data = collector.collect_raw(
                    pair, strategy=extraction_strategy,
                    layers=layer_strs,
                )
                raw_pairs_data.append({
                    'pair': pair,
                    'raw_data': raw_data,
                })

            print(f"   ✓ Collected raw hidden states for {len(raw_pairs_data)} pairs")

            # Convert to JSON format (raw mode)
            print(f"\n💾 Saving raw activations to '{args.output}'...")
            output_data = {
                'task_name': task_name,
                'trait_label': trait_label,
                'model': args.model,
                'layers': layers,
                'extraction_strategy': extraction_strategy.value,
                'text_family': text_family,
                'raw_mode': True,
                'num_pairs': len(raw_pairs_data),
                'pairs': []
            }

            for item in raw_pairs_data:
                pair = item['pair']
                raw_data = item['raw_data']
                
                pair_dict = {
                    'prompt': pair.prompt,
                    'positive_response': {
                        'model_response': pair.positive_response.model_response,
                        'answer_text': raw_data['pos_answer_text'],
                        'prompt_len': raw_data['pos_prompt_len'],
                        'layers_hidden_states': {}
                    },
                    'negative_response': {
                        'model_response': pair.negative_response.model_response,
                        'answer_text': raw_data['neg_answer_text'],
                        'prompt_len': raw_data['neg_prompt_len'],
                        'layers_hidden_states': {}
                    },
                    'label': pair.label,
                    'trait_description': pair.trait_description,
                }

                # Convert raw hidden states to lists
                for layer_str, hs in raw_data['pos_hidden_states'].items():
                    pair_dict['positive_response']['layers_hidden_states'][layer_str] = hs.cpu().tolist()
                for layer_str, hs in raw_data['neg_hidden_states'].items():
                    pair_dict['negative_response']['layers_hidden_states'][layer_str] = hs.cpu().tolist()

                output_data['pairs'].append(pair_dict)

        else:
            # STANDARD MODE: Collect extracted vectors [hidden_size]
            print(f"\n⚡ Collecting activations...")
            
            enriched_pairs = []
            # Track norms for calibration
            layer_norms = {l: [] for l in layer_strs}
            
            for i, pair in enumerate(pair_set.pairs):
                if args.verbose:
                    print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...")

                # Collect activations for all requested layers at once
                updated_pair = collector.collect(
                    pair, strategy=extraction_strategy,
                    layers=layer_strs,
                )

                enriched_pairs.append(updated_pair)
                
                # Record norms for calibration
                for layer_str in layer_strs:
                    if updated_pair.positive_response.layers_activations:
                        pos_act = updated_pair.positive_response.layers_activations.get(layer_str)
                        if pos_act is not None:
                            layer_norms[layer_str].append(pos_act.norm().item())
                    if updated_pair.negative_response.layers_activations:
                        neg_act = updated_pair.negative_response.layers_activations.get(layer_str)
                        if neg_act is not None:
                            layer_norms[layer_str].append(neg_act.norm().item())

            print(f"   ✓ Collected activations for {len(enriched_pairs)} pairs")
            
            # Compute calibration norms (average per layer)
            calibration_norms = {}
            for layer_str, norms in layer_norms.items():
                if norms:
                    calibration_norms[layer_str] = sum(norms) / len(norms)
            
            if calibration_norms:
                print(f"   📊 Calibration norms: {', '.join(f'L{k}={v:.1f}' for k, v in calibration_norms.items())}")

            # Convert to JSON format (standard mode)
            print(f"\n💾 Saving enriched pairs to '{args.output}'...")
            output_data = {
                'task_name': task_name,
                'trait_label': trait_label,
                'model': args.model,
                'layers': layers,
                'extraction_strategy': extraction_strategy.value,
                'raw_mode': False,
                'num_pairs': len(enriched_pairs),
                'calibration_norms': calibration_norms,  # For auto-calibrated steering strength
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
