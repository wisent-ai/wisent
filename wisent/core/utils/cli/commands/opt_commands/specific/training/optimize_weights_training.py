"""Multi-direction method training for optimize-weights."""
import os

import torch

from wisent.core.constants import DEFAULT_CHECKPOINT_INTERVAL, DEFAULT_LAYER_WEIGHT, DEFAULT_LIMIT


def _train_multi_direction_method(
    args,
    caa_vectors: dict[int, torch.Tensor],
    intermediate_dir: str,
    method: str,
) -> dict[int, torch.Tensor]:
    """Train GROM/TECZA/TETNO on pairs with activations and return combined directions.
    
    Args:
        args: Command line arguments
        caa_vectors: CAA vectors (used as fallback if training fails)
        intermediate_dir: Directory containing pairs with activations
        method: 'grom', 'tecza', or 'tetno'
        
    Returns:
        Combined steering vectors per layer
    """
    import glob
    
    # Find enriched pairs file (with activations)
    enriched_files = glob.glob(os.path.join(intermediate_dir, "*_with_activations.json"))
    
    if not enriched_files:
        print(f"   Warning: No enriched pairs found for {method}, using CAA vectors")
        return caa_vectors
    
    enriched_file = enriched_files[0]
    print(f"\n   Training {method.upper()} on enriched pairs...")
    print(f"   Pairs file: {enriched_file}")
    
    try:
        from wisent.core.contrastive_pairs.io.serialization import load_contrastive_pair_set
        from wisent.core.weight_modification.multi_direction import (
            MultiDirectionConfig,
            combine_directions,
        )
        
        # Load pair set with activations
        pair_set = load_contrastive_pair_set(enriched_file)
        print(f"   Loaded {len(pair_set.pairs)} pairs with activations")
        
        # Get config from args
        num_directions = getattr(args, 'num_directions', DEFAULT_CHECKPOINT_INTERVAL)
        combination_strategy = getattr(args, 'combination_strategy', 'learned')
        optimization_steps = getattr(args, 'multi_optimization_steps', DEFAULT_LIMIT)
        
        # Train the method
        if method == 'grom':
            from wisent.core.steering_methods.methods.grom import GROMMethod, GROMConfig
            config = GROMConfig(
                num_directions=num_directions,
                optimization_steps=optimization_steps,
            )
            trainer = GROMMethod(config=config)
            result = trainer.train_grom(pair_set)
            directions = result.directions
            weights = result.direction_weights
            
        elif method == 'tecza':
            from wisent.core.steering_methods.methods.advanced import TECZAMethod, TECZAConfig
            config = TECZAConfig(
                num_directions=num_directions,
                optimization_steps=optimization_steps,
            )
            trainer = TECZAMethod(config=config)
            result = trainer.train_tecza(pair_set)
            directions = result.directions
            weights = None  # TECZA doesn't have learned weights
            
        elif method == 'tetno':
            from wisent.core.steering_methods.methods.advanced import TETNOMethod, TETNOConfig
            config = TETNOConfig(
                optimization_steps=optimization_steps,
            )
            trainer = TETNOMethod(config=config)
            result = trainer.train_tetno(pair_set)
            # TETNO has single direction per layer
            directions = {k: v.unsqueeze(0) for k, v in result.behavior_vectors.items()}
            weights = {k: torch.tensor([result.layer_scales.get(k, DEFAULT_LAYER_WEIGHT)])
                      for k in directions} if result.layer_scales else None
        
        print(f"   Trained {len(directions)} layers with {method.upper()}")
        
        # Combine directions into single vector per layer
        combined_vectors = {}
        for layer_name, layer_dirs in directions.items():
            # Get layer index
            try:
                layer_idx = int(layer_name.replace("layer_", "")) - 1
            except ValueError:
                layer_idx = int(layer_name) - 1
            
            layer_weights = weights.get(layer_name) if weights else None
            combined = combine_directions(layer_dirs, layer_weights, strategy=combination_strategy)
            combined_vectors[layer_idx] = combined
            
        print(f"   Combined into {len(combined_vectors)} steering vectors")
        print(f"   Combination strategy: {combination_strategy}")
        
        return combined_vectors
        
    except Exception as e:
        print(f"   Warning: {method.upper()} training failed: {e}")
        print(f"   Falling back to CAA vectors")
        return caa_vectors

