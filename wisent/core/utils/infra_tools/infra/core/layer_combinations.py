"""Generate all layer combinations for hyperparameter search."""

from itertools import combinations
from math import comb
from typing import List

from wisent.core.utils.config_tools.constants import COMBO_OFFSET


def get_layer_combinations(num_layers: int, max_combo_size: int, single_and_all_only: bool = True) -> List[List[int]]:
    """
    Generate layer combinations up to a maximum combination size.
    
    Args:
        num_layers: Total number of layers in the model
        max_combo_size: Maximum number of layers in a combination (e.g., 3)
        single_and_all_only: If True, only return single layers and all layers together
                             (skip 2-layer, 3-layer combinations). Default: True
        
    Returns:
        List of layer combinations:
        - All layers together: [0, 1, 2, ..., num_layers-1]
        - All individual layers: [0], [1], ..., [num_layers-1]
        - (if not single_and_all_only) All combinations of 2, 3, ..., max_combo_size layers
    """
    all_layers = list(range(num_layers))
    result = []
    
    # All layers together (always included)
    result.append(all_layers)
    
    # All individual layers
    for layer in all_layers:
        result.append([layer])
    
    # All combinations of 2, 3, ..., max_combo_size layers (unless single_and_all_only)
    if not single_and_all_only:
        for r in range(2, max_combo_size + 1):
            for combo in combinations(all_layers, r):
                result.append(list(combo))
    
    return result


def get_layer_combinations_count(num_layers: int, max_combo_size: int) -> int:
    """
    Calculate total number of layer combinations without generating them.
    
    Total = 1 (all layers) + C(n,1) + C(n,2) + ... + C(n, max_combo_size)
    """
    total = 1  # all layers
    for r in range(1, max_combo_size + 1):
        total += comb(num_layers, r)
    return total


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate layer combinations")
    parser.add_argument("num_layers", type=int, help="Number of layers in the model")
    parser.add_argument("--max-combo-size", type=int, required=True, help="Max combination size")
    parser.add_argument("--preview-limit", type=int, required=True, help="Number of combos to preview")
    cli_args = parser.parse_args()
    num_layers = cli_args.num_layers
    max_combo_size = cli_args.max_combo_size
    preview_limit = cli_args.preview_limit
    combos = get_layer_combinations(num_layers, max_combo_size)

    print(f"Model with {num_layers} layers, max_combo_size={max_combo_size}:")
    print(f"Total combinations: {len(combos)}")
    print(f"Expected: {get_layer_combinations_count(num_layers, max_combo_size)}")
    print()

    print(f"First {preview_limit} combinations:")
    for i, combo in enumerate(combos[:preview_limit]):
        print(f"  {i+COMBO_OFFSET}: {combo}")
    if len(combos) > preview_limit:
        print(f"  ... and {len(combos) - preview_limit} more")
