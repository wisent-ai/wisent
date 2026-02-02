"""Generate all layer combinations for hyperparameter search."""

from itertools import combinations
from math import comb
from typing import List


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
    # Test with 16 layers (like Llama-3.2-1B) and max_combo_size=3
    num_layers = 16
    max_combo_size = 3
    combos = get_layer_combinations(num_layers, max_combo_size)
    
    print(f"Model with {num_layers} layers, max_combo_size={max_combo_size}:")
    print(f"Total combinations: {len(combos)}")
    print(f"Expected: {get_layer_combinations_count(num_layers, max_combo_size)}")
    print()
    
    print("First 20 combinations:")
    for i, combo in enumerate(combos[:20]):
        print(f"  {i+1}: {combo}")
    if len(combos) > 20:
        print(f"  ... and {len(combos) - 20} more")
