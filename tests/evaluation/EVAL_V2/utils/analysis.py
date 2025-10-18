import json
import random


# ============================================================================
# Core data retrieval functions - return raw data
# ============================================================================

def get_top_k_answers(path, k):
    """Get top k entries and return as list."""
    with open(path) as f:
        file_json = json.load(f)
    return file_json[:k], len(file_json)


def get_bot_k_answers(path, k):
    """Get bottom k entries and return as list."""
    with open(path) as f:
        file_json = json.load(f)
    return file_json[-k:], len(file_json)


def get_top_k_stats(path, k):
    """Get top k statistics configurations and return as list."""
    with open(path) as f:
        stats_json = json.load(f)
    return stats_json[:k], len(stats_json)


def get_specific_tuple(path, layer, strength, aggregation, steering):
    """Get entries matching specific configuration."""
    with open(path) as f:
        file_json = json.load(f)

    filtered = [entry for entry in file_json
                if entry['layer'] == layer
                and entry['strength'] == strength
                and entry['aggregation_method'] == aggregation
                and entry['steering'] == steering]

    return filtered


def get_random_entry(path):
    """Get a random entry from the file."""
    with open(path) as f:
        file_json = json.load(f)

    if not file_json:
        return None

    return random.choice(file_json)


# ============================================================================
# Formatting functions - convert data to display strings
# ============================================================================

def format_entry(entry, side_by_side=False):
    """Format a single entry as a string.

    Args:
        entry: Entry dictionary
        side_by_side: If True, return (config, baseline, steered, scores) tuple for side-by-side display
    """
    if side_by_side:
        # Return components separately for side-by-side layout
        config = f"""
**Layer:** {entry['layer']} | **Strength:** {entry['strength']} | **Aggregation:** {entry['aggregation_method']} | **Steering:** {entry['steering']}

**Question:** {entry['question']}
"""

        baseline = entry['baseline_response']
        steered = entry['steered_response']

        scores = f"""
**Scores:**

- Differentiation Score: {entry['differentiation_score']}
- Coherence Score: {entry['coherence_score']}
- Trait Alignment Score: {entry['trait_alignment_score']}
- Overall Score: {entry['overall_score']}

**Evaluation Results:**

- Choose Result: {entry['choose_result']}
- Open Traits: {entry['open_traits']}
"""
        return config, baseline, steered, scores
    else:
        # Original vertical layout
        output = f"""
Layer: {entry['layer']}, Strength: {entry['strength']}, Aggregation: {entry['aggregation_method']}, Steering: {entry['steering']}

Question: {entry['question']}

Baseline Response: {entry['baseline_response']}

Steered Response: {entry['steered_response']}

Scores:

- Differentiation Score: {entry['differentiation_score']}
- Coherence Score: {entry['coherence_score']}
- Trait Alignment Score: {entry['trait_alignment_score']}
- Overall Score: {entry['overall_score']}

Evaluation Results:

- Choose Result: {entry['choose_result']}
- Open Traits: {entry['open_traits']}
"""
        return output


def format_stat(config):
    """Format a single statistics configuration as a string."""
    output = f"""
Layer: {config['layer']}, Strength: {config['strength']}, Aggregation: {config['aggregation_method']}, Steering: {config['steering']}

Average Scores:

- Average Overall Score: {config['avg_overall_score']}
- Average Differentiation Score: {config['avg_differentiation_score']}
- Average Coherence Score: {config['avg_coherence_score']}
- Average Trait Alignment Score: {config['avg_trait_alignment_score']}

Choose Results:

- Correct (B): {config['choose_correct']}
- Incorrect (A): {config['choose_incorrect']}
- Equal: {config['choose_equal']}
- Total: {config['choose_total']}
"""
    return output


# ============================================================================
# Print functions for notebook/console use
# ============================================================================

def print_top_k_answers(path, k):
    entries, total = get_top_k_answers(path, k)
    print(f"Total entries: {total}")
    print(f"\nFirst {k} entries:")
    for i, entry in enumerate(entries, 1):
        print(f"\n--- Entry {i} ---")
        print(format_entry(entry))


def print_bot_k_answers(path, k):
    entries, total = get_bot_k_answers(path, k)
    print(f"Total entries: {total}")
    print(f"\nLast {k} entries:")
    for i, entry in enumerate(entries, 1):
        print(f"\n--- Entry {total - k + i} ---")
        print(format_entry(entry))


def print_top_k_stats(path, k):
    configs, total = get_top_k_stats(path, k)
    print(f"Total configurations: {total}")
    print(f"\nTop {k} configurations:")
    for i, config in enumerate(configs, 1):
        print(f"\n--- Configuration {i} ---")
        print(format_stat(config))


def print_specific_tuple(path, layer, strength, aggregation, steering):
    filtered = get_specific_tuple(path, layer, strength, aggregation, steering)

    print(f"Total entries matching (Layer={layer}, Strength={strength}, Aggregation={aggregation}, Steering={steering}): {len(filtered)}")

    if not filtered:
        print("No entries found matching this configuration.")
        return

    print("\nAll matching entries:")
    for i, entry in enumerate(filtered, 1):
        print(f"\n--- Entry {i} ---")
        print(format_entry(entry))


def print_random_entry(path):
    entry = get_random_entry(path)

    if entry is None:
        print("No entries found in file.")
        return

    print("Random entry:")
    print(format_entry(entry))
