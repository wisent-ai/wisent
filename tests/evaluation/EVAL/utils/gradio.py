"""Helper functions for Gradio interface."""

import json
import random
import sys
import os

# Add EVAL directory to path to import generate.py
eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, eval_dir)

from generate import config, num_questions
from utils.analysis import (
    get_top_k_answers,
    get_bot_k_answers,
    get_top_k_stats,
    get_random_entry,
    get_specific_tuple,
    format_entry,
    format_stat
)

# Extract unique values from config for each trait
happy_layers = config["happy"]["layers"]
happy_strengths = config["happy"]["strengths"]
happy_aggregations = config["happy"]["aggregations"]
happy_steerings = config["happy"]["steering"]

evil_layers = config["evil"]["layers"]
evil_strengths = config["evil"]["strengths"]
evil_aggregations = config["evil"]["aggregations"]
evil_steerings = config["evil"]["steering"]

# File choices for dropdowns
SCORES_FILE_CHOICES = [
    "tests/EVAL/output/happy_scores_json.json",
    "tests/EVAL/output/happy_scores_markdown.json",
    "tests/EVAL/output/happy_scores_txt.json",
    "tests/EVAL/output/evil_scores_json.json",
    "tests/EVAL/output/evil_scores_markdown.json",
    "tests/EVAL/output/evil_scores_txt.json",
]

STATS_FILE_CHOICES = [
    "tests/EVAL/output/happy_stats_json.json",
    "tests/EVAL/output/happy_stats_markdown.json",
    "tests/EVAL/output/happy_stats_txt.json",
    "tests/EVAL/output/evil_stats_json.json",
    "tests/EVAL/output/evil_stats_markdown.json",
    "tests/EVAL/output/evil_stats_txt.json",
]


def get_config_for_file(file_path):
    """Determine which config (happy/evil) based on file path."""
    if "happy" in file_path:
        return happy_layers, happy_strengths, happy_aggregations, happy_steerings
    elif "evil" in file_path:
        return evil_layers, evil_strengths, evil_aggregations, evil_steerings
    else:
        # Default to combined if unclear
        all_layers = sorted(set(happy_layers + evil_layers))
        all_strengths = sorted(set(happy_strengths + evil_strengths))
        all_aggregations = sorted(set(happy_aggregations + evil_aggregations))
        all_steerings = sorted(set(happy_steerings + evil_steerings))
        return all_layers, all_strengths, all_aggregations, all_steerings


def load_scores_file(file_path):
    """Load a scores JSON file and return the entries."""
    with open(file_path) as f:
        return json.load(f)


def browse_entries(file_path, entry_index):
    """Browse entries by index (0-based)."""
    try:
        entries = load_scores_file(file_path)

        if not entries:
            return "No entries found in file.", "", "", "", 0, 0

        # Clamp index to valid range
        entry_index = max(0, min(entry_index, len(entries) - 1))

        entry = entries[entry_index]
        config, baseline, steered, scores = format_entry(entry, side_by_side=True)

        return config, baseline, steered, scores, entry_index, len(entries)
    except Exception as e:
        return f"Error loading file: {e}", "", "", "", 0, 0


def get_random_entry_display(file_path):
    """Get a random entry from the file for display."""
    try:
        entry = get_random_entry(file_path)

        if entry is None:
            return "No entries found in file.", "", "", "", 0, 0

        # Find the index of this entry
        with open(file_path) as f:
            entries = json.load(f)

        random_index = entries.index(entry)
        config, baseline, steered, scores = format_entry(entry, side_by_side=True)

        return config, baseline, steered, scores, random_index, len(entries)
    except Exception as e:
        return f"Error loading file: {e}", "", "", "", 0, 0


def display_top_k(file_path, k):
    """Display top k entries with highest overall scores."""
    try:
        entries, total = get_top_k_answers(file_path, k)

        if not entries:
            return "No entries found in file.", []

        summary = f"# Top {k} Entries (Highest Overall Scores)\n\nTotal entries in file: {total}\n\n"

        # Return list of entries data with raw entry info
        entries_data = []
        for i, entry in enumerate(entries, 1):
            config, baseline, steered, scores = format_entry(entry, side_by_side=True)
            entries_data.append({
                'header': f"## Entry {i}\n\n{config}",
                'baseline': baseline,
                'steered': steered,
                'scores': scores,
                'raw_entry': entry  # Include raw entry for state tracking
            })

        return summary, entries_data
    except Exception as e:
        return f"Error: {e}", []


def display_bottom_k(file_path, k):
    """Display bottom k entries with lowest overall scores."""
    try:
        entries, total = get_bot_k_answers(file_path, k)

        if not entries:
            return "No entries found in file.", []

        summary = f"# Bottom {k} Entries (Lowest Overall Scores)\n\nTotal entries in file: {total}\n\n"

        # Return list of entries data with raw entry info
        entries_data = []
        for i, entry in enumerate(entries, 1):
            config, baseline, steered, scores = format_entry(entry, side_by_side=True)
            entries_data.append({
                'header': f"## Entry {total - k + i}\n\n{config}",
                'baseline': baseline,
                'steered': steered,
                'scores': scores,
                'raw_entry': entry  # Include raw entry for state tracking
            })

        return summary, entries_data
    except Exception as e:
        return f"Error: {e}", []


def display_top_stats(stats_file_path, k):
    """Display top k configurations by average overall score."""
    try:
        configs, total = get_top_k_stats(stats_file_path, k)

        output = f"# Top {k} Configurations (Highest Average Overall Scores)\n\n"
        output += f"Total configurations: {total}\n\n"

        for i, config in enumerate(configs, 1):
            output += f"## Configuration {i}\n"
            output += format_stat(config)  # Use shared function directly
            output += "\n---\n\n"

        return output
    except Exception as e:
        return f"Error: {e}"


def display_filtered_entries(file_path, layer, strength, aggregation, steering):
    """Display entries matching specific (layer, strength, aggregation, steering) configuration."""
    try:
        filtered = get_specific_tuple(file_path, layer, strength, aggregation, steering)

        if not filtered:
            summary = f"# Filtered Entries\n\nFilter: Layer={layer}, Strength={strength}, Aggregation={aggregation}, Steering={steering}\n\nTotal matching entries: 0\n\nNo entries found matching this configuration."
            return summary, []

        summary = f"# Filtered Entries\n\nFilter: Layer={layer}, Strength={strength}, Aggregation={aggregation}, Steering={steering}\n\nTotal matching entries: {len(filtered)}\n\n"

        # Return list of entries data with raw entry info
        entries_data = []
        for i, entry in enumerate(filtered, 1):
            config, baseline, steered, scores = format_entry(entry, side_by_side=True)
            entries_data.append({
                'header': f"## Entry {i}\n\n{config}",
                'baseline': baseline,
                'steered': steered,
                'scores': scores,
                'raw_entry': entry  # Include raw entry for state tracking
            })

        return summary, entries_data
    except Exception as e:
        return f"Error: {e}", []


def get_initial_game_state():
    """Get initial game state dictionary."""
    return {
        "file_path": None,
        "current_entry": None,
        "is_left_steered": None,
        "round": 0,
        "score": 0,
        "total_rounds": 10,
        "game_active": False
    }


def start_blind_game(file_path, state):
    """Start a new 10-round blind test game."""
    # Reset state
    state["file_path"] = file_path
    state["round"] = 1
    state["score"] = 0
    state["game_active"] = True

    # Load first question
    return load_next_question(state)


def load_next_question(state):
    """Load the next question pair."""
    try:
        import gradio as gr

        entry = get_random_entry(state["file_path"])
        if entry is None:
            return "No entries found.", "", "", "", "", gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), state

        # Randomly decide if steered is on left or right
        is_left_steered = random.choice([True, False])

        # Store state
        state["current_entry"] = entry
        state["is_left_steered"] = is_left_steered

        # Prepare displays
        question = f"**Round {state['round']}/{state['total_rounds']}**\n\n**Question:** {entry['question']}"

        if is_left_steered:
            left_response = entry['steered_response']
            right_response = entry['baseline_response']
        else:
            left_response = entry['baseline_response']
            right_response = entry['steered_response']

        progress = f"**Score:** {state['score']}/{state['round']-1}" if state['round'] > 1 else "**Score:** 0/0"

        # Hide start button, show guess buttons
        return question, left_response, right_response, "", progress, gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=True), state

    except Exception as e:
        import gradio as gr
        return f"Error: {e}", "", "", "", "", gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), state


def check_guess(choice, current_file_path, state):
    """Check user's guess and load next question or end game."""
    import gradio as gr

    if not state["game_active"]:
        return "Game not started!", "", "", "", "", gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), state

    is_left_steered = state["is_left_steered"]

    # Check if correct
    is_correct = (choice == "A" and is_left_steered) or (choice == "B" and not is_left_steered)

    if is_correct:
        state["score"] += 1
        result = "#  Correct!\n\n"
    else:
        result = "#  Incorrect\n\n"

    # Show which was which
    if is_left_steered:
        result += "Response A was **steered**, Response B was **baseline**."
    else:
        result += "Response A was **baseline**, Response B was **steered**."

    # Check if game is over
    if state["round"] >= state["total_rounds"]:
        # Game over
        state["game_active"] = False
        final_score = state["score"]
        total = state["total_rounds"]
        percentage = (final_score / total) * 100

        result += f"\n\n# Game Over!\n\n**Final Score: {final_score}/{total} ({percentage:.1f}%)**"

        progress = f"**Final Score:** {final_score}/{total}"

        # Clear question and responses, show start button, hide guess buttons
        return result, "", "", "", progress, gr.Button(visible=True), gr.Button(visible=False), gr.Button(visible=False), state
    else:
        # Load next round
        state["round"] += 1

        # Get next question from CURRENT file path (allows switching mid-game)
        entry = get_random_entry(current_file_path)
        is_left_steered = random.choice([True, False])

        state["current_entry"] = entry
        state["is_left_steered"] = is_left_steered

        question = f"**Round {state['round']}/{state['total_rounds']}**\n\n**Question:** {entry['question']}"

        if is_left_steered:
            left_response = entry['steered_response']
            right_response = entry['baseline_response']
        else:
            left_response = entry['baseline_response']
            right_response = entry['steered_response']

        progress = f"**Score:** {state['score']}/{state['round']-1}"

        # Keep guess buttons visible
        return result, question, left_response, right_response, progress, gr.Button(visible=False), gr.Button(visible=True), gr.Button(visible=True), state


def plot_score_histogram(file_path, score_type='overall_score'):
    """
    Generate histogram of scores from a scores file.

    Args:
        file_path: Path to the JSON scores file
        score_type: Type of score to plot. Options:
                   - 'overall_score'
                   - 'differentiation_score'
                   - 'coherence_score'
                   - 'trait_alignment_score'

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Score type to display name mapping
    score_names = {
        'overall_score': 'Overall Score',
        'differentiation_score': 'Differentiation Score',
        'coherence_score': 'Coherence Score',
        'trait_alignment_score': 'Trait Alignment Score'
    }

    score_display_name = score_names.get(score_type, score_type.replace('_', ' ').title())

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract scores (filter out None values)
        scores = []
        for entry in data:
            if score_type in entry and entry[score_type] is not None:
                scores.append(entry[score_type])

        if not scores:
            # Return empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No {score_display_name.lower()} found',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine bins - use integer bins if scores are in 0-100 range
        min_score = min(scores)
        max_score = max(scores)

        if min_score >= 0 and max_score <= 100:
            bins = np.arange(0, 105, 5)  # Bins of 5 from 0 to 100
        else:
            bins = 30  # Default number of bins

        ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(score_display_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {score_display_name}\n{os.path.basename(file_path)}',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add mean, median, and standard deviation
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_dev = np.std(scores)

        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_score:.2f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_score:.2f}')

        # Add invisible line to include std dev in legend
        ax.plot([], [], ' ', label=f'Std Dev: {std_dev:.2f}')

        # Use 'best' location to automatically find the best position
        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        return fig

    except Exception as e:
        # Return error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


# Keep backward compatibility
def plot_overall_score_histogram(file_path):
    """Wrapper for backward compatibility."""
    return plot_score_histogram(file_path, 'overall_score')


def plot_score_correlations(file_path):
    """
    Generate pairwise correlation scatter plots between different score types.

    Args:
        file_path: Path to the JSON scores file

    Returns:
        matplotlib figure object with 6 scatter plots (one for each pair)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract all score types
        score_types = ['differentiation_score', 'coherence_score', 'trait_alignment_score', 'overall_score']
        scores_data = {score_type: [] for score_type in score_types}

        # Collect scores (only include entries where all scores are present)
        for entry in data:
            if all(score_type in entry and entry[score_type] is not None for score_type in score_types):
                for score_type in score_types:
                    scores_data[score_type].append(entry[score_type])

        if not scores_data['overall_score']:
            # Return empty plot with message
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No complete score data found',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        # Create figure with 2x3 subplots for pairwise comparisons
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Pairwise Score Correlations\n{os.path.basename(file_path)}',
                    fontsize=16, fontweight='bold')

        score_labels = {
            'differentiation_score': 'Differentiation',
            'coherence_score': 'Coherence',
            'trait_alignment_score': 'Trait Alignment',
            'overall_score': 'Overall'
        }

        # Define pairs for correlation plots
        pairs = [
            ('differentiation_score', 'coherence_score'),
            ('differentiation_score', 'trait_alignment_score'),
            ('differentiation_score', 'overall_score'),
            ('coherence_score', 'trait_alignment_score'),
            ('coherence_score', 'overall_score'),
            ('trait_alignment_score', 'overall_score')
        ]

        for idx, (score_x, score_y) in enumerate(pairs):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            x_data = scores_data[score_x]
            y_data = scores_data[score_y]

            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.5, s=20, color='steelblue')

            # Calculate and display correlation coefficient
            correlation = np.corrcoef(x_data, y_data)[0, 1]

            # Add trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(score_labels[score_x], fontsize=10)
            ax.set_ylabel(score_labels[score_y], fontsize=10)
            ax.set_title(f'Correlation: {correlation:.3f}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        # Return error plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def plot_config_score_histogram(file_path, layer, strength, aggregation, steering, score_type='overall_score'):
    """
    Generate histogram of scores for a specific configuration.

    Args:
        file_path: Path to the JSON scores file
        layer: Layer value to filter
        strength: Strength value to filter
        aggregation: Aggregation method to filter
        steering: Steering method to filter
        score_type: Type of score to plot

    Returns:
        matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Score type to display name mapping
    score_names = {
        'overall_score': 'Overall Score',
        'differentiation_score': 'Differentiation Score',
        'coherence_score': 'Coherence Score',
        'trait_alignment_score': 'Trait Alignment Score'
    }

    score_display_name = score_names.get(score_type, score_type.replace('_', ' ').title())

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Filter entries by configuration and extract scores
        scores = []
        for entry in data:
            if (entry.get('layer') == layer and
                entry.get('strength') == strength and
                entry.get('aggregation_method') == aggregation and
                entry.get('steering') == steering):
                if score_type in entry and entry[score_type] is not None:
                    scores.append(entry[score_type])

        if not scores:
            # Return empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'No {score_display_name.lower()} found\nfor this configuration',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        # Create histogram
        fig, ax = plt.subplots(figsize=(10, 6))

        # Determine bins
        min_score = min(scores)
        max_score = max(scores)

        if min_score >= 0 and max_score <= 100:
            bins = np.arange(0, 105, 5)
        else:
            bins = 30

        ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(score_display_name, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Distribution of {score_display_name}\nLayer: {layer}, Strength: {strength}, Aggregation: {aggregation}, Steering: {steering}',
                    fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add statistics
        mean_score = np.mean(scores)
        median_score = np.median(scores)
        std_dev = np.std(scores)

        ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_score:.2f}')
        ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_score:.2f}')

        # Add invisible line to include std dev in legend
        ax.plot([], [], ' ', label=f'Std Dev: {std_dev:.2f}')
        ax.plot([], [], ' ', label=f'N: {len(scores)}')

        ax.legend(loc='best', framealpha=0.9)

        plt.tight_layout()
        return fig

    except Exception as e:
        # Return error plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

def plot_config_score_correlations(file_path, layer, strength, aggregation, steering):
    """
    Generate pairwise correlation scatter plots for a specific configuration.

    Args:
        file_path: Path to the JSON scores file
        layer: Layer value to filter
        strength: Strength value to filter
        aggregation: Aggregation method to filter
        steering: Steering method to filter

    Returns:
        matplotlib figure object with 6 scatter plots (one for each pair)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Filter entries by configuration
        score_types = ['differentiation_score', 'coherence_score', 'trait_alignment_score', 'overall_score']
        scores_data = {score_type: [] for score_type in score_types}

        # Collect scores for matching configuration
        for entry in data:
            if (entry.get('layer') == layer and
                entry.get('strength') == strength and
                entry.get('aggregation_method') == aggregation and
                entry.get('steering') == steering):
                if all(score_type in entry and entry[score_type] is not None for score_type in score_types):
                    for score_type in score_types:
                        scores_data[score_type].append(entry[score_type])

        if not scores_data['overall_score']:
            # Return empty plot with message
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(0.5, 0.5, 'No complete score data found\nfor this configuration',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig

        # Create figure with 2x3 subplots for pairwise comparisons
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Pairwise Score Correlations\nLayer: {layer}, Strength: {strength}, Aggregation: {aggregation}, Steering: {steering}',
                    fontsize=14, fontweight='bold')

        score_labels = {
            'differentiation_score': 'Differentiation',
            'coherence_score': 'Coherence',
            'trait_alignment_score': 'Trait Alignment',
            'overall_score': 'Overall'
        }

        # Define pairs for correlation plots
        pairs = [
            ('differentiation_score', 'coherence_score'),
            ('differentiation_score', 'trait_alignment_score'),
            ('differentiation_score', 'overall_score'),
            ('coherence_score', 'trait_alignment_score'),
            ('coherence_score', 'overall_score'),
            ('trait_alignment_score', 'overall_score')
        ]

        for idx, (score_x, score_y) in enumerate(pairs):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]

            x_data = scores_data[score_x]
            y_data = scores_data[score_y]

            # Scatter plot
            ax.scatter(x_data, y_data, alpha=0.5, s=20, color='steelblue')

            # Calculate and display correlation coefficient
            correlation = np.corrcoef(x_data, y_data)[0, 1]

            # Add trend line
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            ax.set_xlabel(score_labels[score_x], fontsize=10)
            ax.set_ylabel(score_labels[score_y], fontsize=10)
            ax.set_title(f'Correlation: {correlation:.3f} (N={len(x_data)})', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    except Exception as e:
        # Return error plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig


def get_questions_ranked_by_score(file_path, score_type='overall_score'):
    """
    Get all questions ranked by average score across all configurations.

    Args:
        file_path: Path to the JSON scores file
        score_type: Type of score to rank by

    Returns:
        List of tuples (question, average_score, num_configs) sorted by score descending
    """
    import json

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Group scores by question
        question_scores = {}
        for entry in data:
            question = entry.get('question', '')
            score = entry.get(score_type)
            
            if question and score is not None:
                if question not in question_scores:
                    question_scores[question] = []
                question_scores[question].append(score)

        # Calculate averages
        question_averages = []
        for question, scores in question_scores.items():
            avg_score = sum(scores) / len(scores)
            question_averages.append((question, avg_score, len(scores)))

        # Sort by average score descending
        question_averages.sort(key=lambda x: x[1], reverse=True)

        return question_averages

    except Exception as e:
        print(f"Error processing file: {e}")
        return []


def save_favorite_answer(file_path, question, steered_response, layer, strength, aggregation, notes=""):
    """
    Save a favorite steered answer to a JSON file.

    Args:
        file_path: Path to save favorites (will be created if doesn't exist)
        question: The question text
        steered_response: The steered response text
        layer: Layer configuration
        strength: Strength configuration
        aggregation: Aggregation method
        notes: Optional user notes
    """
    import json
    import os
    from datetime import datetime

    favorites = []
    
    # Load existing favorites if file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                favorites = json.load(f)
        except:
            favorites = []

    # Add new favorite
    favorite = {
        'question': question,
        'steered_response': steered_response,
        'layer': layer,
        'strength': strength,
        'aggregation': aggregation,
        'notes': notes,
        'timestamp': datetime.now().isoformat()
    }

    favorites.append(favorite)

    # Save back to file
    with open(file_path, 'w') as f:
        json.dump(favorites, f, indent=2)

    return True


def load_favorites(file_path):
    """
    Load saved favorite answers.

    Args:
        file_path: Path to favorites file

    Returns:
        List of favorite entries
    """
    import json
    import os

    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return []


def get_all_entries_for_browsing(file_path):
    """
    Get all entries from a scores file for browsing.

    Args:
        file_path: Path to the JSON scores file

    Returns:
        List of all entries
    """
    import json

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return []


def delete_favorite_by_index(index):
    """
    Delete a specific favorite by its index.

    Args:
        index: Index of the favorite to delete (0-based, from the original order)

    Returns:
        True if successful, False otherwise
    """
    import json
    import os

    favorites_file = "tests/EVAL/output/favorites.json"

    if not os.path.exists(favorites_file):
        return False

    try:
        with open(favorites_file, 'r') as f:
            favorites = json.load(f)

        # Delete the item at the specified index
        if 0 <= index < len(favorites):
            favorites.pop(index)

            # Save back to file
            with open(favorites_file, 'w') as f:
                json.dump(favorites, f, indent=2)

            return True
        return False

    except Exception as e:
        print(f"Error deleting favorite: {e}")
        return False
