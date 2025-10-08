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

evil_layers = config["evil"]["layers"]
evil_strengths = config["evil"]["strengths"]
evil_aggregations = config["evil"]["aggregations"]

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
        return happy_layers, happy_strengths, happy_aggregations
    elif "evil" in file_path:
        return evil_layers, evil_strengths, evil_aggregations
    else:
        # Default to combined if unclear
        all_layers = sorted(set(happy_layers + evil_layers))
        all_strengths = sorted(set(happy_strengths + evil_strengths))
        all_aggregations = sorted(set(happy_aggregations + evil_aggregations))
        return all_layers, all_strengths, all_aggregations


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

        # Return list of entries data
        entries_data = []
        for i, entry in enumerate(entries, 1):
            config, baseline, steered, scores = format_entry(entry, side_by_side=True)
            entries_data.append({
                'header': f"## Entry {i}\n\n{config}",
                'baseline': baseline,
                'steered': steered,
                'scores': scores
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

        # Return list of entries data
        entries_data = []
        for i, entry in enumerate(entries, 1):
            config, baseline, steered, scores = format_entry(entry, side_by_side=True)
            entries_data.append({
                'header': f"## Entry {total - k + i}\n\n{config}",
                'baseline': baseline,
                'steered': steered,
                'scores': scores
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


def display_filtered_entries(file_path, layer, strength, aggregation):
    """Display entries matching specific (layer, strength, aggregation) configuration."""
    try:
        filtered = get_specific_tuple(file_path, layer, strength, aggregation)

        if not filtered:
            summary = f"# Filtered Entries\n\nFilter: Layer={layer}, Strength={strength}, Aggregation={aggregation}\n\nTotal matching entries: 0\n\nNo entries found matching this configuration."
            return summary, []

        summary = f"# Filtered Entries\n\nFilter: Layer={layer}, Strength={strength}, Aggregation={aggregation}\n\nTotal matching entries: {len(filtered)}\n\n"

        # Return list of entries data
        entries_data = []
        for i, entry in enumerate(filtered, 1):
            config, baseline, steered, scores = format_entry(entry, side_by_side=True)
            entries_data.append({
                'header': f"## Entry {i}\n\n{config}",
                'baseline': baseline,
                'steered': steered,
                'scores': scores
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
