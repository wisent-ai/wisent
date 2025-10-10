import gradio as gr
from utils.gradio import (
    # Config and constants
    happy_layers, happy_strengths, happy_aggregations, happy_steerings,
    SCORES_FILE_CHOICES, STATS_FILE_CHOICES, num_questions,
    # Functions
    get_config_for_file, browse_entries, get_random_entry_display,
    display_top_k, display_bottom_k, display_top_stats, display_filtered_entries,
    start_blind_game, check_guess, get_initial_game_state,
    plot_score_histogram, plot_score_correlations, plot_config_score_histogram,
    plot_config_score_correlations, get_questions_ranked_by_score,
    save_favorite_answer, load_favorites, get_all_entries_for_browsing,
    delete_favorite_by_index
)


# Shared save favorite function
def save_to_favorites(question, steered_response, layer, strength, aggregation, steering, notes="",
                      overall_score=None, diff_score=None, coh_score=None, trait_score=None):
    """Save an answer to favorites."""
    import os
    import json
    from datetime import datetime

    favorites_file = "tests/EVAL/output/favorites.json"
    os.makedirs(os.path.dirname(favorites_file), exist_ok=True)

    try:
        # Load existing favorites
        favorites = []
        if os.path.exists(favorites_file):
            try:
                with open(favorites_file, 'r') as f:
                    favorites = json.load(f)
            except:
                favorites = []

        # Add new favorite with scores
        favorite = {
            'question': question,
            'steered_response': steered_response,
            'layer': layer,
            'strength': strength,
            'aggregation': aggregation,
            'steering': steering,
            'notes': notes,
            'overall_score': overall_score,
            'differentiation_score': diff_score,
            'coherence_score': coh_score,
            'trait_alignment_score': trait_score,
            'timestamp': datetime.now().isoformat()
        }

        favorites.append(favorite)

        # Save back to file
        with open(favorites_file, 'w') as f:
            json.dump(favorites, f, indent=2)

        return "✅ Saved to favorites!"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_save_favorite_handler(entry_state_var):
    """
    Create a save favorite handler function for a given entry state variable.

    Args:
        entry_state_var: A gr.State variable containing entry data

    Returns:
        Tuple of (save_button, status_markdown, save_function)
    """
    save_btn = gr.Button("⭐ Save as Favorite", variant="secondary")
    status_md = gr.Markdown("")

    def save_handler(entry_data):
        """Save current entry to favorites."""
        if not entry_data or 'question' not in entry_data:
            return "⚠️ No entry loaded"
        return save_to_favorites(
            entry_data['question'],
            entry_data['steered_response'],
            entry_data['layer'],
            entry_data['strength'],
            entry_data['aggregation'],
            entry_data['steering'],
            notes="",
            overall_score=entry_data.get('overall_score'),
            diff_score=entry_data.get('differentiation_score'),
            coh_score=entry_data.get('coherence_score'),
            trait_score=entry_data.get('trait_alignment_score')
        )

    save_btn.click(
        fn=save_handler,
        inputs=[entry_state_var],
        outputs=[status_md]
    )

    return save_btn, status_md


# Create the Gradio interface
with gr.Blocks(title="LLM Steering Evaluation Results") as demo:
    gr.Markdown("# LLM Steering Evaluation Results Browser")
    gr.Markdown("Browse evaluation results from different steering configurations.")

    # Tabs for different views
    with gr.Tabs():
        # Tab 1: Browse Individual Entries
        with gr.Tab("Browse"):
            gr.Markdown("""
            **For a given trait and prompt format browse individual evaluation entries one at a time.**
            """)
            with gr.Row():
                # File selector
                file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File",
                    value=SCORES_FILE_CHOICES[0]
                )

            with gr.Row():
                # Entry navigation
                entry_index = gr.Number(label="Entry Index (0-based)", value=0, precision=0)
                total_entries = gr.Number(label="Total Entries", value=0, interactive=False)

            with gr.Row():
                # Navigation buttons
                prev_btn = gr.Button("◀ Previous")
                next_btn = gr.Button("Next ▶")
                random_btn = gr.Button("🎲 Random")
                load_btn = gr.Button("Load Entry")

            # Display area - config and question
            entry_config = gr.Markdown(label="Configuration & Question")

            # Side-by-side responses
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Baseline Response")
                    baseline_display = gr.Textbox(label="", lines=15, max_lines=25, interactive=False)

                with gr.Column():
                    gr.Markdown("### Steered Response")
                    steered_display = gr.Textbox(label="", lines=15, max_lines=25, interactive=False)

                    # Hidden state to track current entry (must be created before save button)
                    current_entry = gr.State(value={})

                    # Reusable save button
                    save_fav_btn, save_status = create_save_favorite_handler(current_entry)

            # Scores below
            entry_scores = gr.Markdown(label="Scores & Evaluation")

            # Button actions
            def browse_and_track(file_path, entry_index):
                """Browse entries and track current entry data."""
                result = browse_entries(file_path, entry_index)

                # Load the actual entry to track its data
                entries = get_all_entries_for_browsing(file_path)
                if entries and 0 <= entry_index < len(entries):
                    entry = entries[entry_index]
                    entry_data = {
                        'question': entry.get('question', ''),
                        'steered_response': entry.get('steered_response', ''),
                        'layer': entry.get('layer'),
                        'strength': entry.get('strength'),
                        'aggregation': entry.get('aggregation_method'),
                        'steering': entry.get('steering'),
                        'overall_score': entry.get('overall_score'),
                        'differentiation_score': entry.get('differentiation_score'),
                        'coherence_score': entry.get('coherence_score'),
                        'trait_alignment_score': entry.get('trait_alignment_score')
                    }
                    return (*result, entry_data, "")
                return (*result, {}, "")

            def go_previous(file_path, current_index):
                new_index = max(0, current_index - 1)
                return browse_and_track(file_path, new_index)

            def go_next(file_path, current_index):
                new_index = current_index + 1
                return browse_and_track(file_path, new_index)

            def go_random(file_path):
                """Get random entry and track it."""
                result = get_random_entry_display(file_path)
                # Extract the index from result
                random_index = result[4] if len(result) > 4 else 0

                entries = get_all_entries_for_browsing(file_path)
                if entries and 0 <= random_index < len(entries):
                    entry = entries[random_index]
                    entry_data = {
                        'question': entry.get('question', ''),
                        'steered_response': entry.get('steered_response', ''),
                        'layer': entry.get('layer'),
                        'strength': entry.get('strength'),
                        'aggregation': entry.get('aggregation_method'),
                        'steering': entry.get('steering'),
                        'overall_score': entry.get('overall_score'),
                        'differentiation_score': entry.get('differentiation_score'),
                        'coherence_score': entry.get('coherence_score'),
                        'trait_alignment_score': entry.get('trait_alignment_score')
                    }
                    return (*result, entry_data, "")
                return (*result, {}, "")

            prev_btn.click(
                fn=go_previous,
                inputs=[file_dropdown, entry_index],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries, current_entry, save_status]
            )

            next_btn.click(
                fn=go_next,
                inputs=[file_dropdown, entry_index],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries, current_entry, save_status]
            )

            random_btn.click(
                fn=go_random,
                inputs=[file_dropdown],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries, current_entry, save_status]
            )

            load_btn.click(
                fn=browse_and_track,
                inputs=[file_dropdown, entry_index],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries, current_entry, save_status]
            )

            # Load initial entry when file changes
            file_dropdown.change(
                fn=lambda file_path: browse_and_track(file_path, 0),
                inputs=[file_dropdown],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries, current_entry, save_status]
            )

        # Tab 2: Top K Entries
        with gr.Tab("Top K"):
            gr.Markdown("""
            **For a given trait and prompt format view the top K entries with the highest overall score.**
            """)
            with gr.Row():
                top_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File",
                    value=SCORES_FILE_CHOICES[0]
                )
                top_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Entries (K)")

            top_k_btn = gr.Button("Load Top K Entries")

            # Pre-create slots for up to 10 entries
            top_k_summary = gr.Markdown("")
            top_k_outputs = []

            for i in range(10):
                with gr.Column(visible=False) as entry_col:
                    entry_header = gr.Markdown("")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Baseline Response**")
                            baseline_text = gr.Textbox(label="", lines=10, max_lines=15, interactive=False)
                        with gr.Column():
                            gr.Markdown("**Steered Response**")
                            steered_text = gr.Textbox(label="", lines=10, max_lines=15, interactive=False)

                            # Create state and save button for this entry
                            entry_state = gr.State(value={})
                            save_btn, save_status = create_save_favorite_handler(entry_state)

                    scores_md = gr.Markdown("")
                    gr.Markdown("---")

                    top_k_outputs.append({
                        'col': entry_col,
                        'header': entry_header,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'scores': scores_md,
                        'state': entry_state,
                        'save_status': save_status
                    })

            def update_top_k(file_path, k):
                """Update display with top k entries."""
                summary, entries_data = display_top_k(file_path, k)

                updates = [gr.Markdown(value=summary)]

                for i in range(10):
                    if i < len(entries_data):
                        entry_display = entries_data[i]
                        raw_entry = entry_display.get('raw_entry', {})

                        # Extract entry data for state
                        entry_state_data = {
                            'question': raw_entry.get('question', ''),
                            'steered_response': raw_entry.get('steered_response', ''),
                            'layer': raw_entry.get('layer'),
                            'strength': raw_entry.get('strength'),
                            'aggregation': raw_entry.get('aggregation_method'),
                            'steering': raw_entry.get('steering'),
                            'overall_score': raw_entry.get('overall_score'),
                            'differentiation_score': raw_entry.get('differentiation_score'),
                            'coherence_score': raw_entry.get('coherence_score'),
                            'trait_alignment_score': raw_entry.get('trait_alignment_score')
                        }

                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=entry_display['header']),
                            gr.Textbox(value=entry_display['baseline']),
                            gr.Textbox(value=entry_display['steered']),
                            gr.Markdown(value=entry_display['scores']),
                            entry_state_data,  # Update state
                            ""  # Clear save status
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Textbox(value=""),
                            gr.Textbox(value=""),
                            gr.Markdown(value=""),
                            {},  # Empty state
                            ""  # Empty save status
                        ])

                return updates

            outputs_list = [top_k_summary]
            for entry in top_k_outputs:
                outputs_list.extend([entry['col'], entry['header'], entry['baseline'], entry['steered'], entry['scores'], entry['state'], entry['save_status']])

            top_k_btn.click(
                fn=update_top_k,
                inputs=[top_file_dropdown, top_k_slider],
                outputs=outputs_list
            )

        # Tab 3: Bottom K Entries
        with gr.Tab("Bottom K"):
            gr.Markdown("""
            **For a given trait and prompt format view the bottom K entries with the lowest overall score.**
            """)
            with gr.Row():
                bot_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File",
                    value=SCORES_FILE_CHOICES[0]
                )
                bot_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Entries (K)")

            bot_k_btn = gr.Button("Load Bottom K Entries")

            # Pre-create slots for up to 10 entries
            bot_k_summary = gr.Markdown("")
            bot_k_outputs = []

            for i in range(10):
                with gr.Column(visible=False) as entry_col:
                    entry_header = gr.Markdown("")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Baseline Response**")
                            baseline_text = gr.Textbox(label="", lines=10, max_lines=15, interactive=False)
                        with gr.Column():
                            gr.Markdown("**Steered Response**")
                            steered_text = gr.Textbox(label="", lines=10, max_lines=15, interactive=False)

                            # Create state and save button for this entry
                            entry_state = gr.State(value={})
                            save_btn, save_status = create_save_favorite_handler(entry_state)

                    scores_md = gr.Markdown("")
                    gr.Markdown("---")

                    bot_k_outputs.append({
                        'col': entry_col,
                        'header': entry_header,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'scores': scores_md,
                        'state': entry_state,
                        'save_status': save_status
                    })

            def update_bottom_k(file_path, k):
                """Update display with bottom k entries."""
                summary, entries_data = display_bottom_k(file_path, k)

                updates = [gr.Markdown(value=summary)]

                for i in range(10):
                    if i < len(entries_data):
                        entry_display = entries_data[i]
                        raw_entry = entry_display.get('raw_entry', {})

                        # Extract entry data for state
                        entry_state_data = {
                            'question': raw_entry.get('question', ''),
                            'steered_response': raw_entry.get('steered_response', ''),
                            'layer': raw_entry.get('layer'),
                            'strength': raw_entry.get('strength'),
                            'aggregation': raw_entry.get('aggregation_method'),
                            'steering': raw_entry.get('steering'),
                            'overall_score': raw_entry.get('overall_score'),
                            'differentiation_score': raw_entry.get('differentiation_score'),
                            'coherence_score': raw_entry.get('coherence_score'),
                            'trait_alignment_score': raw_entry.get('trait_alignment_score')
                        }

                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=entry_display['header']),
                            gr.Textbox(value=entry_display['baseline']),
                            gr.Textbox(value=entry_display['steered']),
                            gr.Markdown(value=entry_display['scores']),
                            entry_state_data,  # Update state
                            ""  # Clear save status
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Textbox(value=""),
                            gr.Textbox(value=""),
                            gr.Markdown(value=""),
                            {},  # Empty state
                            ""  # Empty save status
                        ])

                return updates

            outputs_list = [bot_k_summary]
            for entry in bot_k_outputs:
                outputs_list.extend([entry['col'], entry['header'], entry['baseline'], entry['steered'], entry['scores'], entry['state'], entry['save_status']])

            bot_k_btn.click(
                fn=update_bottom_k,
                inputs=[bot_file_dropdown, bot_k_slider],
                outputs=outputs_list
            )

        # Tab 4: Top Statistics
        with gr.Tab("Top Configurations"):
            gr.Markdown("""
            **For a given trait and prompt format view top k tuples of (layer, strength, aggregation) with the highest average overall score across all questions.**
            """)
            with gr.Row():
                stats_file_dropdown = gr.Dropdown(
                    choices=STATS_FILE_CHOICES,
                    label="Select Stats File",
                    value=STATS_FILE_CHOICES[0]
                )
                stats_k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Configurations (K)")

            stats_k_btn = gr.Button("Load Top K Configurations")
            stats_k_display = gr.Markdown(label="Top K Configurations")

            stats_k_btn.click(
                fn=display_top_stats,
                inputs=[stats_file_dropdown, stats_k_slider],
                outputs=[stats_k_display]
            )

        # Tab 5: Filter by Configuration
        with gr.Tab("Filter by Configuration"):
            gr.Markdown("""
            **For a given trait and prompt format view specific tuple (layers, strength, aggregation).**
            """)
            with gr.Row():
                filter_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File",
                    value=SCORES_FILE_CHOICES[0]
                )

            with gr.Row():
                layer_dropdown = gr.Dropdown(
                    choices=happy_layers,
                    label="Layer",
                    value=happy_layers[0] if happy_layers else None
                )
                strength_dropdown = gr.Dropdown(
                    choices=happy_strengths,
                    label="Strength",
                    value=happy_strengths[0] if happy_strengths else None
                )
                aggregation_dropdown = gr.Dropdown(
                    choices=happy_aggregations,
                    label="Aggregation",
                    value=happy_aggregations[0] if happy_aggregations else None
                )
                steering_dropdown = gr.Dropdown(
                    choices=happy_steerings,
                    label="Steering",
                    value=happy_steerings[0] if happy_steerings else None
                )

            filter_btn = gr.Button("Filter Entries")

            # Pre-create slots for num_questions * 2 entries (exact number for filter results)
            filter_summary = gr.Markdown("")
            filter_outputs = []
            max_filter_entries = num_questions * 2

            for i in range(max_filter_entries):
                with gr.Column(visible=False) as entry_col:
                    entry_header = gr.Markdown("")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**Baseline Response**")
                            baseline_text = gr.Textbox(label="", lines=10, max_lines=15, interactive=False)
                        with gr.Column():
                            gr.Markdown("**Steered Response**")
                            steered_text = gr.Textbox(label="", lines=10, max_lines=15, interactive=False)

                            # Create state and save button for this entry
                            entry_state = gr.State(value={})
                            save_btn, save_status = create_save_favorite_handler(entry_state)

                    scores_md = gr.Markdown("")
                    gr.Markdown("---")

                    filter_outputs.append({
                        'col': entry_col,
                        'header': entry_header,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'scores': scores_md,
                        'state': entry_state,
                        'save_status': save_status
                    })

            def update_filtered(file_path, layer, strength, aggregation, steering):
                """Update display with filtered entries."""
                summary, entries_data = display_filtered_entries(file_path, layer, strength, aggregation, steering)

                updates = [gr.Markdown(value=summary)]

                for i in range(max_filter_entries):
                    if i < len(entries_data):
                        entry_display = entries_data[i]
                        raw_entry = entry_display.get('raw_entry', {})

                        # Extract entry data for state
                        entry_state_data = {
                            'question': raw_entry.get('question', ''),
                            'steered_response': raw_entry.get('steered_response', ''),
                            'layer': raw_entry.get('layer'),
                            'strength': raw_entry.get('strength'),
                            'aggregation': raw_entry.get('aggregation_method'),
                            'steering': raw_entry.get('steering'),
                            'overall_score': raw_entry.get('overall_score'),
                            'differentiation_score': raw_entry.get('differentiation_score'),
                            'coherence_score': raw_entry.get('coherence_score'),
                            'trait_alignment_score': raw_entry.get('trait_alignment_score')
                        }

                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=entry_display['header']),
                            gr.Textbox(value=entry_display['baseline']),
                            gr.Textbox(value=entry_display['steered']),
                            gr.Markdown(value=entry_display['scores']),
                            entry_state_data,  # Update state
                            ""  # Clear save status
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Textbox(value=""),
                            gr.Textbox(value=""),
                            gr.Markdown(value=""),
                            {},  # Empty state
                            ""  # Empty save status
                        ])

                return updates

            # Update dropdowns when file changes
            def update_dropdowns(file_path):
                layers, strengths, aggregations, steerings = get_config_for_file(file_path)
                return (
                    gr.Dropdown(choices=layers, value=layers[0] if layers else None),
                    gr.Dropdown(choices=strengths, value=strengths[0] if strengths else None),
                    gr.Dropdown(choices=aggregations, value=aggregations[0] if aggregations else None),
                    gr.Dropdown(choices=steerings, value=steerings[0] if steerings else None)
                )

            filter_file_dropdown.change(
                fn=update_dropdowns,
                inputs=[filter_file_dropdown],
                outputs=[layer_dropdown, strength_dropdown, aggregation_dropdown, steering_dropdown]
            )

            outputs_list = [filter_summary]
            for entry in filter_outputs:
                outputs_list.extend([entry['col'], entry['header'], entry['baseline'], entry['steered'], entry['scores'], entry['state'], entry['save_status']])

            filter_btn.click(
                fn=update_filtered,
                inputs=[filter_file_dropdown, layer_dropdown, strength_dropdown, aggregation_dropdown, steering_dropdown],
                outputs=outputs_list
            )

        # Tab 6: Blind Test Game
        with gr.Tab("Blind A/B Test Game"):
            gr.Markdown("""
            ## Can you identify which response is steered?

            **Test your ability to distinguish baseline from steered responses.**

            Play 10 rounds! In each round, you'll see two responses (A and B) to the same question.
            One is from the baseline model, the other is steered. Try to guess which one is steered.
            You get 1 point for each correct guess. This helps evaluate how detectable the steering is.
            """)

            with gr.Row():
                blind_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File",
                    value=SCORES_FILE_CHOICES[0]
                )

            start_game_btn = gr.Button("Start Game", variant="primary")

            blind_progress = gr.Markdown("**Score:** 0/0")
            blind_question = gr.Markdown("")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Response A")
                    left_response = gr.Textbox(label="", lines=15, max_lines=20, interactive=False)

                with gr.Column():
                    gr.Markdown("### Response B")
                    right_response = gr.Textbox(label="", lines=15, max_lines=20, interactive=False)

            with gr.Row():
                guess_a_btn = gr.Button("Response A is Steered", visible=False)
                guess_b_btn = gr.Button("Response B is Steered", visible=False)

            blind_result = gr.Markdown("")

            # Per-user game state
            game_state = gr.State(value=get_initial_game_state())

            # Start game
            start_game_btn.click(
                fn=start_blind_game,
                inputs=[blind_file_dropdown, game_state],
                outputs=[blind_question, left_response, right_response, blind_result, blind_progress, start_game_btn, guess_a_btn, guess_b_btn, game_state]
            )

            # Check guesses - pass current file path to allow switching mid-game
            guess_a_btn.click(
                fn=lambda file_path, state: check_guess("A", file_path, state),
                inputs=[blind_file_dropdown, game_state],
                outputs=[blind_result, blind_question, left_response, right_response, blind_progress, start_game_btn, guess_a_btn, guess_b_btn, game_state]
            )

            guess_b_btn.click(
                fn=lambda file_path, state: check_guess("B", file_path, state),
                inputs=[blind_file_dropdown, game_state],
                outputs=[blind_result, blind_question, left_response, right_response, blind_progress, start_game_btn, guess_a_btn, guess_b_btn, game_state]
            )

        # Tab 7: Statistical Analysis
        with gr.Tab("General Statistical Analysis"):
            gr.Markdown("""
            **Statistical analysis and visualization of evaluation results.**

            View distributions, summary statistics, and trends across different configurations
            and evaluation metrics.
            """)

            with gr.Row():
                stats_analysis_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File (Trait & Prompt Format)",
                    value=SCORES_FILE_CHOICES[0]
                )

            gr.Markdown("### Score Distributions")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Overall Score**")
                    overall_score_plot = gr.Plot(label="")

                with gr.Column():
                    gr.Markdown("**Differentiation Score**")
                    differentiation_plot = gr.Plot(label="")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Coherence Score**")
                    coherence_plot = gr.Plot(label="")

                with gr.Column():
                    gr.Markdown("**Trait Alignment Score**")
                    trait_alignment_plot = gr.Plot(label="")

            # Function to update all histograms
            def update_all_histograms(file_path):
                return (
                    plot_score_histogram(file_path, 'overall_score'),
                    plot_score_histogram(file_path, 'differentiation_score'),
                    plot_score_histogram(file_path, 'coherence_score'),
                    plot_score_histogram(file_path, 'trait_alignment_score')
                )

            # Update histograms when file changes
            stats_analysis_file_dropdown.change(
                fn=update_all_histograms,
                inputs=[stats_analysis_file_dropdown],
                outputs=[overall_score_plot, differentiation_plot, coherence_plot, trait_alignment_plot]
            )

            # Load initial histograms
            demo.load(
                fn=lambda: update_all_histograms(SCORES_FILE_CHOICES[0]),
                outputs=[overall_score_plot, differentiation_plot, coherence_plot, trait_alignment_plot]
            )

            gr.Markdown("---")
            gr.Markdown("### Pairwise Score Correlations")
            gr.Markdown("Scatter plots showing relationships between different score types with correlation coefficients and trend lines.")

            correlation_plot = gr.Plot(label="Correlation Analysis")

            # Update correlation plot when file changes
            stats_analysis_file_dropdown.change(
                fn=plot_score_correlations,
                inputs=[stats_analysis_file_dropdown],
                outputs=[correlation_plot]
            )

            # Load initial correlation plot
            demo.load(
                fn=lambda: plot_score_correlations(SCORES_FILE_CHOICES[0]),
                outputs=[correlation_plot]
            )

        # Tab 8: Configuration Analysis
        with gr.Tab("Configuration Statistical Analysis"):
            gr.Markdown("""
            **Analyze score distributions for specific steering configurations.**

            Select a trait, prompt format, and specific configuration (layer, strength, aggregation)
            to view the distribution of scores across all questions for that configuration.
            """)

            with gr.Row():
                config_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File (Trait & Prompt Format)",
                    value=SCORES_FILE_CHOICES[0]
                )

            with gr.Row():
                config_layer_dropdown = gr.Dropdown(
                    choices=happy_layers,
                    label="Layer",
                    value=happy_layers[0] if happy_layers else None
                )
                config_strength_dropdown = gr.Dropdown(
                    choices=happy_strengths,
                    label="Strength",
                    value=happy_strengths[0] if happy_strengths else None
                )
                config_aggregation_dropdown = gr.Dropdown(
                    choices=happy_aggregations,
                    label="Aggregation",
                    value=happy_aggregations[0] if happy_aggregations else None
                )
                config_steering_dropdown = gr.Dropdown(
                    choices=happy_steerings,
                    label="Steering",
                    value=happy_steerings[0] if happy_steerings else None
                )

            gr.Markdown("### Score Distributions for Selected Configuration")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Overall Score**")
                    config_overall_plot = gr.Plot(label="")

                with gr.Column():
                    gr.Markdown("**Differentiation Score**")
                    config_diff_plot = gr.Plot(label="")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Coherence Score**")
                    config_coh_plot = gr.Plot(label="")

                with gr.Column():
                    gr.Markdown("**Trait Alignment Score**")
                    config_trait_plot = gr.Plot(label="")

            # Function to update all config histograms
            def update_config_histograms(file_path, layer, strength, aggregation, steering):
                return (
                    plot_config_score_histogram(file_path, layer, strength, aggregation, steering, 'overall_score'),
                    plot_config_score_histogram(file_path, layer, strength, aggregation, steering, 'differentiation_score'),
                    plot_config_score_histogram(file_path, layer, strength, aggregation, steering, 'coherence_score'),
                    plot_config_score_histogram(file_path, layer, strength, aggregation, steering, 'trait_alignment_score')
                )

            # Function to update config correlation plot
            def update_config_correlations(file_path, layer, strength, aggregation, steering):
                return plot_config_score_correlations(file_path, layer, strength, aggregation, steering)

            gr.Markdown("---")
            gr.Markdown("### Pairwise Score Correlations for Selected Configuration")
            gr.Markdown("Scatter plots showing relationships between different score types with correlation coefficients and trend lines.")

            config_correlation_plot = gr.Plot(label="Correlation Analysis")

            # Update dropdowns when file changes
            def update_config_dropdowns(file_path):
                layers, strengths, aggregations, steerings = get_config_for_file(file_path)
                return (
                    gr.Dropdown(choices=layers, value=layers[0] if layers else None),
                    gr.Dropdown(choices=strengths, value=strengths[0] if strengths else None),
                    gr.Dropdown(choices=aggregations, value=aggregations[0] if aggregations else None),
                    gr.Dropdown(choices=steerings, value=steerings[0] if steerings else None)
                )

            config_file_dropdown.change(
                fn=update_config_dropdowns,
                inputs=[config_file_dropdown],
                outputs=[config_layer_dropdown, config_strength_dropdown, config_aggregation_dropdown, config_steering_dropdown]
            )

            # Update plots when any parameter changes
            for component in [config_file_dropdown, config_layer_dropdown, config_strength_dropdown, config_aggregation_dropdown, config_steering_dropdown]:
                component.change(
                    fn=update_config_histograms,
                    inputs=[config_file_dropdown, config_layer_dropdown, config_strength_dropdown, config_aggregation_dropdown, config_steering_dropdown],
                    outputs=[config_overall_plot, config_diff_plot, config_coh_plot, config_trait_plot]
                )
                component.change(
                    fn=update_config_correlations,
                    inputs=[config_file_dropdown, config_layer_dropdown, config_strength_dropdown, config_aggregation_dropdown, config_steering_dropdown],
                    outputs=[config_correlation_plot]
                )

            # Load initial plots
            demo.load(
                fn=lambda: update_config_histograms(
                    SCORES_FILE_CHOICES[0],
                    happy_layers[0] if happy_layers else None,
                    happy_strengths[0] if happy_strengths else None,
                    happy_aggregations[0] if happy_aggregations else None,
                    happy_steerings[0] if happy_steerings else None
                ),
                outputs=[config_overall_plot, config_diff_plot, config_coh_plot, config_trait_plot]
            )

            # Load initial correlation plot
            demo.load(
                fn=lambda: update_config_correlations(
                    SCORES_FILE_CHOICES[0],
                    happy_layers[0] if happy_layers else None,
                    happy_strengths[0] if happy_strengths else None,
                    happy_aggregations[0] if happy_aggregations else None,
                    happy_steerings[0] if happy_steerings else None
                ),
                outputs=[config_correlation_plot]
            )

        # Tab 9: Question Rankings
        with gr.Tab("Question Rankings"):
            gr.Markdown("""
            **Rank questions by average score across all configurations.**

            Select a trait, prompt format, and score type to see which questions
            achieve the highest average scores across all steering configurations
            (all combinations of layer, strength, and aggregation).
            """)

            with gr.Row():
                rank_file_dropdown = gr.Dropdown(
                    choices=SCORES_FILE_CHOICES,
                    label="Select Scores File (Trait & Prompt Format)",
                    value=SCORES_FILE_CHOICES[0]
                )
                rank_score_dropdown = gr.Dropdown(
                    choices=[
                        ("Overall Score", "overall_score"),
                        ("Differentiation Score", "differentiation_score"),
                        ("Coherence Score", "coherence_score"),
                        ("Trait Alignment Score", "trait_alignment_score")
                    ],
                    label="Sort by Score Type",
                    value="overall_score"
                )

            rank_btn = gr.Button("Load Rankings")

            rank_output = gr.Markdown("")

            def display_question_rankings(file_path, score_type):
                """Display questions ranked by average score."""
                rankings = get_questions_ranked_by_score(file_path, score_type)

                if not rankings:
                    return "No data found."

                # Map score type to display name
                score_names = {
                    'overall_score': 'Overall Score',
                    'differentiation_score': 'Differentiation Score',
                    'coherence_score': 'Coherence Score',
                    'trait_alignment_score': 'Trait Alignment Score'
                }
                score_display = score_names.get(score_type, score_type)

                import os
                output = f"# Question Rankings by {score_display}\n\n"
                output += f"**File:** {os.path.basename(file_path)}\n\n"
                output += f"**Total Questions:** {len(rankings)}\n\n"
                output += "---\n\n"

                for idx, (question, avg_score, num_configs) in enumerate(rankings, 1):
                    output += f"## {idx}. {question}\n\n"
                    output += f"**Average {score_display}:** {avg_score:.2f}\n\n"
                    output += "---\n\n"

                return output

            rank_btn.click(
                fn=display_question_rankings,
                inputs=[rank_file_dropdown, rank_score_dropdown],
                outputs=[rank_output]
            )

        # Tab 10: View Favorite Answers
        with gr.Tab("Saved Favorites"):
            gr.Markdown("""
            **View your saved favorite steered answers.**

            Browse other tabs and click "⭐ Save as Favorite" on answers you like.
            They will appear here.
            """)

            with gr.Row():
                fav_refresh_btn = gr.Button("🔄 Refresh Favorites", variant="primary")
                fav_clear_btn = gr.Button("🗑️ Clear All Favorites", variant="stop")

            fav_count = gr.Markdown("")

            # Container for favorite entries (max 20 for performance)
            MAX_FAV_DISPLAY = 20
            fav_entries = []

            for i in range(MAX_FAV_DISPLAY):
                with gr.Column(visible=False) as fav_col:
                    with gr.Row():
                        fav_header = gr.Markdown("")
                        fav_delete_btn = gr.Button("🗑️ Delete", variant="stop", scale=0)
                    fav_content = gr.Markdown("")
                    gr.Markdown("---")

                    fav_entries.append({
                        'col': fav_col,
                        'header': fav_header,
                        'content': fav_content,
                        'delete_btn': fav_delete_btn
                    })

            def display_favorites():
                """Load and display all favorites."""
                favorites = load_favorites("tests/EVAL/output/favorites.json")

                if not favorites:
                    updates = [gr.Markdown(value="No favorites saved yet. Browse entries in other tabs and click '⭐ Save as Favorite' to add them here.")]
                    for i in range(MAX_FAV_DISPLAY):
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Markdown(value="")
                        ])
                    return updates

                # Reverse to show most recent first
                favorites_reversed = list(reversed(favorites))
                count_text = f"**Total Favorites:** {len(favorites)}"

                updates = [gr.Markdown(value=count_text)]

                for i in range(MAX_FAV_DISPLAY):
                    if i < len(favorites_reversed):
                        fav = favorites_reversed[i]
                        # Calculate original index (for deletion)
                        original_idx = len(favorites) - 1 - i

                        header = f"## {i+1}. {fav['question']}"
                        content = f"**Configuration:** Layer {fav['layer']}, Strength {fav['strength']}, Aggregation {fav['aggregation']}, Steering {fav['steering']}\n\n"

                        # Add scores if available
                        scores_available = any([
                            fav.get('overall_score') is not None,
                            fav.get('differentiation_score') is not None,
                            fav.get('coherence_score') is not None,
                            fav.get('trait_alignment_score') is not None
                        ])

                        if scores_available:
                            content += "**Scores:**\n"
                            if fav.get('overall_score') is not None:
                                content += f"- Overall: {fav['overall_score']:.2f}\n"
                            if fav.get('differentiation_score') is not None:
                                content += f"- Differentiation: {fav['differentiation_score']:.2f}\n"
                            if fav.get('coherence_score') is not None:
                                content += f"- Coherence: {fav['coherence_score']:.2f}\n"
                            if fav.get('trait_alignment_score') is not None:
                                content += f"- Trait Alignment: {fav['trait_alignment_score']:.2f}\n"
                            content += "\n"

                        content += f"**Steered Response:**\n\n{fav['steered_response']}\n\n"
                        if fav.get('notes'):
                            content += f"**Notes:** {fav['notes']}\n\n"
                        content += f"*Saved: {fav.get('timestamp', 'Unknown')}*"

                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=header),
                            gr.Markdown(value=content)
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Markdown(value="")
                        ])

                return updates

            def clear_all_favorites():
                """Delete all saved favorites."""
                import os
                favorites_file = "tests/EVAL/output/favorites.json"

                try:
                    if os.path.exists(favorites_file):
                        os.remove(favorites_file)
                except:
                    pass

                return display_favorites()

            def delete_single_favorite(btn_index):
                """Delete a single favorite by button index."""
                favorites = load_favorites("tests/EVAL/output/favorites.json")
                if not favorites:
                    return display_favorites()

                # btn_index corresponds to reversed display order
                # Calculate original index
                original_idx = len(favorites) - 1 - btn_index

                delete_favorite_by_index(original_idx)
                return display_favorites()

            # Wire up refresh button
            outputs_list = [fav_count]
            for entry in fav_entries:
                outputs_list.extend([entry['col'], entry['header'], entry['content']])

            fav_refresh_btn.click(
                fn=display_favorites,
                outputs=outputs_list
            )

            fav_clear_btn.click(
                fn=clear_all_favorites,
                outputs=outputs_list
            )

            # Wire up individual delete buttons
            for i, entry in enumerate(fav_entries):
                entry['delete_btn'].click(
                    fn=lambda idx=i: delete_single_favorite(idx),
                    outputs=outputs_list
                )

            # Load on tab open
            demo.load(
                fn=display_favorites,
                outputs=outputs_list
            )


if __name__ == "__main__":
    demo.launch(share=True)
