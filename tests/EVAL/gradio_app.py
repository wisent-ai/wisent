import gradio as gr
from utils.gradio import (
    # Config and constants
    happy_layers, happy_strengths, happy_aggregations,
    SCORES_FILE_CHOICES, STATS_FILE_CHOICES, num_questions,
    # Functions
    get_config_for_file, browse_entries, get_random_entry_display,
    display_top_k, display_bottom_k, display_top_stats, display_filtered_entries,
    start_blind_game, check_guess, get_initial_game_state
)


# Create the Gradio interface
with gr.Blocks(title="LLM Steering Evaluation Results") as demo:
    gr.Markdown("# LLM Steering Evaluation Results Browser")
    gr.Markdown("Browse evaluation results from different steering configurations.")

    # Tabs for different views
    with gr.Tabs():
        # Tab 1: Browse Individual Entries
        with gr.Tab("Browse Entries"):
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

            # Scores below
            entry_scores = gr.Markdown(label="Scores & Evaluation")

            # Button actions
            def go_previous(file_path, current_index):
                new_index = max(0, current_index - 1)
                return browse_entries(file_path, new_index)

            def go_next(file_path, current_index):
                new_index = current_index + 1
                return browse_entries(file_path, new_index)

            prev_btn.click(
                fn=go_previous,
                inputs=[file_dropdown, entry_index],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries]
            )

            next_btn.click(
                fn=go_next,
                inputs=[file_dropdown, entry_index],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries]
            )

            random_btn.click(
                fn=get_random_entry_display,
                inputs=[file_dropdown],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries]
            )

            load_btn.click(
                fn=browse_entries,
                inputs=[file_dropdown, entry_index],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries]
            )

            # Load initial entry when file changes
            file_dropdown.change(
                fn=lambda file_path: browse_entries(file_path, 0),
                inputs=[file_dropdown],
                outputs=[entry_config, baseline_display, steered_display, entry_scores, entry_index, total_entries]
            )

        # Tab 2: Top K Entries
        with gr.Tab("Top K Entries"):
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
                    scores_md = gr.Markdown("")
                    gr.Markdown("---")

                    top_k_outputs.append({
                        'col': entry_col,
                        'header': entry_header,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'scores': scores_md
                    })

            def update_top_k(file_path, k):
                """Update display with top k entries."""
                summary, entries_data = display_top_k(file_path, k)

                updates = [gr.Markdown(value=summary)]

                for i in range(10):
                    if i < len(entries_data):
                        entry = entries_data[i]
                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=entry['header']),
                            gr.Textbox(value=entry['baseline']),
                            gr.Textbox(value=entry['steered']),
                            gr.Markdown(value=entry['scores'])
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Textbox(value=""),
                            gr.Textbox(value=""),
                            gr.Markdown(value="")
                        ])

                return updates

            outputs_list = [top_k_summary]
            for entry in top_k_outputs:
                outputs_list.extend([entry['col'], entry['header'], entry['baseline'], entry['steered'], entry['scores']])

            top_k_btn.click(
                fn=update_top_k,
                inputs=[top_file_dropdown, top_k_slider],
                outputs=outputs_list
            )

        # Tab 3: Bottom K Entries
        with gr.Tab("Bottom K Entries"):
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
                    scores_md = gr.Markdown("")
                    gr.Markdown("---")

                    bot_k_outputs.append({
                        'col': entry_col,
                        'header': entry_header,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'scores': scores_md
                    })

            def update_bottom_k(file_path, k):
                """Update display with bottom k entries."""
                summary, entries_data = display_bottom_k(file_path, k)

                updates = [gr.Markdown(value=summary)]

                for i in range(10):
                    if i < len(entries_data):
                        entry = entries_data[i]
                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=entry['header']),
                            gr.Textbox(value=entry['baseline']),
                            gr.Textbox(value=entry['steered']),
                            gr.Markdown(value=entry['scores'])
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Textbox(value=""),
                            gr.Textbox(value=""),
                            gr.Markdown(value="")
                        ])

                return updates

            outputs_list = [bot_k_summary]
            for entry in bot_k_outputs:
                outputs_list.extend([entry['col'], entry['header'], entry['baseline'], entry['steered'], entry['scores']])

            bot_k_btn.click(
                fn=update_bottom_k,
                inputs=[bot_file_dropdown, bot_k_slider],
                outputs=outputs_list
            )

        # Tab 4: Top Statistics
        with gr.Tab("Top Configurations"):
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
                    scores_md = gr.Markdown("")
                    gr.Markdown("---")

                    filter_outputs.append({
                        'col': entry_col,
                        'header': entry_header,
                        'baseline': baseline_text,
                        'steered': steered_text,
                        'scores': scores_md
                    })

            def update_filtered(file_path, layer, strength, aggregation):
                """Update display with filtered entries."""
                summary, entries_data = display_filtered_entries(file_path, layer, strength, aggregation)

                updates = [gr.Markdown(value=summary)]

                for i in range(max_filter_entries):
                    if i < len(entries_data):
                        entry = entries_data[i]
                        updates.extend([
                            gr.Column(visible=True),
                            gr.Markdown(value=entry['header']),
                            gr.Textbox(value=entry['baseline']),
                            gr.Textbox(value=entry['steered']),
                            gr.Markdown(value=entry['scores'])
                        ])
                    else:
                        updates.extend([
                            gr.Column(visible=False),
                            gr.Markdown(value=""),
                            gr.Textbox(value=""),
                            gr.Textbox(value=""),
                            gr.Markdown(value="")
                        ])

                return updates

            # Update dropdowns when file changes
            def update_dropdowns(file_path):
                layers, strengths, aggregations = get_config_for_file(file_path)
                return (
                    gr.Dropdown(choices=layers, value=layers[0] if layers else None),
                    gr.Dropdown(choices=strengths, value=strengths[0] if strengths else None),
                    gr.Dropdown(choices=aggregations, value=aggregations[0] if aggregations else None)
                )

            filter_file_dropdown.change(
                fn=update_dropdowns,
                inputs=[filter_file_dropdown],
                outputs=[layer_dropdown, strength_dropdown, aggregation_dropdown]
            )

            outputs_list = [filter_summary]
            for entry in filter_outputs:
                outputs_list.extend([entry['col'], entry['header'], entry['baseline'], entry['steered'], entry['scores']])

            filter_btn.click(
                fn=update_filtered,
                inputs=[filter_file_dropdown, layer_dropdown, strength_dropdown, aggregation_dropdown],
                outputs=outputs_list
            )

        # Tab 6: Blind Test Game
        with gr.Tab("Blind A/B Test Game"):
            gr.Markdown("## Can you identify which response is steered?")
            gr.Markdown("Play 10 rounds! Try to guess which response is from the steered model. You get 1 point for each correct guess.")

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


if __name__ == "__main__":
    demo.launch(share=True)
