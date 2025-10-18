"""Refactored Gradio app for EVALOOP evaluation results."""
import gradio as gr
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.EVALOOP.core.config import ConfigManager
from tests.EVALOOP.core.data_manager import DataManager
from tests.EVALOOP.ui.components.favorites import FavoritesManager
from tests.EVALOOP.ui.components.displays import EntryDisplay, MultiEntryDisplay, StatisticsDisplay
from tests.EVALOOP.ui.components.controls import FileSelector, ConfigDropdowns, NavigationButtons, FavoriteButton
from tests.EVALOOP.ui.components.plots import PlotGenerator
from tests.EVALOOP.ui.components.game import BlindTestGame


# Initialize managers
config_manager = ConfigManager()
data_manager = DataManager(config_manager)
favorites_manager = FavoritesManager()
plot_generator = PlotGenerator()


def create_browse_tab():
    """Create the Browse tab."""
    gr.Markdown("""
    **For a given trait and prompt format browse individual evaluation entries one at a time.**
    """)

    # File selector
    file_selector = FileSelector(data_manager, "scores")
    file_dropdown = file_selector.get_component()

    # Entry index input
    with gr.Row():
        entry_index = gr.Number(label="Entry Index (0-based)", value=0, precision=0)
        total_entries = gr.Number(label="Total Entries", value=0, interactive=False)

    # Navigation buttons
    nav_buttons = NavigationButtons()
    buttons = nav_buttons.get_components()

    # Entry display
    entry_display = EntryDisplay(show_save_button=True)

    # Wire up save button
    save_fn = FavoriteButton.create_save_function(favorites_manager)
    entry_display.get_save_button().click(
        fn=save_fn,
        inputs=[entry_display.get_state()],
        outputs=[entry_display.components['save_status']]
    )

    # Event handlers
    def browse_entry(file_path, index):
        """Browse to a specific entry."""
        entry = data_manager.get_entry(file_path, index)
        total = data_manager.get_entry_count(file_path)

        if entry:
            config, baseline, steered, scores, explanations, state = entry_display.format_entry(entry)
            return config, baseline, steered, scores, explanations, state, "", index, total
        return "No entry found", "", "", "", "", {}, "", index, total

    def go_previous(file_path, current_index):
        new_index = max(0, current_index - 1)
        return browse_entry(file_path, new_index)

    def go_next(file_path, current_index):
        return browse_entry(file_path, current_index + 1)

    def go_random(file_path):
        entry, index = data_manager.get_random_entry(file_path)
        total = data_manager.get_entry_count(file_path)

        if entry:
            config, baseline, steered, scores, explanations, state = entry_display.format_entry(entry)
            return config, baseline, steered, scores, explanations, state, "", index, total
        return "No entries found", "", "", "", "", {}, "", 0, total

    # Wire up buttons
    # get_output_components returns: [config, baseline, steered, scores, explanations, state, save_status]
    outputs = entry_display.get_output_components() + [entry_index, total_entries]

    buttons['prev'].click(go_previous, [file_dropdown, entry_index], outputs)
    buttons['next'].click(go_next, [file_dropdown, entry_index], outputs)
    buttons['random'].click(go_random, [file_dropdown], outputs)
    buttons['load'].click(browse_entry, [file_dropdown, entry_index], outputs)
    file_dropdown.change(lambda fp: browse_entry(fp, 0), [file_dropdown], outputs)


def create_topk_tab():
    """Create the Top K tab."""
    gr.Markdown("""
    **For a given trait and prompt format view the top K entries with the highest overall score.**
    """)

    with gr.Row():
        file_selector = FileSelector(data_manager, "scores")
        file_dropdown = file_selector.get_component()
        k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Entries (K)")

    load_btn = gr.Button("Load Top K Entries")

    # Multi-entry display
    multi_display = MultiEntryDisplay(max_entries=10, show_save_buttons=True)

    def load_top_k(file_path, k):
        entries = data_manager.get_top_k(file_path, k)
        summary = f"# Top {k} Entries (Highest Overall Scores)\n\nTotal entries: {data_manager.get_entry_count(file_path)}"
        return multi_display.format_entries(entries, summary)

    load_btn.click(load_top_k, [file_dropdown, k_slider], multi_display.get_output_components())

    # Wire up save buttons
    save_fn = FavoriteButton.create_save_function(favorites_manager)
    multi_display.wire_save_buttons(save_fn)


def create_bottomk_tab():
    """Create the Bottom K tab."""
    gr.Markdown("""
    **For a given trait and prompt format view the bottom K entries with the lowest overall score.**
    """)

    with gr.Row():
        file_selector = FileSelector(data_manager, "scores")
        file_dropdown = file_selector.get_component()
        k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Entries (K)")

    load_btn = gr.Button("Load Bottom K Entries")

    # Multi-entry display
    multi_display = MultiEntryDisplay(max_entries=10, show_save_buttons=True)

    # Wire up save buttons
    save_fn = FavoriteButton.create_save_function(favorites_manager)
    multi_display.wire_save_buttons(save_fn)

    def load_bottom_k(file_path, k):
        entries = data_manager.get_bottom_k(file_path, k)
        summary = f"# Bottom {k} Entries (Lowest Overall Scores)\n\nTotal entries: {data_manager.get_entry_count(file_path)}"
        return multi_display.format_entries(entries, summary)

    load_btn.click(load_bottom_k, [file_dropdown, k_slider], multi_display.get_output_components())


def create_stats_tab():
    """Create the Top Configurations tab."""
    gr.Markdown("""
    **For a given trait and prompt format view top k tuples of (layer, strength, aggregation) with the highest average overall score across all questions.**
    """)

    with gr.Row():
        file_selector = FileSelector(data_manager, "stats")
        file_dropdown = file_selector.get_component()
        k_slider = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of Configurations (K)")

    load_btn = gr.Button("Load Top K Configurations")
    stats_display = gr.Markdown(label="Top K Configurations")

    def load_stats(file_path, k):
        configs = data_manager.load_stats(file_path)[:k]
        return StatisticsDisplay.format_stats(configs, k)

    load_btn.click(load_stats, [file_dropdown, k_slider], [stats_display])


def create_filter_tab():
    """Create the Filter by Configuration tab."""
    gr.Markdown("""
    **For a given trait and prompt format view specific tuple (layers, strength, aggregation).**
    """)

    file_selector = FileSelector(data_manager, "scores")
    file_dropdown = file_selector.get_component()

    # Get initial file for dropdowns
    files = data_manager.get_available_files()
    initial_file = files['scores'][0] if files['scores'] else None

    # Config dropdowns
    config_dropdowns = ConfigDropdowns(data_manager, initial_file)
    dropdown_components = config_dropdowns.get_values()

    filter_btn = gr.Button("Filter Entries")

    # Multi-entry display
    multi_display = MultiEntryDisplay(max_entries=20, show_save_buttons=True)

    # Wire up save buttons
    save_fn = FavoriteButton.create_save_function(favorites_manager)
    multi_display.wire_save_buttons(save_fn)

    def filter_entries(file_path, layer, strength, aggregation):
        entries = data_manager.filter_by_config(file_path, layer, strength, aggregation)
        summary = f"# Filtered Entries\n\nFilter: Layer={layer}, Strength={strength}, Aggregation={aggregation}\n\nTotal matching: {len(entries)}"
        return multi_display.format_entries(entries, summary)

    # Update dropdowns when file changes
    file_dropdown.change(
        config_dropdowns.update_from_file,
        [file_dropdown],
        dropdown_components
    )

    filter_btn.click(
        filter_entries,
        [file_dropdown] + dropdown_components,
        multi_display.get_output_components()
    )


def create_analysis_tab():
    """Create the Statistical Analysis tab."""
    gr.Markdown("""
    **Statistical analysis and visualization of evaluation results.**
    """)

    file_selector = FileSelector(data_manager, "scores")
    file_dropdown = file_selector.get_component()

    gr.Markdown("### Score Distributions")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Overall Score**")
            overall_plot = gr.Plot()
        with gr.Column():
            gr.Markdown("**Differentiation Score**")
            diff_plot = gr.Plot()

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Coherence Score**")
            coh_plot = gr.Plot()
        with gr.Column():
            gr.Markdown("**Trait Alignment Score**")
            trait_plot = gr.Plot()

    gr.Markdown("### Pairwise Score Correlations")
    corr_plot = gr.Plot()

    def update_plots(file_path):
        return (
            plot_generator.plot_histogram(file_path, 'overall_score'),
            plot_generator.plot_histogram(file_path, 'differentiation_score'),
            plot_generator.plot_histogram(file_path, 'coherence_score'),
            plot_generator.plot_histogram(file_path, 'trait_alignment_score'),
            plot_generator.plot_correlations(file_path)
        )

    file_dropdown.change(
        update_plots,
        [file_dropdown],
        [overall_plot, diff_plot, coh_plot, trait_plot, corr_plot]
    )


def create_blind_game_tab():
    """Create the Blind A/B Test Game tab."""
    game = BlindTestGame(data_manager, total_rounds=10)
    game.wire_events()


def create_config_analysis_tab():
    """Create the Configuration Statistical Analysis tab."""
    gr.Markdown("""
    **Analyze score distributions for specific steering configurations.**

    Select a trait, prompt format, and specific configuration (layer, strength, aggregation)
    to view the distribution of scores across all questions for that configuration.
    """)

    file_selector = FileSelector(data_manager, "scores")
    file_dropdown = file_selector.get_component()

    # Get initial file for dropdowns
    files = data_manager.get_available_files()
    initial_file = files['scores'][0] if files['scores'] else None

    # Config dropdowns
    config_dropdowns = ConfigDropdowns(data_manager, initial_file)
    dropdown_components = config_dropdowns.get_values()

    gr.Markdown("### Score Distributions for Selected Configuration")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Overall Score**")
            overall_plot = gr.Plot()
        with gr.Column():
            gr.Markdown("**Differentiation Score**")
            diff_plot = gr.Plot()

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Coherence Score**")
            coh_plot = gr.Plot()
        with gr.Column():
            gr.Markdown("**Trait Alignment Score**")
            trait_plot = gr.Plot()

    gr.Markdown("### Pairwise Score Correlations for Selected Configuration")
    corr_plot = gr.Plot()

    def update_config_plots(file_path, layer, strength, aggregation):
        return (
            plot_generator.plot_config_histogram(file_path, layer, strength, aggregation, 'overall_score'),
            plot_generator.plot_config_histogram(file_path, layer, strength, aggregation, 'differentiation_score'),
            plot_generator.plot_config_histogram(file_path, layer, strength, aggregation, 'coherence_score'),
            plot_generator.plot_config_histogram(file_path, layer, strength, aggregation, 'trait_alignment_score'),
            plot_generator.plot_config_correlations(file_path, layer, strength, aggregation)
        )

    # Update dropdowns when file changes
    file_dropdown.change(
        config_dropdowns.update_from_file,
        [file_dropdown],
        dropdown_components
    )

    # Update plots when any parameter changes
    all_inputs = [file_dropdown] + dropdown_components
    all_plots = [overall_plot, diff_plot, coh_plot, trait_plot, corr_plot]

    for component in all_inputs:
        component.change(
            update_config_plots,
            all_inputs,
            all_plots
        )


def create_question_rankings_tab():
    """Create the Question Rankings tab."""
    gr.Markdown("""
    **Rank questions by average score across all configurations.**

    Select a trait, prompt format, and score type to see which questions
    achieve the highest average scores across all steering configurations.
    """)

    with gr.Row():
        file_selector = FileSelector(data_manager, "scores")
        file_dropdown = file_selector.get_component()
        score_dropdown = gr.Dropdown(
            choices=[
                ("Overall Score", "overall_score"),
                ("Differentiation Score", "differentiation_score"),
                ("Coherence Score", "coherence_score"),
                ("Trait Alignment Score", "trait_alignment_score")
            ],
            label="Sort by Score Type",
            value="overall_score"
        )

    load_btn = gr.Button("Load Rankings")
    rankings_display = gr.Markdown("")

    def display_rankings(file_path, score_type):
        """Display questions ranked by average score."""
        rankings = data_manager.get_questions_ranked_by_score(file_path, score_type)

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
            output += f"**Average {score_display}:** {avg_score:.2f} (across {num_configs} configurations)\n\n"
            output += "---\n\n"

        return output

    load_btn.click(
        display_rankings,
        [file_dropdown, score_dropdown],
        [rankings_display]
    )


def create_interactive_eval_tab():
    """Create the Interactive Evaluation tab."""
    gr.Markdown("""
    ## Interactive Evaluation

    **Evaluate a single custom question in real-time.**

    Choose between two modes:
    1. **Generate Mode**: Automatically generate baseline and steered responses
    2. **Manual Mode**: Provide your own baseline and steered responses
    """)

    # Mode selection
    mode = gr.Radio(
        choices=[("Generate Responses (automatic)", "generate"), ("Provide Responses (manual)", "manual")],
        label="Evaluation Mode",
        value="generate"
    )

    # Common settings for both modes
    gr.Markdown("### Evaluation Settings")

    # Get trait choices from ConfigManager
    trait_choices = config_manager.list_traits()

    with gr.Row():
        trait_dropdown = gr.Dropdown(
            choices=trait_choices,
            label="Trait (determines evaluation prompts)",
            value=trait_choices[0] if trait_choices else None
        )

        format_dropdown = gr.Dropdown(
            choices=["txt", "markdown", "json"],
            label="Prompt Format",
            value="txt"
        )

    # Question input (common to both modes)
    question_input = gr.Textbox(
        label="Question",
        placeholder="Enter your question here...",
        lines=3
    )

    # Generate mode controls
    with gr.Column(visible=True) as generate_controls:
        gr.Markdown("### Generation Settings (layer, strength, aggregation)")

        # Get initial trait config for dropdowns
        initial_trait = trait_choices[0] if trait_choices else None
        if initial_trait:
            trait_config = config_manager.get_trait_config(initial_trait)
            layers = [str(l) for l in trait_config.layers]
            strengths = [float(s) for s in trait_config.strengths]
            aggregations = trait_config.aggregations
        else:
            layers, strengths, aggregations = [], [], []

        with gr.Row():
            layer_dropdown = gr.Dropdown(
                choices=layers,
                label="Layer",
                value=layers[0] if layers else None
            )
            strength_dropdown = gr.Dropdown(
                choices=strengths,
                label="Strength",
                value=strengths[0] if strengths else None
            )
            aggregation_dropdown = gr.Dropdown(
                choices=aggregations,
                label="Aggregation",
                value=aggregations[0] if aggregations else None
            )

    # Manual mode controls
    with gr.Column(visible=False) as manual_controls:
        gr.Markdown("### Provide Your Responses")
        baseline_input = gr.Textbox(
            label="Baseline Response",
            placeholder="Enter baseline response...",
            lines=5
        )
        steered_input = gr.Textbox(
            label="Steered Response",
            placeholder="Enter steered response...",
            lines=5
        )

    # Evaluate button
    eval_button = gr.Button("🔍 Evaluate", variant="primary", size="lg")

    # Results display
    gr.Markdown("### Evaluation Results")

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Baseline Response**")
            result_baseline = gr.Textbox(label="", lines=10, interactive=False)

        with gr.Column():
            gr.Markdown("**Steered Response**")
            result_steered = gr.Textbox(label="", lines=10, interactive=False)

    result_scores = gr.Markdown("")
    result_explanations = gr.Markdown("")
    result_status = gr.Markdown("")

    # Toggle visibility based on mode
    def toggle_mode(selected_mode):
        if selected_mode == "generate":
            return gr.Column(visible=True), gr.Column(visible=False)
        else:
            return gr.Column(visible=False), gr.Column(visible=True)

    mode.change(
        toggle_mode,
        inputs=[mode],
        outputs=[generate_controls, manual_controls]
    )

    # Update config dropdowns when trait changes
    def update_configs_for_trait(trait):
        """Update layer/strength/aggregation dropdowns based on selected trait."""
        try:
            # Get trait configuration from ConfigManager
            trait_config = config_manager.get_trait_config(trait)
            layers = [str(l) for l in trait_config.layers]
            strengths = [float(s) for s in trait_config.strengths]
            aggregations = trait_config.aggregations

            return (
                gr.Dropdown(choices=layers, value=layers[0] if layers else None),
                gr.Dropdown(choices=strengths, value=strengths[0] if strengths else None),
                gr.Dropdown(choices=aggregations, value=aggregations[0] if aggregations else None)
            )
        except Exception as e:
            print(f"Error loading config for trait {trait}: {e}")
            # Return empty dropdowns if error
            return (
                gr.Dropdown(choices=[], value=None),
                gr.Dropdown(choices=[], value=None),
                gr.Dropdown(choices=[], value=None)
            )

    trait_dropdown.change(
        update_configs_for_trait,
        inputs=[trait_dropdown],
        outputs=[layer_dropdown, strength_dropdown, aggregation_dropdown]
    )

    # Evaluation function
    def evaluate_question(
        eval_mode, question, trait, layer, strength, aggregation,
        baseline_manual, steered_manual, format_type
    ):
        """Evaluate a single question."""
        try:
            # Validate inputs
            if not question or not question.strip():
                return "", "", "", "⚠️ Please enter a question"

            if eval_mode == "generate":
                # Generate mode - use existing generation pipeline
                import torch

                # Get vector path from ConfigManager
                vector_dir = config_manager.get_vector_path(trait, layer, aggregation)
                vector_file = vector_dir / "steering_vectors.pt"

                if not vector_file.exists():
                    return "", "", "", f"❌ Steering vectors not found at {vector_file}. Please train vectors first using the EVALOOP pipeline."

                # Load vectors (PyTorch format)
                steering_vectors = torch.load(vector_file)

                # Initialize generator
                from tests.EVALOOP.core.generator import ResponseGenerator
                from wisent_guard.core.models.wisent_model import WisentModel

                model = WisentModel(
                    model_name=config_manager.config.model_name,
                    layers={},
                    device=config_manager.config.device
                )

                generator = ResponseGenerator(model, max_new_tokens=config_manager.config.max_new_tokens)

                # Generate responses
                gen_result = generator.generate_responses(
                    questions=[question],
                    steering_vectors=steering_vectors.to_dict() if hasattr(steering_vectors, 'to_dict') else steering_vectors,
                    strength=strength,
                    layer=layer,
                    aggregation=aggregation
                )[0]

                baseline_response = gen_result.baseline_response
                steered_response = gen_result.steered_response

                status_msg = "✅ Responses generated! Now evaluating..."

            else:
                # Manual mode - use provided responses
                if not baseline_manual or not steered_manual:
                    return "", "", "", "⚠️ Please provide both baseline and steered responses"

                baseline_response = baseline_manual
                steered_response = steered_manual
                status_msg = "🔄 Evaluating responses..."

            # Evaluate using existing evaluator
            from tests.EVALOOP.core.evaluator import Evaluator
            from tests.EVALOOP.core.models import GenerationResult

            # Get trait config from ConfigManager (has the instruction_prompts for evaluation)
            trait_cfg = config_manager.get_trait_config(trait)

            evaluator = Evaluator(config_manager.config, trait_cfg)

            # Create generation result
            gen_result = GenerationResult(
                layer=layer if eval_mode == "generate" else "manual",
                strength=strength if eval_mode == "generate" else 0.0,
                aggregation_method=aggregation if eval_mode == "generate" else "manual",
                question=question,
                baseline_response=baseline_response,
                steered_response=steered_response
            )

            # Evaluate
            eval_result = evaluator.evaluate_single(gen_result, format_type)

            # Format scores with proper conditionals
            overall_str = f"{eval_result.overall_score:.2f}" if eval_result.overall_score is not None else "N/A"
            diff_str = f"{eval_result.differentiation_score:.2f}" if eval_result.differentiation_score is not None else "N/A"
            coh_str = f"{eval_result.coherence_score:.2f}" if eval_result.coherence_score is not None else "N/A"
            trait_str = f"{eval_result.trait_alignment_score:.2f}" if eval_result.trait_alignment_score is not None else "N/A"
            choose_str = eval_result.choose_result if eval_result.choose_result else "N/A"
            open_str = eval_result.open_traits if eval_result.open_traits else "N/A"

            scores_md = f"""
## Evaluation Scores

**Overall Score:** {overall_str}

### Individual Metrics:
- **Differentiation Score:** {diff_str}
- **Coherence Score:** {coh_str}
- **Trait Alignment Score:** {trait_str}

### Qualitative Results:
- **Choose Result:** {choose_str}
- **Open Traits:** {open_str}

---

**Configuration:** Layer={eval_result.layer}, Strength={eval_result.strength}, Aggregation={eval_result.aggregation_method}
"""

            # Format explanations using the static method from EntryDisplay
            explanations_md = EntryDisplay.format_explanations(eval_result.explanations)

            return baseline_response, steered_response, scores_md, explanations_md, "✅ Evaluation complete!"

        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            return "", "", "", "", error_msg

    # Wire evaluation button
    eval_button.click(
        evaluate_question,
        inputs=[
            mode, question_input, trait_dropdown, layer_dropdown, strength_dropdown,
            aggregation_dropdown, baseline_input, steered_input, format_dropdown
        ],
        outputs=[result_baseline, result_steered, result_scores, result_explanations, result_status]
    )


def create_favorites_tab():
    """Create the Saved Favorites tab."""
    gr.Markdown("""
    **View your saved favorite steered answers.**

    Browse other tabs and click "⭐ Save as Favorite" on answers you like.
    """)

    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh Favorites", variant="primary")
        clear_btn = gr.Button("🗑️ Clear All Favorites", variant="stop")

    fav_display = gr.Markdown("")

    def display_favorites():
        favorites = favorites_manager.load()

        if not favorites:
            return "No favorites saved yet. Browse entries in other tabs and click '⭐ Save as Favorite'."

        # Reverse to show most recent first
        favorites = list(reversed(favorites))
        output = f"**Total Favorites:** {len(favorites)}\n\n---\n\n"

        for i, fav in enumerate(favorites, 1):
            output += f"## {i}. {fav['question']}\n\n"
            output += f"**Configuration:** Layer {fav['layer']}, Strength {fav['strength']}, Aggregation {fav['aggregation']}\n\n"

            if fav.get('overall_score') is not None:
                output += f"**Scores:** Overall: {fav['overall_score']:.2f}"
                if fav.get('differentiation_score') is not None:
                    output += f", Diff: {fav['differentiation_score']:.2f}"
                if fav.get('coherence_score') is not None:
                    output += f", Coh: {fav['coherence_score']:.2f}"
                if fav.get('trait_alignment_score') is not None:
                    output += f", Trait: {fav['trait_alignment_score']:.2f}"
                output += "\n\n"

            output += f"**Steered Response:**\n\n{fav['steered_response']}\n\n"
            output += f"*Saved: {fav.get('timestamp', 'Unknown')}*\n\n---\n\n"

        return output

    def clear_favorites():
        favorites_manager.clear()
        return display_favorites()

    refresh_btn.click(display_favorites, outputs=[fav_display])
    clear_btn.click(clear_favorites, outputs=[fav_display])


# Create the Gradio interface
with gr.Blocks(title="LLM Steering Evaluation Results - EVALOOP") as demo:
    gr.Markdown("# LLM Steering Evaluation Results Browser (EVALOOP)")
    gr.Markdown("Browse evaluation results from different steering configurations.")

    with gr.Tabs():
        with gr.Tab("Browse"):
            create_browse_tab()

        with gr.Tab("Top K"):
            create_topk_tab()

        with gr.Tab("Bottom K"):
            create_bottomk_tab()

        with gr.Tab("Top Configurations"):
            create_stats_tab()

        with gr.Tab("Filter by Configuration"):
            create_filter_tab()

        with gr.Tab("Blind A/B Test Game"):
            create_blind_game_tab()

        with gr.Tab("Statistical Analysis"):
            create_analysis_tab()

        with gr.Tab("Configuration Analysis"):
            create_config_analysis_tab()

        with gr.Tab("Question Rankings"):
            create_question_rankings_tab()

        with gr.Tab("Interactive Evaluation"):
            create_interactive_eval_tab()

        with gr.Tab("Saved Favorites"):
            create_favorites_tab()


if __name__ == "__main__":
    demo.launch(share=True)
