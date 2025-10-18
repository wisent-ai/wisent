"""Display components for evaluation results."""
import gradio as gr
from typing import Dict, Optional, Tuple


class EntryDisplay:
    """Displays a single evaluation entry with baseline and steered responses."""

    def __init__(self, show_save_button: bool = True):
        """
        Initialize EntryDisplay.

        Args:
            show_save_button: Whether to show the save favorite button
        """
        self.show_save_button = show_save_button
        self.components = {}
        self._create_components()

    @staticmethod
    def format_explanations(explanations_dict: Dict[str, str]) -> str:
        """
        Format explanations dictionary into markdown string.

        Args:
            explanations_dict: Dictionary mapping metric names to explanation strings

        Returns:
            Formatted markdown string with explanations
        """
        if not explanations_dict:
            return "No explanations available"

        explanations = "### Judge Explanations\n\n"
        if 'differentiation' in explanations_dict:
            explanations += f"**Differentiation:** {explanations_dict['differentiation']}\n\n"
        if 'coherence' in explanations_dict:
            explanations += f"**Coherence:** {explanations_dict['coherence']}\n\n"
        if 'trait_alignment' in explanations_dict:
            explanations += f"**Trait Alignment:** {explanations_dict['trait_alignment']}\n\n"
        if 'open' in explanations_dict:
            explanations += f"**Open Traits:** {explanations_dict['open']}\n\n"
        if 'choose' in explanations_dict:
            explanations += f"**Choose:** {explanations_dict['choose']}\n\n"

        return explanations

    def _create_components(self):
        """Create Gradio components for entry display."""
        # Configuration and question header
        self.components['config'] = gr.Markdown(label="Configuration & Question")

        # Side-by-side responses
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Baseline Response")
                self.components['baseline'] = gr.Textbox(
                    label="",
                    lines=15,
                    max_lines=25,
                    interactive=False
                )

            with gr.Column():
                gr.Markdown("### Steered Response")
                self.components['steered'] = gr.Textbox(
                    label="",
                    lines=15,
                    max_lines=25,
                    interactive=False
                )

                if self.show_save_button:
                    # State to track current entry
                    self.components['state'] = gr.State(value={})

                    # Save button
                    self.components['save_button'] = gr.Button(
                        "⭐ Save as Favorite",
                        variant="secondary"
                    )
                    self.components['save_status'] = gr.Markdown("")

        # Scores display
        self.components['scores'] = gr.Markdown(label="Scores & Evaluation")

        # Explanations display
        self.components['explanations'] = gr.Markdown(label="Judge Explanations")

    def format_entry(self, entry: Dict) -> Tuple[str, str, str, str, str, Dict]:
        """
        Format an entry for display.

        Args:
            entry: Entry dictionary

        Returns:
            Tuple of (config_text, baseline_text, steered_text, scores_text, explanations_text, state_dict)
        """
        # Configuration and question
        config = f"""
**Layer:** {entry.get('layer')} | **Strength:** {entry.get('strength')} | **Aggregation:** {entry.get('aggregation_method')}

**Question:** {entry.get('question', '')}
"""

        baseline = entry.get('baseline_response', '')
        steered = entry.get('steered_response', '')

        # Scores
        scores = f"""
**Scores:**

- Differentiation Score: {entry.get('differentiation_score', 'N/A')}
- Coherence Score: {entry.get('coherence_score', 'N/A')}
- Trait Alignment Score: {entry.get('trait_alignment_score', 'N/A')}
- Overall Score: {entry.get('overall_score', 'N/A')}

**Evaluation Results:**

- Choose Result: {entry.get('choose_result', 'N/A')}
- Open Traits: {entry.get('open_traits', 'N/A')}
"""

        # Explanations - use static method
        explanations_dict = entry.get('explanations', {})
        explanations = EntryDisplay.format_explanations(explanations_dict)

        # State for save button
        state = {
            'question': entry.get('question', ''),
            'steered_response': steered,
            'layer': entry.get('layer'),
            'strength': entry.get('strength'),
            'aggregation': entry.get('aggregation_method'),
            'overall_score': entry.get('overall_score'),
            'differentiation_score': entry.get('differentiation_score'),
            'coherence_score': entry.get('coherence_score'),
            'trait_alignment_score': entry.get('trait_alignment_score')
        }

        return config, baseline, steered, scores, explanations, state

    def get_output_components(self):
        """Get list of output components for Gradio event handlers."""
        outputs = [
            self.components['config'],
            self.components['baseline'],
            self.components['steered'],
            self.components['scores'],
            self.components['explanations']
        ]

        if self.show_save_button:
            outputs.extend([
                self.components['state'],
                self.components['save_status']
            ])

        return outputs

    def get_save_button(self):
        """Get the save button component."""
        return self.components.get('save_button')

    def get_state(self):
        """Get the state component."""
        return self.components.get('state')


class MultiEntryDisplay:
    """Displays multiple evaluation entries (for Top K, Bottom K, Filter views)."""

    def __init__(self, max_entries: int = 10, show_save_buttons: bool = True):
        """
        Initialize MultiEntryDisplay.

        Args:
            max_entries: Maximum number of entries to display
            show_save_buttons: Whether to show save favorite buttons
        """
        self.max_entries = max_entries
        self.show_save_buttons = show_save_buttons
        self.entry_widgets = []
        self._create_components()

    def _create_components(self):
        """Create Gradio components for multiple entries."""
        # Summary at top
        self.summary = gr.Markdown("")

        # Create slots for entries
        for i in range(self.max_entries):
            with gr.Column(visible=False) as entry_col:
                entry_header = gr.Markdown("")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Baseline Response**")
                        baseline_text = gr.Textbox(
                            label="",
                            lines=10,
                            max_lines=15,
                            interactive=False
                        )

                    with gr.Column():
                        gr.Markdown("**Steered Response**")
                        steered_text = gr.Textbox(
                            label="",
                            lines=10,
                            max_lines=15,
                            interactive=False
                        )

                        # Save button right below steered response
                        if self.show_save_buttons:
                            entry_state = gr.State(value={})
                            save_btn = gr.Button("⭐ Save as Favorite", variant="secondary", visible=False)
                            save_status = gr.Markdown("")

                scores_md = gr.Markdown("")
                explanations_md = gr.Markdown("")
                gr.Markdown("---")

                # Create widget dict
                widget_dict = {
                    'col': entry_col,
                    'header': entry_header,
                    'baseline': baseline_text,
                    'steered': steered_text,
                    'scores': scores_md,
                    'explanations': explanations_md
                }

                if self.show_save_buttons:
                    widget_dict['state'] = entry_state
                    widget_dict['save_btn'] = save_btn
                    widget_dict['save_status'] = save_status

                self.entry_widgets.append(widget_dict)

    def format_entries(self, entries: list, summary_text: str) -> list:
        """
        Format entries for display.

        Args:
            entries: List of entry dictionaries
            summary_text: Summary text to show at top

        Returns:
            List of updates for all components
        """
        updates = [gr.Markdown(value=summary_text)]

        for i in range(self.max_entries):
            if i < len(entries):
                entry = entries[i]

                # Format entry
                header = f"## Entry {i+1}\n\n"
                header += f"**Layer:** {entry.get('layer')} | **Strength:** {entry.get('strength')} | **Aggregation:** {entry.get('aggregation_method')}\n\n"
                header += f"**Question:** {entry.get('question', '')}"

                baseline = entry.get('baseline_response', '')
                steered = entry.get('steered_response', '')

                scores = f"""
**Scores:**

- Differentiation: {entry.get('differentiation_score', 'N/A')}
- Coherence: {entry.get('coherence_score', 'N/A')}
- Trait Alignment: {entry.get('trait_alignment_score', 'N/A')}
- Overall: {entry.get('overall_score', 'N/A')}

**Choose Result:** {entry.get('choose_result', 'N/A')} | **Open Traits:** {entry.get('open_traits', 'N/A')}
"""

                # Explanations - use static method from EntryDisplay
                explanations_dict = entry.get('explanations', {})
                explanations = EntryDisplay.format_explanations(explanations_dict) if explanations_dict else ""

                state_data = {
                    'question': entry.get('question', ''),
                    'steered_response': steered,
                    'layer': entry.get('layer'),
                    'strength': entry.get('strength'),
                    'aggregation': entry.get('aggregation_method'),
                    'overall_score': entry.get('overall_score'),
                    'differentiation_score': entry.get('differentiation_score'),
                    'coherence_score': entry.get('coherence_score'),
                    'trait_alignment_score': entry.get('trait_alignment_score')
                }

                updates.extend([
                    gr.Column(visible=True),
                    gr.Markdown(value=header),
                    gr.Textbox(value=baseline),
                    gr.Textbox(value=steered),
                    gr.Markdown(value=scores),
                    gr.Markdown(value=explanations)
                ])

                if self.show_save_buttons:
                    updates.extend([
                        state_data,  # state
                        gr.update(visible=True),  # save button visible
                        ""  # status
                    ])
            else:
                # Hide unused slots
                updates.extend([
                    gr.Column(visible=False),
                    gr.Markdown(value=""),
                    gr.Textbox(value=""),
                    gr.Textbox(value=""),
                    gr.Markdown(value=""),
                    gr.Markdown(value="")
                ])

                if self.show_save_buttons:
                    updates.extend([
                        {},  # state
                        gr.update(visible=False),  # save button hidden
                        ""  # status
                    ])

        return updates

    def get_output_components(self):
        """Get list of output components for Gradio event handlers."""
        outputs = [self.summary]

        for widget in self.entry_widgets:
            outputs.extend([
                widget['col'],
                widget['header'],
                widget['baseline'],
                widget['steered'],
                widget['scores'],
                widget['explanations']
            ])

            if self.show_save_buttons:
                outputs.extend([
                    widget['state'],
                    widget['save_btn'],  # Add button to outputs
                    widget['save_status']
                ])

        return outputs

    def get_save_buttons(self):
        """Get all save button components."""
        if self.show_save_buttons:
            return [widget['save_btn'] for widget in self.entry_widgets]
        return []

    def get_states(self):
        """Get all state components."""
        if self.show_save_buttons:
            return [widget['state'] for widget in self.entry_widgets]
        return []

    def wire_save_buttons(self, save_handler_fn):
        """
        Wire up all save buttons with a save handler function.

        Args:
            save_handler_fn: Function that takes entry_data and returns status message
        """
        if not self.show_save_buttons:
            return

        for widget in self.entry_widgets:
            widget['save_btn'].click(
                fn=save_handler_fn,
                inputs=[widget['state']],
                outputs=[widget['save_status']]
            )


class StatisticsDisplay:
    """Displays statistics for top configurations."""

    @staticmethod
    def format_stat(config: Dict) -> str:
        """
        Format a single statistics configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Formatted markdown string
        """
        output = f"""
**Layer:** {config.get('layer')} | **Strength:** {config.get('strength')} | **Aggregation:** {config.get('aggregation_method')}

**Average Scores:**

- Overall: {config.get('avg_overall_score', 'N/A')}
- Differentiation: {config.get('avg_differentiation_score', 'N/A')}
- Coherence: {config.get('avg_coherence_score', 'N/A')}
- Trait Alignment: {config.get('avg_trait_alignment_score', 'N/A')}

**Choose Results:**

- Correct (B): {config.get('choose_correct', 0)}
- Incorrect (A): {config.get('choose_incorrect', 0)}
- Equal: {config.get('choose_equal', 0)}
- Total: {config.get('choose_total', 0)}
"""
        return output

    @staticmethod
    def format_stats(configs: list, k: int) -> str:
        """
        Format multiple statistics configurations.

        Args:
            configs: List of configuration dictionaries
            k: Number of configurations

        Returns:
            Formatted markdown string
        """
        output = f"# Top {k} Configurations (Highest Average Overall Scores)\n\n"
        output += f"Total configurations: {len(configs)}\n\n"
        output += "---\n\n"

        for i, config in enumerate(configs, 1):
            output += f"## Configuration {i}\n\n"
            output += StatisticsDisplay.format_stat(config)
            output += "\n---\n\n"

        return output
