"""Control components (dropdowns, buttons) for Gradio UI."""
import gradio as gr
from typing import List, Callable, Optional, Tuple
from tests.EVALOOP.core.data_manager import DataManager
from tests.EVALOOP.ui.components.favorites import FavoritesManager


class FileSelector:
    """File selector dropdown with automatic file discovery."""

    def __init__(self, data_manager: DataManager, file_type: str = "scores"):
        """
        Initialize FileSelector.

        Args:
            data_manager: DataManager instance
            file_type: Type of files to show ("scores" or "stats")
        """
        self.data_manager = data_manager
        self.file_type = file_type
        self.dropdown = self._create_dropdown()

    def _create_dropdown(self):
        """Create the dropdown component."""
        files = self.data_manager.get_available_files()
        choices = files.get(self.file_type, [])

        if not choices:
            choices = ["No files found"]

        return gr.Dropdown(
            choices=choices,
            label=f"Select {self.file_type.capitalize()} File",
            value=choices[0] if choices else None
        )

    def get_component(self):
        """Get the Gradio dropdown component."""
        return self.dropdown


class ConfigDropdowns:
    """Configuration dropdowns for layer, strength, and aggregation."""

    def __init__(self, data_manager: DataManager, initial_file: Optional[str] = None):
        """
        Initialize ConfigDropdowns.

        Args:
            data_manager: DataManager instance
            initial_file: Initial file to load configs from
        """
        self.data_manager = data_manager
        self.initial_file = initial_file
        self.components = {}
        self._create_components()

    def _create_components(self):
        """Create the dropdown components."""
        # Get initial values
        if self.initial_file:
            layers, strengths, aggregations = self.data_manager.get_unique_configs(self.initial_file)
        else:
            layers, strengths, aggregations = [], [], []

        self.components['layer'] = gr.Dropdown(
            choices=layers,
            label="Layer",
            value=layers[0] if layers else None
        )

        self.components['strength'] = gr.Dropdown(
            choices=strengths,
            label="Strength",
            value=strengths[0] if strengths else None
        )

        self.components['aggregation'] = gr.Dropdown(
            choices=aggregations,
            label="Aggregation",
            value=aggregations[0] if aggregations else None
        )

    def update_from_file(self, file_path: str) -> Tuple:
        """
        Update dropdown choices based on a scores file.

        Args:
            file_path: Path to scores file

        Returns:
            Tuple of dropdown updates
        """
        layers, strengths, aggregations = self.data_manager.get_unique_configs(file_path)

        return (
            gr.Dropdown(choices=layers, value=layers[0] if layers else None),
            gr.Dropdown(choices=strengths, value=strengths[0] if strengths else None),
            gr.Dropdown(choices=aggregations, value=aggregations[0] if aggregations else None)
        )

    def get_components(self):
        """Get all dropdown components as a dict."""
        return self.components

    def get_values(self) -> list:
        """Get list of components in order [layer, strength, aggregation]."""
        return [
            self.components['layer'],
            self.components['strength'],
            self.components['aggregation']
        ]


class NavigationButtons:
    """Navigation buttons for browsing entries."""

    def __init__(self):
        """Initialize NavigationButtons."""
        self.components = {}
        self._create_components()

    def _create_components(self):
        """Create button components."""
        with gr.Row():
            self.components['prev'] = gr.Button("◀ Previous")
            self.components['next'] = gr.Button("Next ▶")
            self.components['random'] = gr.Button("🎲 Random")
            self.components['load'] = gr.Button("Load Entry")

    def get_components(self):
        """Get all button components as a dict."""
        return self.components


class FavoriteButton:
    """Reusable favorite button with state management."""

    def __init__(self, favorites_manager: FavoritesManager):
        """
        Initialize FavoriteButton.

        Args:
            favorites_manager: FavoritesManager instance
        """
        self.favorites_manager = favorites_manager
        self.button = gr.Button("⭐ Save as Favorite", variant="secondary")
        self.status = gr.Markdown("")

    def create_save_handler(self, state_component):
        """
        Create and wire up the save handler.

        Args:
            state_component: Gradio State component containing entry data
        """
        def save_handler(entry_data):
            """Save current entry to favorites."""
            if not entry_data or 'question' not in entry_data:
                return "⚠️ No entry loaded"

            success = self.favorites_manager.add(
                question=entry_data['question'],
                steered_response=entry_data['steered_response'],
                layer=entry_data['layer'],
                strength=entry_data['strength'],
                aggregation=entry_data['aggregation'],
                notes="",
                overall_score=entry_data.get('overall_score'),
                differentiation_score=entry_data.get('differentiation_score'),
                coherence_score=entry_data.get('coherence_score'),
                trait_alignment_score=entry_data.get('trait_alignment_score')
            )

            return "✅ Saved to favorites!" if success else "❌ Error saving"

        self.button.click(
            fn=save_handler,
            inputs=[state_component],
            outputs=[self.status]
        )

    def get_components(self):
        """Get button and status components."""
        return self.button, self.status

    @staticmethod
    def create_save_function(favorites_manager: FavoritesManager):
        """
        Create a save handler function that can be used with any button.

        Args:
            favorites_manager: FavoritesManager instance

        Returns:
            Function that saves entry_data to favorites and returns status message
        """
        def save_handler(entry_data):
            """Save current entry to favorites."""
            if not entry_data or 'question' not in entry_data:
                return "⚠️ No entry loaded"

            success = favorites_manager.add(
                question=entry_data['question'],
                steered_response=entry_data['steered_response'],
                layer=entry_data['layer'],
                strength=entry_data['strength'],
                aggregation=entry_data['aggregation'],
                notes="",
                overall_score=entry_data.get('overall_score'),
                differentiation_score=entry_data.get('differentiation_score'),
                coherence_score=entry_data.get('coherence_score'),
                trait_alignment_score=entry_data.get('trait_alignment_score')
            )

            return "✅ Saved to favorites!" if success else "❌ Error saving"

        return save_handler
