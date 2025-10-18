"""Plotting components for statistical analysis."""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


class PlotGenerator:
    """Generates statistical plots for evaluation data."""

    SCORE_NAMES = {
        'overall_score': 'Overall Score',
        'differentiation_score': 'Differentiation Score',
        'coherence_score': 'Coherence Score',
        'trait_alignment_score': 'Trait Alignment Score'
    }

    @staticmethod
    def plot_histogram(file_path: str, score_type: str = 'overall_score'):
        """
        Generate histogram of scores.

        Args:
            file_path: Path to scores JSON file
            score_type: Type of score to plot

        Returns:
            matplotlib figure object
        """
        score_display_name = PlotGenerator.SCORE_NAMES.get(
            score_type, score_type.replace('_', ' ').title()
        )

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract scores (filter out None values)
            scores = [
                entry[score_type] for entry in data
                if score_type in entry and entry[score_type] is not None
            ]

            if not scores:
                return PlotGenerator._empty_plot(f'No {score_display_name.lower()} found')

            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))

            # Determine bins
            min_score, max_score = min(scores), max(scores)
            bins = np.arange(0, 105, 5) if 0 <= min_score and max_score <= 100 else 30

            ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(score_display_name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(
                f'Distribution of {score_display_name}\n{os.path.basename(file_path)}',
                fontsize=14, fontweight='bold'
            )
            ax.grid(axis='y', alpha=0.3)

            # Add statistics
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_dev = np.std(scores)

            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_score:.2f}')
            ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_score:.2f}')
            ax.plot([], [], ' ', label=f'Std Dev: {std_dev:.2f}')

            ax.legend(loc='best', framealpha=0.9)
            plt.tight_layout()
            return fig

        except Exception as e:
            return PlotGenerator._error_plot(str(e))

    @staticmethod
    def plot_correlations(file_path: str):
        """
        Generate pairwise correlation scatter plots.

        Args:
            file_path: Path to scores JSON file

        Returns:
            matplotlib figure object
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract all score types
            score_types = ['differentiation_score', 'coherence_score',
                          'trait_alignment_score', 'overall_score']
            scores_data = {score_type: [] for score_type in score_types}

            # Collect scores (only entries where all scores are present)
            for entry in data:
                if all(score_type in entry and entry[score_type] is not None
                      for score_type in score_types):
                    for score_type in score_types:
                        scores_data[score_type].append(entry[score_type])

            if not scores_data['overall_score']:
                return PlotGenerator._empty_plot('No complete score data found')

            # Create figure with 2x3 subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                f'Pairwise Score Correlations\n{os.path.basename(file_path)}',
                fontsize=16, fontweight='bold'
            )

            score_labels = {
                'differentiation_score': 'Differentiation',
                'coherence_score': 'Coherence',
                'trait_alignment_score': 'Trait Alignment',
                'overall_score': 'Overall'
            }

            # Define pairs
            pairs = [
                ('differentiation_score', 'coherence_score'),
                ('differentiation_score', 'trait_alignment_score'),
                ('differentiation_score', 'overall_score'),
                ('coherence_score', 'trait_alignment_score'),
                ('coherence_score', 'overall_score'),
                ('trait_alignment_score', 'overall_score')
            ]

            for idx, (score_x, score_y) in enumerate(pairs):
                row, col = idx // 3, idx % 3
                ax = axes[row, col]

                x_data = scores_data[score_x]
                y_data = scores_data[score_y]

                # Scatter plot
                ax.scatter(x_data, y_data, alpha=0.5, s=20, color='steelblue')

                # Calculate correlation
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
            return PlotGenerator._error_plot(str(e))

    @staticmethod
    def plot_config_histogram(
        file_path: str,
        layer: str,
        strength: float,
        aggregation: str,
        score_type: str = 'overall_score'
    ):
        """
        Generate histogram for a specific configuration.

        Args:
            file_path: Path to scores JSON file
            layer: Layer value
            strength: Strength value
            aggregation: Aggregation method
            score_type: Type of score to plot

        Returns:
            matplotlib figure object
        """
        score_display_name = PlotGenerator.SCORE_NAMES.get(
            score_type, score_type.replace('_', ' ').title()
        )

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Filter and extract scores
            scores = [
                entry[score_type] for entry in data
                if entry.get('layer') == layer
                and entry.get('strength') == strength
                and entry.get('aggregation_method') == aggregation
                and score_type in entry and entry[score_type] is not None
            ]

            if not scores:
                return PlotGenerator._empty_plot(
                    f'No {score_display_name.lower()} found\nfor this configuration'
                )

            # Create histogram
            fig, ax = plt.subplots(figsize=(10, 6))

            min_score, max_score = min(scores), max(scores)
            bins = np.arange(0, 105, 5) if 0 <= min_score and max_score <= 100 else 30

            ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
            ax.set_xlabel(score_display_name, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(
                f'Distribution of {score_display_name}\n'
                f'Layer: {layer}, Strength: {strength}, Aggregation: {aggregation}',
                fontsize=12, fontweight='bold'
            )
            ax.grid(axis='y', alpha=0.3)

            # Add statistics
            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_dev = np.std(scores)

            ax.axvline(mean_score, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_score:.2f}')
            ax.axvline(median_score, color='green', linestyle='--', linewidth=2,
                      label=f'Median: {median_score:.2f}')
            ax.plot([], [], ' ', label=f'Std Dev: {std_dev:.2f}')
            ax.plot([], [], ' ', label=f'N: {len(scores)}')

            ax.legend(loc='best', framealpha=0.9)
            plt.tight_layout()
            return fig

        except Exception as e:
            return PlotGenerator._error_plot(str(e))

    @staticmethod
    def plot_config_correlations(
        file_path: str,
        layer: str,
        strength: float,
        aggregation: str
    ):
        """
        Generate correlation plots for a specific configuration.

        Args:
            file_path: Path to scores JSON file
            layer: Layer value
            strength: Strength value
            aggregation: Aggregation method

        Returns:
            matplotlib figure object
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Filter and collect scores
            score_types = ['differentiation_score', 'coherence_score',
                          'trait_alignment_score', 'overall_score']
            scores_data = {score_type: [] for score_type in score_types}

            for entry in data:
                if (entry.get('layer') == layer
                    and entry.get('strength') == strength
                    and entry.get('aggregation_method') == aggregation):
                    if all(score_type in entry and entry[score_type] is not None
                          for score_type in score_types):
                        for score_type in score_types:
                            scores_data[score_type].append(entry[score_type])

            if not scores_data['overall_score']:
                return PlotGenerator._empty_plot(
                    'No complete score data found\nfor this configuration'
                )

            # Create figure with 2x3 subplots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                f'Pairwise Score Correlations\n'
                f'Layer: {layer}, Strength: {strength}, Aggregation: {aggregation}',
                fontsize=14, fontweight='bold'
            )

            score_labels = {
                'differentiation_score': 'Differentiation',
                'coherence_score': 'Coherence',
                'trait_alignment_score': 'Trait Alignment',
                'overall_score': 'Overall'
            }

            pairs = [
                ('differentiation_score', 'coherence_score'),
                ('differentiation_score', 'trait_alignment_score'),
                ('differentiation_score', 'overall_score'),
                ('coherence_score', 'trait_alignment_score'),
                ('coherence_score', 'overall_score'),
                ('trait_alignment_score', 'overall_score')
            ]

            for idx, (score_x, score_y) in enumerate(pairs):
                row, col = idx // 3, idx % 3
                ax = axes[row, col]

                x_data = scores_data[score_x]
                y_data = scores_data[score_y]

                ax.scatter(x_data, y_data, alpha=0.5, s=20, color='steelblue')

                correlation = np.corrcoef(x_data, y_data)[0, 1]

                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(x_data), max(x_data), 100)
                ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

                ax.set_xlabel(score_labels[score_x], fontsize=10)
                ax.set_ylabel(score_labels[score_y], fontsize=10)
                ax.set_title(
                    f'Correlation: {correlation:.3f} (N={len(x_data)})',
                    fontsize=11, fontweight='bold'
                )
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            return PlotGenerator._error_plot(str(e))

    @staticmethod
    def _empty_plot(message: str):
        """Create an empty plot with a message."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig

    @staticmethod
    def _error_plot(error_message: str):
        """Create an error plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Error loading data:\n{error_message}',
               ha='center', va='center', fontsize=12, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
