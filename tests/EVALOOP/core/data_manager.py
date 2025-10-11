"""Data management for loading evaluation results."""
import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from tests.EVALOOP.core.config import ConfigManager
from tests.EVALOOP.core.models import EvaluationResult, ConfigStatistics


class DataManager:
    """Manages loading and accessing evaluation data."""

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize DataManager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.base_dir = self.config_manager.eval_config.base_dir / "output"

    def get_available_files(self) -> Dict[str, List[str]]:
        """
        Get available scores and stats files.

        Returns:
            Dictionary with 'scores' and 'stats' lists of file paths
        """
        scores_files = []
        stats_files = []

        # Get all traits
        for trait_name in self.config_manager.list_traits():
            # Get all output formats
            for format_type in self.config_manager.eval_config.output_formats:
                scores_file = self.base_dir / f"{trait_name}_scores_{format_type}.json"
                stats_file = self.base_dir / f"{trait_name}_stats_{format_type}.json"

                if scores_file.exists():
                    scores_files.append(str(scores_file))
                if stats_file.exists():
                    stats_files.append(str(stats_file))

        return {
            "scores": scores_files,
            "stats": stats_files
        }

    def load_scores(self, file_path: str) -> List[Dict]:
        """
        Load evaluation scores from JSON file.

        Args:
            file_path: Path to scores file

        Returns:
            List of evaluation result dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_stats(self, file_path: str) -> List[Dict]:
        """
        Load statistics from JSON file.

        Args:
            file_path: Path to stats file

        Returns:
            List of statistics dictionaries
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_entry(self, file_path: str, index: int) -> Optional[Dict]:
        """
        Get a specific entry by index.

        Args:
            file_path: Path to scores file
            index: Entry index (0-based)

        Returns:
            Entry dictionary or None if index out of range
        """
        entries = self.load_scores(file_path)
        if 0 <= index < len(entries):
            return entries[index]
        return None

    def get_random_entry(self, file_path: str) -> Tuple[Optional[Dict], int]:
        """
        Get a random entry.

        Args:
            file_path: Path to scores file

        Returns:
            Tuple of (entry dictionary, index) or (None, -1) if no entries
        """
        entries = self.load_scores(file_path)
        if not entries:
            return None, -1

        index = random.randint(0, len(entries) - 1)
        return entries[index], index

    def get_top_k(self, file_path: str, k: int) -> List[Dict]:
        """
        Get top k entries with highest overall scores.

        Args:
            file_path: Path to scores file
            k: Number of entries to return

        Returns:
            List of top k entries
        """
        entries = self.load_scores(file_path)
        return entries[:k]

    def get_bottom_k(self, file_path: str, k: int) -> List[Dict]:
        """
        Get bottom k entries with lowest overall scores.

        Args:
            file_path: Path to scores file
            k: Number of entries to return

        Returns:
            List of bottom k entries
        """
        entries = self.load_scores(file_path)
        return entries[-k:]

    def filter_by_config(
        self,
        file_path: str,
        layer: str,
        strength: float,
        aggregation: str
    ) -> List[Dict]:
        """
        Filter entries by configuration.

        Args:
            file_path: Path to scores file
            layer: Layer value
            strength: Strength value
            aggregation: Aggregation method

        Returns:
            List of matching entries
        """
        entries = self.load_scores(file_path)
        return [
            entry for entry in entries
            if entry.get('layer') == layer
            and entry.get('strength') == strength
            and entry.get('aggregation_method') == aggregation
        ]

    def get_unique_configs(self, file_path: str) -> Tuple[List[str], List[float], List[str]]:
        """
        Get unique configuration values from a scores file.

        Args:
            file_path: Path to scores file

        Returns:
            Tuple of (unique layers, unique strengths, unique aggregations)
        """
        entries = self.load_scores(file_path)

        layers = sorted(set(entry.get('layer') for entry in entries if entry.get('layer')))
        strengths = sorted(set(entry.get('strength') for entry in entries if entry.get('strength') is not None))
        aggregations = sorted(set(entry.get('aggregation_method') for entry in entries if entry.get('aggregation_method')))

        return layers, strengths, aggregations

    def get_questions_ranked_by_score(
        self,
        file_path: str,
        score_type: str = 'overall_score'
    ) -> List[Tuple[str, float, int]]:
        """
        Get questions ranked by average score across all configurations.

        Args:
            file_path: Path to scores file
            score_type: Score type to rank by

        Returns:
            List of (question, avg_score, num_configs) tuples, sorted descending
        """
        entries = self.load_scores(file_path)

        # Group scores by question
        question_scores: Dict[str, List[float]] = {}
        for entry in entries:
            question = entry.get('question', '')
            score = entry.get(score_type)

            if question and score is not None:
                if question not in question_scores:
                    question_scores[question] = []
                question_scores[question].append(score)

        # Calculate averages
        rankings = []
        for question, scores in question_scores.items():
            avg_score = sum(scores) / len(scores)
            rankings.append((question, avg_score, len(scores)))

        # Sort by average score descending
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def get_entry_count(self, file_path: str) -> int:
        """
        Get total number of entries in a scores file.

        Args:
            file_path: Path to scores file

        Returns:
            Number of entries
        """
        entries = self.load_scores(file_path)
        return len(entries)
