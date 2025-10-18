"""Favorites management for saving and loading favorite answers."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class FavoritesManager:
    """Manages saving and loading favorite steered responses."""

    def __init__(self, favorites_file: str = "tests/EVALOOP/output/favorites.json"):
        """
        Initialize FavoritesManager.

        Args:
            favorites_file: Path to favorites JSON file
        """
        self.favorites_file = Path(favorites_file)
        self._ensure_dir()

    def _ensure_dir(self):
        """Ensure the favorites directory exists."""
        os.makedirs(self.favorites_file.parent, exist_ok=True)

    def load(self) -> List[Dict]:
        """
        Load all favorites.

        Returns:
            List of favorite dictionaries
        """
        if not self.favorites_file.exists():
            return []

        try:
            with open(self.favorites_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save(self, favorites: List[Dict]) -> bool:
        """
        Save favorites list to file.

        Args:
            favorites: List of favorite dictionaries

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.favorites_file, 'w', encoding='utf-8') as f:
                json.dump(favorites, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def add(
        self,
        question: str,
        steered_response: str,
        layer: str,
        strength: float,
        aggregation: str,
        notes: str = "",
        overall_score: Optional[float] = None,
        differentiation_score: Optional[float] = None,
        coherence_score: Optional[float] = None,
        trait_alignment_score: Optional[float] = None
    ) -> bool:
        """
        Add a new favorite.

        Args:
            question: Question text
            steered_response: Steered response text
            layer: Layer configuration
            strength: Strength configuration
            aggregation: Aggregation method
            notes: Optional user notes
            overall_score: Overall score
            differentiation_score: Differentiation score
            coherence_score: Coherence score
            trait_alignment_score: Trait alignment score

        Returns:
            True if successful, False otherwise
        """
        favorites = self.load()

        favorite = {
            'question': question,
            'steered_response': steered_response,
            'layer': layer,
            'strength': strength,
            'aggregation': aggregation,
            'notes': notes,
            'overall_score': overall_score,
            'differentiation_score': differentiation_score,
            'coherence_score': coherence_score,
            'trait_alignment_score': trait_alignment_score,
            'timestamp': datetime.now().isoformat()
        }

        favorites.append(favorite)
        return self.save(favorites)

    def delete(self, index: int) -> bool:
        """
        Delete a favorite by index.

        Args:
            index: Index of favorite to delete (0-based)

        Returns:
            True if successful, False otherwise
        """
        favorites = self.load()

        if 0 <= index < len(favorites):
            favorites.pop(index)
            return self.save(favorites)

        return False

    def clear(self) -> bool:
        """
        Delete all favorites.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.favorites_file.exists():
                os.remove(self.favorites_file)
            return True
        except Exception:
            return False

    def count(self) -> int:
        """
        Get number of favorites.

        Returns:
            Number of favorites
        """
        return len(self.load())
