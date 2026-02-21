"""Pair management mixin for Wisent."""
from __future__ import annotations
from typing import List
from wisent.core.modalities import ContentType, wrap_content


class WisentPairsMixin:
    """Mixin providing contrastive pair management."""

    def add_pair(
        self,
        positive: ContentType,
        negative: ContentType,
        trait: str,
        description: str | None = None,
    ) -> "Wisent":
        """
        Add a contrastive pair for a trait.

        Args:
            positive: Desired behavior example
            negative: Undesired behavior example
            trait: Name of the trait being steered
            description: Optional description of the trait

        Returns:
            Self for chaining
        """
        # Wrap raw content
        pos_content = wrap_content(positive) if isinstance(positive, str) else positive
        neg_content = wrap_content(negative) if isinstance(negative, str) else negative

        # Initialize trait if needed
        if trait not in self._pairs:
            self._pairs[trait] = []
            self._traits[trait] = TraitConfig(
                name=trait,
                description=description or f"Steering trait: {trait}",
            )

        self._pairs[trait].append((pos_content, neg_content))
        self._trained = False  # Need to retrain

        return self

    def add_pairs(
        self,
        pairs: List[tuple[ContentType, ContentType]],
        trait: str,
        description: str | None = None,
    ) -> "Wisent":
        """
        Add multiple contrastive pairs for a trait.

        Args:
            pairs: List of (positive, negative) tuples
            trait: Name of the trait
            description: Optional description

        Returns:
            Self for chaining
        """
        for pos, neg in pairs:
            self.add_pair(pos, neg, trait, description)
        return self

    def clear_pairs(self, trait: str | None = None) -> "Wisent":
        """
        Clear stored pairs.

        Args:
            trait: Specific trait to clear (None = all)

        Returns:
            Self for chaining
        """
        if trait is None:
            self._pairs.clear()
            self._traits.clear()
        elif trait in self._pairs:
            del self._pairs[trait]
            del self._traits[trait]

        self._trained = False
        return self

    # ==================== Training ====================

