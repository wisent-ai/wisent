"""IO and utility mixin for Wisent."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import logging
import torch

from wisent.core.constants import DEFAULT_STRENGTH

logger = logging.getLogger(__name__)


class WisentIOMixin:
    """Mixin providing IO and utility methods."""

    def act(
        self,
        state: RobotState,
        steer: Dict[str, float] | None = None,
        config: SteeringConfig | None = None,
    ) -> Any:
        """
        Convenience method for robotics: get action from state.

        Args:
            state: Robot state observation
            steer: Steering configuration
            config: Additional config

        Returns:
            Robot action
        """
        if not isinstance(self.adapter, RoboticsAdapter):
            raise AdapterError("act() is only available for robotics adapter")

        return self.generate(state, steer=steer, config=config)

    def transcribe(
        self,
        audio: AudioContent,
        steer: Dict[str, float] | None = None,
        language: str | None = None,
    ) -> str:
        """
        Convenience method for audio: transcribe with steering.

        Args:
            audio: Audio content
            steer: Steering configuration
            language: Target language

        Returns:
            Transcribed text
        """
        if not isinstance(self.adapter, AudioAdapter):
            raise AdapterError("transcribe() is only available for audio adapter")

        if steer is None or not steer:
            return self.adapter.transcribe(audio, language=language)

        combined_vectors = self._combine_steering_vectors(steer)
        return self.adapter.transcribe(
            audio,
            steering_vectors=combined_vectors,
            language=language,
        )

    # ==================== Introspection ====================

    @property
    def traits(self) -> List[str]:
        """Get list of defined traits."""
        return list(self._traits.keys())

    @property
    def is_trained(self) -> bool:
        """Check if steering vectors have been trained."""
        return self._trained

    def get_trait_info(self, trait: str) -> TraitConfig | None:
        """Get configuration for a trait."""
        return self._traits.get(trait)

    def get_intervention_points(self) -> List[str]:
        """Get available steering intervention points."""
        return [ip.name for ip in self.adapter.get_intervention_points()]

    def get_recommended_layers(self) -> List[str]:
        """Get recommended layers for steering."""
        return [
            ip.name for ip in self.adapter.get_intervention_points()
            if ip.recommended
        ]

    # ==================== Persistence ====================

    def save_vectors(self, path: str) -> None:
        """
        Save trained steering vectors to file.

        Args:
            path: File path to save to
        """
        if not self._trained:
            raise NoTrainedVectorsError()

        data = {}
        for trait, config in self._traits.items():
            if config.steering_vectors is not None:
                data[trait] = {
                    "description": config.description,
                    "default_scale": config.default_scale,
                    "layers": config.layers,
                    "vectors": {
                        k: v.cpu() for k, v in config.steering_vectors.items()
                        if v is not None
                    },
                }

        torch.save(data, path)
        logger.info(f"Saved steering vectors to {path}")

    def load_vectors(self, path: str) -> "Wisent":
        """
        Load trained steering vectors from file.

        Args:
            path: File path to load from

        Returns:
            Self for chaining
        """
        data = torch.load(path, map_location=self.device)

        for trait, trait_data in data.items():
            self._traits[trait] = TraitConfig(
                name=trait,
                description=trait_data.get("description", ""),
                default_scale=trait_data.get("default_scale", DEFAULT_STRENGTH),
                layers=trait_data.get("layers"),
                steering_vectors=LayerActivations(trait_data["vectors"]),
            )

        self._trained = True
        logger.info(f"Loaded steering vectors from {path}")
        return self

    def __repr__(self) -> str:
        return (
            f"Wisent(\n"
            f"  adapter={self.adapter!r},\n"
            f"  traits={self.traits},\n"
            f"  trained={self._trained}\n"
            f")"
        )
