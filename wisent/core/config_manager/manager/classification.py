"""Classification configuration mixin for WisentConfigManager."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..types import ClassificationConfig, TaskConfig


class ClassificationMixin:
    """Mixin providing classification config save/get methods."""

    def save_classification_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
        layer: int = 12,
        token_aggregation: str = "average",
        detection_threshold: float = 0.6,
        classifier_type: str = "logistic",
        prompt_construction_strategy: str = "multiple_choice",
        token_targeting_strategy: str = "last_token",
        accuracy: float = 0.0,
        f1_score: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        optimization_method: str = "manual",
        set_as_default: bool = False,
    ) -> Path:
        """Save classification config for a model/task."""
        config = self._load_model_config(model_name)

        classification = ClassificationConfig(
            layer=layer,
            token_aggregation=token_aggregation,
            detection_threshold=detection_threshold,
            classifier_type=classifier_type,
            prompt_construction_strategy=prompt_construction_strategy,
            token_targeting_strategy=token_targeting_strategy,
            accuracy=accuracy,
            f1_score=f1_score,
            precision=precision,
            recall=recall,
        )

        if task_name:
            if task_name not in config.tasks:
                config.tasks[task_name] = TaskConfig(task_name=task_name)
            config.tasks[task_name].classification = classification
            config.tasks[task_name].optimization_method = optimization_method
            config.tasks[task_name].updated_at = datetime.now().isoformat()

        if set_as_default or not task_name:
            config.default_classification = classification

        return self._save_model_config(config)

    def get_classification_config(
        self,
        model_name: str,
        task_name: Optional[str] = None,
    ) -> Optional[ClassificationConfig]:
        """Get classification config for a model/task."""
        config = self._load_model_config(model_name)

        if task_name and task_name in config.tasks:
            task_config = config.tasks[task_name]
            if task_config.classification:
                return task_config.classification

        return config.default_classification
