"""Base classifier types and abstract class for wisent classifiers."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from torch.nn.modules.loss import _Loss
from wisent.core.utils.infra_tools.errors import DuplicateNameError, InvalidRangeError, UnknownTypeError
from wisent.core.utils import preferred_dtype
from wisent.core.utils.config_tools.constants import CLASSIFIER_THRESHOLD, BASE_CLASS_NAME

# Re-export config types
from wisent.core.reading.classifiers.core._atoms_config import (
    ClassifierTrainConfig,
    ClassifierMetrics,
    ClassifierTrainReport,
    ClassifierError,
)

# Import mixins
from wisent.core.reading.classifiers.core._atoms_training import ClassifierTrainingMixin
from wisent.core.reading.classifiers.core._atoms_inference import ClassifierInferenceMixin
from wisent.core.reading.classifiers.core._atoms_io import ClassifierIOMixin

__all__ = [
    "ClassifierTrainConfig",
    "ClassifierMetrics",
    "ClassifierTrainReport",
    "ClassifierError",
    "BaseClassifier",
]

logger = logging.getLogger(__name__)


class BaseClassifier(ClassifierTrainingMixin, ClassifierInferenceMixin, ClassifierIOMixin, ABC):
    name: str = BASE_CLASS_NAME
    description: str = "Abstract classifier"

    _REGISTRY: dict[str, type[BaseClassifier]] = {}

    model: nn.Module | None
    device: str
    dtype: torch.dtype
    threshold: float

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls is BaseClassifier:
            return
        if not getattr(cls, "name", None):
            raise TypeError("Classifier subclasses must define class attribute `name`.")
        if cls.name in BaseClassifier._REGISTRY:
            raise DuplicateNameError(name=cls.name, context="classifier registry")
        BaseClassifier._REGISTRY[cls.name] = cls

    def __init__(
        self,
        threshold: float = CLASSIFIER_THRESHOLD,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise InvalidRangeError(param_name="threshold", actual=threshold, min_val=0.0, max_val=1.0)
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype if dtype is not None else preferred_dtype(self.device)
        self.model = None

    @abstractmethod
    def build_model(self, input_dim: int, **model_params: Any) -> nn.Module:
        """Return a torch.nn.Module that outputs P(y=1) ∈ [0,1]."""
        raise NotImplementedError

    def model_hyperparams(self) -> dict[str, Any]:
        return {}


    # Training methods from ClassifierTrainingMixin:
    # fit, _make_dataloaders, _train_one_epoch, _eval_one_epoch, _forward_probs

    # Inference methods from ClassifierInferenceMixin:
    # predict, predict_proba, evaluate, configure_criterion, _make_criterion,
    # configure_optimizer, _make_optimizer

    # IO methods from ClassifierIOMixin:
    # save_model, load_model, _require_model, to_2d_tensor, to_1d_tensor,
    # _basic_prf, _roc_auc
