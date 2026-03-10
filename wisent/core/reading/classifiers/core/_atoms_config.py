from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from torch.nn.modules.loss import _Loss
from wisent.core.utils.infra_tools.errors import DuplicateNameError, InvalidRangeError, UnknownTypeError
from wisent.core.utils import preferred_dtype
from typing import Optional as _Optional
from wisent.core.utils.config_tools.constants import (
    VIZ_MLP_EPOCHS,
    CLASSIFIER_BATCH_SIZE_MIN,
    CLASSIFIER_BATCH_SIZE_MAX,
    CLASSIFIER_BATCH_DIVISOR,
    CLASSIFIER_TEST_SIZE_MIN,
    CLASSIFIER_TEST_SIZE_MAX,
    CLASSIFIER_MIN_TEST_SAMPLES,
    CLASSIFIER_DEFAULT_LR,
)

__all__ = [
    "ClassifierTrainConfig",
    "ClassifierMetrics",
    "ClassifierTrainReport",
    "ClassifierError",
    "BaseClassifier",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ClassifierTrainConfig:
    """
    Training configuration for classifiers.
    
    attributes:
        test_size:
            fraction of data to hold out for testing
        num_epochs:
            maximum number of training epochs
        batch_size:
            training batch size
        learning_rate:
            optimizer learning rate
        monitor:
            which metric to monitor for best epoch selection
        random_state:
            random seed for data shuffling and initialization
    """
    num_epochs: int
    batch_size: int
    learning_rate: float
    test_size: float
    monitor: Optional[str] = None
    random_state: _Optional[int] = None

    @classmethod
    def from_data_shape(cls, n_samples: int, n_features: int) -> ClassifierTrainConfig:
        """Derive training config from data dimensions.

        Same pattern as derive_geometry_params: clamped formulas from data shape.
        """
        batch_size = min(
            CLASSIFIER_BATCH_SIZE_MAX,
            max(CLASSIFIER_BATCH_SIZE_MIN, n_samples // CLASSIFIER_BATCH_DIVISOR),
        )
        test_size = min(
            CLASSIFIER_TEST_SIZE_MAX,
            max(CLASSIFIER_TEST_SIZE_MIN, CLASSIFIER_MIN_TEST_SAMPLES / n_samples),
        )
        return cls(
            num_epochs=VIZ_MLP_EPOCHS,
            batch_size=batch_size,
            learning_rate=CLASSIFIER_DEFAULT_LR,
            test_size=test_size,
            monitor="auc",
        )

@dataclass(slots=True, frozen=True)
class ClassifierMetrics:
    """
    Evaluation metrics for classifiers.

    attributes:
        accuracy: float
            Overall accuracy of predictions.
        precision: float
            Precision (positive predictive value).
        recall: float
            Recall (sensitivity).
        f1: float
            F1 score (harmonic mean of precision and recall).
        auc: float
            Area under the ROC curve.
    """
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float

@dataclass(slots=True, frozen=True)
class ClassifierTrainReport:
    """
    Training report for classifiers.

    attributes:
        classifier_name: str
            Name of the classifier.
        input_dim: int
            Dimensionality of the input features.
        best_epoch: int
            Epoch number of the best model.
        epochs_ran: int
            Total number of epochs run.
        final: ClassifierMetrics
            Final evaluation metrics on the test set. It contains accuracy, precision, recall, f1, and auc.
        history: dict[str, list[float]]

    
    """
    classifier_name: str
    input_dim: int
    best_epoch: int
    epochs_ran: int
    final: ClassifierMetrics
    history: dict[str, list[float]]

    def asdict(self) -> dict[str, str | int | float | dict]:
        """
        Return a dictionary representation of the report.
        
        returns:
            A dictionary with all report fields, including nested metrics.

        example:
            >>> report.asdict()
            {
                "classifier_name": "mlp",
                "input_dim": 4,
                "best_epoch": 23,
                "epochs_ran": 30,
                "final": {
                    "accuracy": 0.95,
                    "precision": 0.96,
                    "recall": 0.94,
                    "f1": 0.95,
                    "auc": 0.98
                },
                "history": {
                    "train_loss": [...],
                    "test_loss": [...],
                    "accuracy": [...],
                    "precision": [...],
                    "recall": [...],
                    "f1": [...],
                    "auc": [...]
                }
            }   
        """
        d = asdict(self); d["final"] = asdict(self.final); return d

class ClassifierError(RuntimeError):
    pass
