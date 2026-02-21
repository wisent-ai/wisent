"""IO and utility mixin for BaseClassifier."""
from __future__ import annotations
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from wisent.core.utils import preferred_dtype

logger = logging.getLogger(__name__)


class ClassifierIOMixin:
    """Mixin providing IO and utility methods for BaseClassifier."""

    def save_model(self, path: str) -> None:
        """
        Save the model state and metadata to a file.

        arguments:
            path:
                the file path to save the model.

        raises:
            ClassifierError:
                if the model is not initialized."""
        self._require_model()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        input_dim = int(next(self.model.parameters()).shape[1])
        torch.save({
            "classifier_name": self.name,
            "state_dict": self.model.state_dict(),
            "input_dim": input_dim,
            "threshold": self.threshold,
            "model_hyperparams": self.model_hyperparams(),
        }, path)
        logger.info("Saved %s to %s", self.name, path)

    def load_model(self, path: str) -> None:
        """
        Load the model state and metadata from a file.

        arguments:
            path:
                the file path to load the model from.

        raises:
            FileNotFoundError:
                if the model file does not exist.
            ClassifierError:
                if the checkpoint format is unsupported.
        """
        if not os.path.exists(path): raise FileNotFoundError(path)
        data = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(data, dict) or "state_dict" not in data or "input_dim" not in data:
            raise ClassifierError("Unsupported checkpoint format.")
        self.threshold = float(data.get("threshold", self.threshold))
        input_dim = int(data["input_dim"])
        hyper = dict(data.get("model_hyperparams", {}))
        self.model = self.build_model(input_dim, **hyper).to(self.device)
        self.model.load_state_dict(data["state_dict"]); self.model.eval()

    def _require_model(self) -> None:
        if self.model is None:
            raise ClassifierError("Model not initialized. Call fit() or load_model() first.")

    @classmethod
    def to_2d_tensor(cls, X, device: str, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert input to a 2D tensor on the specified device and dtype.
        
        arguments:
            X:
                input data as array-like or tensor.
            device:
                target device string.
            dtype:
                target torch dtype.
        
        returns:
            2D torch tensor.

        raises:
            ClassifierError:
                if the input cannot be converted to 2D tensor.
        """
        if isinstance(X, torch.Tensor):
            t = X.to(device=device, dtype=dtype)
            if t.ndim == 1: t = t.view(1, -1)
            if t.ndim != 2: raise ClassifierError(f"Expected 2D features, got {tuple(t.shape)}")
            return t
        t = torch.tensor(X, device=device, dtype=dtype)
        if t.ndim == 1: t = t.view(1, -1)
        if t.ndim != 2: raise ClassifierError(f"Expected 2D features, got {tuple(t.shape)}")
        return t

    @staticmethod
    def to_1d_tensor(y, *, device: str, dtype: torch.dtype) -> torch.Tensor:
        """
        Convert input to a 1D tensor on the specified device and dtype.

        arguments:
            y:
                input data as array-like or tensor.
            device:
                target device string.
            dtype:
                target torch dtype.

        returns:
            1D torch tensor.

        raises:
            ClassifierError:
                if the input cannot be converted to 1D tensor.
        """
        if isinstance(y, torch.Tensor):
            return y.to(device=device, dtype=dtype).view(-1)
        return torch.tensor(list(y), device=device, dtype=dtype).view(-1)

    @staticmethod
    def _basic_prf(preds: list[float], labels: list[float]) -> tuple[float, float, float, float]:
        """
        Compute basic precision, recall, and F1 score.

        arguments:
            preds:
                list of predicted labels (0.0 or 1.0).
            labels:
                list of true labels (0.0 or 1.0).
        
        returns:
            tuple of (accuracy, precision, recall, f1).
        """
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
        total = max(len(labels), 1)
        acc = sum(1 for p, l in zip(preds, labels) if p == l) / total
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        return float(acc), float(prec), float(rec), float(f1)

    @staticmethod
    def _roc_auc(labels: list[float], scores: list[float]) -> float:
        """
        Compute ROC AUC using the Mann-Whitney U statistic.

        arguments:
            labels:
                list of true binary labels (0.0 or 1.0).
            scores:
                list of predicted scores or probabilities.  
        
        returns:
            ROC AUC value.
        """
        if len(scores) < 2 or len(set(labels)) < 2: return 0.0
        pairs = sorted(zip(scores, labels), key=lambda x: x[0])
        pos = sum(1 for _, y in pairs if y == 1); neg = sum(1 for _, y in pairs if y == 0)
        if pos == 0 or neg == 0: return 0.0
        rank_sum = 0.0; i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]: j += 1
            avg_rank = (i + j + 2) / 2.0
            rank_sum += avg_rank * sum(1 for k in range(i, j + 1) if pairs[k][1] == 1)
            i = j + 1
        U = rank_sum - pos * (pos + 1) / 2.0
