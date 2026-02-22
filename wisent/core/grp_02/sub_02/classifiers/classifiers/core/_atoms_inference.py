"""Inference and evaluation mixin for BaseClassifier."""
from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)


class ClassifierInferenceMixin:
    """Mixin providing inference/evaluation methods for BaseClassifier."""

    def predict(self, X: torch.Tensor | np.ndarray) -> int | list[int]:
        """
        Predict class labels for the given input.

        arguments:
            X:
                2D feature array or tensor.
        
        returns:
            predicted class label(s) as int or list of int.
        
        example:
            >>> X = np.random.randn(5, 2).astype(np.float32)
            >>> print(X)
            [[ 0.123 -1.456]
             [ 0.789  0.012]
             [-0.345  0.678]
             [ 1.234 -0.567]
             [-0.890 -1.234]]
            >>> preds = self.predict(X)
            >>> print(preds)
            [0, 1, 1, 0, 0]
        """
        self._require_model()

        X_tensor = self.to_2d_tensor(X, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            probs = self._forward_probs(self.model, X_tensor).view(-1).cpu().tolist()
        preds = [1 if p >= self.threshold else 0 for p in probs]
        return preds[0] if len(preds) == 1 else preds

    def predict_proba(self, X: torch.Tensor | np.ndarray) -> float | list[float]:
        """
        Predict class probabilities for the given input.

        arguments:
            X: 2D feature array or tensor.
        
        returns:
            predicted class probability

        example:
            >>> X = np.random.randn(5, 2).astype(np.float32)
            >>> print(X)
            [[ 0.123 -1.456]
             [ 0.789  0.012]
             [-0.345  0.678]
             [ 1.234 -0.567]
             [-0.890 -1.234]]
            >>> probs = self.predict_proba(X)
            >>> print(probs)
            [0.23, 0.76, 0.54, 0.12, 0.34]
        """
        self._require_model()

        X_tensor = self.to_2d_tensor(X, device=self.device, dtype=self.dtype)

        with torch.no_grad():
            probs = self._forward_probs(self.model, X_tensor).view(-1).cpu().tolist()
        return probs[0] if len(probs) == 1 else probs

    def evaluate(self, X: torch.Tensor | np.ndarray, y: torch.Tensor | np.ndarray) -> dict[str, float]:
        """
        Evaluate the model on the given dataset and return metrics.

        arguments:
            X:
                2D feature array or tensor.
            y:
                1D label array or tensor.

        returns:
            dictionary of evaluation metrics.

        flow:
            >>> X = np.random.randn(2, 2).astype(np.float32)
            >>> y = np.random.randint(0, 2, size=(2,)).astype(np.int64)
            >>> print(X)
            [[ 0.123 -1.456]
             [ 0.789  0.012]]
            >>> print(y)
            [1, 0]
            >>> y_pred = self.predict(X)
            >>> print(y_pred)
            [0, 0]
            >>> y_prob = self.predict_proba(X)
            >>> print(y_prob)
            [0.34, 0.12]
            >>> metrics = self.evaluate(X, y)
            >>> print(metrics)
            {'accuracy': 0.5, ...}
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        preds = [float(y_pred)] if isinstance(y_pred, int) else [float(v) for v in y_pred]
        probs = [float(y_prob)] if isinstance(y_prob, float) else [float(v) for v in y_prob]
        labels = y.detach().cpu().view(-1).tolist() if isinstance(y, torch.Tensor) else list(y)
        acc, prec, rec, f1 = self._basic_prf(preds, labels)
        auc = self._roc_auc(labels, probs)
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

    def configure_criterion(self) -> nn.Module: return nn.BCELoss()

    def _make_criterion(self, spec: nn.Module | str) -> nn.Module:
        """
         Create a loss criterion from a string or module.
         
         arguments:
             spec:
                 loss specification, either a string or a torch.nn.Module instance.
        
         returns:
             a torch.nn.Module loss function.
             
        raises:
            ValueError:
                if the string specification is unknown.
        """
        if isinstance(spec, nn.Module): return spec
        key = str(spec).strip().lower()
        if key in {"bce", "bceloss"}: return nn.BCELoss()
        if key in {"bcewithlogits", "bcewithlogitsloss"}: return nn.BCEWithLogitsLoss()
        raise UnknownTypeError(entity_type="criterion", value=spec)

    def configure_optimizer(self, model: nn.Module, lr: float) -> optim.Optimizer:
        """
        Default optimizer configuration: Adam with given learning rate.

        arguments:
            model:
                the model to optimize.
            lr:
                the learning rate.
        returns:
            an Adam optimizer instance.
        """
        return optim.Adam(model.parameters(), lr=lr)

    def _make_optimizer(self, model: nn.Module, spec: str | optim.Optimizer | None, lr: float, extra: dict) -> optim.Optimizer:
        """
        Create an optimizer from a specification.

        arguments:
            model:
                the model to optimize.
            spec:
                optimizer specification: string, instance, callable, or None for default.
            lr:
                learning rate.
            extra:
                extra keyword arguments for optimizer constructor.

        returns:
            an optimizer instance.

        raises:
            ValueError:
                if the string specification is unknown.
            TypeError:
                if the specification type is unsupported.
        """
        if isinstance(spec, optim.Optimizer): return spec
        if spec is None: return self.configure_optimizer(model, lr)
        if isinstance(spec, str):
            try: cls = getattr(optim, spec)
            except AttributeError as exc: raise UnknownTypeError(entity_type="optimizer", value=spec) from exc
            return cls(model.parameters(), lr=lr, **extra)
        if callable(spec): return spec(model.parameters(), lr=lr, **extra)
        raise TypeError(f"Unsupported optimizer spec: {type(spec)}")

