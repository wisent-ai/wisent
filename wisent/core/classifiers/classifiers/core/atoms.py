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
from wisent.core.errors import DuplicateNameError, InvalidRangeError, UnknownTypeError
from wisent.core.utils.device import preferred_dtype

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
    test_size: float = 0.2
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    monitor: str = "accuracy"  
    random_state: int = 42

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

class BaseClassifier(ABC):
    name: str = "base"
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
        threshold: float = 0.5,
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

    def fit(
            self,
            X,
            y,
            config: ClassifierTrainConfig | None = None,
            optimizer: str | optim.Optimizer | callable | None = None,
            lr: float | None = None,
            optimizer_kwargs: dict | None = None,
            criterion: nn.Module | str | None = None,
            on_epoch_end: Callable[[int, dict[str, float]], bool | None] | None = None,
            **model_params: Any,
    ) -> ClassifierTrainReport:
       
        #1 creating 
        cfg = config or ClassifierTrainConfig()
        torch.manual_seed(cfg.random_state)

        #2 creating tensors
        X_tensor = self.to_2d_tensor(X, device=self.device, dtype=self.dtype)
        y_tensor = self.to_1d_tensor(y, device=self.device, dtype=self.dtype)

        #3 checking dimensions
        if X_tensor.shape[0] != y_tensor.shape[0]:
            raise ClassifierError(f"X and y length mismatch: {X_tensor.shape[0]} vs {y_tensor.shape[0]}")

        if self.model is None:
            input_dim = int(X_tensor.shape[1])
            self.model = self.build_model(input_dim, **model_params).to(self.device)

        
        # 4 creating dataloaders
        train_loader, test_loader = self._make_dataloaders(X_tensor, y_tensor, cfg)

        # 5 creating criterion and optimizer
        crit = self._make_criterion(criterion) if criterion is not None else self.configure_criterion()
        learn_rate = lr if lr is not None else cfg.learning_rate
        opt = self._make_optimizer(self.model, optimizer, learn_rate, optimizer_kwargs or {})

        # 6 training loop
        best_metric = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None

        # 7 history
        history: dict[str, list[float]] = {
            "train_loss": [], "test_loss": [],
            "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": [],
        }

        # 8 main loop
        for epoch in range(cfg.num_epochs):
            # one epoch
            train_loss = self._train_one_epoch(self.model, train_loader, opt, crit)
            test_loss, probs, labels = self._eval_one_epoch(self.model, test_loader, crit)

            preds = [1.0 if p >= self.threshold else 0.0 for p in probs]
            acc, prec, rec, f1 = self._basic_prf(preds, labels)
            auc = self._roc_auc(labels, probs)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["accuracy"].append(acc)
            history["precision"].append(prec)
            history["recall"].append(rec)
            history["f1"].append(f1)
            history["auc"].append(auc)

            # keep best checkpoint by cfg.monitor
            monitored = history[cfg.monitor][-1]
            if monitored > best_metric:
                best_metric = monitored
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            # optional external observer/pruner
            if on_epoch_end is not None:
                stop = on_epoch_end(epoch, {k: history[k][-1] for k in history})
                if stop:
                    break

            if (epoch == 0) or ((epoch + 1) % 10 == 0) or (epoch == cfg.num_epochs - 1):
                logger.info("[%s] epoch %d/%d  train=%.4f  test=%.4f  acc=%.4f  f1=%.4f",
                            self.name, epoch + 1, cfg.num_epochs, train_loss, test_loss, acc, f1)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # final pass
        test_loss, probs, labels = self._eval_one_epoch(self.model, test_loader, crit)
        preds = [1.0 if p >= self.threshold else 0.0 for p in probs]
        acc, prec, rec, f1 = self._basic_prf(preds, labels)
        auc = self._roc_auc(labels, probs)
        final = ClassifierMetrics(acc, prec, rec, f1, auc)

        best_epoch = int(max(range(len(history[cfg.monitor])), key=history[cfg.monitor].__getitem__) + 1)
        return ClassifierTrainReport(
            classifier_name=self.name,
            input_dim=input_dim,
            best_epoch=best_epoch,
            epochs_ran=len(history["accuracy"]),
            final=final,
            history={k: [float(v) for v in vs] for k, vs in history.items()},
        )
    
    def _make_dataloaders(
        self,
        X: torch.Tensor | np.ndarray,
        y: torch.Tensor | np.ndarray,
        cfg: ClassifierTrainConfig,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Split (X, y) into train/test using a seeded random split and wrap each in DataLoaders.

        arguments:
            X:
                2D feature array or tensor.
            y:
                1D label array or tensor.
            cfg:
                training configuration with test_size, batch_size, and random_state.
        
        returns:
            tuple of (train_dataloader, test_dataloader)

        example:
            >>> X = np.random.randn(100, 2).astype(np.float32)
            >>> print(X.shape)
            (100, 2)
            >>> print(X[0])
            [ 0.123 -1.456]
            >>> y = np.random.randint(0, 2, size=(100,)).astype(np.int64)
            >>> print(y.shape)
            (100,)
            >>> print(y[0])
            1
            >>> cfg = ClassifierTrainConfig(test_size=0.2, batch_size=16, random_state=42)
            >>> train_loader, test_loader = self._make_dataloaders(X, y, cfg)
            >>> print(len(train_loader.dataset), len(test_loader.dataset))
            (80, 20)
            >>> xb, yb = next(iter(train_loader))
            >>> print(xb.shape, yb.shape)
            (16, 2) (16,)
        """

        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        if isinstance(y, np.ndarray): y = torch.from_numpy(y)

        ds = TensorDataset(X, y)

        if len(ds) < 2:
            return (
                DataLoader(ds, batch_size=cfg.batch_size, shuffle=True),
                DataLoader(ds, batch_size=cfg.batch_size, shuffle=False),
            )

        test_count = max(1, int(round(cfg.test_size * len(ds))))
        test_count = min(test_count, len(ds) - 1)
        train_count = len(ds) - test_count

        gen = torch.Generator().manual_seed(cfg.random_state)
        train_ds, test_ds = random_split(ds, [train_count, test_count], generator=gen)

        return (
            DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False),
        )


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

    def _train_one_epoch(self, model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion: _Loss) -> float:
        """
        Train the model for one epoch over the given DataLoader.
        
        arguments:
            model:
                the model to train.
            loader:
                DataLoader for training data.
            optimizer:
                optimizer instance.
            criterion:
                loss function.

        returns:
            average training loss over the epoch.
        """
        model.train(); total = 0.0; steps = 0
        xb: torch.Tensor; yb: torch.Tensor

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            out = self._forward_probs(model, xb)
            loss = criterion(out.view(-1), yb.view(-1))
            loss.backward(); optimizer.step()
            total += float(loss.item()); steps += 1
        return total / max(steps, 1)

    def _eval_one_epoch(self, model: nn.Module, loader: DataLoader, criterion: _Loss) -> float:
        """
        Evaluate the model for one epoch over the given DataLoader.

        arguments:
            model:
                the model to evaluate.
            loader:
                DataLoader for evaluation data.
            criterion:
                loss function.

        returns:
            average evaluation loss over the epoch.
        """
        model.eval(); total = 0.0; steps = 0; probs_all=[]; labels_all=[]
        with torch.no_grad():
            xb: torch.Tensor; yb: torch.Tensor
            for xb, yb in loader:
                out = self._forward_probs(model, xb)
                loss = criterion(out.view(-1), yb.view(-1))
                total += float(loss.item()); steps += 1
                probs_all.extend(out.detach().cpu().view(-1).tolist())
                labels_all.extend(yb.detach().cpu().view(-1).tolist())
        return (total / max(steps, 1), probs_all, labels_all)

    def _forward_probs(self, model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get predicted probabilities.
        
        arguments:
            model:
                the model to use.
            xb:
                input feature tensor.
                
        returns:
            tensor of predicted probabilities.
        """
        if xb.device.type != self.device: xb = xb.to(self.device)
        if xb.dtype != self.dtype: xb = xb.to(self.dtype)
        out = model(xb)
        return out.view(-1, 1) if out.ndim == 1 else out

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
        return float(U / (pos * neg))