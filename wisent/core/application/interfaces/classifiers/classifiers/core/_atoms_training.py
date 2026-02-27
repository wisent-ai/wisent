"""Training mixin for BaseClassifier."""
from __future__ import annotations
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
from torch.nn.modules.loss import _Loss
from wisent.core.classifiers.classifiers.core._atoms_config import (
    ClassifierTrainConfig, ClassifierTrainReport, ClassifierMetrics,
)
from wisent.core.constants import TRAINING_LOG_FREQUENCY

logger = logging.getLogger(__name__)


class ClassifierTrainingMixin:
    """Mixin providing training methods for BaseClassifier."""

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

            if (epoch == 0) or ((epoch + 1) % TRAINING_LOG_FREQUENCY == 0) or (epoch == cfg.num_epochs - 1):
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
        """Run model forward and return sigmoid probabilities."""
        logits = model(xb)
        return torch.sigmoid(logits)
