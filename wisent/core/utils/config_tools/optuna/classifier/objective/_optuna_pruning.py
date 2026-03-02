"""Pruning callback and summary for Optuna optimizer."""
import logging
from typing import Any
import optuna
from wisent.core.utils.config_tools.optuna.classifier._optuna_config import OptimizationResult

logger = logging.getLogger(__name__)


class OptunaPruningMixin:
    """Mixin providing pruning callback and summary."""

    def _create_pruning_callback(self, trial: optuna.Trial):
        """
        Create a pruning callback for early stopping during classifier training.
        
        This callback is called at each epoch during training and reports
        intermediate values to Optuna. If the trial is performing poorly
        compared to other trials, Optuna will prune it early.
        
        Args:
            trial: The Optuna trial object
            
        Returns:
            A callback function that can be passed to classifier.fit()
        """
        pruning_patience = self.opt_config.pruning_patience
        best_score = 0.0
        patience_counter = 0
        
        def callback(epoch: int, metrics: dict[str, Any]) -> bool:
            """
            Pruning callback called at each training epoch.
            
            Args:
                epoch: Current epoch number (0-indexed)
                metrics: Dict containing training metrics like 'val_accuracy', 'val_loss', etc.
                
            Returns:
                True to continue training, False to stop early
            """
            nonlocal best_score, patience_counter
            
            # Get the validation metric to report
            # Try common metric names in order of preference
            score = None
            for metric_name in ["val_f1", "val_accuracy", "accuracy", "f1", "train_accuracy"]:
                if metric_name in metrics:
                    score = metrics[metric_name]
                    break
            
            if score is None:
                # If no validation metric found, use train loss inverted
                if "train_loss" in metrics:
                    score = 1.0 / (1.0 + metrics["train_loss"])
                else:
                    score = 0.5  # Default neutral score
            
            # Report intermediate value to Optuna
            try:
                trial.report(score, epoch)
            except Exception:
                pass  # Trial may already be finished
            
            # Check if trial should be pruned by Optuna
            try:
                if trial.should_prune():
                    self.logger.debug(f"Trial {trial.number} pruned at epoch {epoch} (score={score:.4f})")
                    raise optuna.TrialPruned()
            except optuna.TrialPruned:
                return False  # Stop training
            except Exception:
                pass  # Continue if pruning check fails
            
            # Local early stopping based on patience
            if score > best_score:
                best_score = score
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= pruning_patience:
                self.logger.debug(
                    f"Early stopping at epoch {epoch}: no improvement for {pruning_patience} epochs"
                )
                return False  # Stop training
            
            return True  # Continue training
        
        return callback
    
    def get_optimization_summary(self, result: OptimizationResult) -> dict[str, Any]:
        """Get a comprehensive optimization summary."""
        return {
            "best_configuration": result.get_best_config(),
            "best_score": result.best_value,
            "optimization_time_seconds": result.optimization_time,
            "total_trials": len(result.trial_results),
            "cache_efficiency": {
                "hits": result.cache_hits,
                "misses": result.cache_misses,
                "hit_rate": result.cache_hits / (result.cache_hits + result.cache_misses)
                if (result.cache_hits + result.cache_misses) > 0
                else 0,
            },
            "activation_data_info": {
                key: {
                    "samples": data.activations.shape[0],
                    "features": data.activations.shape[1]
                    if len(data.activations.shape) > 1
                    else data.activations.shape[0],
                    "layer": data.layer,
                    "aggregation": data.aggregation,
                }
                for key, data in self.activation_data.items()
            },
            "study_info": {
                "n_trials": len(result.study.trials),
                "best_trial": result.study.best_trial.number,
                "pruned_trials": len([t for t in result.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            },
        }
