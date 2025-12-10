"""
WandB tracking integration for the steering optimization pipeline.
"""

import logging
from datetime import datetime
from typing import Any, Optional

import numpy as np
import optuna

from wisent.core.errors import OptimizationError

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


logger = logging.getLogger(__name__)


class WandBTracker:
    """WandB tracking for optimization experiments."""

    def __init__(self, config: Any, enabled: bool = False):
        """
        Initialize WandB tracker.
        
        Args:
            config: OptimizationConfig with wandb settings
            enabled: Whether WandB tracking is enabled
        """
        self.config = config
        self.enabled = enabled
        self.wandb_run: Optional[Any] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def init(self) -> Optional[Any]:
        """Initialize WandB for experiment tracking."""
        if not self.enabled:
            return None
            
        if not WANDB_AVAILABLE:
            raise ImportError(
                "WandB integration enabled but wandb is not installed. Install with: pip install wandb"
            )
        
        try:
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=f"{self.config.study_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict(),
                tags=["optuna", "steering", "optimization"],
                reinit=True,
            )
            self.logger.info(f"WandB initialized: {wandb.run.url}")
            return self.wandb_run
        except Exception as e:
            raise OptimizationError(
                reason=f"Failed to initialize WandB: {e}. "
                       f"Run 'wandb login' to authenticate or set use_wandb=False",
                cause=e
            )

    def log_trial(self, trial: optuna.Trial, metrics: dict[str, float]):
        """Log trial results to WandB."""
        if not self.enabled or self.wandb_run is None:
            return

        try:
            log_data = {f"trial/{k}": v for k, v in trial.params.items()}
            log_data.update({f"metrics/{k}": v for k, v in metrics.items()})
            log_data["trial/number"] = trial.number
            wandb.log(log_data)
        except Exception as e:
            self.logger.warning(f"Failed to log trial to WandB: {e}")

    def log_final_results(self, study: optuna.Study, final_results: dict[str, Any]):
        """Log final optimization results to WandB."""
        if not self.enabled or self.wandb_run is None:
            return

        try:
            best_params = {f"best/{k}": v for k, v in study.best_params.items()}
            best_metrics = {
                "best/validation_accuracy": study.best_value,
                "best/baseline_accuracy": final_results["baseline_benchmark_metrics"]["accuracy"],
                "best/steered_accuracy": final_results["steered_benchmark_metrics"]["accuracy"],
                "best/accuracy_improvement": final_results["accuracy_improvement"],
                "study/n_trials": len(study.trials),
                "study/n_complete_trials": len(
                    [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                ),
            }

            wandb.log({**best_params, **best_metrics})

            trial_values = [t.value for t in study.trials if t.value is not None]
            if trial_values:
                wandb.log(
                    {
                        "optimization/best_value_so_far": max(trial_values),
                        "optimization/mean_trial_value": np.mean(trial_values),
                        "optimization/std_trial_value": np.std(trial_values),
                    }
                )

        except Exception as e:
            self.logger.warning(f"Failed to log final results to WandB: {e}")

    def finish(self):
        """Finish WandB run."""
        if self.wandb_run is not None:
            wandb.finish()
            self.wandb_run = None
