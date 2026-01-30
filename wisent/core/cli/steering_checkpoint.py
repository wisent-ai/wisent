"""Checkpoint and logging for steering autotune - enables resume and progress tracking."""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SteeringConfig:
    """Optimal steering configuration found by tuning."""
    layer_strengths: Dict[int, float]
    steering_method: str
    accuracy_threshold: float
    base_strength: float
    val_score: float
    layer_accuracies: Dict[int, float]

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "layer_strengths": {str(k): v for k, v in self.layer_strengths.items()},
            "steering_method": self.steering_method,
            "accuracy_threshold": self.accuracy_threshold,
            "base_strength": self.base_strength,
            "val_score": self.val_score,
            "layer_accuracies": {str(k): v for k, v in self.layer_accuracies.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringConfig":
        """Load from dict."""
        return cls(
            layer_strengths={int(k): v for k, v in d["layer_strengths"].items()},
            steering_method=d["steering_method"],
            accuracy_threshold=d["accuracy_threshold"],
            base_strength=d["base_strength"],
            val_score=d["val_score"],
            layer_accuracies={int(k): v for k, v in d["layer_accuracies"].items()},
        )


@dataclass
class AutotuneCheckpoint:
    """Checkpoint for resumable autotune."""
    completed_configs: List[dict]
    best_config: Optional[dict]
    best_score: int
    last_combo_idx: int
    total_combos: int
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AutotuneCheckpoint":
        return cls(**d)

    def save(self, path: Path):
        """Save checkpoint to file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Optional["AutotuneCheckpoint"]:
        """Load checkpoint from file if exists."""
        if not path.exists():
            return None
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


def log_config_result(log_file: Path, result: dict):
    """Append a single config result to the log file."""
    with open(log_file, 'a') as f:
        f.write(json.dumps(result) + "\n")


def setup_checkpoint_paths(checkpoint_dir: str, log_file: str = None):
    """Setup checkpoint directory and log file paths."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    checkpoint_file = checkpoint_path / "autotune_checkpoint.json"

    if log_file is None:
        log_file_path = checkpoint_path / f"autotune_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    else:
        log_file_path = Path(log_file)

    return checkpoint_file, log_file_path
