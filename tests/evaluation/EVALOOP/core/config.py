"""Configuration management for evaluation pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any
import json


@dataclass
class TraitConfig:
    """Configuration for a single trait evaluation."""
    name: str
    contrastive_pairs_path: Path
    test_questions_paths: List[Path]
    layers: List[str]
    strengths: List[float]
    aggregations: List[str]
    output_file: Path
    instruction_prompts: Dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "TraitConfig":
        """Create TraitConfig from dictionary."""
        return cls(
            name=name,
            contrastive_pairs_path=Path(data["contrastive_pairs"]),
            test_questions_paths=[Path(p) for p in data["test_questions"]],
            layers=data["layers"],
            strengths=data["strengths"],
            aggregations=data["aggregations"],
            output_file=Path(data["output_file"]),
            instruction_prompts={
                metric: Path(path)
                for metric, path in data.get("instruction_prompt", {}).items()
            }
        )


@dataclass
class EvaluationConfig:
    """Main evaluation configuration."""
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    device: str = "cuda"
    num_questions: int = 4
    max_new_tokens: int = 400
    judge_model: str = "claude-sonnet-4-5"
    judge_max_tokens: int = 512
    output_formats: List[str] = field(default_factory=lambda: ["txt", "markdown", "json"])

    # Metric weights for overall score
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "differentiation": 0.2,
        "coherence": 0.3,
        "trait_alignment": 0.5
    })

    # Paths
    base_dir: Path = field(default_factory=lambda: Path("tests/EVALOOP"))

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)


class ConfigManager:
    """Manages loading and accessing configuration."""

    def __init__(self, config_path: Path = None):
        """
        Initialize ConfigManager.

        Args:
            config_path: Optional path to JSON config file
        """
        self.eval_config = EvaluationConfig()
        self.trait_configs: Dict[str, TraitConfig] = {}

        if config_path and config_path.exists():
            self.load_from_file(config_path)
        else:
            self._load_default_config()

    def _load_default_config(self):
        """Load default hardcoded configuration."""
        default_traits = {
            "happy": {
                "contrastive_pairs": "tests/EVALOOP/contrastive_pairs/happy.json",
                "test_questions": [
                    "tests/EVALOOP/test_questions/base_questions.txt",
                    "tests/EVALOOP/test_questions/happy_questions.txt"
                ],
                "layers": ["6", "7", "8"],
                "strengths": [2.0, 3.0, 4.0, 5.0],
                "aggregations": ["continuation_token"],
                "output_file": "tests/EVALOOP/output/happy_output.json",
                "instruction_prompt": {
                    "differentiation": "tests/EVALOOP/instruction_prompts/general/differentiation",
                    "coherence": "tests/EVALOOP/instruction_prompts/general/coherence",
                    "trait_alignment": "tests/EVALOOP/instruction_prompts/happy/trait_alignment",
                    "open": "tests/EVALOOP/instruction_prompts/general/open",
                    "choose": "tests/EVALOOP/instruction_prompts/happy/choose"
                }
            },
            "evil": {
                "contrastive_pairs": "tests/EVALOOP/contrastive_pairs/evil.json",
                "test_questions": [
                    "tests/EVALOOP/test_questions/base_questions.txt",
                    "tests/EVALOOP/test_questions/evil_questions.txt"
                ],
                "layers": ["6", "7", "8"],
                "strengths": [-2.0, -3.0, -4.0, -5.0],
                "aggregations": ["continuation_token"],
                "output_file": "tests/EVALOOP/output/evil_output.json",
                "instruction_prompt": {
                    "differentiation": "tests/EVALOOP/instruction_prompts/general/differentiation",
                    "coherence": "tests/EVALOOP/instruction_prompts/general/coherence",
                    "trait_alignment": "tests/EVALOOP/instruction_prompts/evil/trait_alignment",
                    "open": "tests/EVALOOP/instruction_prompts/general/open",
                    "choose": "tests/EVALOOP/instruction_prompts/evil/choose"
                }
            }
        }

        for trait_name, trait_data in default_traits.items():
            self.trait_configs[trait_name] = TraitConfig.from_dict(trait_name, trait_data)

    def load_from_file(self, path: Path):
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Load evaluation config
        if "evaluation" in data:
            for key, value in data["evaluation"].items():
                if hasattr(self.eval_config, key):
                    setattr(self.eval_config, key, value)

        # Load trait configs
        if "traits" in data:
            for trait_name, trait_data in data["traits"].items():
                self.trait_configs[trait_name] = TraitConfig.from_dict(trait_name, trait_data)

    def get_trait(self, name: str) -> TraitConfig:
        """Get configuration for a specific trait."""
        if name not in self.trait_configs:
            raise ValueError(f"Trait '{name}' not found in configuration")
        return self.trait_configs[name]

    def list_traits(self) -> List[str]:
        """List all available trait names."""
        return list(self.trait_configs.keys())

    def get_trait_config(self, name: str) -> TraitConfig:
        """Alias for get_trait() for consistency."""
        return self.get_trait(name)

    @property
    def config(self) -> EvaluationConfig:
        """Get evaluation config."""
        return self.eval_config

    @property
    def traits(self) -> Dict[str, TraitConfig]:
        """Get all trait configs."""
        return self.trait_configs

    def get_vector_path(self, trait: str, layer: str, aggregation: str) -> Path:
        """
        Get the path to steering vectors for a specific configuration.

        Args:
            trait: Trait name
            layer: Layer identifier
            aggregation: Aggregation method

        Returns:
            Path to steering vectors directory
        """
        base_output = self.eval_config.base_dir / "output"
        vector_dir = base_output / f"{trait}_vectors" / f"steering_output_layer{layer}_aggregation{aggregation}"
        return vector_dir
