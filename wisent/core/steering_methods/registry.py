"""
Centralized Steering Method Registry.

This module is the SINGLE SOURCE OF TRUTH for all steering method definitions.
All CLI arguments, optimization parameters, and method-specific settings
are defined here and automatically propagated throughout the codebase.

================================================================================
HOW TO ADD A NEW STEERING METHOD
================================================================================

1. Create the method class in steering_methods/methods/your_method.py:

    from wisent.core.steering_methods.core.atoms import PerLayerBaseSteeringMethod
    
    class YourMethod(PerLayerBaseSteeringMethod):
        name = "your_method"  # Must be unique
        description = "Description of your method"
        
        def train_for_layer(self, pos_list, neg_list) -> torch.Tensor:
            # Your implementation here
            return steering_vector

2. Add the method type to SteeringMethodType enum below:

    class SteeringMethodType(Enum):
        CAA = "caa"
        YOUR_METHOD = "your_method"  # Add this

3. Create the method definition in this file:

    YOUR_METHOD_DEFINITION = SteeringMethodDefinition(
        name="your_method",
        method_type=SteeringMethodType.YOUR_METHOD,
        description="Your method description",
        method_class_path="wisent.core.steering_methods.methods.your_method.YourMethod",
        parameters=[
            SteeringMethodParameter(
                name="param_name",
                type=float,
                default=1.0,
                help="Parameter description",
            ),
        ],
        default_strength=1.0,
        strength_range=(0.1, 5.0),
    )

4. Register it in the SteeringMethodRegistry._REGISTRY dict:

    _REGISTRY: Dict[str, SteeringMethodDefinition] = {
        "caa": CAA_DEFINITION,
        "your_method": YOUR_METHOD_DEFINITION,  # Add this
    }

That's it! The method will automatically:
- Appear in CLI --steering-method choices
- Have its parameters added to CLI
- Be available in optimization pipelines
- Work with all existing steering infrastructure

================================================================================
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod


class SteeringMethodType(Enum):
    """Enumeration of all supported steering methods."""
    CAA = "caa"
    HYPERPLANE = "hyperplane"
    PRISM = "prism"
    PULSE = "pulse"
    TITAN = "titan"


@dataclass
class SteeringMethodParameter:
    """Definition of a single parameter for a steering method."""
    name: str
    type: Type
    default: Any
    help: str
    cli_flag: Optional[str] = None  # e.g., "--caa-normalize"
    choices: Optional[List[Any]] = None
    action: Optional[str] = None  # e.g., "store_true" for boolean flags
    
    def get_cli_flag(self, method_name: str) -> str:
        """Get the CLI flag name for this parameter."""
        if self.cli_flag:
            return self.cli_flag
        return f"--{method_name}-{self.name.replace('_', '-')}"


@dataclass
class SteeringMethodDefinition:
    """Complete definition of a steering method including all configuration."""
    
    name: str
    method_type: SteeringMethodType
    description: str
    method_class_path: str  # e.g., "wisent.core.steering_methods.methods.caa.CAAMethod"
    
    # Method-specific parameters
    parameters: List[SteeringMethodParameter] = field(default_factory=list)
    
    # Optimization configuration
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    
    # Default values for common settings
    default_strength: float = 1.0
    strength_range: tuple = (0.1, 5.0)
    
    def get_method_class(self) -> Type[BaseSteeringMethod]:
        """Dynamically import and return the method class."""
        module_path, class_name = self.method_class_path.rsplit(".", 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameter values as a dictionary."""
        return {p.name: p.default for p in self.parameters}
    
    def add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add this method's CLI arguments to a parser."""
        for param in self.parameters:
            flag = param.get_cli_flag(self.name)
            kwargs: Dict[str, Any] = {
                "default": param.default,
                "help": f"[{self.name.upper()}] {param.help}",
            }
            
            if param.action:
                kwargs["action"] = param.action
            else:
                kwargs["type"] = param.type
                
            if param.choices:
                kwargs["choices"] = param.choices
                
            parser.add_argument(flag, **kwargs)
    
    def extract_params_from_args(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract this method's parameters from parsed CLI args."""
        params = {}
        for param in self.parameters:
            # Convert CLI flag to attribute name
            flag = param.get_cli_flag(self.name)
            attr_name = flag.lstrip("-").replace("-", "_")
            if hasattr(args, attr_name):
                params[param.name] = getattr(args, attr_name)
            else:
                params[param.name] = param.default
        return params


# =============================================================================
# STEERING METHOD DEFINITIONS
# =============================================================================
# Add new steering methods here. This is the ONLY place you need to define them.

CAA_DEFINITION = SteeringMethodDefinition(
    name="caa",
    method_type=SteeringMethodType.CAA,
    description="Contrastive Activation Addition - computes mean(positive) - mean(negative) steering vectors",
    method_class_path="wisent.core.steering_methods.methods.caa.CAAMethod",
    parameters=[
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--caa-normalize",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 5.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 5.0),
)


HYPERPLANE_DEFINITION = SteeringMethodDefinition(
    name="hyperplane",
    method_type=SteeringMethodType.HYPERPLANE,
    description="Classifier-based steering using logistic regression decision boundary. Works better than CAA when geometry is orthogonal (each pair has unique direction rather than shared direction).",
    method_class_path="wisent.core.steering_methods.methods.hyperplane.HyperplaneMethod",
    parameters=[
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the steering vector",
            action="store_true",
            cli_flag="--hyperplane-normalize",
        ),
        SteeringMethodParameter(
            name="max_iter",
            type=int,
            default=1000,
            help="Maximum iterations for logistic regression",
            cli_flag="--hyperplane-max-iter",
        ),
        SteeringMethodParameter(
            name="C",
            type=float,
            default=1.0,
            help="Regularization strength (inverse). Smaller values = stronger regularization.",
            cli_flag="--hyperplane-C",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 5.0),
        "default_strength": 1.0,
    },
    default_strength=1.0,
    strength_range=(0.1, 5.0),
)


PRISM_DEFINITION = SteeringMethodDefinition(
    name="prism",
    method_type=SteeringMethodType.PRISM,
    description="PRISM - Projected Representations for Independent Steering Manifolds. Gradient-optimized multi-directional steering.",
    method_class_path="wisent.core.steering_methods.methods.prism.PRISMMethod",
    parameters=[
        SteeringMethodParameter(
            name="num_directions",
            type=int,
            default=3,
            help="Number of directions to discover per layer",
            cli_flag="--prism-num-directions",
        ),
        SteeringMethodParameter(
            name="optimization_steps",
            type=int,
            default=100,
            help="Number of gradient descent steps for direction optimization",
            cli_flag="--prism-optimization-steps",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            default=0.01,
            help="Learning rate for direction optimization",
            cli_flag="--prism-learning-rate",
        ),
        SteeringMethodParameter(
            name="retain_weight",
            type=float,
            default=0.1,
            help="Weight for retain loss (preserving behavior on harmless examples)",
            cli_flag="--prism-retain-weight",
        ),
        SteeringMethodParameter(
            name="independence_weight",
            type=float,
            default=0.05,
            help="Weight for representational independence loss between directions",
            cli_flag="--prism-independence-weight",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize the final directions",
            action="store_true",
            cli_flag="--prism-normalize",
        ),
        SteeringMethodParameter(
            name="use_caa_init",
            type=bool,
            default=True,
            help="Initialize first direction using CAA (difference-in-means)",
            action="store_true",
            cli_flag="--prism-use-caa-init",
        ),
        SteeringMethodParameter(
            name="cone_constraint",
            type=bool,
            default=True,
            help="Constrain directions to form a polyhedral cone",
            action="store_true",
            cli_flag="--prism-cone-constraint",
        ),
        SteeringMethodParameter(
            name="min_cosine_similarity",
            type=float,
            default=0.3,
            help="Minimum cosine similarity between directions",
            cli_flag="--prism-min-cosine-similarity",
        ),
        SteeringMethodParameter(
            name="max_cosine_similarity",
            type=float,
            default=0.95,
            help="Maximum cosine similarity between directions (avoid redundancy)",
            cli_flag="--prism-max-cosine-similarity",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 3.0),
        "default_strength": 1.0,
        "num_directions_range": (1, 7),
    },
    default_strength=1.0,
    strength_range=(0.1, 3.0),
)


PULSE_DEFINITION = SteeringMethodDefinition(
    name="pulse",
    method_type=SteeringMethodType.PULSE,
    description="PULSE - Probabilistic Uncertainty-guided Layer Steering Engine. Layer-adaptive conditional steering with uncertainty-guided intensity.",
    method_class_path="wisent.core.steering_methods.methods.pulse.PULSEMethod",
    parameters=[
        SteeringMethodParameter(
            name="sensor_layer",
            type=int,
            default=None,
            help="Layer index where condition gating is computed (auto-computed if not set)",
            cli_flag="--pulse-sensor-layer",
        ),
        SteeringMethodParameter(
            name="steering_layers",
            type=str,
            default=None,
            help="Comma-separated layer indices where steering is applied (auto-computed if not set)",
            cli_flag="--pulse-steering-layers",
        ),
        SteeringMethodParameter(
            name="per_layer_scaling",
            type=bool,
            default=True,
            help="Learn different scaling per layer",
            action="store_true",
            cli_flag="--pulse-per-layer-scaling",
        ),
        SteeringMethodParameter(
            name="condition_threshold",
            type=float,
            default=0.5,
            help="Threshold for condition activation (0-1)",
            cli_flag="--pulse-condition-threshold",
        ),
        SteeringMethodParameter(
            name="gate_temperature",
            type=float,
            default=0.1,
            help="Temperature for sigmoid gating (lower = sharper)",
            cli_flag="--pulse-gate-temperature",
        ),
        SteeringMethodParameter(
            name="learn_threshold",
            type=bool,
            default=True,
            help="Learn optimal threshold via grid search",
            action="store_true",
            cli_flag="--pulse-learn-threshold",
        ),
        SteeringMethodParameter(
            name="use_entropy_scaling",
            type=bool,
            default=True,
            help="Enable entropy-based intensity modulation",
            action="store_true",
            cli_flag="--pulse-use-entropy-scaling",
        ),
        SteeringMethodParameter(
            name="entropy_floor",
            type=float,
            default=0.5,
            help="Minimum entropy to trigger scaling",
            cli_flag="--pulse-entropy-floor",
        ),
        SteeringMethodParameter(
            name="entropy_ceiling",
            type=float,
            default=2.0,
            help="Entropy at which max_alpha is reached",
            cli_flag="--pulse-entropy-ceiling",
        ),
        SteeringMethodParameter(
            name="max_alpha",
            type=float,
            default=2.0,
            help="Maximum steering strength",
            cli_flag="--pulse-max-alpha",
        ),
        SteeringMethodParameter(
            name="optimization_steps",
            type=int,
            default=100,
            help="Steps for condition vector optimization",
            cli_flag="--pulse-optimization-steps",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            default=0.01,
            help="Learning rate for optimization",
            cli_flag="--pulse-learning-rate",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize vectors",
            action="store_true",
            cli_flag="--pulse-normalize",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 3.0),
        "default_strength": 1.0,
        "threshold_search_range": (0.0, 1.0),
    },
    default_strength=1.0,
    strength_range=(0.1, 3.0),
)


TITAN_DEFINITION = SteeringMethodDefinition(
    name="titan",
    method_type=SteeringMethodType.TITAN,
    description="TITAN - Total Integrated Targeted Activation Navigation. Joint optimization of manifold, gating, and intensity.",
    method_class_path="wisent.core.steering_methods.methods.titan.TITANMethod",
    parameters=[
        SteeringMethodParameter(
            name="num_directions",
            type=int,
            default=5,
            help="Number of directions per layer in the steering manifold",
            cli_flag="--titan-num-directions",
        ),
        SteeringMethodParameter(
            name="steering_layers",
            type=str,
            default=None,
            help="Comma-separated layer indices for steering (auto-computed if not set)",
            cli_flag="--titan-steering-layers",
        ),
        SteeringMethodParameter(
            name="sensor_layer",
            type=int,
            default=None,
            help="Primary layer for gating decisions (auto-computed if not set)",
            cli_flag="--titan-sensor-layer",
        ),
        SteeringMethodParameter(
            name="gate_hidden_dim",
            type=int,
            default=None,
            help="Hidden dimension for gating network (auto-computed as hidden_dim//16 if not set)",
            cli_flag="--titan-gate-hidden-dim",
        ),
        SteeringMethodParameter(
            name="intensity_hidden_dim",
            type=int,
            default=None,
            help="Hidden dimension for intensity network (auto-computed as hidden_dim//32 if not set)",
            cli_flag="--titan-intensity-hidden-dim",
        ),
        SteeringMethodParameter(
            name="optimization_steps",
            type=int,
            default=200,
            help="Total optimization steps",
            cli_flag="--titan-optimization-steps",
        ),
        SteeringMethodParameter(
            name="learning_rate",
            type=float,
            default=0.005,
            help="Learning rate for all components",
            cli_flag="--titan-learning-rate",
        ),
        SteeringMethodParameter(
            name="behavior_weight",
            type=float,
            default=1.0,
            help="Weight for behavior effectiveness loss",
            cli_flag="--titan-behavior-weight",
        ),
        SteeringMethodParameter(
            name="retain_weight",
            type=float,
            default=0.2,
            help="Weight for retain loss (minimize side effects)",
            cli_flag="--titan-retain-weight",
        ),
        SteeringMethodParameter(
            name="sparse_weight",
            type=float,
            default=0.05,
            help="Weight for sparsity loss",
            cli_flag="--titan-sparse-weight",
        ),
        SteeringMethodParameter(
            name="max_alpha",
            type=float,
            default=3.0,
            help="Maximum steering intensity",
            cli_flag="--titan-max-alpha",
        ),
        SteeringMethodParameter(
            name="normalize",
            type=bool,
            default=True,
            help="L2-normalize directions",
            action="store_true",
            cli_flag="--titan-normalize",
        ),
    ],
    optimization_config={
        "strength_search_range": (0.1, 5.0),
        "default_strength": 1.0,
        "num_directions_range": (3, 10),
    },
    default_strength=1.0,
    strength_range=(0.1, 5.0),
)


# =============================================================================
# REGISTRY CLASS
# =============================================================================

class SteeringMethodRegistry:
    """
    Central registry for all steering methods.
    
    Usage:
        # Get all registered methods
        methods = SteeringMethodRegistry.list_methods()
        
        # Get a specific method definition
        caa = SteeringMethodRegistry.get("caa")
        
        # Add CLI arguments for all methods
        SteeringMethodRegistry.add_all_cli_arguments(parser)
        
        # Get method class
        method_class = SteeringMethodRegistry.get_method_class("caa")
    """
    
    _REGISTRY: Dict[str, SteeringMethodDefinition] = {
        "caa": CAA_DEFINITION,
        "hyperplane": HYPERPLANE_DEFINITION,
        "prism": PRISM_DEFINITION,
        "pulse": PULSE_DEFINITION,
        "titan": TITAN_DEFINITION,
    }
    
    @classmethod
    def register(cls, definition: SteeringMethodDefinition) -> None:
        """Register a new steering method definition."""
        cls._REGISTRY[definition.name] = definition
    
    @classmethod
    def get(cls, name: str) -> SteeringMethodDefinition:
        """Get a steering method definition by name."""
        name_lower = name.lower()
        if name_lower not in cls._REGISTRY:
            available = ", ".join(cls._REGISTRY.keys())
            raise ValueError(f"Unknown steering method: {name}. Available: {available}")
        return cls._REGISTRY[name_lower]
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all registered method names."""
        return list(cls._REGISTRY.keys())
    
    @classmethod
    def list_definitions(cls) -> List[SteeringMethodDefinition]:
        """List all registered method definitions."""
        return list(cls._REGISTRY.values())
    
    @classmethod
    def get_method_class(cls, name: str) -> Type[BaseSteeringMethod]:
        """Get the method class for a given method name."""
        return cls.get(name).get_method_class()
    
    @classmethod
    def get_default_method(cls) -> str:
        """Get the default steering method name."""
        return "caa"
    
    @classmethod
    def add_all_cli_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for all registered steering methods."""
        # Add method selection argument
        parser.add_argument(
            "--steering-method",
            type=str,
            default=cls.get_default_method(),
            choices=[m.upper() for m in cls.list_methods()] + [m.lower() for m in cls.list_methods()],
            help=f"Steering method to use (default: {cls.get_default_method().upper()})",
        )
        
        # Add method-specific arguments
        for definition in cls.list_definitions():
            definition.add_cli_arguments(parser)
    
    @classmethod
    def extract_method_params(cls, method_name: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Extract method-specific parameters from parsed CLI args."""
        return cls.get(method_name).extract_params_from_args(args)
    
    @classmethod
    def create_method_instance(cls, name: str, **kwargs) -> BaseSteeringMethod:
        """Create an instance of a steering method with given parameters."""
        definition = cls.get(name)
        method_class = definition.get_method_class()
        
        # Merge default params with provided kwargs
        params = definition.get_default_params()
        params.update(kwargs)
        
        return method_class(**params)
    
    @classmethod
    def get_optimization_config(cls, name: str) -> Dict[str, Any]:
        """Get optimization configuration for a method."""
        return cls.get(name).optimization_config
    
    @classmethod
    def get_strength_range(cls, name: str) -> tuple:
        """Get the strength search range for a method."""
        return cls.get(name).strength_range
    
    @classmethod
    def get_default_strength(cls, name: str) -> float:
        """Get the default strength for a method."""
        return cls.get(name).default_strength
    
    @classmethod
    def validate_method(cls, name: str) -> bool:
        """Check if a method name is valid."""
        return name.lower() in cls._REGISTRY
    
    @classmethod
    def get_method_info(cls) -> List[Dict[str, Any]]:
        """Get detailed info about all methods for documentation/help."""
        return [
            {
                "name": d.name,
                "description": d.description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.type.__name__,
                        "default": p.default,
                        "help": p.help,
                    }
                    for p in d.parameters
                ],
                "default_strength": d.default_strength,
                "strength_range": d.strength_range,
            }
            for d in cls.list_definitions()
        ]


# Convenience function for backward compatibility
def get_steering_method(name: str, **kwargs) -> BaseSteeringMethod:
    """Create a steering method instance by name."""
    return SteeringMethodRegistry.create_method_instance(name, **kwargs)


def list_steering_methods() -> List[str]:
    """List all available steering methods."""
    return SteeringMethodRegistry.list_methods()


def is_valid_steering_method(name: str) -> bool:
    """Check if a steering method name is valid."""
    return SteeringMethodRegistry.validate_method(name)
