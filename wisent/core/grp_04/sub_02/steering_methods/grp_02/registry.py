"""
Centralized Steering Method Registry.

Single source of truth for all steering method definitions.
See _definitions_part1/2/3.py for method-specific definitions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

from wisent.core.steering_methods.core.atoms import BaseSteeringMethod
from wisent.core.constants import DEFAULT_STRENGTH, STEERING_STRENGTH_RANGE_WIDE


class SteeringMethodType(Enum):
    """Enumeration of all supported steering methods."""
    CAA = "caa"
    OSTRZE = "ostrze"
    MLP = "mlp"
    TECZA = "tecza"
    TETNO = "tetno"
    GROM = "grom"
    NURT = "nurt"
    SZLAK = "szlak"
    WICHER = "wicher"
    PRZELOM = "przelom"


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
    default_strength: float = DEFAULT_STRENGTH
    strength_range: tuple = STEERING_STRENGTH_RANGE_WIDE
    
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
# STEERING METHOD DEFINITIONS (imported from sibling modules)
# =============================================================================
from wisent.core.steering_methods._definitions_part1 import (
    CAA_DEFINITION,
    OSTRZE_DEFINITION,
    TECZA_DEFINITION,
)
from wisent.core.steering_methods._definitions_part2 import (
    TETNO_DEFINITION,
    GROM_DEFINITION,
    PRZELOM_DEFINITION,
)
from wisent.core.steering_methods._definitions_part3 import (
    MLP_DEFINITION,
    NURT_DEFINITION,
    SZLAK_DEFINITION,
    WICHER_DEFINITION,
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
        "ostrze": OSTRZE_DEFINITION,
        "mlp": MLP_DEFINITION,
        "tecza": TECZA_DEFINITION,
        "tetno": TETNO_DEFINITION,
        "grom": GROM_DEFINITION,
        "nurt": NURT_DEFINITION,
        "szlak": SZLAK_DEFINITION,
        "wicher": WICHER_DEFINITION,
        "przelom": PRZELOM_DEFINITION,
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
