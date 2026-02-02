"""Layer representation for model extraction."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Layer:
    """Represents a model layer for activation extraction."""
    
    index: int
    type: str = "transformer"
    name: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = f"layer_{self.index}"
