# coding/providers/core/atoms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Protocol, Literal

Language = Literal["python", "cpp", "java"]

@dataclass(frozen=True)
class CodingTask:
    """A normalized task with language + harness files to be executed."""
    language: Language
    files: Dict[str, str]          # e.g., {"solution.py": "...", "tests.py": "..."} or C++/Java equivalents
    options: Dict[str, object]     # e.g., {"cxx_std": "c++20", "java_main": "MainTest"}

class Provider(Protocol):
    """Dataset provider yields tasks (codegen or self-repair compatible)."""
    name: str
    def iter_tasks(self, split: str = "test") -> Iterable[CodingTask]: ...
