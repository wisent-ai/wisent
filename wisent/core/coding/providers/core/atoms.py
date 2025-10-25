from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Protocol, Literal

Language = Literal["python", "cpp", "java"]

@dataclass(frozen=True)
class CodingTask:
    """
    A normalized task with language + harness files to be executed.
    
    attributes:
        language:
            The programming language of the task.
        files:
            A dictionary mapping filenames to their content. For example,
            {"solution.py": "...", "tests.py": "..."} for Python tasks,
            or equivalent files for C++/Java tasks.
        options:
            A dictionary of additional options that may be required for
            execution. For example, {"cxx_std": "c++20"} for C++ tasks,
            or {"java_main": "MainTest"} for Java tasks.
    """
    language: Language
    files: dict[str, str]          
    options: dict[str, object]

class Provider(Protocol):
    """Dataset provider yields tasks (codegen or self-repair compatible)."""
    name: str
    def iter_tasks(self, split: str = "test") -> Iterable[CodingTask]: ...