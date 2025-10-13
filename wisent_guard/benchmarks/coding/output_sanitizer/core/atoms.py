# coding/llm_sanitizer/core/atoms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Protocol, Literal, Optional

Language = Literal["python", "cpp", "java"]

@dataclass(frozen=True)
class TaskSchema:
    """What the sandbox expects for this task."""
    language: Language
    file_name: str               # e.g., "solution.py" | "solution.cpp" | "Solution.java"
    entry_point: str             # function/method name tests will call (e.g., "add", "solve")
    java_class: str = "Solution" # only for Java; expected public class name
    # Optional hints:
    allow_wrapper: bool = True   # may synthesize thin wrapper instead of renaming
    prefer_rename: bool = False  # if True and safe, rename single top-level function to entry_point

@dataclass(frozen=True)
class NormalizeResult:
    files: Dict[str, str]        # filename -> normalized source
    notes: str                   # human-readable log of what was done
    ok: bool                     # True if we think it’s valid / parseable

class CodeStandardizer(Protocol):
    language: Language
    def normalize(self, raw: str, schema: TaskSchema) -> NormalizeResult: ...
