from __future__ import annotations
from dataclasses import dataclass
from typing import  Iterable, Protocol

@dataclass(frozen=True)
class SampleOutcome:
    """
    Result of executing a single sample (possibly after self-repair).

    attributes:
        task_id: 
            The unique identifier for the task.
        status:
            One of "ok", "compile_error", "runtime_error", or "timeout".
        passed:
            True if the code passed all tests, False otherwise.
        elapsed:
            Time taken to execute the code in seconds.
    """
    task_id: str
    status: str         
    passed: bool
    elapsed: float

class Metric(Protocol):
    """
    Metric computes a score from an iterable of SampleOutcome.
    """
    def compute(self, outcomes: Iterable[SampleOutcome]) -> float: ...


class Evaluator(Protocol):
    """
    Runs tasks end-to-end (codegen + optional self-repair) and yields SampleOutcome.
    """
    def evaluate(self) -> Iterable[SampleOutcome]: ...