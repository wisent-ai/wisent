from __future__ import annotations

from abc import ABC, abstractmethod

class CleanStep(ABC):
    """
    Single step in a cleaning pipeline.

    attributes:
        name:
            Name of the step, used in stats and logging.

    methods:
        stats():
            Return a dict of statistics about the last run of `apply()`.
        apply(items):
            Apply the cleaning step to a list of items. 
    """
    name: str = "step"

    def stats(self) -> dict[str, int | float | str]:
        return {}

    @abstractmethod
    def apply(self, items: list[dict[str, str]]) -> list[dict[str, str]]:
        ...

class Cleaner(ABC):
    """
    Cleaning pipeline composed of multiple `CleanStep`s.
    """
    @abstractmethod
    def clean(
        self, items: list[dict[str, str]]
    ) -> tuple[list[dict[str, str]], dict[str, int | float | str]]:
        ...
