
from abc import ABC, abstractmethod

__all__ = ["DB_Instructions"]

class DB_Instructions(ABC):
    """
    Interface for database instructions storage and retrieval.

    methods:
        get(key: str) -> str:
            Retrieve instructions by key.
        set(key: str, value: str) -> None:
            Store instructions by key.

        notes:

        This is an abstract base class for the database instructions for the system prompts.
        System prompts need to describe the what kind of contrastive pairs we want to generate.
        or for example instructions for fixing negative examples.
    """
    @abstractmethod
    def get(self, key: str) -> str: ...
    @abstractmethod
    def set(self, key: str, value: str) -> None: ...