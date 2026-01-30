"""Contrastive pairs generation for steering optimization - wraps wisent generate-pairs-from-task CLI."""

from wisent.core.cli.generate_pairs_from_task import execute_generate_pairs_from_task

__all__ = ["execute_generate_pairs_from_task"]
