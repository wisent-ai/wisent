"""
Task Relevance Selection for Wisent Guard.

This module provides functionality to select the most relevant tasks from the
lm-evaluation-harness library based on a user query or issue type.

Uses model-driven decisions instead of hardcoded patterns.
"""

from typing import List, Dict, Set, Tuple
from .task_manager import get_available_tasks
from wisent.core.utils.infra_tools.errors import MissingParameterError


class TaskRelevanceSelector:
    """Selects tasks based on model-driven relevance analysis."""

    def __init__(self, model, layer: int, task_search_limit: int, max_results: int = None, min_relevance_score: float = None):
        if max_results is None:
            raise ValueError("max_results is required")
        if min_relevance_score is None:
            raise ValueError("min_relevance_score is required")
        self.model = model
        self.layer = layer
        self._task_search_limit = task_search_limit
        self._max_results = max_results
        self._min_relevance_score = min_relevance_score
        
    def find_relevant_tasks(
        self,
        query: str,
        max_results: int = None,
        min_relevance_score: float = None
    ) -> List[Tuple[str, float]]:
        """
        Find tasks most relevant to the given query using model decisions.
        
        Args:
            query: The search query (e.g., "hallucination detection", "bias", "truthfulness")
            max_results: Maximum number of tasks to return
            min_relevance_score: Minimum relevance score threshold (0.0 to 1.0)
            
        Returns:
            List of (task_name, relevance_score) tuples, sorted by relevance
        """
        if max_results is None:
            max_results = self._max_results
        if min_relevance_score is None:
            min_relevance_score = self._min_relevance_score
        available_tasks = get_available_tasks()
        task_scores = []
        for task_name in available_tasks[:self._task_search_limit]:
            score = self._get_model_relevance_score(query, task_name)
            if score >= min_relevance_score:
                task_scores.append((task_name, score))
        
        # Sort by relevance score (descending)
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        return task_scores[:max_results]
    
    def _get_model_relevance_score(self, query: str, task_name: str) -> float:
        """Get relevance score from the model."""
        prompt = f"""Rate the relevance of this task for the given query.
        
Query: {query}
Task: {task_name}

Rate relevance from 0.0 to 1.0 (1.0 = highly relevant, 0.0 = not relevant).
Respond with only the number:"""
        
        try:
            response = self.model.generate(prompt, layer_index=self.layer)
            score_str = response.strip()
            
            # Extract number from response
            import re
            match = re.search(r'(\d+\.?\d*)', score_str)
            if match:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))  # Clamp to [0,1]
            return 0.0
        except:
            return 0.0


def find_relevant_tasks(
    query: str, task_search_limit: int,
    max_results: int = None, min_relevance_score: float = None,
    model=None, layer: int = None,
) -> List[Tuple[str, float]]:
    """Standalone function for task relevance selection."""
    if model is None:
        from ....model import Model
        model = Model("meta-llama/Llama-3.1-8B-Instruct")
    if layer is None:
        raise MissingParameterError(params=["layer"], context="find_relevant_tasks")
    selector = TaskRelevanceSelector(model, layer=layer, task_search_limit=task_search_limit, max_results=max_results, min_relevance_score=min_relevance_score)
    return selector.find_relevant_tasks(query, max_results, min_relevance_score)


def get_top_relevant_tasks(query: str, count: int, task_search_limit: int, max_results: int = None, min_relevance_score: float = None, model=None, layer: int = None) -> List[str]:
    """Get top N relevant tasks for a query."""
    results = find_relevant_tasks(query, task_search_limit=task_search_limit, max_results=count, min_relevance_score=min_relevance_score, model=model, layer=layer)
    return [task_name for task_name, _ in results]
