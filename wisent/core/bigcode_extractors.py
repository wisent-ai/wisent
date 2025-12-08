"""BigCode extractors stub module.

This provides stub functions for bigcode extractor functionality.
"""

from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.mbpp import MBPPExtractor


def get_bigcode_extractor(task_name: str):
    """Get a bigcode extractor for the given task.
    
    Args:
        task_name: Name of the task (e.g., "mbpp", "humaneval")
        
    Returns:
        An extractor instance for the task.
    """
    extractors = {
        "mbpp": MBPPExtractor,
    }
    
    extractor_class = extractors.get(task_name.lower())
    if extractor_class is None:
        raise ValueError(f"Unknown bigcode task: {task_name}")
    
    return extractor_class()
