"""Extracted from model_persistence.py - create_steering_vector_metadata tail."""


def complete_steering_vector_metadata(metadata, kwargs):
    """Complete the steering vector metadata by adding extra kwargs and returning it.

    This function finishes the metadata dictionary construction started by
    create_steering_vector_metadata by merging any additional keyword arguments
    and returning the completed metadata dictionary.

    Args:
        metadata: Partially constructed metadata dict
        kwargs: Additional metadata fields to merge

    Returns:
        Completed metadata dictionary
    """
    metadata.update(kwargs)
    return metadata
