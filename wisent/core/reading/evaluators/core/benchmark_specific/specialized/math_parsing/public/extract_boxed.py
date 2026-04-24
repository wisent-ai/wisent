"""Extract boxed answers from LaTeX text.

Common utility for extracting answers from \\boxed{} notation used in
math competition problems and solutions.
"""

import re


def extract_boxed_answer(text: str) -> str | None:
    """Extract the LAST \\boxed{} answer from text (final answer convention).

    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}}).

    Args:
        text: The text containing \\boxed{answer}

    Returns:
        The extracted answer from the last \\boxed{} or None if not found
    """
    # Find all \boxed{ occurrences
    start_pattern = r'\\boxed\{'
    matches = list(re.finditer(start_pattern, text))

    if not matches:
        return None

    # Process the LAST match (final answer convention)
    last_match = matches[-1]

    # Start after \boxed{
    start_idx = last_match.end()
    brace_count = 1
    idx = start_idx

    # Find the matching closing brace
    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        # Extract content between the braces
        return text[start_idx:idx-1].strip()

    return None
