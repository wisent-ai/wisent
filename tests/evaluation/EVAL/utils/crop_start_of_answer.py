def crop_to_answer(response_text):
    """
    Crops the model response to remove the system prompt and user question,
    returning only the assistant's actual answer.

    The format is:
    system\n\nCutting Knowledge Date: ...\n\nuser\n\n<question>assistant\n\n<answer>

    We want to extract only <answer>.

    Args:
        response_text: The full response text including system prompt

    Returns:
        str: Only the assistant's answer, or original text if pattern not found
    """
    # Look for "assistant\n\n" which marks the start of the actual answer
    marker = "assistant\n\n"

    if marker in response_text:
        # Find the position after "assistant\n\n"
        start_pos = response_text.find(marker) + len(marker)
        return response_text[start_pos:]

    # Fallback: if the marker isn't found, return original text
    return response_text


def crop_responses_in_data(data):
    """
    Crops baseline_response and steered_response in a list of result dictionaries.

    Args:
        data: List of dictionaries with 'baseline_response' and 'steered_response' keys

    Returns:
        List of dictionaries with cropped responses
    """
    for item in data:
        if 'baseline_response' in item:
            item['baseline_response'] = crop_to_answer(item['baseline_response'])
        if 'steered_response' in item:
            item['steered_response'] = crop_to_answer(item['steered_response'])

    return data
