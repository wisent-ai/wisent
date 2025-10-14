def build_prompt_from_template(template, format_type, **kwargs):
    """
    Build a prompt from a template by replacing placeholders.

    Args:
        template: String template with <<<placeholder>>> markers
        format_type: The format type ('txt', 'markdown', or 'json') - used for logging only
        **kwargs: Variables to replace in the template (question, response, response_a, response_b)

    Returns:
        str: The formatted prompt string with placeholders replaced
    """
    result = template
    for key, value in kwargs.items():
        result = result.replace(f"<<<{key}>>>", value)
    return result
