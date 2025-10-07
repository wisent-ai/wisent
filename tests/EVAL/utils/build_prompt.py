import json
import re
import copy


def build_prompt_from_template(template, format_type, **kwargs):
    """
    Build a prompt from a template based on the format type.

    Args:
        template: Either a string (for txt/markdown) or dict (for json)
        format_type: The format type ('txt', 'markdown', or 'json')
        **kwargs: Variables to format into the template (question, response, response_a, response_b)

    Returns:
        str: The formatted prompt string
    """
    if format_type == "json":
        # For JSON: fill in prompt_template and return as JSON string
        filled = copy.deepcopy(template)
        if 'prompt_template' in filled:
            for key in filled['prompt_template']:
                if key in kwargs:
                    filled['prompt_template'][key] = kwargs[key]
        return json.dumps(filled, indent=2, ensure_ascii=False)
    else:
        # For txt/markdown format, we need to handle curly braces in the template
        # Find all placeholders like {question}, {response}, etc.
        placeholders = re.findall(r'\{(\w+)\}', template)

        # Temporarily replace known placeholders with a unique marker
        temp_template = template
        temp_markers = {}
        for i, placeholder in enumerate(set(placeholders)):
            if placeholder in kwargs:
                marker = f"<<<PLACEHOLDER_{i}>>>"
                temp_markers[marker] = placeholder
                temp_template = temp_template.replace(f"{{{placeholder}}}", marker)

        # Escape remaining curly braces
        temp_template = temp_template.replace("{", "{{").replace("}", "}}")

        # Restore placeholders
        for marker, placeholder in temp_markers.items():
            temp_template = temp_template.replace(marker, f"{{{placeholder}}}")

        # Now use format
        return temp_template.format(**kwargs)
