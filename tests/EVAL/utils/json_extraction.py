import json
import re


def extract_json_from_response(response_text, expected_keys=None):
    """
    Extract JSON object from model response text.
    Handles markdown code blocks and various JSON formats.

    Args:
        response_text: The text response from the model
        expected_keys: Optional list of keys that should be in the JSON

    Returns:
        Parsed JSON object (dict), or None if extraction fails
    """
    # First, try to extract from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response_text, re.DOTALL)
    if code_block_match:
        try:
            json_str = code_block_match.group(1).strip()
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Try to find JSON object with balanced braces
    # Use a more robust approach that handles nested objects
    brace_stack = []
    start_idx = None

    for i, char in enumerate(response_text):
        if char == '{':
            if not brace_stack:  # First opening brace
                start_idx = i
            brace_stack.append(char)
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:  # Found matching closing brace
                    json_str = response_text[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        # If expected_keys provided, verify they exist
                        if expected_keys:
                            if all(key in parsed for key in expected_keys):
                                return parsed
                        else:
                            return parsed
                    except json.JSONDecodeError:
                        continue  # Try next JSON-like structure

    return None


def extract_score(response_text, score_key):
    """
    Extract a numeric score from model response.

    Args:
        response_text: The text response from the model
        score_key: The key name for the score (e.g., "differentiation_score")

    Returns:
        The score as a float/int, or None if extraction fails
    """
    try:
        # Try to extract JSON with the score key
        json_obj = extract_json_from_response(response_text, expected_keys=[score_key])
        if json_obj and score_key in json_obj:
            score = json_obj[score_key]
            # Convert to appropriate numeric type
            if isinstance(score, (int, float)):
                return score
            # Try to parse string as number
            if isinstance(score, str):
                # Remove any placeholder markers like "<0-100>" or angle brackets
                score = score.strip().replace('<', '').replace('>', '')
                try:
                    # Extract just the number if it contains non-numeric chars
                    number_match = re.search(r'\d+(?:\.\d+)?', score)
                    if number_match:
                        value = number_match.group(0)
                        return float(value) if '.' in value else int(value)
                except ValueError:
                    pass

        # Fallback: try to extract from key-value pattern in text
        # Look for patterns like "score_key": 85 or "score_key": "85"
        pattern = rf'"{score_key}"\s*:\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, response_text)
        if match:
            value = match.group(1)
            return float(value) if '.' in value else int(value)

        # Last resort: extract first number in the response
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response_text)
        if numbers:
            value = numbers[0]
            return float(value) if '.' in value else int(value)

    except Exception as e:
        print(f"Warning: Error extracting score for '{score_key}': {e}")
        print(f"Response excerpt: {response_text[:200]}...")

    return None


def extract_traits(response_text):
    """
    Extract list of traits from model response.

    Args:
        response_text: The text response from the model

    Returns:
        List of trait strings, or None if extraction fails
    """
    try:
        # Try to extract JSON with traits key
        json_obj = extract_json_from_response(response_text, expected_keys=["traits"])
        if json_obj and "traits" in json_obj:
            traits = json_obj["traits"]
            if isinstance(traits, list):
                return traits

        # Fallback: look for array pattern in text
        # Match patterns like "traits": ["trait1", "trait2", "trait3"]
        pattern = r'"traits"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            traits_str = match.group(1)
            # Extract quoted strings
            traits = re.findall(r'"([^"]+)"', traits_str)
            if traits:
                return traits

    except Exception as e:
        print(f"Warning: Error extracting traits: {e}")
        print(f"Response excerpt: {response_text[:200]}...")

    return None


def extract_choice(response_text):
    """
    Extract choice result from model response.

    Args:
        response_text: The text response from the model

    Returns:
        Choice string ('A', 'B', or 'equal'), or None if extraction fails
    """
    try:
        # Try to extract JSON with choice key
        json_obj = extract_json_from_response(response_text, expected_keys=["choice"])
        if json_obj and "choice" in json_obj:
            choice = json_obj["choice"]
            if isinstance(choice, str):
                choice = choice.strip().upper()
                if choice in ['A', 'B', 'EQUAL']:
                    return choice

        # Fallback: look for choice pattern in text
        pattern = r'"choice"\s*:\s*"([AaBb]|equal)"'
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            choice = match.group(1).strip().upper()
            if choice in ['A', 'B', 'EQUAL']:
                return choice

        # Last resort: look for explicit statements
        response_lower = response_text.lower()
        if 'response a' in response_lower and 'response b' not in response_lower:
            return 'A'
        elif 'response b' in response_lower and 'response a' not in response_lower:
            return 'B'
        elif 'equal' in response_lower or 'same' in response_lower:
            return 'equal'

    except Exception as e:
        print(f"Warning: Error extracting choice: {e}")
        print(f"Response excerpt: {response_text[:200]}...")

    return None
