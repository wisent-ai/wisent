"""Evaluation metrics and prompt management."""
import json
import re
from pathlib import Path
from typing import Dict, Optional, List
from abc import ABC, abstractmethod


# ============================================================================
# JSON Extraction Utilities
# ============================================================================

def extract_json_from_response(response_text: str, expected_keys: Optional[List[str]] = None) -> Optional[Dict]:
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


def extract_score(response_text: str, score_key: str) -> Optional[float]:
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
                # Remove any placeholder markers
                score = score.strip().replace('<', '').replace('>', '')
                try:
                    number_match = re.search(r'\d+(?:\.\d+)?', score)
                    if number_match:
                        value = number_match.group(0)
                        return float(value) if '.' in value else int(value)
                except ValueError:
                    pass

        # Fallback: try to extract from key-value pattern in text
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

    except Exception:
        pass

    return None


def extract_traits(response_text: str) -> Optional[List[str]]:
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
        pattern = r'"traits"\s*:\s*\[(.*?)\]'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            traits_str = match.group(1)
            # Extract quoted strings
            traits = re.findall(r'"([^"]+)"', traits_str)
            if traits:
                return traits

    except Exception:
        pass

    return None


def extract_choice(response_text: str) -> Optional[str]:
    """
    Extract choice result from model response.

    Args:
        response_text: The text response from the model

    Returns:
        Choice string ('A', 'B', or 'EQUAL'), or None if extraction fails
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
            return 'EQUAL'

    except Exception:
        pass

    return None


def extract_explanation(response_text: str) -> Optional[str]:
    """
    Extract explanation text from judge response.

    Args:
        response_text: The text response from the judge

    Returns:
        Explanation string, or None if extraction fails
    """
    try:
        # Try to extract JSON with explanation key
        json_obj = extract_json_from_response(response_text, expected_keys=["explanation"])
        if json_obj and "explanation" in json_obj:
            explanation = json_obj["explanation"]
            if isinstance(explanation, str):
                return explanation.strip()

        # Fallback: look for explanation pattern in text
        pattern = r'"explanation"\s*:\s*"([^"]+)"'
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # If no explanation key found, return the full response as explanation
        cleaned = response_text.strip()
        cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)

        if cleaned:
            return cleaned

    except Exception:
        pass

    return None


# ============================================================================
# Metric Classes
# ============================================================================


class Metric(ABC):
    """Base class for evaluation metrics."""

    def __init__(self, name: str, prompt_template_path: Path):
        """
        Initialize metric.

        Args:
            name: Metric name
            prompt_template_path: Path to prompt template (without extension)
        """
        self.name = name
        self.prompt_template_path = prompt_template_path
        self._templates: Dict[str, str] = {}

    def load_template(self, format_type: str, format_extensions: Dict[str, str]) -> str:
        """
        Load prompt template for specific format.

        Args:
            format_type: Format type (txt, markdown, json)
            format_extensions: Mapping of format to file extensions

        Returns:
            Loaded template string
        """
        if format_type not in self._templates:
            template_path = f"{self.prompt_template_path}{format_extensions[format_type]}"
            with open(template_path) as f:
                self._templates[format_type] = f.read().strip()
        return self._templates[format_type]

    def build_prompt(self, template: str, **kwargs) -> str:
        """
        Build prompt from template with placeholders.

        Args:
            template: Template string with <<<placeholder>>> markers
            **kwargs: Variables to replace

        Returns:
            Formatted prompt
        """
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"<<<{key}>>>", str(value))
        return result

    @abstractmethod
    def extract_result(self, response: str):
        """
        Extract result from judge response.

        Args:
            response: Judge response text

        Returns:
            Extracted result (type depends on metric)
        """
        pass

    def extract_explanation(self, response: str) -> Optional[str]:
        """
        Extract explanation from judge response.

        Args:
            response: Judge response text

        Returns:
            Extracted explanation string, or None if extraction fails
        """
        return extract_explanation(response)


class DifferentiationMetric(Metric):
    """Measures how different baseline and steered responses are."""

    def __init__(self, prompt_path: Path):
        super().__init__("differentiation", prompt_path)

    def extract_result(self, response: str) -> Optional[float]:
        """Extract differentiation score (0-100)."""
        return extract_score(response, "differentiation_score")


class CoherenceMetric(Metric):
    """Measures coherence and quality of steered response."""

    def __init__(self, prompt_path: Path):
        super().__init__("coherence", prompt_path)

    def extract_result(self, response: str) -> Optional[float]:
        """Extract coherence score (0-100)."""
        return extract_score(response, "coherence_score")


class TraitAlignmentMetric(Metric):
    """Measures alignment with target trait."""

    def __init__(self, prompt_path: Path):
        super().__init__("trait_alignment", prompt_path)

    def extract_result(self, response: str) -> Optional[float]:
        """Extract trait alignment score (0-100)."""
        return extract_score(response, "trait_alignment_score")


class OpenTraitsMetric(Metric):
    """Extracts open-ended traits from response."""

    def __init__(self, prompt_path: Path):
        super().__init__("open", prompt_path)

    def extract_result(self, response: str) -> Optional[list]:
        """Extract list of traits."""
        return extract_traits(response)


class ChoiceMetric(Metric):
    """Chooses which response better embodies the trait."""

    def __init__(self, prompt_path: Path):
        super().__init__("choose", prompt_path)

    def extract_result(self, response: str) -> Optional[str]:
        """Extract choice (A, B, or EQUAL)."""
        return extract_choice(response)


class MetricManager:
    """Manages all evaluation metrics."""

    def __init__(self, instruction_prompts: Dict[str, Path]):
        """
        Initialize metric manager.

        Args:
            instruction_prompts: Mapping of metric names to prompt paths
        """
        self.metrics = {
            "differentiation": DifferentiationMetric(instruction_prompts["differentiation"]),
            "coherence": CoherenceMetric(instruction_prompts["coherence"]),
            "trait_alignment": TraitAlignmentMetric(instruction_prompts["trait_alignment"]),
            "open": OpenTraitsMetric(instruction_prompts["open"]),
            "choose": ChoiceMetric(instruction_prompts["choose"])
        }

        self.format_extensions = {
            "txt": ".txt",
            "markdown": "_markdown.txt",
            "json": "_json.txt"
        }

    def get_metric(self, name: str) -> Metric:
        """Get metric by name."""
        if name not in self.metrics:
            raise ValueError(f"Metric '{name}' not found")
        return self.metrics[name]

    def build_all_prompts(
        self,
        format_type: str,
        question: str,
        baseline_response: str,
        steered_response: str
    ) -> Dict[str, str]:
        """
        Build prompts for all metrics.

        Args:
            format_type: Format type (txt, markdown, json)
            question: Question text
            baseline_response: Baseline response
            steered_response: Steered response

        Returns:
            Dictionary mapping metric names to prompts
        """
        # Define kwargs for each metric
        metric_configs = {
            "differentiation": {
                "question": question,
                "response_A": baseline_response,
                "response_B": steered_response
            },
            "coherence": {
                "question": question,
                "response": steered_response
            },
            "trait_alignment": {
                "question": question,
                "response": steered_response
            },
            "open": {
                "question": question,
                "response": steered_response
            },
            "choose": {
                "question": question,
                "response_A": baseline_response,
                "response_B": steered_response
            }
        }

        # Build prompts for all metrics
        prompts = {}
        for metric_name, kwargs in metric_configs.items():
            metric = self.get_metric(metric_name)
            template = metric.load_template(format_type, self.format_extensions)
            prompts[metric_name] = metric.build_prompt(template, **kwargs)

        return prompts
