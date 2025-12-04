from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger
import json

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["BFCLExtractor"]

log = setup_logger(__name__)

# BFCL categories
BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
    "relevance",
    "multi_turn_base",
    "multi_turn_augmented",
]


class BFCLExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Berkeley Function Calling Leaderboard (BFCL).

    BFCL is the first comprehensive evaluation of LLM's ability to call functions
    and tools, built to be representative of most users' function calling use-cases.

    Categories:
    - Simple: Single function evaluation with basic format
    - Multiple: One function call from 2-4 available functions
    - Parallel: Multiple function calls in parallel
    - Parallel Multiple: Multiple function calls from multiple functions
    - Irrelevance: No function should be called
    - Multi-turn: Conversational function calling

    Schema (AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix):
        - id: str (unique identifier)
        - question: list[dict] (user queries with role/content)
        - function: list[dict] (function definitions with parameters)
        - split: str (dataset split)

    Note: The original gorilla-llm/Berkeley-Function-Calling-Leaderboard dataset
    is not compatible with HuggingFace load_dataset, so we use the fixed version.
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "function_calling"

    def __init__(self, category: str | None = None):
        """
        Initialize BFCL extractor.

        Args:
            category: Optional filter for specific category
        """
        super().__init__()
        self.category = category

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from BFCL examples.

        For function calling:
        - Positive (correct) = Proper function call with correct parameters
        - Negative (incorrect) = Wrong function or incorrect parameters

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        pairs: list[ContrastivePair] = []

        # Try the fixed version first
        try:
            docs = self.load_dataset(
                dataset_name="AndyChen123/Berkeley-Function-Calling-Leaderboard-Fix",
                split="train",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from BFCL (fixed version)")
        except Exception as e:
            log.warning(f"Failed to load fixed BFCL dataset: {e}")
            # Try loading from the original with specific config
            try:
                docs = self.load_dataset(
                    dataset_name="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                    config="BFCL_v3_exec_simple",
                    split="train",
                    limit=max_items,
                    trust_remote_code=True,
                )
                log.info(f"Loaded {len(docs)} examples from original BFCL")
            except Exception as e2:
                log.error(f"Failed to load any BFCL dataset: {e2}")
                return []

        for doc in docs:
            # Filter by category if specified
            if self.category:
                doc_split = doc.get("split", "").lower()
                if self.category.lower() not in doc_split:
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid BFCL pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            doc_id = doc.get("id", "")
            question_data = doc.get("question", [])
            function_data = doc.get("function", [])
            split = doc.get("split", "")

            # Extract the user question
            user_question = self._extract_question(question_data)
            if not user_question:
                log.debug("Skipping: missing question")
                return None

            # Extract function definitions
            functions = self._extract_functions(function_data)
            if not functions:
                log.debug("Skipping: missing function definitions")
                return None

            # Build the prompt with function context
            prompt = self._build_function_call_prompt(user_question, functions)

            # Correct answer: Proper function call (we simulate expected output)
            correct_answer = self._create_correct_function_call(user_question, functions)

            # Incorrect answer: Wrong function call
            incorrect_answer = self._create_incorrect_function_call(functions)

            metadata = {
                "label": "bfcl",
                "source": "Berkeley-Function-Calling-Leaderboard",
                "id": doc_id,
                "split": split,
                "num_functions": len(functions),
                "is_function_calling_benchmark": True,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _extract_question(self, question_data: list | dict | str) -> str:
        """Extract the user question from various formats."""
        if isinstance(question_data, str):
            return question_data.strip()
        elif isinstance(question_data, list):
            # Format: [{"role": "user", "content": "..."}]
            for item in question_data:
                if isinstance(item, dict):
                    if item.get("role") == "user":
                        return item.get("content", "").strip()
                    # Sometimes content is nested
                    content = item.get("content", "")
                    if content:
                        return str(content).strip()
            # If no user role found, try first item
            if question_data and isinstance(question_data[0], dict):
                return question_data[0].get("content", "").strip()
        elif isinstance(question_data, dict):
            return question_data.get("content", "").strip()
        return ""

    def _extract_functions(self, function_data: list | dict) -> list[dict]:
        """Extract function definitions."""
        if isinstance(function_data, list):
            return [f for f in function_data if isinstance(f, dict)]
        elif isinstance(function_data, dict):
            return [function_data]
        return []

    def _build_function_call_prompt(self, question: str, functions: list[dict]) -> str:
        """Build a prompt that includes function definitions."""
        function_docs = []
        for func in functions:
            name = func.get("name", "unknown")
            description = func.get("description", "")
            parameters = func.get("parameters", {})

            func_doc = f"Function: {name}\nDescription: {description}"
            if parameters:
                props = parameters.get("properties", {})
                required = parameters.get("required", [])
                if props:
                    params_str = ", ".join(
                        f"{k} ({'required' if k in required else 'optional'})"
                        for k in props.keys()
                    )
                    func_doc += f"\nParameters: {params_str}"
            function_docs.append(func_doc)

        functions_text = "\n\n".join(function_docs)

        return f"""Available functions:
{functions_text}

User query: {question}

Please call the appropriate function with the correct parameters."""

    def _create_correct_function_call(self, question: str, functions: list[dict]) -> str:
        """Create a correct function call response."""
        if not functions:
            return "No function should be called for this query."

        # Use the first function as an example of correct usage
        func = functions[0]
        name = func.get("name", "unknown_function")
        params = func.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        # Build example arguments
        args = {}
        for param_name in required:
            param_info = props.get(param_name, {})
            param_type = param_info.get("type", "string")
            if param_type == "number" or param_type == "integer":
                args[param_name] = 10
            elif param_type == "boolean":
                args[param_name] = True
            elif param_type == "array":
                args[param_name] = []
            else:
                args[param_name] = f"<{param_name}_value>"

        args_str = json.dumps(args)
        return f'{{"name": "{name}", "arguments": {args_str}}}'

    def _create_incorrect_function_call(self, functions: list[dict]) -> str:
        """Create an incorrect function call response."""
        if len(functions) > 1:
            # Use wrong function
            func = functions[1]
            name = func.get("name", "wrong_function")
            return f'{{"name": "{name}", "arguments": {{}}}}'
        elif functions:
            # Use correct function but wrong parameters
            func = functions[0]
            name = func.get("name", "unknown_function")
            return f'{{"name": "{name}", "arguments": {{"wrong_param": "wrong_value"}}}}'
        else:
            # Call a function when none should be called
            return '{"name": "nonexistent_function", "arguments": {}}'

