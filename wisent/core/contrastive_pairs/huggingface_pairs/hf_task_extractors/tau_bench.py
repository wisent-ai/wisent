from __future__ import annotations

import requests
from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["TauBenchExtractor"]

log = setup_logger(__name__)

# TAU-bench domains and their GitHub URLs
TAU_BENCH_DOMAINS = ["retail", "airline"]

# GitHub URLs for the Python task files
TAU_BENCH_GITHUB_URLS = {
    "retail": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/envs/retail/tasks.py",
    "airline": "https://raw.githubusercontent.com/sierra-research/tau-bench/main/tau_bench/envs/airline/tasks.py",
}


class TauBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for τ-bench - Tool-Agent-User Interaction Benchmark (Sierra 2024).

    τ-bench evaluates language agents on completing complex tasks while
    interacting with simulated users and tools in real-world domains.

    Domains:
    - retail: E-commerce customer service scenarios
    - airline: Airline booking and support scenarios
    - telecom: Telecommunications support (τ²-bench extension)

    Dataset: HuggingFaceH4/tau2-bench-data (or GitHub sierra-research/tau-bench)

    Schema:
        - id: str (task identifier)
        - user_scenario: str (user situation description)
        - description: str (task description)
        - evaluation_criteria: list (success criteria)
        - initial_state: dict (starting database state)

    For agent task evaluation:
    - Positive (correct) = Successfully completes user task
    - Negative (incorrect) = Fails to complete or makes errors
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "agent_task_completion"

    def __init__(self, domain: str = "retail"):
        """
        Initialize TAU-bench extractor.

        Args:
            domain: Domain to use ("retail", "airline", "telecom")
        """
        super().__init__()
        self.domain = domain

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from TAU-bench examples.

        Creates pairs for agent task completion:
        - Positive (correct) = Successfully completes the task
        - Negative (incorrect) = Fails or makes errors

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load from GitHub (HuggingFace dataset is broken)
        docs = self._load_from_github()
        if not docs:
            log.error("Failed to load TAU-bench from GitHub")
            return []

        log.info(f"Loaded {len(docs)} examples from TAU-bench GitHub")

        pairs: list[ContrastivePair] = []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid TAU-bench pairs extracted")

        return pairs

    def _load_from_github(self) -> list[dict[str, Any]]:
        """Load TAU-bench tasks from GitHub Python files."""
        all_docs = []

        for domain in TAU_BENCH_DOMAINS:
            url = TAU_BENCH_GITHUB_URLS.get(domain)
            if not url:
                continue

            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                content = response.text

                # Parse the Python file to extract the 'tasks' list
                # The file contains: tasks = [...]
                # We use exec to safely load the list
                local_vars: dict[str, Any] = {}
                exec(content, {"__builtins__": {}}, local_vars)
                tasks = local_vars.get("tasks", [])

                # Add domain to each task
                for task in tasks:
                    task["domain"] = domain

                all_docs.extend(tasks)
                log.info(f"Loaded {len(tasks)} tasks from {domain} domain")

            except Exception as e:
                log.warning(f"Failed to load {domain} domain from GitHub: {e}")
                continue

        return all_docs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        TAU-bench GitHub schema:
            - user_id: str (user identifier)
            - instruction: str (user's request/scenario)
            - actions: list[dict] (expected tool call sequence)
            - outputs: list[str] (optional expected outputs)
            - annotator: int (annotator id)
        """
        try:
            user_id = doc.get("user_id", "")
            instruction = doc.get("instruction", "").strip()
            actions = doc.get("actions", [])
            outputs = doc.get("outputs", [])
            domain = doc.get("domain", self.domain)

            if not instruction:
                log.debug("Skipping: missing instruction")
                return None

            # Extract tool names from actions
            tool_names = [a.get("name", "") for a in actions if a.get("name")]

            # Build the agent task prompt
            task_prompt = self._build_task_prompt(instruction, tool_names, domain)

            # Positive = successful task completion with correct tool sequence
            correct_response = self._create_successful_response(actions, outputs)
            # Negative = failed or incomplete task
            incorrect_response = self._create_failed_response()

            metadata = {
                "label": "tau_bench",
                "source": "sierra-research/tau-bench",
                "user_id": user_id,
                "domain": domain,
                "num_actions": len(actions),
                "tool_names": tool_names,
                "is_agent_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_task_prompt(
        self,
        instruction: str,
        tool_names: list[str],
        domain: str,
    ) -> str:
        """Build the agent task prompt."""
        domain_context = {
            "retail": "You are a customer service agent for an e-commerce platform.",
            "airline": "You are a customer service agent for an airline booking system.",
        }

        parts = [domain_context.get(domain, "You are a helpful customer service agent.")]
        parts.append(f"\nUser Request: {instruction}")

        if tool_names:
            unique_tools = list(dict.fromkeys(tool_names))  # Preserve order, remove duplicates
            tools_str = ", ".join(unique_tools)
            parts.append(f"\nAvailable Tools: {tools_str}")

        parts.append("\nPlease help the user complete their request by using the appropriate tools.")

        return "\n".join(parts)

    def _create_successful_response(
        self,
        actions: list[dict[str, Any]],
        outputs: list[str],
    ) -> str:
        """Create a successful task completion response."""
        steps = []
        for i, action in enumerate(actions[:5]):  # Limit to first 5 actions for brevity
            name = action.get("name", "")
            args = action.get("arguments", {})
            args_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
            steps.append(f"{i+1}. Called {name}({args_str})")

        steps_str = "\n".join(steps) if steps else "Completed the requested actions"

        output_str = ""
        if outputs:
            output_str = f"\n\nResult: {outputs[0]}"

        return (
            f"I'll help you with this request. Let me work through the necessary steps:\n\n"
            f"{steps_str}\n"
            f"{output_str}\n\n"
            "I have successfully completed all the required actions. Is there "
            "anything else I can help you with?"
        )

    def _create_failed_response(self) -> str:
        """Create a failed task response."""
        return (
            "I apologize, but I'm having trouble completing this request. "
            "I attempted to process your request but encountered an issue. "
            "The system isn't responding as expected and I couldn't complete "
            "all the necessary steps. Please try again later or contact support."
        )

