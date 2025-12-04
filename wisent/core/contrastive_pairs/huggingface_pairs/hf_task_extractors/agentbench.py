from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["AgentBenchExtractor"]

log = setup_logger(__name__)

# AgentBench/AgentInstruct task domains
AGENTBENCH_DOMAINS = [
    "os",        # Operating System (Linux shell)
    "db",        # Database (SQL)
    "alfworld",  # ALFWorld (household tasks)
    "webshop",   # WebShop (e-commerce)
    "kg",        # Knowledge Graph
    "mind2web",  # Mind2Web (web browsing)
]


class AgentBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for AgentBench/AgentInstruct - LLM Agent Evaluation Benchmark (ICLR 2024).

    AgentBench is the first benchmark designed to evaluate LLM-as-Agent across
    a diverse spectrum of 8 different environments, assessing reasoning and
    decision-making abilities in multi-turn open-ended generation settings.

    Task Domains:
    - OS: Operating system shell commands
    - DB: Database SQL queries
    - ALFWorld: Household task simulation
    - WebShop: E-commerce product search
    - KG: Knowledge graph queries
    - Mind2Web: Web browsing tasks

    Uses THUDM/AgentInstruct which contains 1,866 high-quality agent interactions
    with detailed thought explanations for each action.

    Schema (THUDM/AgentInstruct):
        - split: str (task domain identifier)
        - conversations: list[dict] (dialogue turns with from/value/loss fields)
        - id: str (unique identifier)

    Each conversation follows Think->Act pattern demonstrating chain-of-thought reasoning.
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "agent_task"

    def __init__(self, domain: str | None = None):
        """
        Initialize AgentBench extractor.

        Args:
            domain: Optional filter for specific task domain
                   (os, db, alfworld, webshop, kg, mind2web)
        """
        super().__init__()
        self.domain = domain

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from AgentInstruct examples.

        For agent tasks:
        - Positive (correct) = Correct action sequence with reasoning
        - Negative (incorrect) = Wrong action or missing reasoning

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        pairs: list[ContrastivePair] = []

        # Load from different splits based on domain filter
        splits_to_load = [self.domain] if self.domain else AGENTBENCH_DOMAINS

        for split in splits_to_load:
            try:
                docs = self.load_dataset(
                    dataset_name="THUDM/AgentInstruct",
                    split=split,
                    limit=max_items - len(pairs) if max_items else None,
                )
                log.info(f"Loaded {len(docs)} examples from AgentInstruct ({split})")

                for doc in docs:
                    pair = self._extract_pair_from_doc(doc, split)
                    if pair is not None:
                        pairs.append(pair)
                        if max_items is not None and len(pairs) >= max_items:
                            break

            except Exception as e:
                log.warning(f"Failed to load AgentInstruct split '{split}': {e}")
                continue

            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            log.warning("No valid AgentBench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], split: str) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            doc_id = doc.get("id", "")
            conversations = doc.get("conversations", [])

            if not conversations or len(conversations) < 2:
                log.debug("Skipping: insufficient conversation turns")
                return None

            # Extract the task description (first human message)
            task_description = self._extract_task(conversations)
            if not task_description:
                log.debug("Skipping: missing task description")
                return None

            # Extract the agent's response with reasoning
            correct_response = self._extract_agent_response(conversations)
            if not correct_response:
                log.debug("Skipping: missing agent response")
                return None

            # Create incorrect response (missing reasoning or wrong action)
            incorrect_response = self._create_incorrect_response(conversations, split)

            metadata = {
                "label": "agentbench",
                "source": "THUDM/AgentInstruct",
                "id": doc_id,
                "domain": split,
                "num_turns": len(conversations),
                "is_agent_benchmark": True,
            }

            return self._build_pair(
                question=task_description,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _extract_task(self, conversations: list[dict]) -> str:
        """Extract the task description from conversations."""
        for turn in conversations:
            if turn.get("from") == "human":
                value = turn.get("value", "")
                if value:
                    return value.strip()
        return ""

    def _extract_agent_response(self, conversations: list[dict]) -> str:
        """Extract the first agent response with reasoning."""
        for turn in conversations:
            if turn.get("from") == "gpt":
                value = turn.get("value", "")
                if value:
                    return value.strip()
        return ""

    def _extract_full_response_chain(self, conversations: list[dict]) -> str:
        """Extract the full response chain for complex multi-turn tasks."""
        responses = []
        for turn in conversations:
            if turn.get("from") == "gpt":
                value = turn.get("value", "")
                if value:
                    responses.append(value.strip())
        return "\n\n".join(responses)

    def _create_incorrect_response(self, conversations: list[dict], domain: str) -> str:
        """Create an incorrect response based on domain."""
        # Get the correct response to modify
        correct = self._extract_agent_response(conversations)

        # Domain-specific incorrect responses
        if domain == "os":
            # Remove reasoning, give command without Think
            if "Act:" in correct:
                act_part = correct.split("Act:")[-1].strip()
                return f"Act: rm -rf / # this will fix it"
            return "Act: bash[echo 'done']"

        elif domain == "db":
            return "Act: answer[SELECT * FROM table]"

        elif domain == "alfworld":
            return "Act: go to random location"

        elif domain == "webshop":
            return "Act: click[random product]"

        elif domain == "kg":
            return "Act: answer[I don't know the answer]"

        elif domain == "mind2web":
            return "Act: click[body]"

        else:
            # Generic incorrect: skip reasoning
            return "Act: finish[Task completed without proper analysis]"

