from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

class TerminalBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Terminal-Bench - AI agent benchmark for terminal tasks.

    Dataset: ia03/terminal-bench on HuggingFace

    Terminal-Bench evaluates AI agents in real terminal environments on tasks
    ranging from compiling code and training models to setting up servers
    and debugging systems.

    For terminal task evaluation:
    - Positive (correct) = Correct terminal commands that accomplish the task
    - Negative (incorrect) = Commands that fail or produce wrong results
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "terminal_bench"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Terminal-Bench dataset.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="ia03/terminal-bench",
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from terminal-bench")
        except Exception as e:
            log.error(f"Failed to load terminal-bench dataset: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid Terminal-Bench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.

        terminal-bench schema:
        - task_id: str (unique task identifier)
        - base_description: str (task description)
        - category: str (task category)
        - difficulty: str (easy, medium, hard)
        - task_yaml: str (YAML with task details)
        """
        try:
            import yaml
            task_id = doc.get("task_id", "")
            description = doc.get("base_description", "").strip()
            category = doc.get("category", "")
            difficulty = doc.get("difficulty", "")
            task_yaml_str = doc.get("task_yaml", "")

            # Parse task_yaml for additional details
            task_details = ""
            if task_yaml_str:
                try:
                    task_yaml = yaml.safe_load(task_yaml_str)
                    if isinstance(task_yaml, dict):
                        task_details = task_yaml.get("instructions", task_yaml.get("description", ""))
                except Exception:
                    pass

            # Use description as the task prompt
            if not description:
                return None

            task_text = task_details if task_details else description
            task_prompt = f"""Terminal Task: {task_id}
Category: {category}
Difficulty: {difficulty}

{task_text}

Provide the correct sequence of terminal commands to accomplish this task."""

            # For terminal tasks, create synthetic correct/incorrect responses
            correct = f"# Correct solution for {task_id}\n# Following best practices for {category} tasks"
            incorrect = f"# Incorrect approach for {task_id}\necho 'Task not completed correctly'"

            metadata = {
                "label": "terminal_bench",
                "source": "ia03/terminal-bench",
                "task_id": task_id,
                "category": category,
                "difficulty": difficulty,
                "is_terminal_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Terminal-Bench pair: {exc}", exc_info=True)
            return None

