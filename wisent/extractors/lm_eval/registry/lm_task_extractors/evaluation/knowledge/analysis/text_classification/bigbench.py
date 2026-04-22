from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["BigBenchExtractor"]
_LOG = setup_logger(__name__)

task_names = ("bigbench",)

class BigBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for BIG-Bench tasks."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from BIG-Bench docs.

        BIG-Bench schema varies by task but commonly includes:
            - input/question/text: str (the question)
            - target/answer/output: str (the answer)
            - choices: list[str] (optional multiple choice options)

        Args:
            lm_eval_task_data: lm-eval task instance for BIG-Bench.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)

        # Handle problematic BigBench tasks that have empty 'train' splits,
        # or cases where the lm-eval task fails to load entirely.
        if lm_eval_task_data is not None:
            task_name = getattr(lm_eval_task_data, "NAME", None) or lm_eval_task_data.config.task
        else:
            task_name = getattr(self, "task_name", "")
        problematic_tasks = {
            'bigbench_simple_arithmetic_json_multiple_choice_generate_until',
            'bigbench_simple_arithmetic_multiple_targets_json_generate_until',
        }

        if task_name in problematic_tasks:
            import datasets
            if lm_eval_task_data is not None:
                dataset_path = lm_eval_task_data.config.dataset_path
                dataset_name = lm_eval_task_data.config.dataset_name
            else:
                # Both simple_arithmetic_* bigbench variants map to the
                # 'simple_arithmetic_json' config on tasksource/bigbench.
                dataset_path = "tasksource/bigbench"
                dataset_name = "simple_arithmetic_json"
            log.info(f"Loading problematic task {task_name}")
            docs = []
            for split in ("validation", "train", "test"):
                try:
                    ds = datasets.load_dataset(path=dataset_path, name=dataset_name, split=split)
                    docs.extend(list(ds))
                except Exception:
                    continue
            if max_items and len(docs) > max_items:
                docs = docs[:max_items]
        else:
            docs = self.load_docs(lm_eval_task_data, max_items, train_ratio=train_ratio)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, lm_eval_task_data)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid BIG-Bench pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            # BigBench uses "inputs" (not "input") — may be str or list
            raw_q = doc.get("inputs", doc.get("input", doc.get("question", doc.get("text", ""))))
            if isinstance(raw_q, list):
                question = " ".join(str(x) for x in raw_q).strip()
            else:
                question = str(raw_q).strip()

            if hasattr(task_data, "doc_to_text"):
                formatted_question = task_data.doc_to_text(doc)
            else:
                formatted_question = question

            # BigBench uses "targets" which is a list, and "multiple_choice_targets" for choices
            targets = doc.get("targets", [])
            if isinstance(targets, list) and len(targets) > 0:
                correct_answer = str(targets[0]).strip()
            else:
                correct_answer = str(doc.get("target", doc.get("answer", doc.get("output", "")))).strip()

            # Some BigBench tasks have empty "inputs" (e.g. misconceptions_russian)
            # where the statement to judge IS the target. Use the correct_answer as
            # the question text when inputs is empty.
            if not question and correct_answer:
                question = correct_answer
                formatted_question = f"Is this statement correct? {correct_answer}"

            if not all([question, correct_answer]):
                _LOG.debug("Skipping: missing question or answer")
                return None

            # For BIG-Bench, create incorrect answer
            incorrect_answer = "incorrect response"

            # Try multiple_choice_targets first, then fall back to choices
            choices = doc.get("multiple_choice_targets", doc.get("choices", []))
            if choices:
                # Use multiple_choice_scores if available to find incorrect answer
                scores = doc.get("multiple_choice_scores", [])
                for idx, choice in enumerate(choices):
                    # If we have scores, use them to identify incorrect answers (score = 0)
                    # Otherwise just pick one that doesn't match correct_answer
                    if scores and len(scores) > idx:
                        if scores[idx] == 0 and str(choice).strip() != correct_answer:
                            incorrect_answer = str(choice).strip()
                            break
                    elif str(choice).strip() != correct_answer:
                        incorrect_answer = str(choice).strip()
                        break

            metadata = {
                "label": "bigbench",
                "source": getattr(task_data, "NAME", "bigbench"),
            }

            return self._build_pair(
                question=formatted_question,
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            _LOG.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )
