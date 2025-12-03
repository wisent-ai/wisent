from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["MMLUProExtractor"]
_LOG = setup_logger(__name__)

task_names = ("mmlu-pro", "mmlu-pro-plus")
class MMLUProExtractor(LMEvalBenchmarkExtractor):
    """Extractor for MMLU-Pro benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MMLU-Pro docs.

        Args:
            lm_eval_task_data: lm-eval task instance.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        log = bind(_LOG, doc_id=doc.get("question_id", "unknown"))

        try:
            # MMLU-Pro format: question, options (list), answer (letter), answer_index
            # mmlu_prox format: question, option_0...option_9, answer (letter), answer_index
            question = doc.get("question", "").strip()
            options = doc.get("options", [])

            # Handle option_0, option_1, etc. format (mmlu_prox)
            if not options:
                options = []
                for i in range(10):
                    opt = doc.get(f"option_{i}")
                    if opt is not None and opt != "None" and str(opt).strip():
                        options.append(str(opt).strip())
                    else:
                        break  # Stop at first None/empty

            answer = doc.get("answer", "").strip().upper()
            answer_index = doc.get("answer_index")

            if not question or not options:
                log.debug("Skipping: missing question or options")
                return None

            # Convert answer letter to index if needed
            if answer and not answer_index:
                answer_index = ord(answer) - ord('A')
            elif answer_index is not None:
                answer_index = int(answer_index)
            else:
                log.debug("Skipping: no answer information")
                return None

            if not (0 <= answer_index < len(options)):
                log.debug(f"Skipping: answer index {answer_index} out of range for {len(options)} options")
                return None

            correct_answer = options[answer_index]

            # Get an incorrect answer (any other option)
            incorrect_index = (answer_index + 1) % len(options)
            incorrect_answer = options[incorrect_index]

            # Format question with options
            formatted_question = f"Question: {question}\nOptions:\n"
            for i, opt in enumerate(options):
                formatted_question += f"{chr(ord('A') + i)}. {opt}\n"

            metadata = {
                "label": "mmlu_pro",
                "category": doc.get("category", ""),
            }

            return self._build_pair(
                question=formatted_question.strip(),
                correct=correct_answer,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
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
