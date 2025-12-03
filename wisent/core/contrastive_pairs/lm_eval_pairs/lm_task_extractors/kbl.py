from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["KblExtractor"]
_LOG = setup_logger(__name__)

task_names = ("kbl",)
class KblExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Kbl benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Kbl docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Kbl.
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
            log.warning("No valid Kbl pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Kbl doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        KBL documents have structure:
        - 'question': question text
        - 'A', 'B', 'C', 'D', 'E': individual choice fields
        - 'gt': ground truth answer (e.g. 'B')
        - 'no': question number
        - 'meta': metadata
        """
        log = bind(_LOG, doc_id=doc.get("no", "unknown"))

        try:
            # Extract question
            question = doc.get("question", "").strip()
            if not question:
                log.debug("Skipping doc - missing question", extra={"doc": doc})
                return None

            # Extract choices from individual fields (A, B, C, D, E)
            choice_letters = ['A', 'B', 'C', 'D', 'E']
            choices = {}
            for letter in choice_letters:
                if letter in doc and doc[letter]:
                    choices[letter] = str(doc[letter]).strip()

            if not choices:
                log.debug("Skipping doc - no choices found", extra={"doc": doc})
                return None

            # Extract ground truth answer (can be either 'gt' or 'label')
            gt = doc.get("label", doc.get("gt", "")).strip().upper()
            if not gt or gt not in choices:
                log.debug("Skipping doc - invalid ground truth", extra={"doc": doc, "gt": gt})
                return None

            # Get correct answer text
            correct_answer = choices[gt]

            # Get an incorrect answer (pick the first choice that isn't the correct one)
            incorrect_answer = None
            for letter, text in choices.items():
                if letter != gt:
                    incorrect_answer = text
                    break

            if not incorrect_answer:
                log.debug("Skipping doc - only one choice available", extra={"doc": doc})
                return None

            # Build the prompt using doc_to_text if available, otherwise manually
            # The prompt should just be the question since doc_to_text handles formatting
            prompt = question

            metadata = {"label": "kbl"}

            return self._build_pair(
                question=prompt,
                correct=gt,  # Return the letter (A/B/C/D/E) as the answer
                incorrect="Z",  # Invalid choice letter as negative
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(prompt=question, positive_response=positive_response, negative_response=negative_response, label=metadata.get("label"))
