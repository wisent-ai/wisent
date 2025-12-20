from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["C4Extractor"]
_LOG = setup_logger(__name__)


class C4Extractor(LMEvalBenchmarkExtractor):
    """Extractor for the C4 benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from C4 docs.

        Args:
            lm_eval_task_data: lm-eval task instance for C4.
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
            log.warning("No valid C4 pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single C4 doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        C4 is a language modeling benchmark with plain text documents.
        We create contrastive pairs by taking sentences from the text.
        """
        log = bind(_LOG, doc_id=doc.get("id", doc.get("url", "unknown")))

        try:
            # C4 format: just has 'text' field with raw text
            if "text" in doc:
                text = str(doc.get("text", "")).strip()
                if not text or len(text) < 50:  # Skip very short texts
                    return None

                # Split text into sentences (simple split on periods)
                sentences = [s.strip() for s in text.split('.') if s.strip()]

                # Need at least 2 sentences to create a pair
                if len(sentences) < 2:
                    return None

                # Use first sentence as context/prompt and second as correct completion
                prompt = sentences[0]
                correct_completion = sentences[1]

                # Create an incorrect completion by using a generic wrong answer
                incorrect_completion = "This sentence does not follow from the previous context."

                metadata = {"label": "c4", "source": "text_completion"}

                return self._build_pair(
                    question=f"Complete the following text:\n{prompt}.",
                    correct=correct_completion,
                    incorrect=incorrect_completion,
                    metadata=metadata,
                )

            # Legacy formats for backwards compatibility
            # Format 1: question + choices + answer
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                else:
                    return None

                answer = doc.get("answer", doc.get("answerKey", ""))
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    answer_idx = int(answer) if answer else 0

                if not question or not choices or not (0 <= answer_idx < len(choices)):
                    return None

                correct = choices[answer_idx]
                incorrect_idx = (answer_idx + 1) % len(choices)
                incorrect = choices[incorrect_idx]
                metadata = {"label": "c4"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            return None

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
