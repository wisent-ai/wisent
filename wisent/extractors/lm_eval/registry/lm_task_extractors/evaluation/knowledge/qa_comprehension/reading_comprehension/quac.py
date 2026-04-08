from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["QuACExtractor"]
_LOG = setup_logger(__name__)


class QuACExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for QuAC benchmark (HF-only — quac removed from lm-eval)."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from QuAC docs. lm_eval lost the quac task, so
        we load the HF dataset (allenai/quac) directly.

        Args:
            lm_eval_task_data: optional lm-eval task instance (unused; quac is HF-only).
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task="quac")

        max_items = self._normalize_limit(limit)

        # Load directly from HF since quac is no longer a registered lm-eval task
        from datasets import load_dataset
        try:
            ds = load_dataset("allenai/quac", split="validation", trust_remote_code=True)
        except Exception as exc:
            log.error(f"Failed to load allenai/quac: {exc}")
            return []
        docs = list(ds)[: (max_items * 4 if max_items else None)]

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = "quac"
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], task_data: Any = None
    ) -> ContrastivePair | None:
        """Convert a single QuAC doc into a ContrastivePair.

        QuAC schema: dialogue_id, wikipedia_page_title, background, section_title,
        context, turn_ids, questions (list), followups, yesnos, answers (dict
        with text/answer_start lists), orig_answers
        """
        try:
            context = str(doc.get("context", "")).strip()
            questions = doc.get("questions", [])
            answers = doc.get("answers", {}) or {}
            answer_texts = answers.get("texts") if isinstance(answers, dict) else None
            if answer_texts is None and isinstance(answers, dict):
                answer_texts = answers.get("text", [])

            if not context or not questions or not answer_texts:
                return None

            # Use the first question/answer pair
            question = str(questions[0]).strip() if questions else ""
            answer_list = answer_texts[0] if answer_texts and isinstance(answer_texts[0], list) else answer_texts
            correct_answer = str(answer_list[0]).strip() if answer_list else ""

            if not question or not correct_answer:
                return None

            return self._build_pair(
                question=f"Context: {context[:1500]}\nQuestion: {question}",
                correct=correct_answer,
                incorrect="I don't know.",
                metadata={"label": "quac", "source": "allenai/quac"},
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
