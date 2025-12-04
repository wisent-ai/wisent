from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["DocVQAExtractor"]
_LOG = setup_logger(__name__)

task_names = ("doc_vqa",)

class DocVQAExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for DocVQA (Document Visual Question Answering) benchmark.

    This is a Unitxt multimodal task where models answer questions about document images.
    Format: source (question with <img> tags) + target (answer text)
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from DocVQA docs.

        Args:
            lm_eval_task_data: lm-eval task instance for DocVQA.
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

        # Group docs by their questions to create pairs
        # For each question, pair correct answer with a random incorrect answer from other docs
        answers_by_question: dict[str, list[str]] = {}

        for doc in docs:
            source = doc.get("source", "").strip()
            target = doc.get("target", "").strip()

            if not source or not target:
                continue

            if source not in answers_by_question:
                answers_by_question[source] = []
            answers_by_question[source].append(target)

        # Create pairs: for each question with its correct answer,
        # pair it with an incorrect answer from a different question
        all_answers = [ans for ans_list in answers_by_question.values() for ans in ans_list]

        for source, correct_answers in answers_by_question.items():
            for correct_answer in correct_answers:
                # Find an incorrect answer (from different question)
                incorrect_answer = None
                for other_source, other_answers in answers_by_question.items():
                    if other_source != source and other_answers:
                        incorrect_answer = other_answers[0]
                        break

                if incorrect_answer:
                    metadata = {"label": "doc_vqa"}
                    pair = self._build_pair(
                        question=source,
                        correct=correct_answer,
                        incorrect=incorrect_answer,
                        metadata=metadata,
                    )
                    pairs.append(pair)

                    if max_items is not None and len(pairs) >= max_items:
                        break

            if max_items is not None and len(pairs) >= max_items:
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid DocVQA pairs extracted", extra={"task": task_name})

        return pairs

