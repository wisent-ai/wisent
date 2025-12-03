from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CabreuExtractor"]
_LOG = setup_logger(__name__)

task_names = ("cabreu", "cabreu_extractive", "cabreu_abstractive", "cabreu_extreme")

class CabreuExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Cabreu benchmark (Catalan summarization)."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc)
        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        # Determine which summary type to use based on task name
        task_name = getattr(lm_eval_task_data, "NAME", "")
        if "extractive" in task_name:
            summary_type = "extractive"
        elif "abstractive" in task_name:
            summary_type = "abstractive"
        elif "extreme" in task_name:
            summary_type = "extreme"
        else:
            summary_type = "extractive"  # default

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, summary_type)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], summary_type: str = "extractive") -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Cabreu summarization format: content + summaries
            content = doc.get("content", "").strip()
            summaries = doc.get("summaries", {})

            if not content or not summaries:
                log.debug("Skipping doc - missing content or summaries", extra={"doc": doc})
                return None

            # Get the correct summary based on type
            summary_dict = summaries.get(summary_type, {})
            if not summary_dict:
                log.debug(f"Skipping doc - no {summary_type} summary found", extra={"doc": doc})
                return None

            # Use 'a1' as the reference summary
            correct_summary = summary_dict.get("a1", "").strip()
            if not correct_summary:
                log.debug(f"Skipping doc - empty {summary_type} summary 'a1'", extra={"doc": doc})
                return None

            # Create synthetic negative by shuffling sentences
            import random
            sentences = [s.strip() for s in correct_summary.split('.') if s.strip()]
            if len(sentences) <= 1:
                # Can't shuffle if only one sentence, skip
                log.debug("Skipping doc - summary has only one sentence", extra={"doc": doc})
                return None

            shuffled_sentences = sentences.copy()
            random.shuffle(shuffled_sentences)

            # Ensure shuffled is actually different
            if shuffled_sentences == sentences:
                # Reverse if shuffle didn't change order
                shuffled_sentences = list(reversed(sentences))

            incorrect_summary = '. '.join(shuffled_sentences) + '.'

            # Format prompt
            prompt = f"Text: {content}\n\nGenerate a {summary_type} summary:"

            metadata = {"label": "cabreu", "summary_type": summary_type}

            return self._build_pair(
                question=prompt,
                correct=correct_summary,
                incorrect=incorrect_summary,
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
