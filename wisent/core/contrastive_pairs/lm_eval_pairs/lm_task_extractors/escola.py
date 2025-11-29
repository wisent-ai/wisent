from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EscolaExtractor"]
_LOG = setup_logger(__name__)

task_names = ("escola",)

class EscolaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Escola benchmark - grammatical acceptability task."""


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

        # Escola format: Sentence + Label (1=acceptable, 0=unacceptable)
        # Group sentences by label
        acceptable_sentences = []
        unacceptable_sentences = []

        for doc in docs:
            sentence = doc.get("Sentence", "").strip()
            label = doc.get("Label")

            if not sentence or label is None:
                continue

            if label == 1:
                acceptable_sentences.append(sentence)
            elif label == 0:
                unacceptable_sentences.append(sentence)

        # Create pairs by pairing acceptable with unacceptable sentences
        num_pairs = min(len(acceptable_sentences), len(unacceptable_sentences))
        if max_items is not None:
            num_pairs = min(num_pairs, max_items)

        for i in range(num_pairs):
            prompt = "acceptable"  # Placeholder prompt for log-likelihood evaluation
            metadata = {"label": "escola"}

            pair = self._build_pair(
                question=prompt,
                correct=acceptable_sentences[i],
                incorrect=unacceptable_sentences[i],
                metadata=metadata,
            )
            pairs.append(pair)

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

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
