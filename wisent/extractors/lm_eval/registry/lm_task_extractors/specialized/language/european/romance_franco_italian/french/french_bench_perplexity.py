from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FrenchBenchPerplexityExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "french_bench_opus_perplexity",
    "french_bench_wikitext_fr"
)
class FrenchBenchPerplexityExtractor(LMEvalBenchmarkExtractor):
    """Extractor for French Bench perplexity benchmarks (loglikelihood_rolling)."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)

        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)

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
            log.warning("No valid French Bench perplexity pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Perplexity docs may use either 'text' (opus) or 'paragraph' (wikitext_fr).
            text = str(doc.get("text", doc.get("paragraph", ""))).strip()

            if not text:
                log.debug("Skipping doc due to missing text", extra={"doc": doc})
                return None

            import random
            words = text.split()
            # Cap length so pairs stay manageable
            if len(words) > 80:
                words = words[:80]
                text = " ".join(words)
            random.seed(hash(text) % (2**32))
            shuffled = words.copy()
            random.shuffle(shuffled)
            incorrect = " ".join(shuffled)
            if incorrect == text:
                incorrect = " ".join(words[::-1])
                if incorrect == text:
                    # Fall back to char reversal for short single-word docs
                    incorrect = text[::-1]
                if incorrect == text:
                    incorrect = text + " (garbled)"

            positive_response = PositiveResponse(model_response=text)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt="Continue this text:",
                positive_response=positive_response,
                negative_response=negative_response,
                label="french_bench_perplexity",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
