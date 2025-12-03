from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

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
    ) -> list[ContrastivePair]:
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
            log.warning("No valid French Bench perplexity pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            text = str(doc.get("text", "")).strip()

            if not text:
                log.debug("Skipping doc due to missing text", extra={"doc": doc})
                return None

            # For perplexity tasks, we use the text as both prompt and correct response
            # Negative response is a corrupted version
            import random
            words = text.split()
            if len(words) > 3:
                shuffled = words.copy()
                random.shuffle(shuffled)
                incorrect = " ".join(shuffled)
            else:
                incorrect = "texte incorrect"

            positive_response = PositiveResponse(model_response=text)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt="",
                positive_response=positive_response,
                negative_response=negative_response,
                label="french_bench_perplexity",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
