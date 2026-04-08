from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["Bec2016euExtractor"]
_LOG = setup_logger(__name__)

task_names = ("bec2016eu",)

class Bec2016euExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Bec2016Eu benchmark (basqueGLUE bec subtask)."""


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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # bec2016eu schema: text + label (3-class sentiment: 0=N, 1=NEU, 2=P)
            text = str(doc.get("text", "")).strip()
            label = doc.get("label", None)
            choices = ["negatiboa", "neutrala", "positiboa"]

            if not text or label is None:
                return None

            try:
                answer_idx = int(label)
            except (TypeError, ValueError):
                return None

            if not (0 <= answer_idx < len(choices)):
                return None

            correct = choices[answer_idx]
            incorrect = choices[(answer_idx + 1) % len(choices)]

            prompt = f"Testua: {text}\nGaldera: Nolako jarrera agertzen du aurreko testuak?\nErantzuna:"

            return ContrastivePair(
                prompt=prompt,
                positive_response=PositiveResponse(model_response=correct),
                negative_response=NegativeResponse(model_response=incorrect),
                label="bec2016eu",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

