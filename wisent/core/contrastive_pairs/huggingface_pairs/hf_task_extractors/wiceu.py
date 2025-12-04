from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["WiceuExtractor"]
_LOG = setup_logger(__name__)

task_names = ("wiceu",)

class WiceuExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Wiceu benchmark."""


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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("idx", doc.get("id", "unknown")))

        try:
            # WiC (Word in Context) task: determine if a word has the same meaning in two sentences
            sentence1 = doc.get("sentence1", "").strip()
            sentence2 = doc.get("sentence2", "").strip()
            word = doc.get("word", "").strip()
            label = doc.get("label", None)

            if not sentence1 or not sentence2 or not word or label is None:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # label: 0 = different meaning, 1 = same meaning
            wic_choices = ["Different meaning", "Same meaning"]

            correct_idx = int(label)
            incorrect_idx = 1 - correct_idx

            correct = wic_choices[correct_idx]
            incorrect = wic_choices[incorrect_idx]

            # Format the WiC task
            formatted_question = f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDoes the word '{word}' have the same meaning in both sentences?\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "wiceu"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

