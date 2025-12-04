from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["T0EvalExtractor"]
_LOG = setup_logger(__name__)

task_names = ("t0_eval",)

class T0EvalExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for T0 Eval - zero-shot prompted tasks."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="t0_eval")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("bigscience/t0_eval", split="train")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load t0_eval dataset: {e}")
            return []

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(dataset)})

        for doc in dataset:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "t0_eval"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # T0 format: inputs (prompt) + targets (answer)
            inputs = doc.get("inputs", doc.get("input", "")).strip()
            targets = doc.get("targets", doc.get("target", "")).strip()

            if not inputs or not targets:
                log.debug("Skipping doc due to missing inputs or targets", extra={"doc": doc})
                return None

            prompt = inputs
            correct = targets
            incorrect = "incorrect answer"

            metadata = {"label": "t0_eval"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

