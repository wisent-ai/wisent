from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["SelfConsistencyExtractor"]
_LOG = setup_logger(__name__)

task_names = ("self_consistency",)

class SelfConsistencyExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Self-Consistency - multiple choice questions."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="self_consistency")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("openai/gsm8k", "main", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load self_consistency dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "self_consistency"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", doc.get("input", "")).strip()
            answer = doc.get("answer", "").strip()

            if not question or not answer:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Extract numeric answer if present
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                correct = numbers[-1]  # Usually the final number is the answer
            else:
                correct = answer

            # Create incorrect answer
            try:
                num = float(correct)
                incorrect = str(num + 1)
            except ValueError:
                incorrect = f"{correct} + 1"

            prompt = f"Question: {question}\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "self_consistency"}

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

