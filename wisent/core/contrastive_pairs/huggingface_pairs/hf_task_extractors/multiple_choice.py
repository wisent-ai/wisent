from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["MultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = ("multiple_choice",)

class MultipleChoiceExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Multiple Choice - generic multiple choice questions."""


    evaluator_name = "exact_match"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="multiple_choice")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("EleutherAI/bigbench", "multiple_choice", split="train")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load multiple_choice dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "multiple_choice"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("input", doc.get("question", doc.get("text", ""))).strip()
            choices = doc.get("choices", doc.get("multiple_choice_targets", []))
            target = doc.get("target", doc.get("answer", ""))

            if not question or not choices or not target:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Find correct answer index
            try:
                answer_idx = choices.index(target)
            except ValueError:
                log.debug("Target not in choices", extra={"doc": doc})
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            formatted_question = f"Question: {question}\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "multiple_choice"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

