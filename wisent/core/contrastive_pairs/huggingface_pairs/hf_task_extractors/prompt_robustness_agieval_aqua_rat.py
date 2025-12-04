from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["PromptRobustnessAgievalAquaRatExtractor"]
_LOG = setup_logger(__name__)

task_names = ("prompt_robustness_agieval_aqua_rat",)

class PromptRobustnessAgievalAquaRatExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Prompt Robustness AGIEval AQUA-RAT - robustness testing."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="prompt_robustness_agieval_aqua_rat")
        max_items = self._normalize_limit(limit)

        from datasets import load_dataset
        try:
            dataset = load_dataset("genbench/score", "prompt_robustness_agieval_aqua_rat", split="test")
            if max_items:
                dataset = dataset.select(range(min(max_items, len(dataset))))
        except Exception as e:
            log.error(f"Failed to load prompt_robustness_agieval_aqua_rat dataset: {e}")
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
            log.warning("No valid pairs extracted", extra={"task": "prompt_robustness_agieval_aqua_rat"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = doc.get("question", doc.get("input", "")).strip()
            choices = doc.get("choices", doc.get("options", []))
            answer = doc.get("answer", doc.get("label", ""))

            if not question or not choices or not answer:
                log.debug("Skipping doc due to missing fields", extra={"doc": doc})
                return None

            # Convert answer label to index
            if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                answer_idx = ord(answer.upper()) - ord('A')
            elif isinstance(answer, int):
                answer_idx = answer
            else:
                return None

            if not (0 <= answer_idx < len(choices)):
                log.debug("Invalid answer index", extra={"doc": doc})
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            formatted_question = f"Question: {question}\nA. {incorrect}\nB. {correct}"
            metadata = {"label": "prompt_robustness_agieval_aqua_rat"}

            return self._build_pair(
                question=formatted_question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

