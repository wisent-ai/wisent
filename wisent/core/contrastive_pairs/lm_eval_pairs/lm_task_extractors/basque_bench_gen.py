from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["BasqueBenchGenerationExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "mgsm_native_cot_eu",
    "flores_ca-eu",
    "flores_de-eu",
    "flores_en-eu",
    "flores_es-eu",
    "flores_eu-ca",
    "flores_eu-de",
    "flores_eu-en",
    "flores_eu-es",
    "flores_eu-fr",
    "flores_eu-gl",
    "flores_eu-it",
    "flores_eu-pt",
    "flores_eu",
    "flores_fr-eu",
    "flores_gl-eu",
    "flores_it-eu",
    "flores_pt-eu",
)
class BasqueBenchGenerationExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Basque Bench generation benchmarks (MGSM CoT and FLORES translation)."""


    evaluator_name = "generation"
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
            log.warning("No valid Basque Bench generation pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: MGSM CoT - {question, answer or answer_number}
            if "question" in doc and ("answer" in doc or "answer_number" in doc):
                question = str(doc.get("question", "")).strip()
                answer = doc.get("answer")
                answer_number = doc.get("answer_number")

                if not question:
                    log.debug("Skipping doc due to missing question", extra={"doc": doc})
                    return None

                # Extract correct answer
                if answer is not None:
                    answer_str = str(answer).strip()
                    # Remove prefix if present
                    if len(answer_str) > 27:
                        correct = answer_str[27:]
                    else:
                        correct = answer_str
                elif answer_number is not None:
                    correct = str(answer_number).strip()
                else:
                    return None

                # Create synthetic negative by shuffling
                incorrect = self._create_shuffled_text(correct)

                formatted_question = f"Galdera: {question}\nErantzuna urratsez urrats:"

                return self._build_pair(
                    question=formatted_question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "basque_bench_mgsm"},
                )

            # Format 2: FLORES translation - {sentence_<lang1>, sentence_<lang2>}
            else:
                source_sent = None
                target_sent = None

                for key in doc.keys():
                    if key.startswith("sentence_"):
                        if source_sent is None:
                            source_sent = doc[key]
                        elif target_sent is None:
                            target_sent = doc[key]

                if not source_sent or not target_sent:
                    log.debug("Skipping doc due to missing sentence fields", extra={"doc": doc})
                    return None

                source_sent = str(source_sent).strip()
                target_sent = str(target_sent).strip()

                # Create synthetic negative by shuffling
                incorrect = self._create_shuffled_text(target_sent)

                formatted_question = f"Translate: {source_sent}"

                return self._build_pair(
                    question=formatted_question,
                    correct=target_sent,
                    incorrect=incorrect,
                    metadata={"label": "basque_bench_translation"},
                )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    @staticmethod
    def _create_shuffled_text(text: str) -> str:
        """Create a synthetic negative response by shuffling words."""
        words = text.split()
        if len(words) > 3:
            shuffled = words.copy()
            random.shuffle(shuffled)
            return " ".join(shuffled)
        return "erantzun okerra"

    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
        )
