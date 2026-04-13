from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["XStoryCloze"]
_LOG = setup_logger(__name__)

task_names = (
    "xstorycloze_ar",
    "xstorycloze_en",
    "xstorycloze_es",
    "xstorycloze_eu",
    "xstorycloze_hi",
    "xstorycloze_id",
    "xstorycloze_my",
    "xstorycloze_ru",
    "xstorycloze_sw",
    "xstorycloze_te",
    "xstorycloze_zh",
)

class XStoryClozeExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the XStoryCloze benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from XStoryCloze docs.

        XStoryCloze schema:
            - input_sentence_1, input_sentence_2, input_sentence_3, input_sentence_4: str
            - sentence_quiz1, sentence_quiz2: str
            - answer_right_ending: 1 or 2 or 3 or 4
            
        Args:
            lm_eval_task_data: lm-eval task instance for XStoryCloze.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, train_ratio=train_ratio)

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
            log.warning("No valid XStoryCloze pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single XStoryCloze doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try both key formats: snake_case (standard) and CamelCase (galician_bench)
            i1 = str(doc.get("input_sentence_1", doc.get("InputSentence1", ""))).strip()
            i2 = str(doc.get("input_sentence_2", doc.get("InputSentence2", ""))).strip()
            i3 = str(doc.get("input_sentence_3", doc.get("InputSentence3", ""))).strip()
            i4 = str(doc.get("input_sentence_4", doc.get("InputSentence4", ""))).strip()
            inputs = [i1, i2, i3, i4]
            e1 = str(doc.get("sentence_quiz1", doc.get("RandomFifthSentenceQuiz1", ""))).strip()
            e2 = str(doc.get("sentence_quiz2", doc.get("RandomFifthSentenceQuiz2", ""))).strip()
            endings = [e1, e2]
            raw_answer = doc.get("answer_right_ending", doc.get("AnswerRightEnding"))
            if raw_answer is None:
                return None
            answer = int(raw_answer) - 1

            if not inputs or not endings or answer is None:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None
            
            correct = endings[answer]
            incorrect = endings[(answer+1)%len(endings)]

            prompt = " ".join(s.strip() for s in inputs if s)

            metadata = {
                "label": "xstorycloze",
            }

            return self._build_pair(
                question=prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:  
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

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
