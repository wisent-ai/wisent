from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["JapaneseLeaderboardMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "ja_leaderboard_jcommonsenseqa",
    "ja_leaderboard_jnli",
    "ja_leaderboard_marc_ja",
    "ja_leaderboard_xwinograd",
)
class JapaneseLeaderboardMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for Japanese Leaderboard multiple-choice benchmarks."""


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
            log.warning("No valid Japanese Leaderboard MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices (JCommonsenseQA)
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", [])
                if isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 2: sentence1 + sentence2 (JNLI)
            elif "sentence1" in doc and "sentence2" in doc:
                premise = str(doc.get("sentence1", "")).strip()
                hypothesis = str(doc.get("sentence2", "")).strip()
                question = f"Premise: {premise}\nHypothesis: {hypothesis}"
                choices = ["含意", "矛盾", "中立"]
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 3: sentence + label (MARC-ja)
            elif "sentence" in doc and "label" in doc:
                question = str(doc.get("sentence", "")).strip()
                choices = ["ポジティブ", "ネガティブ"]
                answer = doc.get("label", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            # Format 4: XWinograd format (sentence with options)
            elif "sentence" in doc and "option1" in doc and "option2" in doc:
                sentence = str(doc.get("sentence", "")).strip()
                option1 = str(doc.get("option1", "")).strip()
                option2 = str(doc.get("option2", "")).strip()
                question = sentence
                choices = [option1, option2]
                answer = doc.get("answer", 0)
                answer_idx = int(answer) if isinstance(answer, (int, str)) else 0

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to missing/invalid fields", extra={"doc": doc})
                return None

            correct = str(choices[answer_idx]).strip()
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = str(choices[incorrect_idx]).strip()

            prompt = f"Question: {question}"

            positive_response = PositiveResponse(model_response=correct)
            negative_response = NegativeResponse(model_response=incorrect)

            return ContrastivePair(
                prompt=formatted_question,
                positive_response=positive_response,
                negative_response=negative_response,
                label="japanese_leaderboard_mc",
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None
