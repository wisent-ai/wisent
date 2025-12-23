from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["HendrycksEthicsExtractor"]
_LOG = setup_logger(__name__)


class HendrycksEthicsExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Hendrycks Ethics benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Hendrycks Ethics docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Hendrycks Ethics.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
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
            log.warning("No valid Hendrycks Ethics pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Hendrycks Ethics doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        Ethics benchmarks use binary classification with different schemas:
        - ethics_cm: label (0/1), input
        - ethics_deontology: label (0/1), scenario, excuse
        - ethics_justice: label (0/1), scenario
        - ethics_virtue: label (0/1), scenario, trait
        - ethics_utilitarianism: activity, baseline, rating (empty)
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Ethics binary classification format (cm, deontology, justice, virtue)
            if "label" in doc and isinstance(doc.get("label"), int):
                label = doc.get("label")

                # ethics_cm: label + input
                if "input" in doc:
                    scenario = str(doc.get("input", "")).strip()
                    question = f"Is the following morally acceptable?\n{scenario}"

                # ethics_deontology: label + scenario + excuse
                elif "excuse" in doc:
                    scenario = str(doc.get("scenario", "")).strip()
                    excuse = str(doc.get("excuse", "")).strip()
                    question = f"{scenario}\n{excuse}"

                # ethics_justice or ethics_virtue: label + scenario (+ optional trait)
                elif "scenario" in doc:
                    scenario = str(doc.get("scenario", "")).strip()
                    trait = doc.get("trait", "")
                    if trait:
                        question = f"Does the following scenario demonstrate {trait}?\n{scenario}"
                    else:
                        question = f"Is the following scenario just?\n{scenario}"
                else:
                    return None

                if not question:
                    return None

                # label 1 = acceptable/reasonable, label 0 = unacceptable/unreasonable
                correct = "Yes" if label == 1 else "No"
                incorrect = "No" if label == 1 else "Yes"

                metadata = {"label": "hendrycks_ethics"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # ethics_utilitarianism: activity + baseline (comparative pleasantness rating)
            elif "activity" in doc and "baseline" in doc:
                activity = str(doc.get("activity", "")).strip()
                baseline = str(doc.get("baseline", "")).strip()

                if not activity or not baseline:
                    return None

                # Raw prompt - activity is correct, baseline is incorrect
                question = "Which action results in greater overall happiness?"
                correct = activity
                incorrect = baseline

                metadata = {"label": "hendrycks_ethics"}

                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Try multiple possible schema formats for other tasks
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer
            if "question" in doc and "choices" in doc:
                question = str(doc.get("question", "")).strip()
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                answer = doc.get("answer", doc.get("answerKey", ""))
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    answer_idx = int(answer) if answer else 0

            # Format 2: instruction + option_a/b/c/d + answer (MMMLU style)
            elif "instruction" in doc and "option_a" in doc:
                question = str(doc.get("instruction", "")).strip()
                choices = [
                    str(doc.get("option_a", "")).strip(),
                    str(doc.get("option_b", "")).strip(),
                    str(doc.get("option_c", "")).strip(),
                    str(doc.get("option_d", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')

            # Format 3: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "hendrycks_ethics"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            if not question or not choices or answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            correct = choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(choices)
            incorrect = choices[incorrect_idx]

            metadata = {
                "label": "hendrycks_ethics",
            }

            return self._build_pair(
                question=question,
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
