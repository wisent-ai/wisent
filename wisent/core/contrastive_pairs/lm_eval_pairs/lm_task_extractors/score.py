from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["ScoreExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "score_robustness",
    "score_robustness_agieval",
    "score_robustness_math",
    "score_robustness_mmlu_pro",
    "score_non_greedy_robustness_agieval",
    "score_non_greedy_robustness_math",
    "score_non_greedy_robustness_mmlu_pro",
    "score_option_order_robustness_agieval",
    "score_option_order_robustness_mmlu_pro",
    "score_prompt_robustness_agieval",
    "score_prompt_robustness_math",
    "score_prompt_robustness_mmlu_pro",
)

class ScoreExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the SCORE robustness benchmarks.

    SCORE (Systematic Consistency and Robustness Evaluation) tests
    model consistency across different prompt variations, option orders,
    and sampling strategies. Uses the same data format as AGIEval.
    """


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Score docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Score.
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
            log.warning("No valid Score pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single SCORE doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.

        SCORE format: {question, choices, gold, answer, options, ...}
        """
        log = bind(_LOG, doc_id=doc.get("question_id", doc.get("id", "unknown")))

        try:
            # SCORE uses standard multiple-choice format
            if "question" not in doc:
                log.debug("Skipping doc due to missing question", extra={"doc": doc})
                return None

            question = str(doc["question"]).strip()

            # Try to get choices from either "choices" or "options" field
            choices_data = doc.get("choices", doc.get("options", []))

            # Parse choices (can be list or dict)
            if isinstance(choices_data, dict):
                choices = choices_data.get("text", [])
            elif isinstance(choices_data, list):
                choices = choices_data
            else:
                log.debug("Skipping doc due to invalid choices format", extra={"doc": doc})
                return None

            if not question or not choices:
                log.debug("Skipping doc due to empty question/choices", extra={"doc": doc})
                return None

            # Get answer index
            answer_idx = None
            if "gold" in doc:
                gold = doc["gold"]
                if isinstance(gold, list) and gold:
                    answer_idx = gold[0]
                else:
                    answer_idx = int(gold)
            elif "answer_index" in doc:
                answer_idx = int(doc["answer_index"])
            elif "answer" in doc:
                answer = doc["answer"]
                if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
                    answer_idx = ord(answer.upper()) - ord('A')
                else:
                    try:
                        answer_idx = int(answer)
                    except (ValueError, TypeError):
                        log.debug("Could not parse answer", extra={"answer": answer})
                        return None

            if answer_idx is None or not (0 <= answer_idx < len(choices)):
                log.debug("Skipping doc due to invalid answer index", extra={"doc": doc, "answer_idx": answer_idx})
                return None

            # Extract correct and incorrect answers
            # Remove option labels like "(A)" from choices if present
            cleaned_choices = []
            for choice in choices:
                choice_str = str(choice).strip()
                # Remove leading "(A)", "(B)", etc. if present
                if choice_str.startswith("(") and len(choice_str) > 3 and choice_str[2] == ")":
                    choice_str = choice_str[3:].strip()
                cleaned_choices.append(choice_str)

            correct = cleaned_choices[answer_idx]
            incorrect_idx = (answer_idx + 1) % len(cleaned_choices)
            incorrect = cleaned_choices[incorrect_idx]

            metadata = {"label": "score_robustness"}

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
