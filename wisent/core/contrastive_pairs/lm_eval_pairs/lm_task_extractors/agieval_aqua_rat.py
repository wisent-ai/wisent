from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AgievalAquaRatExtractor"]
_LOG = setup_logger(__name__)

task_names = ("prompt_robustness_agieval_aqua_rat", "option_order_robustness_agieval_aqua_rat", "non_greedy_robustness_agieval_aqua_rat")

class AgievalAquaRatExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AGIEval AQUA-RAT robustness benchmarks (prompt, option order, non-greedy)."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from PIQA docs.
        
        PIQA schema:
            - question: str 
            - choices: list
            - gold: list

        Args:
            lm_eval_task_data: lm-eval task instance for prompt_robustness_agieval_aqua_rat.
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items)

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
            log.warning("No valid prompt_robustness_agieval_aqua_rat pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single prompt_robustness_agieval_aqua_rat doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:

            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            options = doc.get("options", [])
            gold = doc.get("gold", [])

            if not question or not choices or not options or not gold:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None

            # Use letter answers (A, B, C, D, E)
            incorrect_map = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}
            correct_letter = doc.get("answer", "")
            incorrect_letter = incorrect_map.get(correct_letter, "B")

            correct = f"The best answer is {correct_letter}"
            incorrect = f"The best answer is {incorrect_letter}"

            choices_str = "\n".join(choices)

            formatted_question = f"""{question}

{choices_str}

Examine the question and choose the correct answer from the options 'A', 'B', 'C', 'D' or 'E'. End your answer with:
The best answer is [the_answer_letter].
where the [the_answer_letter] is a letter from A to E."""

            metadata = {
                "label": "agieval_aqua_rat",
            }

            return self._build_pair(
                question=formatted_question,
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