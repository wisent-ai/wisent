from __future__ import annotations

import random
from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["CoQAExtractor"]
_LOG = setup_logger(__name__)


class CoQAExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the CoQA benchmark."""

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from CoQA docs.

        CoQA schema:
            - story: str 
            - questions: list
            - answers: list
            
        Args:
            lm_eval_task_data: lm-eval task instance for CoQA.
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
            log.warning("No valid CoQA pairs extracted", extra={"task": task_name})

        return pairs
    
    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single CoQA doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            story = str(doc.get("story", ""))
            questions = doc.get("questions", {})
            answers = doc.get("answers", {})

            if not story or not questions or not answers:
                log.debug(
                    "Skipping doc due to missing/invalid fields",
                    extra={"doc": doc},
                )
                return None
            
            qs = questions["input_text"]
            asw = answers["input_text"]

            lines = []
            lines.append(story.strip())
            lines.append("")  

            pairs_count = max(0, min(len(qs) - 1, len(asw)))
            for q, a in zip(qs[:pairs_count], asw[:pairs_count]):
                lines.append(f"Q: {q}")
                lines.append(f"A: {a}")

            if qs:
                lines.append(f"Q: {qs[-1]}")

            prompt = "\n".join(lines)
            prompt = f"{prompt}\nA:"

            correct = asw[-1] if len(asw) == len(qs) else "no"
            incorrect = None
            # Generate incorrect answer
            try:
                # Try to convert to number
                num = float(correct)
                # Check if it's an integer
                if num.is_integer():
                    incorrect = str(int(num) + 1)
                else:
                    incorrect = str(num + 1)
            except ValueError:
                # It's a string, shuffle the letters until different
                letters = list(correct)
                incorrect = correct
                random.shuffle(letters)
                incorrect = ''.join(letters)
                if incorrect == correct:
                    incorrect += "k"

                # Ensure correct answer is not a substring of incorrect answer
                # If it is, replace incorrect answer completely with something different
                if correct.lower() in incorrect.lower() or incorrect.lower() in correct.lower():
                    # Generate a completely different answer
                    if correct.lower() in ["yes", "y"]:
                        incorrect = "no"
                    elif correct.lower() in ["no", "n"]:
                        incorrect = "yes"
                    elif correct.lower() in ["true", "t"]:
                        incorrect = "false"
                    elif correct.lower() in ["false", "f"]:
                        incorrect = "true"
                    else:
                        # Generic fallback: negate or add "not "
                        incorrect = f"not {correct}"

            metadata = {
                "label": "coqa",
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