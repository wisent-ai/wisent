from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["SIQAExtractor"]
_LOG = setup_logger(__name__)

task_names = ("siqa", "siqa_ca", "bigbench_social_iqa_multiple_choice")

class SIQAExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the SIQA (Social IQA) benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from SIQA docs.

        SIQA schema (similar to social_iqa already implemented):
            - context: str
            - question: str
            - answerA, answerB, answerC: str (answer choices)
            - label: str (correct answer: "1", "2", or "3")

        Args:
            lm_eval_task_data: lm-eval task instance for SIQA.
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
            log.warning("No valid SIQA pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single SIQA doc into a ContrastivePair.

        Supports two formats:
        1. Standard SIQA format (siqa, siqa_ca):
            - context: str
            - question: str
            - answerA, answerB, answerC: str (answer choices)
            - label: str (correct answer: "1", "2", or "3")

        2. BigBench format (bigbench_social_iqa_multiple_choice):
            - inputs: Full question text with choices
            - multiple_choice_targets: List of all answer choices
            - multiple_choice_scores: Binary scores (0 for incorrect, 1 for correct)
        """
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"))

        try:
            # Format 1: Standard SIQA format (context + question + answerA/B/C + label)
            if "context" in doc and "question" in doc and "answerA" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answerA = str(doc.get("answerA", "")).strip()
                answerB = str(doc.get("answerB", "")).strip()
                answerC = str(doc.get("answerC", "")).strip()
                label_str = str(doc.get("label", "")).strip()

                if not all([context, question, answerA, answerB, answerC, label_str]):
                    log.debug(
                        "Skipping doc due to missing/invalid fields",
                        extra={"doc": doc},
                    )
                    return None

                # Parse label (1, 2, or 3)
                try:
                    label_idx = int(label_str) - 1  # Convert 1-indexed to 0-indexed
                except (ValueError, TypeError):
                    log.debug(
                        "Skipping doc due to invalid label",
                        extra={"doc": doc, "label": label_str},
                    )
                    return None

                choices = [answerA, answerB, answerC]
                if not (0 <= label_idx < len(choices)):
                    log.debug(
                        "Skipping doc due to label out of range",
                        extra={"doc": doc, "label_idx": label_idx},
                    )
                    return None

                correct = choices[label_idx]
                # Use next choice as incorrect (wrap around)
                incorrect_idx = (label_idx + 1) % len(choices)
                incorrect = choices[incorrect_idx]

                prompt = f"Context: {context}\nQuestion: {question}"

                metadata = {
                    "label": "siqa",
                }

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            # Format 2: BigBench format (inputs + multiple_choice_targets/scores)
            elif "inputs" in doc and "multiple_choice_targets" in doc:
                inputs = str(doc.get("inputs", "")).strip()
                choices = doc.get("multiple_choice_targets", [])
                scores = doc.get("multiple_choice_scores", [])

                if not inputs or not choices or not scores:
                    log.debug(
                        "Skipping doc due to missing/invalid fields",
                        extra={"doc": doc},
                    )
                    return None

                # Find correct and incorrect answers
                correct_indices = [i for i, score in enumerate(scores) if score == 1]
                incorrect_indices = [i for i, score in enumerate(scores) if score == 0]

                if not correct_indices or not incorrect_indices:
                    log.debug(
                        "Skipping doc due to missing correct/incorrect answers",
                        extra={"doc": doc},
                    )
                    return None

                # Use first correct and first incorrect
                correct = str(choices[correct_indices[0]]).strip()
                incorrect = str(choices[incorrect_indices[0]]).strip()

                # Extract question from inputs (remove the "choice:" lines)
                question_lines = []
                for line in inputs.split('\n'):
                    if line.strip() and not line.strip().startswith('choice:'):
                        question_lines.append(line.strip())
                prompt = '\n'.join(question_lines)

                metadata = {
                    "label": "siqa",
                }

                return self._build_pair(
                    question=prompt,
                    correct=correct,
                    incorrect=incorrect,
                    metadata=metadata,
                )

            else:
                log.debug(
                    "Skipping doc due to unknown format",
                    extra={"doc": doc, "keys": list(doc.keys())},
                )
                return None

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
