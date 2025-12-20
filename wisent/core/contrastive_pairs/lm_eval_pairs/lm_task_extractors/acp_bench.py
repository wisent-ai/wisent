from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AcpBenchExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "acp_bench",
    "acp_bench_hard",
    "acp_bench_hard_with_pddl",
    # Bool variants
    "acp_prog_bool", "acp_reach_bool", "acp_app_bool", "acp_just_bool",
    "acp_land_bool", "acp_areach_bool", "acp_val_bool",
    # MCQ variants
    "acp_prog_mcq", "acp_reach_mcq", "acp_app_mcq", "acp_just_mcq",
    "acp_land_mcq", "acp_areach_mcq", "acp_val_mcq",
    # Gen variants (acp_bench_hard subtasks)
    "acp_prog_gen", "acp_reach_gen", "acp_app_gen", "acp_just_gen",
    "acp_land_gen", "acp_nexta_gen", "acp_areach_gen", "acp_val_gen",
)

class AcpBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Acp Bench benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Acp Bench docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Acp Bench.
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
            log.warning("No valid Acp Bench pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Acp Bench doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple possible schema formats
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

            # Format 3: context + question + answer (yes/no format for acp_bench)
            elif "context" in doc and "question" in doc and "answer" in doc:
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answer_raw = doc.get("answer", "")

                # Create full prompt with context
                full_prompt = f"Context: {context}\n\nQuestion: {question}"

                # Format 3a: Yes/no format
                if isinstance(answer_raw, str):
                    answer = answer_raw.strip().lower()
                    if answer in ["yes", "no"]:
                        correct = answer
                        incorrect = "yes" if answer == "no" else "no"
                        metadata = {"label": "acp_bench"}
                        return self._build_pair(
                            question=full_prompt,
                            correct=correct,
                            incorrect=incorrect,
                            metadata=metadata,
                        )

                # Format 3b: Structured dict format (acp_bench_hard _gen tasks)
                elif isinstance(answer_raw, dict) and "neg" in answer_raw and "pos" in answer_raw:
                    # For structured generation tasks, use the dict as-is
                    correct_answer = str(answer_raw)
                    # Create incorrect by swapping pos/neg
                    incorrect_answer = str({"neg": answer_raw.get("pos", []), "pos": answer_raw.get("neg", [])})
                    metadata = {"label": "acp_bench_hard"}
                    return self._build_pair(
                        question=full_prompt,
                        correct=correct_answer,
                        incorrect=incorrect_answer,
                        metadata=metadata,
                    )

                return None

            # Format 4: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "acp_bench"}
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
                "label": "acp_bench",
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
