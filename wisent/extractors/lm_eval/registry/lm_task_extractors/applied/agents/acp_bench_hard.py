from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AcpBenchHardExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "acp_bench_hard",
    # Gen variants (acp_bench_hard subtasks)
    "acp_prog_gen", "acp_reach_gen", "acp_app_gen", "acp_just_gen",
    "acp_land_gen", "acp_nexta_gen", "acp_areach_gen", "acp_val_gen",
    # Gen variants with PDDL
    "acp_prog_gen_with_pddl", "acp_reach_gen_with_pddl", "acp_app_gen_with_pddl", "acp_just_gen_with_pddl",
    "acp_land_gen_with_pddl", "acp_nexta_gen_with_pddl", "acp_areach_gen_with_pddl", "acp_val_gen_with_pddl",
)

class AcpBenchHardExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Acp Bench Hard benchmark."""


    evaluator_name = "generation"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Acp Bench Hard docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Acp Bench Hard.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)

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
            log.warning("No valid Acp Bench Hard pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Acp Bench Hard doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer (only if choices is non-empty)
            # We check if choices would be non-empty before committing to this format
            choices_data = doc.get("choices", {}) if "choices" in doc else None
            if choices_data is not None:
                if isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                else:
                    choices = []
            else:
                choices = None

            if "question" in doc and choices:
                question = str(doc.get("question", "")).strip()
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

            # Format 3: context + question + answer/output (structured dict for _gen tasks or yes/no)
            elif "context" in doc and "question" in doc and ("answer" in doc or "output" in doc):
                context = str(doc.get("context", "")).strip()
                question = str(doc.get("question", "")).strip()
                answer_raw = doc.get("answer", doc.get("output", ""))
                pddl = doc.get("pddl", "")

                # Create full prompt with context and optional PDDL
                if pddl:
                    pddl = str(pddl).strip()
                    full_prompt = f"PDDL:\n{pddl}\n\nContext: {context}\n\nQuestion: {question}"
                else:
                    full_prompt = f"Context: {context}\n\nQuestion: {question}"

                # Format 3a: Array/List of answers (for generative tasks like acp_reach_gen)
                if isinstance(answer_raw, list) and len(answer_raw) > 0:
                    # Convert list to string representation for pair extraction
                    correct_answer = str(answer_raw).strip()
                    if correct_answer:
                        # Create incorrect answer by negating the answer or using alternatives
                        if len(answer_raw) > 1:
                            # If there are multiple answers, use a subset as incorrect
                            incorrect_answer = str(answer_raw[1:]).strip()
                        else:
                            # Single answer - create negation
                            first_answer = str(answer_raw[0]).strip()
                            if first_answer.lower() in ["yes", "true"]:
                                incorrect_answer = "no"
                            elif first_answer.lower() in ["no", "false"]:
                                incorrect_answer = "yes"
                            else:
                                incorrect_answer = f"not {first_answer}"
                        metadata = {"label": "acp_bench_hard"}
                        return self._build_pair(
                            question=full_prompt,
                            correct=correct_answer,
                            incorrect=incorrect_answer,
                            metadata=metadata,
                        )

                # Format 3b: Dict format: {"neg": [...], "pos": [...]} or any other dict
                elif isinstance(answer_raw, dict):
                    if "neg" in answer_raw and "pos" in answer_raw:
                        # For structured generation tasks with explicit neg/pos, use them
                        correct_answer = str(answer_raw)
                        # Create incorrect by swapping pos/neg
                        incorrect_answer = str({"neg": answer_raw.get("pos", []), "pos": answer_raw.get("neg", [])})
                    else:
                        # For any other dict format, use it as-is and create a negated version
                        correct_answer = str(answer_raw)
                        incorrect_answer = "null"  # Negation: expected answer is not the given structure
                    metadata = {"label": "acp_bench_hard"}
                    return self._build_pair(
                        question=full_prompt,
                        correct=correct_answer,
                        incorrect=incorrect_answer,
                        metadata=metadata,
                    )

                # Format 3d: String answers (yes/no or free-form)
                elif isinstance(answer_raw, str):
                    answer = answer_raw.strip().lower()
                    if answer in ["yes", "no"]:
                        correct = answer
                        incorrect = "yes" if answer == "no" else "no"
                        metadata = {"label": "acp_bench_hard"}
                        return self._build_pair(
                            question=full_prompt,
                            correct=correct,
                            incorrect=incorrect,
                            metadata=metadata,
                        )
                    # Free-form string answer
                    elif answer:
                        correct_answer = answer_raw.strip()
                        # Create incorrect by negating or providing placeholder
                        incorrect_answer = "incorrect answer"
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
                    metadata = {"label": "acp_bench_hard"}
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
                "label": "acp_bench_hard",
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
