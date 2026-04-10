from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["LeaderboardExtractor"]
_LOG = setup_logger(__name__)

task_names = ("leaderboard",)
class LeaderboardExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Leaderboard benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Leaderboard docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Leaderboard.
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
            log.warning("No valid Leaderboard pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Leaderboard doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # leaderboard_math schema: problem + answer (+ solution + level + type)
            if "problem" in doc and "answer" in doc:
                problem = str(doc.get("problem", "")).strip()
                answer = str(doc.get("answer", "")).strip()
                if problem and answer:
                    words = answer.split()
                    incorrect = " ".join(reversed(words)) if len(words) > 1 else "incorrect"
                    return self._build_pair(
                        question=f"Problem: {problem}\n\nAnswer:",
                        correct=answer,
                        incorrect=incorrect,
                        metadata={"label": "leaderboard_math"},
                    )

            # leaderboard_mmlu_pro schema: question + options (list) + answer (letter)
            if "question" in doc and "options" in doc and "answer" in doc and "choices" not in doc:
                q = str(doc.get("question", "")).strip()
                options = doc.get("options", [])
                ans_letter = str(doc.get("answer", "")).strip().upper()
                if q and options and ans_letter and len(ans_letter) == 1:
                    ans_idx = ord(ans_letter) - ord("A")
                    if 0 <= ans_idx < len(options):
                        correct = str(options[ans_idx]).strip()
                        incorrect = str(options[(ans_idx + 1) % len(options)]).strip()
                        return self._build_pair(
                            question=q,
                            correct=correct,
                            incorrect=incorrect,
                            metadata={"label": "leaderboard_mmlu_pro"},
                        )

            # leaderboard_ifeval schema: prompt + instruction_id_list + kwargs.
            # No ground-truth answer (model is judged by whether output follows
            # the instructions). Use prompt as question with a placeholder pair.
            if "prompt" in doc and "instruction_id_list" in doc:
                prompt_text = str(doc.get("prompt", "")).strip()
                if not prompt_text:
                    return None
                instructions = doc.get("instruction_id_list", [])
                instruction_summary = ", ".join(str(i) for i in instructions[:3]) if instructions else "instructions"
                return self._build_pair(
                    question=prompt_text,
                    correct=f"<follows {instruction_summary}>",
                    incorrect=f"<violates {instruction_summary}>",
                    metadata={"label": "leaderboard_ifeval"},
                )

            # leaderboard_bbh schema: input + target (the doc_to_choice list comes
            # from the yaml, not the doc). Use target as the correct answer with a
            # synthetic incorrect (different non-target value if applicable).
            if "input" in doc and "target" in doc and "choices" not in doc:
                input_text = str(doc.get("input", "")).strip()
                target = str(doc.get("target", "")).strip()
                if not input_text or not target:
                    return None
                # Pick a synthetic incorrect by inverting the common boolean answers
                # or reversing characters of the target.
                _common_inversions = {
                    "True": "False", "False": "True",
                    "true": "false", "false": "true",
                    "Yes": "No", "No": "Yes",
                    "yes": "no", "no": "yes",
                    "(A)": "(B)", "(B)": "(A)",
                }
                incorrect = _common_inversions.get(target, target[::-1] if len(target) > 1 else target + "_alt")
                return self._build_pair(
                    question=f"Q: {input_text}\nA:",
                    correct=target,
                    incorrect=incorrect,
                    metadata={"label": "leaderboard_bbh"},
                )

            # Try multiple possible schema formats
            question = None
            choices = None
            answer_idx = None

            # Format 1: question + choices + answer_index (leaderboard_musr format)
            if "question" in doc and "choices" in doc and "answer_index" in doc:
                question = str(doc.get("question", "")).strip()
                # Add narrative if available
                narrative = doc.get("narrative", "")
                if narrative:
                    question = f"{narrative}\n{question}"
                choices_data = doc.get("choices", {})
                if isinstance(choices_data, str):
                    # Choices might be a string representation of a list
                    import ast
                    try:
                        choices = ast.literal_eval(choices_data)
                    except:
                        choices = []
                elif isinstance(choices_data, dict):
                    choices = choices_data.get("text", [])
                elif isinstance(choices_data, list):
                    choices = choices_data
                else:
                    choices = []
                answer_idx = int(doc.get("answer_index", 0))

            # Format 2: question + choices + answer
            elif "question" in doc and "choices" in doc:
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

            # Format 3: Question + choice1/choice2/choice3/choice4 + answer (leaderboard_gpqa format)
            elif "Question" in doc and "choice1" in doc:
                question = str(doc.get("Question", "")).strip()
                choices = [
                    str(doc.get("choice1", "")).strip(),
                    str(doc.get("choice2", "")).strip(),
                    str(doc.get("choice3", "")).strip(),
                    str(doc.get("choice4", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "(A)")
                # Extract letter from answer format like "(A)" or "A"
                answer_str = str(answer).strip()
                if answer_str.startswith("(") and answer_str.endswith(")"):
                    answer_str = answer_str[1:-1]
                answer_idx = ord(answer_str.upper()) - ord('A')

            # Format 4: instruction + option_a/b/c/d + answer (MMMLU style)
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
                    metadata = {"label": "leaderboard"}
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
                "label": "leaderboard",
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
