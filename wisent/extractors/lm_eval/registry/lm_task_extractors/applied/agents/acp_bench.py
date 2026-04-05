from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AcpBenchExtractor"]
_LOG = setup_logger(__name__)

# Regex to extract yes/no from end of chain-of-thought answer strings.
# Matches patterns like "**Final Answer**: No." or a bare "Yes"/"No".
_YESNO_PATTERN = re.compile(
    r"\*\*Final Answer\*\*:\s*(yes|no)"
    r"|(?:the answer is|The answer is|The answer:)\s*(yes|no)"
    r"|\b(yes|no)\b",
    re.IGNORECASE,
)

# Regex to extract the final MCQ letter (A/B/C/D) from a chain-of-thought answer.
_MCQ_LETTER_PATTERN = re.compile(
    r"\*\*Final Answer\*\*:\s*([A-D])"
    r"|(?:the answer is|The answer is|answer is)\s*([A-D])"
    r"|\b([A-D])\b",
    re.IGNORECASE,
)

# Regex to parse MCQ choices embedded in the question field.
# Handles formats like "A. text  B. text" or "A) text  B) text"
_MCQ_CHOICE_PATTERN = re.compile(
    r"(?:^|[ \t])([A-D])[.)]\s*(.+?)(?=\s+[A-D][.)]|$)",
    re.DOTALL,
)

task_names = (
    "acp_bench",
    "acp_bench_hard",
    # Bool variants
    "acp_prog_bool", "acp_reach_bool", "acp_app_bool", "acp_just_bool",
    "acp_land_bool", "acp_areach_bool", "acp_val_bool",
    # MCQ variants
    "acp_prog_mcq", "acp_reach_mcq", "acp_app_mcq", "acp_just_mcq",
    "acp_land_mcq", "acp_areach_mcq", "acp_val_mcq",
    # Gen variants (acp_bench_hard subtasks) - handled by AcpBenchHardExtractor
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
        *,
        train_ratio: float,
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
            log.warning("No valid Acp Bench pairs extracted", extra={"task": task_name})

        return pairs

    @staticmethod
    def _extract_yesno(answer_raw: str) -> str | None:
        """Extract 'yes' or 'no' from a chain-of-thought answer string.

        The acp_bench bool tasks store a full CoT reasoning string as the answer,
        e.g. "Let's think step by step. ... **Final Answer**: No."
        This helper extracts the final yes/no decision from such strings.
        """
        match = _YESNO_PATTERN.search(answer_raw)
        if not match:
            return None
        for group in match.groups():
            if group is not None:
                return group.lower()
        return None

    @staticmethod
    def _extract_mcq_letter(answer_raw: str) -> str | None:
        """Extract the final MCQ letter (A-D) from a chain-of-thought answer string.

        The acp_bench mcq tasks store a full CoT reasoning string as the answer,
        e.g. "Let's think step by step. ... **Final Answer**: D."
        This helper extracts the letter from such strings, preferring the
        **Final Answer** marker over any incidental letter mentions.
        """
        best = None
        for match in _MCQ_LETTER_PATTERN.finditer(answer_raw):
            for group in match.groups():
                if group is not None:
                    best = group.upper()
                    break
        return best

    @staticmethod
    def _extract_mcq_choices(question_text: str) -> dict[str, str]:
        """Parse MCQ choices embedded in the question field.

        ACP Bench MCQ questions embed choices in the question text, e.g.:
        "Which facts hold? **Possible Answers**: A. Fact X  B. Fact Y  C. Fact Z  D. None."
        Returns a dict mapping letter to choice text.
        """
        choices: dict[str, str] = {}
        for match in _MCQ_CHOICE_PATTERN.finditer(question_text):
            letter = match.group(1).upper()
            text = match.group(2).strip().rstrip(".")
            if text:
                choices[letter] = text
        return choices

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Acp Bench doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Format 1: question + choices + answer (structured MCQ with separate choices field)
            # Only match if choices is actually non-empty
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
                if not question or not choices or not (0 <= answer_idx < len(choices)):
                    log.debug("Skipping doc (Format 1): missing/invalid fields", extra={"doc": doc})
                    return None
                correct = str(choices[answer_idx]).strip()
                incorrect = str(choices[(answer_idx + 1) % len(choices)]).strip()
                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "acp_bench"},
                )

            # Format 2: instruction + option_a/b/c/d + answer (MMMLU style)
            if "instruction" in doc and "option_a" in doc:
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
                if not question or not choices or not (0 <= answer_idx < len(choices)):
                    log.debug("Skipping doc (Format 2): missing/invalid fields", extra={"doc": doc})
                    return None
                correct = choices[answer_idx]
                incorrect = choices[(answer_idx + 1) % len(choices)]
                return self._build_pair(
                    question=question,
                    correct=correct,
                    incorrect=incorrect,
                    metadata={"label": "acp_bench"},
                )

            # Format 3: context + question + answer/output (primary ACP Bench format)
            # Used by bool tasks (yes/no CoT answer), MCQ tasks (letter CoT answer), and gen tasks.
            if "context" in doc and "question" in doc and ("answer" in doc or "output" in doc):
                context = str(doc.get("context", "")).strip()
                question_text = str(doc.get("question", "")).strip()
                answer_raw = doc.get("answer", doc.get("output", ""))
                full_prompt = f"Context: {context}\n\nQuestion: {question_text}"

                if not isinstance(answer_raw, str):
                    log.debug("Skipping doc (Format 3): non-string answer", extra={"doc": doc})
                    return None

                # Format 3a: Boolean (yes/no) task — extract yes/no from CoT answer.
                # Bool task answers are stored as full chain-of-thought strings ending in
                # "**Final Answer**: Yes." or "**Final Answer**: No."
                yesno = self._extract_yesno(answer_raw)
                if yesno is not None:
                    correct = yesno
                    incorrect = "yes" if yesno == "no" else "no"
                    return self._build_pair(
                        question=full_prompt,
                        correct=correct,
                        incorrect=incorrect,
                        metadata={"label": "acp_bench"},
                    )

                # Format 3b: MCQ task — extract letter from CoT answer, choices from question text.
                # MCQ task answers are stored as full CoT strings ending in "**Final Answer**: D."
                # Choices are embedded in the question field as "A. text  B. text  C. text  D. text".
                mcq_letter = self._extract_mcq_letter(answer_raw)
                if mcq_letter is not None:
                    choices = self._extract_mcq_choices(question_text)
                    correct_text = choices.get(mcq_letter)
                    if correct_text:
                        other_letters = [l for l in ("A", "B", "C", "D") if l != mcq_letter and l in choices]
                        incorrect_text = choices[other_letters[0]] if other_letters else f"Not {mcq_letter}"
                        return self._build_pair(
                            question=full_prompt,
                            correct=correct_text,
                            incorrect=incorrect_text,
                            metadata={"label": "acp_bench"},
                        )
                    # Choices could not be parsed from question; use raw letter as answer text
                    other_letters = [l for l in ("A", "B", "C", "D") if l != mcq_letter]
                    incorrect_letter = other_letters[0] if other_letters else "A"
                    return self._build_pair(
                        question=full_prompt,
                        correct=mcq_letter,
                        incorrect=incorrect_letter,
                        metadata={"label": "acp_bench"},
                    )

                log.debug("Skipping doc (Format 3): could not extract answer", extra={"doc": doc})
                return None

            # Format 4: query/prompt + answer
            if "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata={"label": "acp_bench"},
                    )
                return None

            log.debug("Skipping doc: no recognized field schema", extra={"doc": doc})
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
