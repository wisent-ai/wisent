from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AfroBenchMultipleChoiceExtractor"]
_LOG = setup_logger(__name__)

# This extractor handles all afrobench MC subtasks dynamically
task_names = ()  # Intentionally empty - will match any afrobench MC task

# Ordered list of (condition, handler) used by _extract_pair_from_doc
# Each handler receives (doc, task_choices, log) and returns ContrastivePair|None
_SCHEMA_DISPATCHERS: list[tuple] = []  # populated after class definition


class AfroBenchMultipleChoiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AfroBench multiple-choice benchmarks.

    Handles the variety of document schemas used across afrobench subtasks:

    - belebele:        mc_answer1-4 + correct_answer_num (1-indexed int)
    - openai_mmlu /
      naijarc /
      uhura_arc_easy:  A/B/C/D text fields + Answer/answerKey letter
    - afrisenti /
      nollysenti /
      sib /
      masakhanews /
      injongointent:   text field + label/category/intent int index,
                       choices come from the lm-eval task config (doc_to_choice)
    - generic question/choices list:  question + choices list + answer int/letter
    """

    evaluator_name = "log_likelihoods"

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
        *,
        train_ratio: float,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        max_items = self._normalize_limit(limit)

        docs = self.load_docs(lm_eval_task_data, max_items, preferred_doc=preferred_doc, train_ratio=train_ratio)

        # Extract static choices list from the task YAML config when present
        # (needed for classification tasks like afrisenti, sib, masakhanews, etc.)
        task_choices = _get_task_choices(lm_eval_task_data)

        pairs: list[ContrastivePair] = []
        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, task_choices=task_choices)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid AfroBench MC pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(
        self,
        doc: dict[str, Any],
        task_choices: list[str] | None = None,
    ) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Schema 1: belebele — mc_answer1-4 + correct_answer_num (1-indexed)
            if "mc_answer1" in doc and "correct_answer_num" in doc:
                return _extract_belebele(doc, log)

            # Schema 2: A/B/C/D explicit choice fields (openai_mmlu)
            if "A" in doc and "B" in doc and "C" in doc and "D" in doc:
                return _extract_abcd_choices(doc, log)

            # Schema 2b: options_A/B/C/D explicit choice fields (naijarc)
            if "options_A" in doc and "options_B" in doc:
                return _extract_options_abcd_choices(doc, log)

            # Schema 3: classification — static choices from task YAML + int/str index in doc
            # (afrisenti, nollysenti, sib, masakhanews, injongointent)
            if task_choices is not None:
                pair = _extract_classification(doc, task_choices, log)
                if pair is not None:
                    return pair

            # Schema 4a: uhura-arc-easy — choices dict with text array + answerKey letter
            if "choices" in doc and isinstance(doc.get("choices"), dict) and "answerKey" in doc:
                return _extract_choices_dict(doc, log)

            # Schema 4b: generic list-of-choices in the doc itself
            if "choices" in doc:
                return _extract_choices_list(doc, log)

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})

        return None


# ---------------------------------------------------------------------------
# Module-level helpers (not methods, to keep the class body concise)
# ---------------------------------------------------------------------------

def _get_task_choices(lm_eval_task_data: Any) -> list[str] | None:
    """Return the static choices list from the lm-eval task config, if available.

    In lm-eval, doc_to_choice is typically a callable method. For classification tasks
    (like sib, afrisenti), we need to extract the choices either from:
    1. A static list in config.doc_to_choice (if it exists as an attribute)
    2. By calling the doc_to_choice method with a sample document
    """
    try:
        # Try to get doc_to_choice as a static attribute first (from config)
        config = getattr(lm_eval_task_data, "config", None)
        if config is not None:
            choices = getattr(config, "doc_to_choice", None)
            if isinstance(choices, list) and choices and all(isinstance(c, str) for c in choices):
                return choices

        # Try to call doc_to_choice as a method to get choices
        # This works for classification tasks where doc_to_choice is a callable
        if hasattr(lm_eval_task_data, "doc_to_choice") and callable(lm_eval_task_data.doc_to_choice):
            # Get the first document from the dataset to extract choices
            dataset = getattr(lm_eval_task_data, "dataset", None)
            if dataset is not None:
                # Try different split names
                for split_name in ["train", "validation", "test"]:
                    if split_name in dataset:
                        split_data = dataset[split_name]
                        if len(split_data) > 0:
                            sample_doc = split_data[0]
                            choices = lm_eval_task_data.doc_to_choice(sample_doc)
                            if isinstance(choices, list) and choices:
                                # Convert all choices to strings
                                str_choices = [str(c).strip() for c in choices if c]
                                if str_choices:
                                    return str_choices
    except Exception:
        pass
    return None


def _make_pair(prompt: str, correct: str, incorrect: str) -> ContrastivePair:
    return ContrastivePair(
        prompt=prompt,
        positive_response=PositiveResponse(model_response=correct),
        negative_response=NegativeResponse(model_response=incorrect),
        label="afrobench_mc",
    )


def _extract_belebele(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """belebele schema: mc_answer1-4 + correct_answer_num (1-indexed)."""
    question = str(doc.get("question", "")).strip()
    choices = [
        str(doc.get("mc_answer1", "")).strip(),
        str(doc.get("mc_answer2", "")).strip(),
        str(doc.get("mc_answer3", "")).strip(),
        str(doc.get("mc_answer4", "")).strip(),
    ]
    answer_num = doc.get("correct_answer_num")
    if not question or answer_num is None:
        log.debug("belebele: missing question or correct_answer_num")
        return None
    try:
        answer_idx = int(answer_num) - 1  # 1-indexed → 0-indexed
    except (TypeError, ValueError):
        log.debug("belebele: non-integer correct_answer_num", extra={"val": answer_num})
        return None
    if not (0 <= answer_idx < len(choices)):
        return None
    correct = choices[answer_idx]
    incorrect = choices[(answer_idx + 1) % len(choices)]
    return _make_pair(f"Question: {question}\nAnswer:", correct, incorrect)


def _extract_abcd_choices(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """A/B/C/D schema (openai_mmlu)."""
    question = str(doc.get("Question") or doc.get("question") or "").strip()
    choices_map = {
        "A": str(doc.get("A", "")).strip(),
        "B": str(doc.get("B", "")).strip(),
        "C": str(doc.get("C", "")).strip(),
        "D": str(doc.get("D", "")).strip(),
    }
    answer_key = doc.get("Answer") or doc.get("answerKey") or doc.get("answer")
    if not question or answer_key is None:
        log.debug("abcd: missing question or answer key")
        return None
    answer_letter = str(answer_key).strip().upper()
    if answer_letter not in choices_map:
        log.debug("abcd: answer letter not in A-D", extra={"key": answer_key})
        return None
    correct = choices_map[answer_letter]
    if not correct:
        return None
    other_letters = [l for l in ("A", "B", "C", "D") if l != answer_letter]
    incorrect = choices_map[other_letters[0]]
    return _make_pair(f"Question: {question}\nAnswer:", correct, incorrect)


def _extract_classification(
    doc: dict[str, Any],
    task_choices: list[str],
    log: Any,
) -> ContrastivePair | None:
    """Classification schema: text field + int index in doc, choices from task config.

    Supports: afrisenti (tweet/label), nollysenti (review/label),
    sib (text/category), masakhanews (headline_text/label),
    injongointent (utterance/intent).
    """
    text = str(
        doc.get("tweet")
        or doc.get("review")
        or doc.get("en_review")
        or doc.get("text")
        or doc.get("headline_text")
        or doc.get("headline")
        or doc.get("utterance")
        or doc.get("sentence")
        or doc.get("question")
        or doc.get("Question")
        or ""
    ).strip()

    # Determine answer index from label/category/intent/sentiment
    answer_idx_raw = None
    for field in ("label", "category", "intent", "sentiment"):
        val = doc.get(field)
        if val is not None:
            answer_idx_raw = val
            break

    if not text or answer_idx_raw is None:
        log.debug("classification: missing text or label field")
        return None

    if isinstance(answer_idx_raw, int):
        answer_idx = answer_idx_raw
    elif isinstance(answer_idx_raw, str):
        lower_choices = [c.lower() for c in task_choices]
        if answer_idx_raw.lower() in lower_choices:
            answer_idx = lower_choices.index(answer_idx_raw.lower())
        else:
            try:
                answer_idx = int(answer_idx_raw)
            except ValueError:
                log.debug("classification: cannot parse label", extra={"label": answer_idx_raw})
                return None
    else:
        log.debug("classification: unexpected label type", extra={"type": type(answer_idx_raw).__name__})
        return None

    if not (0 <= answer_idx < len(task_choices)):
        log.debug("classification: idx out of range", extra={"idx": answer_idx})
        return None

    correct = task_choices[answer_idx]
    incorrect = task_choices[(answer_idx + 1) % len(task_choices)]
    return _make_pair(f"Text: {text}\nLabel:", correct, incorrect)


def _extract_choices_list(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """Schema where a 'choices' list is embedded in the document itself."""
    question = str(doc.get("question") or doc.get("Question") or "").strip()
    choices = doc.get("choices", [])
    answer = doc.get("answer")
    if answer is None:
        answer = doc.get("answerKey")
    if answer is None:
        answer = doc.get("label")

    if not question or not choices or answer is None:
        log.debug("choices_list: missing fields")
        return None

    if isinstance(answer, int):
        answer_idx = answer
    elif isinstance(answer, str):
        if answer.upper() in ("A", "B", "C", "D", "E"):
            answer_idx = ord(answer.upper()) - ord("A")
        else:
            try:
                answer_idx = int(answer)
            except ValueError:
                log.debug("choices_list: cannot parse answer", extra={"answer": answer})
                return None
    else:
        log.debug("choices_list: unexpected answer type")
        return None

    if not (0 <= answer_idx < len(choices)):
        return None

    correct = str(choices[answer_idx]).strip()
    incorrect = str(choices[(answer_idx + 1) % len(choices)]).strip()
    return _make_pair(f"Question: {question}\nAnswer:", correct, incorrect)


def _extract_options_abcd_choices(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """options_A/B/C/D schema (NaijaRC): story/question + options_A-D + Answer letter."""
    question = str(doc.get("question") or doc.get("Question") or "").strip()
    story = str(doc.get("story") or doc.get("context") or "").strip()
    prompt_text = f"Context: {story}\nQuestion: {question}" if story else f"Question: {question}"
    choices_map = {
        "A": str(doc.get("options_A", "")).strip(),
        "B": str(doc.get("options_B", "")).strip(),
        "C": str(doc.get("options_C", "")).strip(),
        "D": str(doc.get("options_D", "")).strip(),
    }
    answer_key = doc.get("Answer") or doc.get("answerKey") or doc.get("answer")
    if not question or answer_key is None:
        log.debug("options_abcd: missing question or answer key")
        return None
    answer_letter = str(answer_key).strip().upper()
    if answer_letter not in choices_map or not choices_map.get(answer_letter):
        log.debug("options_abcd: answer letter not in A-D", extra={"key": answer_key})
        return None
    correct = choices_map[answer_letter]
    other_letters = [l for l in ("A", "B", "C", "D") if l != answer_letter and choices_map.get(l)]
    if not other_letters:
        return None
    incorrect = choices_map[other_letters[0]]
    return _make_pair(f"{prompt_text}\nAnswer:", correct, incorrect)


def _extract_choices_dict(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """choices-dict schema (uhura-arc-easy): choices dict with text array + answerKey letter."""
    question = str(doc.get("question") or doc.get("Question") or "").strip()
    choices_container = doc.get("choices")
    if not isinstance(choices_container, dict):
        log.debug("choices_dict: choices is not a dict")
        return None
    choices = choices_container.get("text") or choices_container.get("label") or []
    if not choices:
        log.debug("choices_dict: empty choices text array")
        return None
    answer_key = doc.get("answerKey") or doc.get("Answer") or doc.get("answer")
    if not question or answer_key is None:
        log.debug("choices_dict: missing question or answerKey")
        return None
    answer_letter = str(answer_key).strip().upper()
    if answer_letter in ("A", "B", "C", "D", "E"):
        answer_idx = ord(answer_letter) - ord("A")
    else:
        try:
            answer_idx = int(answer_key)
        except (TypeError, ValueError):
            log.debug("choices_dict: cannot parse answerKey", extra={"key": answer_key})
            return None
    if not (0 <= answer_idx < len(choices)):
        log.debug("choices_dict: answer_idx out of range", extra={"idx": answer_idx})
        return None
    correct = str(choices[answer_idx]).strip()
    incorrect = str(choices[(answer_idx + 1) % len(choices)]).strip()
    return _make_pair(f"Question: {question}\nAnswer:", correct, incorrect)
