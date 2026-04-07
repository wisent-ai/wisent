from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AfroBenchCotExtractor"]
_LOG = setup_logger(__name__)

# This extractor handles all afrobench generation/CoT subtasks dynamically
task_names = ()  # Intentionally empty - will match any afrobench generation task


class AfroBenchCotExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AfroBench generation (generate_until) benchmarks.

    Handles multiple document schemas used across afrobench generation subtasks:

    - adr (afridiacritics):  text (input) + target (restored text)
    - afriqa:                 question_lang + answer_pivot
    - masakhaner:             tokens list + ner_tags (NER)
    - masakhapos:             tokens list + pos_tags (POS)
    - translation tasks
      (ntrex):                source + target
      (mafand):               translation dict with language pairs
      (salt, flores, xlsum):  source_sentence/sentence/article + target/summary/translation (string)
    """

    evaluator_name = "generation"

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
            log.warning("No valid AfroBench CoT pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Schema 0: flores-format sentence_* fields — e.g. sentence_eng_Latn, sentence_fra_Latn
            sentence_fields = [k for k in doc if k.startswith("sentence_")]
            if len(sentence_fields) >= 2:
                source_field = sentence_fields[0]
                target_field = sentence_fields[1]
                source_text = str(doc.get(source_field, "")).strip()
                target_text = str(doc.get(target_field, "")).strip()
                if source_text and target_text:
                    incorrect = _make_wrong_answer(target_text)
                    return ContrastivePair(
                        prompt=f"Input: {source_text}\nOutput:",
                        positive_response=PositiveResponse(model_response=target_text),
                        negative_response=NegativeResponse(model_response=incorrect),
                        label="afrobench_cot",
                    )

            # Schema 0b: SALT translation — *_text fields with eng_source_text + eng_target_text
            # Pick first non-eng *_text as source, eng_target_text as target
            text_fields = [k for k in doc if k.endswith("_text")]
            if "eng_target_text" in text_fields and len(text_fields) >= 2:
                source_field = next(
                    (k for k in text_fields if k != "eng_target_text" and k != "eng_source_text"),
                    None,
                )
                if source_field is None:
                    source_field = "eng_source_text" if "eng_source_text" in text_fields else None
                if source_field:
                    source_text = str(doc.get(source_field, "")).strip()
                    target_text = str(doc.get("eng_target_text", "")).strip()
                    if source_text and target_text:
                        incorrect = _make_wrong_answer(target_text)
                        return ContrastivePair(
                            prompt=f"Input: {source_text}\nOutput:",
                            positive_response=PositiveResponse(model_response=target_text),
                            negative_response=NegativeResponse(model_response=incorrect),
                            label="afrobench_cot",
                        )

            # Schema 1: afriqa — question_lang + answer_pivot
            if "question_lang" in doc and "answer_pivot" in doc:
                return _extract_afriqa(doc, log)

            # Schema 2: adr (afridiacritics) — text + target
            # Skip if source exists (to avoid matching translation schema with source + target)
            if "text" in doc and "target" in doc and "source" not in doc:
                return _extract_text_target(doc, "text", "target", log)

            # Schema 2b: translation — source + target (ntrex, etc.)
            if "source" in doc and "target" in doc:
                return _extract_text_target(doc, "source", "target", log)

            # Schema 2c: translation with nested dict (mafand)
            if "translation" in doc and isinstance(doc["translation"], dict):
                return _extract_translation_dict(doc, log)

            # Schema 3: translation — source_sentence + target_sentence
            if "source_sentence" in doc and "target_sentence" in doc:
                return _extract_text_target(doc, "source_sentence", "target_sentence", log)

            # Schema 4: translation variants — sentence/article + translation/summary (string only)
            if "sentence" in doc and isinstance(doc.get("translation"), str):
                return _extract_text_target(doc, "sentence", "translation", log)

            if "article" in doc and "summary" in doc:
                return _extract_text_target(doc, "article", "summary", log)

            # Schema 4b: xlsum variant — text + summary
            if "text" in doc and "summary" in doc and "target" not in doc:
                return _extract_text_target(doc, "text", "summary", log)

            # Schema 5: NER — tokens list + ner_tags
            if "tokens" in doc and "ner_tags" in doc:
                return _extract_sequence_labeling(doc, "tokens", "ner_tags", log)

            # Schema 6: POS — tokens list + pos_tags or upos
            if "tokens" in doc and "pos_tags" in doc:
                return _extract_sequence_labeling(doc, "tokens", "pos_tags", log)
            if "tokens" in doc and "upos" in doc:
                return _extract_sequence_labeling(doc, "tokens", "upos", log)

            # Schema 7: generic question + answer/target field
            question = str(
                doc.get("question") or doc.get("question_lang") or doc.get("input") or ""
            ).strip()
            answer = doc.get("answer") or doc.get("target") or doc.get("answerKey")
            if question and answer:
                correct = str(answer).strip()
                if correct:
                    incorrect = _make_wrong_answer(correct)
                    return ContrastivePair(
                        prompt=f"Question: {question}\nAnswer:",
                        positive_response=PositiveResponse(model_response=correct),
                        negative_response=NegativeResponse(model_response=incorrect),
                        label="afrobench_cot",
                    )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})

        return None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_wrong_answer(correct: str) -> str:
    """Return a plausible incorrect answer for a generation task."""
    if correct.upper() in ("A", "B", "C", "D", "E"):
        options = [l for l in ("A", "B", "C", "D", "E") if l != correct.upper()]
        return options[0]
    # For free-text answers: reverse the text as a clearly wrong answer
    words = correct.split()
    if len(words) > 1:
        return " ".join(reversed(words))
    return correct + " [incorrect]"


def _extract_afriqa(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """afriqa schema: question_lang (question) + answer_pivot (correct answer)."""
    question = str(doc.get("question_lang", "")).strip()
    answer = str(doc.get("answer_pivot", "")).strip()
    if not question or not answer:
        log.debug("afriqa: missing question_lang or answer_pivot")
        return None
    incorrect = _make_wrong_answer(answer)
    return ContrastivePair(
        prompt=f"Question: {question}\nAnswer:",
        positive_response=PositiveResponse(model_response=answer),
        negative_response=NegativeResponse(model_response=incorrect),
        label="afrobench_cot",
    )


def _extract_translation_dict(doc: dict[str, Any], log: Any) -> ContrastivePair | None:
    """Translation schema with nested dict (mafand)."""
    translation = doc.get("translation")
    if not isinstance(translation, dict):
        log.debug("translation_dict: translation is not a dict")
        return None

    lang_keys = list(translation.keys())
    if len(lang_keys) < 2:
        log.debug("translation_dict: insufficient language pairs")
        return None

    # Use first two language pairs
    source_lang, target_lang = lang_keys[0], lang_keys[1]
    source = str(translation.get(source_lang, "")).strip()
    target = str(translation.get(target_lang, "")).strip()

    if not source or not target:
        log.debug("translation_dict: missing source or target", extra={"src_lang": source_lang, "tgt_lang": target_lang})
        return None

    incorrect = _make_wrong_answer(target)
    return ContrastivePair(
        prompt=f"Input: {source}\nOutput:",
        positive_response=PositiveResponse(model_response=target),
        negative_response=NegativeResponse(model_response=incorrect),
        label="afrobench_cot",
    )


def _extract_text_target(
    doc: dict[str, Any],
    source_field: str,
    target_field: str,
    log: Any,
) -> ContrastivePair | None:
    """Generic source→target schema (adr, translation tasks)."""
    source = str(doc.get(source_field, "")).strip()
    target = str(doc.get(target_field, "")).strip()
    if not source or not target:
        log.debug("text_target: missing source or target", extra={"src": source_field, "tgt": target_field})
        return None
    incorrect = _make_wrong_answer(target)
    return ContrastivePair(
        prompt=f"Input: {source}\nOutput:",
        positive_response=PositiveResponse(model_response=target),
        negative_response=NegativeResponse(model_response=incorrect),
        label="afrobench_cot",
    )


def _extract_sequence_labeling(
    doc: dict[str, Any],
    tokens_field: str,
    tags_field: str,
    log: Any,
) -> ContrastivePair | None:
    """NER/POS schema: tokens list + integer tag list."""
    tokens = doc.get(tokens_field)
    tags = doc.get(tags_field)
    if not tokens or not tags:
        log.debug("seq_labeling: missing tokens or tags")
        return None
    if not isinstance(tokens, (list, tuple)):
        log.debug("seq_labeling: tokens is not a list")
        return None

    sentence = " ".join(str(t) for t in tokens)
    correct_tags = " ".join(str(t) for t in tags)

    # Incorrect: shift each tag by 1 (mod over the tag range)
    try:
        tag_ints = [int(t) for t in tags]
        max_tag = max(tag_ints) + 1
        incorrect_tags = " ".join(str((t + 1) % max_tag) for t in tag_ints)
    except (TypeError, ValueError):
        incorrect_tags = correct_tags + " [shifted]"

    return ContrastivePair(
        prompt=f"Sentence: {sentence}\nLabels:",
        positive_response=PositiveResponse(model_response=correct_tags),
        negative_response=NegativeResponse(model_response=incorrect_tags),
        label="afrobench_cot",
    )
