from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["FrenchBenchExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "french_bench",
    "french_bench_arc_challenge",
    "french_bench_boolqa",
    "french_bench_extra",
    "french_bench_fquadv2",
    "french_bench_fquadv2_bool",
    "french_bench_fquadv2_genq",
    "french_bench_fquadv2_hasAns",
    "french_bench_gen",
    "french_bench_grammar",
    "french_bench_hellaswag",
    "french_bench_mc",
    "french_bench_multifquad",
    "french_bench_opus_perplexity",
    "french_bench_orangesum_abstract",
    "french_bench_orangesum_title",
    "french_bench_perplexity",
    "french_bench_reading_comp",
    "french_bench_topic_based_nli",
    "french_bench_trivia",
    "french_bench_vocab",
    "french_bench_wikitext_fr",
    "french_bench_xnli",
)

class FrenchBenchExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the French Bench benchmark."""


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
        Build contrastive pairs from French Bench docs.

        Args:
            lm_eval_task_data: lm-eval task instance for French Bench.
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
            log.warning("No valid French Bench pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single French Bench doc into a ContrastivePair, if possible.
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

            # Format 2b: question + answerA/B/C/D + answer (french_bench_grammar/vocab/reading)
            elif "question" in doc and "answerA" in doc:
                question = str(doc.get("question", "")).strip()
                choices = [
                    str(doc.get("answerA", "")).strip(),
                    str(doc.get("answerB", "")).strip(),
                    str(doc.get("answerC", "")).strip(),
                    str(doc.get("answerD", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("answer", "A")
                ans_str = str(answer).strip().upper()
                answer_idx = ord(ans_str) - ord('A') if ans_str and ans_str.isalpha() and len(ans_str) == 1 else 0

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

            # Format 3: query + choices + gold (hellaswag style)
            elif "query" in doc and "choices" in doc and "gold" in doc:
                question = str(doc.get("query", "")).strip()
                choices = doc.get("choices", [])
                if isinstance(choices, list):
                    answer_idx = int(doc.get("gold", 0))
                else:
                    return None

            # Format 3b: ctx + endings + label (French hellaswag style)
            elif "ctx" in doc and "endings" in doc and "label" in doc:
                question = str(doc.get("ctx", "")).strip()
                choices = doc.get("endings", [])
                if isinstance(choices, list) and choices:
                    label = doc.get("label", "0")
                    answer_idx = int(label) if isinstance(label, (int, str)) and str(label).isdigit() else 0
                else:
                    return None

            # Format 4: Question + Answer (trivia style, capital letters)
            elif "Question" in doc and "Answer" in doc:
                question = str(doc.get("Question", "")).strip()
                correct_answer = str(doc.get("Answer", "")).strip()
                if question and correct_answer:
                    metadata = {"label": "french_bench"}
                    return self._build_pair(
                        question=f"Question: {question}",
                        correct=correct_answer,
                        incorrect="incorrect answer",
                        metadata=metadata,
                    )
                return None

            # Format 4a: NLI (premise + hypothesis + label, e.g. xnli, topic_based_nli)
            elif "premise" in doc and "hypothesis" in doc and "label" in doc:
                premise = str(doc.get("premise", "")).strip()
                hypothesis = str(doc.get("hypothesis", "")).strip()
                label = doc.get("label")
                label_map = {0: "oui", 1: "peut-être", 2: "non"}
                try: l_idx = int(label)
                except Exception: l_idx = -1
                correct_answer = label_map.get(l_idx)
                if premise and hypothesis and correct_answer:
                    incorrect_answer = [v for k, v in label_map.items() if v != correct_answer][0]
                    return self._build_pair(
                        question=f"Prémisse: {premise}\nHypothèse: {hypothesis}\nLa prémisse implique-t-elle lu0027hypothèse?:",
                        correct=correct_answer,
                        incorrect=incorrect_answer,
                        metadata={"label": "french_bench_nli"},
                    )
                return None

            # Format 4b: SQuAD-style (fquadv2, orangesum): context + question + answers.
            elif ("context" in doc or "text" in doc or "article" in doc) and (
                "answers" in doc or "summary" in doc or "title" in doc
            ):
                passage = str(doc.get("context", doc.get("text", doc.get("article", "")))).strip()
                q = str(doc.get("question", "")).strip()
                answers = doc.get("answers")
                correct_answer = None
                if isinstance(answers, dict):
                    texts = answers.get("text") or []
                    if texts:
                        correct_answer = str(texts[0]).strip()
                elif isinstance(answers, list) and answers:
                    correct_answer = str(answers[0]).strip()
                if not correct_answer and doc.get("summary"):
                    correct_answer = str(doc.get("summary", "")).strip()
                if not correct_answer and doc.get("title"):
                    correct_answer = str(doc.get("title", "")).strip()
                # fquadv2 impossible questions have empty answers.text + is_impossible=True
                if not correct_answer and doc.get("is_impossible") is True:
                    correct_answer = "Impossible"
                if correct_answer and passage:
                    prompt_text = f"{passage}\n\nQuestion: {q}" if q else passage
                    import random as _random
                    words = correct_answer.split()
                    if len(words) > 2:
                        _random.seed(hash(correct_answer) % (2**32))
                        shuffled = words.copy()
                        _random.shuffle(shuffled)
                        incorrect_answer = " ".join(shuffled)
                        if incorrect_answer == correct_answer:
                            incorrect_answer = " ".join(words[::-1])
                    else:
                        incorrect_answer = "Réponse incorrecte"
                    if incorrect_answer == correct_answer:
                        incorrect_answer = "Réponse incorrecte"
                    metadata = {"label": "french_bench"}
                    return self._build_pair(
                        question=prompt_text,
                        correct=correct_answer,
                        incorrect=incorrect_answer,
                        metadata=metadata,
                    )
                return None

            # Format 4c: perplexity-style text corpus (french_bench_perplexity, french_bench_wikitext_fr)
            elif "paragraph" in doc or ("text" in doc and "answers" not in doc and "summary" not in doc):
                text = str(doc.get("paragraph", doc.get("text", ""))).strip()
                if not text or len(text.split()) < 6:
                    return None
                import random as _random
                words = text.split()
                if len(words) > 80:
                    words = words[:80]
                    text = " ".join(words)
                mid_start = len(words) // 3
                mid_end = 2 * len(words) // 3
                middle = words[mid_start:mid_end]
                _random.seed(hash(text) % (2**32))
                shuffled_middle = middle.copy()
                _random.shuffle(shuffled_middle)
                corrupted = " ".join(words[:mid_start] + shuffled_middle + words[mid_end:])
                if corrupted == text:
                    corrupted = " ".join(words[::-1])
                    if corrupted == text:
                        return None
                metadata = {"label": "french_bench"}
                return self._build_pair(
                    question="Continue this text:",
                    correct=text,
                    incorrect=corrupted,
                    metadata=metadata,
                )

            # Format 5: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "french_bench"}
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
                "label": "french_bench",
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
