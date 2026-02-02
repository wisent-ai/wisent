from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["EvalitaLlmExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "evalita-mp",      # Parent group (alias: Evalita-LLM) - all tasks
    "evalita-mp_gen",  # Only generative tasks subgroup
    "evalita-mp_mc",   # Only perplexity-based tasks subgroup
)

class EvalitaLlmExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Evalita-LLM benchmark - Italian LLM evaluation tasks.

    Evalita-LLM is a benchmark for evaluating Large Language Models on Italian.
    It includes both multiple-choice and generative tasks across various domains.

    This extractor handles the parent groups and individual tasks not covered by
    specific extractors (evalita-mp and evalita-sp have their own extractors).
    """


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Evalita Llm docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Evalita Llm.
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
            log.warning("No valid Evalita Llm pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Evalita Llm doc into a ContrastivePair, if possible.
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

            # Format 3a: Question + A/B/C/D/E + Correct (Evalita AT style - uppercase keys, 5 choices)
            elif "Question" in doc and "A" in doc and "Correct" in doc:
                question = str(doc.get("Question", "")).strip()
                choices = [
                    str(doc.get("A", "")).strip(),
                    str(doc.get("B", "")).strip(),
                    str(doc.get("C", "")).strip(),
                    str(doc.get("D", "")).strip(),
                    str(doc.get("E", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("Correct", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')
                if 0 <= answer_idx < len(choices):
                    # Valid format, will be processed below
                    pass
                else:
                    return None

            # Format 3b: question + A/B/C/D + correct_answer (Evalita FAQ style)
            elif "question" in doc and "A" in doc and "correct_answer" in doc:
                question = str(doc.get("question", "")).strip()
                choices = [
                    str(doc.get("A", "")).strip(),
                    str(doc.get("B", "")).strip(),
                    str(doc.get("C", "")).strip(),
                    str(doc.get("D", "")).strip(),
                ]
                choices = [c for c in choices if c]
                answer = doc.get("correct_answer", "A")
                answer_idx = ord(str(answer).upper()) - ord('A')

            # Format 4: Hate speech (hs) - full_text + hs (0/1 label)
            elif "full_text" in doc and "hs" in doc:
                full_text = str(doc.get("full_text", "")).strip()
                hs_label = doc.get("hs")
                if full_text and isinstance(hs_label, int) and hs_label in [0, 1]:
                    question = f"Il seguente testo contiene discorsi d'odio?\n\n{full_text}"
                    choices = ["No, non contiene discorsi d'odio", "Sì, contiene discorsi d'odio"]
                    answer_idx = hs_label

            # Format 5: Sentiment analysis (sa) - text + opos/oneg labels
            elif "text" in doc and ("opos" in doc or "oneg" in doc):
                text = str(doc.get("text", "")).strip()
                # opos/oneg can be int or string "0"/"1"
                try:
                    opos = int(doc.get("opos", 0))  # Overall positive
                    oneg = int(doc.get("oneg", 0))  # Overall negative
                except (ValueError, TypeError):
                    opos, oneg = 0, 0
                if text:
                    question = f"Qual è il sentiment complessivo del seguente testo?\n\n{text}"
                    # Determine sentiment: pos=1 means positive, neg=1 means negative, both 0 = neutral
                    if opos == 1 and oneg == 0:
                        choices = ["Positivo", "Negativo", "Neutro"]
                        answer_idx = 0  # Positive
                    elif oneg == 1 and opos == 0:
                        choices = ["Positivo", "Negativo", "Neutro"]
                        answer_idx = 1  # Negative
                    else:
                        choices = ["Positivo", "Negativo", "Neutro"]
                        answer_idx = 2  # Neutral or mixed

            # Format 6: Text entailment (te) - text1 + text2 + entailment (YES/NO)
            elif "text1" in doc and "text2" in doc and "entailment" in doc:
                text1 = str(doc.get("text1", "")).strip()
                text2 = str(doc.get("text2", "")).strip()
                entailment = str(doc.get("entailment", "")).strip().upper()
                if text1 and text2 and entailment in ["YES", "NO"]:
                    question = f"Il testo 2 è una conseguenza logica del testo 1?\n\nTesto 1: {text1}\n\nTesto 2: {text2}"
                    choices = ["No", "Sì"]
                    answer_idx = 1 if entailment == "YES" else 0

            # Format 7: Word in context (wic) - lemma + sentence1 + sentence2 + label
            elif "lemma" in doc and "sentence1" in doc and "sentence2" in doc and "label" in doc:
                lemma = str(doc.get("lemma", "")).strip()
                sent1 = str(doc.get("sentence1", "")).strip()
                sent2 = str(doc.get("sentence2", "")).strip()
                label = doc.get("label")
                if lemma and sent1 and sent2 and isinstance(label, (bool, int, str)):
                    # label can be True/False, 1/0, or "True"/"False"
                    if isinstance(label, str):
                        is_same = label.lower() in ["true", "1", "yes"]
                    else:
                        is_same = bool(label)
                    question = f"La parola '{lemma}' ha lo stesso significato nelle due frasi?\n\n1: {sent1}\n\n2: {sent2}"
                    choices = ["Significato diverso", "Stesso significato"]
                    answer_idx = 1 if is_same else 0

            # Format 8: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "evalita_llm"}
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
                "label": "evalita_llm",
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
