from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["BasqueglueExtractor"]
_LOG = setup_logger(__name__)

task_names = ("basque-glue", "basqueglue")

# Configs from orai-nlp/basqueGLUE
# intent = fmtod_intent, bhtc = bhtc_v2, bec = bec2016eu, vaxx = vaxx_stance
# qnli = qnli_eu, wic = wiceu, coref = epec_koref_bin
BASQUEGLUE_CONFIGS = ["intent", "bhtc", "bec", "vaxx", "qnli", "wic", "coref"]

# Label names for each classification config
LABEL_NAMES = {
    "intent": [
        "alarm/cancel_alarm", "alarm/modify_alarm", "alarm/set_alarm", "alarm/show_alarms",
        "alarm/snooze_alarm", "alarm/time_left_on_alarm", "reminder/cancel_reminder",
        "reminder/set_reminder", "reminder/show_reminders", "weather/checkSunrise",
        "weather/checkSunset", "weather/find"
    ],
    "bhtc": [
        "Ekonomia", "Euskal Herria", "Euskara", "Gizartea", "Historia", "Ingurumena",
        "Iritzia", "Komunikazioa", "Kultura", "Nazioartea", "Politika", "Zientzia"
    ],
    "bec": ["N", "NEU", "P"],
    "vaxx": ["AGAINST", "NONE", "FAVOR"],
}


class BasqueglueExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for the BasqueGLUE benchmark.

    Dataset: https://huggingface.co/datasets/orai-nlp/basqueGLUE

    Configs:
    - intent (fmtod_intent): text classification
    - bhtc (bhtc_v2): text classification
    - bec (bec2016eu): text classification
    - vaxx (vaxx_stance): text classification
    - qnli (qnli_eu): NLI (question, sentence, label)
    - wic (wiceu): word-in-context (sentence1, sentence2, word, label)
    - coref (epec_koref_bin): coreference (text, span1_text, span2_text, label)
    """


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="basqueglue")
        max_items = self._normalize_limit(limit)

        pairs: list[ContrastivePair] = []

        for config in BASQUEGLUE_CONFIGS:
            try:
                docs = self.load_dataset(
                    dataset_name="orai-nlp/basqueGLUE",
                    dataset_config=config,
                    split="test",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} docs from config '{config}'")

                for doc in docs:
                    pair = self._extract_pair_from_doc(doc, config)
                    if pair is not None:
                        pairs.append(pair)

            except Exception as e:
                log.warning(f"Failed to load config '{config}': {e}")
                continue

        log.info("Extracted contrastive pairs", extra={"pair_count": len(pairs)})

        if not pairs:
            log.warning("No valid pairs extracted", extra={"task": "basqueglue"})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], config: str) -> ContrastivePair | None:
        """Extract a contrastive pair based on the config type."""
        log = bind(_LOG, doc_id=doc.get("idx", "unknown"), config=config)

        try:
            # Text classification: intent, bhtc, bec, vaxx
            if config in ("intent", "bhtc", "bec", "vaxx"):
                return self._extract_text_classification(doc, config)

            # NLI: qnli
            if config == "qnli":
                return self._extract_qnli(doc)

            # Word-in-context: wic
            if config == "wic":
                return self._extract_wic(doc)

            # Coreference: coref
            if config == "coref":
                return self._extract_coref(doc)

            return None

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

    def _extract_text_classification(self, doc: dict[str, Any], config: str) -> ContrastivePair | None:
        """Extract pair from text classification task (intent, bhtc, bec, vaxx)."""
        text = str(doc.get("text", "")).strip()
        label = doc.get("label")

        if not text or label is None:
            return None

        labels = LABEL_NAMES.get(config, [])
        if not labels or label >= len(labels):
            return None

        correct = labels[label]
        incorrect_idx = (label + 1) % len(labels)
        incorrect = labels[incorrect_idx]

        prompt = f"Classify the following text:\n\n{text}\nA. {incorrect}\nB. {correct}"

        return self._build_pair(
            question=prompt,
            correct=correct,
            incorrect=incorrect,
            metadata={"label": f"basqueglue_{config}"},
        )

    def _extract_qnli(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Extract pair from QNLI task (question, sentence, label)."""
        question = str(doc.get("question", "")).strip()
        sentence = str(doc.get("sentence", "")).strip()
        label = doc.get("label")  # 0 = entailment, 1 = not_entailment

        if not question or not sentence or label is None:
            return None

        if label == 0:
            correct = "entailment"
            incorrect = "not_entailment"
        else:
            correct = "not_entailment"
            incorrect = "entailment"

        prompt = f"Question: {question}\nSentence: {sentence}\n\nDoes the sentence entail the question?\nA. {incorrect}\nB. {correct}"

        return self._build_pair(
            question=prompt,
            correct=correct,
            incorrect=incorrect,
            metadata={"label": "basqueglue_qnli"},
        )

    def _extract_wic(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Extract pair from Word-in-Context task."""
        sentence1 = str(doc.get("sentence1", "")).strip()
        sentence2 = str(doc.get("sentence2", "")).strip()
        word = str(doc.get("word", "")).strip()
        label = doc.get("label")  # 0 = different meaning, 1 = same meaning

        if not sentence1 or not sentence2 or not word or label is None:
            return None

        if label == 1:
            correct = "Yes"
            incorrect = "No"
        else:
            correct = "No"
            incorrect = "Yes"

        prompt = f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\n\nDoes the word '{word}' have the same meaning in both sentences?\nA. {incorrect}\nB. {correct}"

        return self._build_pair(
            question=prompt,
            correct=correct,
            incorrect=incorrect,
            metadata={"label": "basqueglue_wic"},
        )

    def _extract_coref(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Extract pair from coreference task."""
        text = str(doc.get("text", "")).strip()
        span1 = str(doc.get("span1_text", "")).strip()
        span2 = str(doc.get("span2_text", "")).strip()
        label = doc.get("label")  # 0 = not coreferent, 1 = coreferent

        if not text or not span1 or not span2 or label is None:
            return None

        if label == 1:
            correct = "Yes"
            incorrect = "No"
        else:
            correct = "No"
            incorrect = "Yes"

        prompt = f"Text: {text}\n\nDo '{span1}' and '{span2}' refer to the same entity?\nA. {incorrect}\nB. {correct}"

        return self._build_pair(
            question=prompt,
            correct=correct,
            incorrect=incorrect,
            metadata={"label": "basqueglue_coref"},
        )

