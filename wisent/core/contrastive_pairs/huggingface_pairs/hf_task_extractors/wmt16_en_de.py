from __future__ import annotations

from typing import Any

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind


__all__ = ["Wmt16EnDeExtractor"]
_LOG = setup_logger(__name__)

task_names = ("wmt16-en-de",)


class Wmt16EnDeExtractor(HuggingFaceBenchmarkExtractor):
    """Extractor for Wmt16 En De benchmark."""

    evaluator_name = "generation"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        log = bind(_LOG, task="wmt16-en-de")
        max_items = self._normalize_limit(limit)

        # Load dataset from HuggingFace
        docs = self.load_dataset(
            dataset_name="wmt16",
            dataset_config="de-en",
            split="test",
            limit=max_items,
        )

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
            log.warning("No valid pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # WMT16 translation format: translation dict with 'en' (source) and 'de' (target)
            translation = doc.get("translation", {})
            source_text = translation.get("en", "").strip()
            target_text = translation.get("de", "").strip()

            if not source_text or not target_text:
                log.debug("Skipping doc due to missing source or target", extra={"doc": doc})
                return None

            # Use English text as prompt and German translation as correct response
            # For incorrect response, use a truncated/corrupted version of the German translation
            words = target_text.split()
            if len(words) > 3:
                # Use first half as truncated translation
                incorrect = " ".join(words[:len(words)//2])
            else:
                # If target is very short, use a single word or empty
                incorrect = words[0] if words else "No translation"

            # Create prompt with instruction to translate from English to German
            prompt = f"Translate from English to German:\n{source_text}"

            metadata = {"label": "wmt16_en_de"}

            return self._build_pair(
                question=prompt,
                correct=target_text,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error("Error extracting pair from doc", exc_info=exc, extra={"doc": doc})
            return None

