from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PalomaExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "paloma_4chan_meta_sep", "paloma_c4_100_domains", "paloma_c4_en",
    "paloma_dolma_100_programing_languages", "paloma_dolma_100_subreddits",
    "paloma_dolma-v1_5", "paloma_falcon-refinedweb", "paloma_gab",
    "paloma_m2d2_s2orc_unsplit", "paloma_m2d2_wikipedia_unsplit",
    "paloma_manosphere_meta_sep", "paloma_mc4", "paloma_ptb",
    "paloma_redpajama", "paloma_twitterAAE_HELM_fixed", "paloma_wikitext_103"
)

# Note: Paloma is a perplexity benchmark (loglikelihood_rolling)
# We use perplexity evaluation with text corruption for contrastive pairs
class PalomaExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Paloma benchmark."""


    evaluator_name = "log_likelihoods"
    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from Paloma docs.

        Note: Paloma docs are very long (hundreds of thousands of characters).
        We split each doc into multiple chunks to generate multiple pairs.

        Args:
            lm_eval_task_data: lm-eval task instance for Paloma.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        # Load fewer docs since we'll split each into multiple pairs
        docs_to_load = 1 if max_items is None else max(1, max_items // 4)
        docs = self.load_docs(lm_eval_task_data, docs_to_load, preferred_doc=preferred_doc)

        pairs: list[ContrastivePair] = []

        log.info("Extracting contrastive pairs", extra={"doc_count": len(docs)})

        for doc in docs:
            # Extract multiple pairs from each long document
            doc_pairs = self._extract_pairs_from_doc(doc, max_pairs=max_items - len(pairs) if max_items else None)
            pairs.extend(doc_pairs)
            if max_items is not None and len(pairs) >= max_items:
                pairs = pairs[:max_items]
                break

        if not pairs:
            task_name = getattr(lm_eval_task_data, "NAME", type(lm_eval_task_data).__name__)
            log.warning("No valid Paloma pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pairs_from_doc(self, doc: dict[str, Any], max_pairs: int | None = None) -> list[ContrastivePair]:
        """
        Extract multiple contrastive pairs from a single long Paloma document.

        Args:
            doc: Paloma document with 'text' field
            max_pairs: Maximum number of pairs to extract from this doc

        Returns:
            List of ContrastivePair objects
        """
        pairs = []
        text = doc.get("text", "").strip()

        if not text or len(text) < 200:
            return pairs

        # Split long text into chunks of ~200 words
        words = text.split()
        chunk_size = 200
        target_pairs = max_pairs if max_pairs else 4  # Default to 4 pairs per doc

        for i in range(target_pairs):
            start_idx = i * chunk_size
            if start_idx >= len(words):
                break

            end_idx = min(start_idx + chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]

            if len(chunk_words) < 50:  # Skip very short chunks
                continue

            chunk_text = " ".join(chunk_words)

            # Create corrupted version
            mid_start = len(chunk_words) // 3
            mid_end = 2 * len(chunk_words) // 3
            middle_words = chunk_words[mid_start:mid_end]

            import random
            random.seed(hash(chunk_text) % (2**32))
            shuffled_middle = middle_words.copy()
            random.shuffle(shuffled_middle)

            corrupted_words = chunk_words[:mid_start] + shuffled_middle + chunk_words[mid_end:]
            corrupted_text = " ".join(corrupted_words)

            metadata = {"label": "paloma", "task": doc.get("source", "paloma"), "chunk": i}

            pair = self._build_pair(
                question="Continue this text:",
                correct=chunk_text,
                incorrect=corrupted_text,
                metadata=metadata,
            )
            pairs.append(pair)

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Paloma doc into a ContrastivePair for perplexity evaluation.

        Paloma is a perplexity benchmark with only text passages. We create contrastive
        pairs by using the original text as positive and a corrupted version as negative.
        The perplexity evaluator will compare both and should prefer the original.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Paloma docs have a 'text' field with the passage
            text = doc.get("text", "").strip()

            if not text or len(text) < 50:  # Skip very short texts
                log.debug("Skipping doc with insufficient text", extra={"text_len": len(text)})
                return None

            # Take a reasonable chunk (first ~200 words or ~1000 chars)
            words = text.split()
            if len(words) > 200:
                text = " ".join(words[:200])
            elif len(text) > 1000:
                text = text[:1000]

            # Create corrupted version by shuffling some words
            # This should have higher perplexity than the original
            words = text.split()
            if len(words) < 10:
                return None

            # Shuffle middle 30% of words to create unnatural text
            mid_start = len(words) // 3
            mid_end = 2 * len(words) // 3
            middle_words = words[mid_start:mid_end]

            import random
            random.seed(hash(text) % (2**32))  # Deterministic shuffle based on text
            shuffled_middle = middle_words.copy()
            random.shuffle(shuffled_middle)

            corrupted_words = words[:mid_start] + shuffled_middle + words[mid_end:]
            corrupted_text = " ".join(corrupted_words)

            # Build the pair with minimal prompt (perplexity is computed on the response)
            metadata = {"label": "paloma", "task": doc.get("source", "paloma")}

            return self._build_pair(
                question="Continue this text:",  # Minimal prompt for perplexity tasks
                correct=text,
                incorrect=corrupted_text,
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
