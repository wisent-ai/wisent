from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.lm_eval.atoms import LMEvalBenchmarkExtractor
from wisent.core.utils.cli.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["PileExtractor"]
_LOG = setup_logger(__name__)

task_names = (
    "pile",
    "pile_arxiv", "pile_bookcorpus2", "pile_books3", "pile_dm-mathematics", "pile_enron",
    "pile_europarl", "pile_freelaw", "pile_github", "pile_gutenberg", "pile_hackernews",
    "pile_nih-exporter", "pile_opensubtitles", "pile_openwebtext2", "pile_philpapers",
    "pile_pile-cc", "pile_pubmed-abstracts", "pile_pubmed-central", "pile_stackexchange",
    "pile_ubuntu-irc", "pile_uspto", "pile_wikipedia", "pile_youtubesubtitles"
)
class PileExtractor(LMEvalBenchmarkExtractor):
    """Extractor for the Pile benchmark."""


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
        Build contrastive pairs from Pile docs.

        Args:
            lm_eval_task_data: lm-eval task instance for Pile.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))

        max_items = self._normalize_limit(limit)
        # Upstream EleutherAI/pile is deprecated/gated. Fall back to NeelNanda/pile-10k
        # (public 10k-row sample) and filter by the pile subset encoded in task_name.
        task_name = getattr(self, "task_name", getattr(lm_eval_task_data, "NAME", "pile") if lm_eval_task_data else "pile")
        subset_map = {
            "pile_arxiv": "ArXiv", "pile_bookcorpus2": "BookCorpus2", "pile_books3": "Books3",
            "pile_dm-mathematics": "DM Mathematics", "pile_enron": "Enron Emails",
            "pile_europarl": "EuroParl", "pile_freelaw": "FreeLaw", "pile_github": "Github",
            "pile_gutenberg": "Gutenberg (PG-19)", "pile_hackernews": "HackerNews",
            "pile_nih-exporter": "NIH ExPorter", "pile_opensubtitles": "OpenSubtitles",
            "pile_openwebtext2": "OpenWebText2", "pile_philpapers": "PhilPapers",
            "pile_pile-cc": "Pile-CC", "pile_pubmed-abstracts": "PubMed Abstracts",
            "pile_pubmed-central": "PubMed Central", "pile_stackexchange": "StackExchange",
            "pile_ubuntu-irc": "Ubuntu IRC", "pile_uspto": "USPTO Backgrounds",
            "pile_wikipedia": "Wikipedia (en)", "pile_youtubesubtitles": "YoutubeSubtitles",
        }
        wanted = subset_map.get(task_name)
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split="train")
        if wanted:
            docs = [r for r in ds if r["meta"].get("pile_set_name") == wanted]
        else:
            docs = list(ds)
        if max_items:
            docs = docs[:max_items]

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
            log.warning("No valid Pile pairs extracted", extra={"task": task_name})

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single Pile doc into a ContrastivePair, if possible.
        Returns None when required fields are missing or malformed.
        """
        log = bind(_LOG, doc_id=doc.get("id", "unknown"))

        try:
            # Pile/pile_10k format: text-only (perplexity task)
            if "text" in doc and len(doc) <= 3:
                text = str(doc.get("text", "")).strip()
                if text:
                    words = text.split()
                    if len(words) >= 2:
                        return self._build_pair(
                            question=" ".join(words[:-1]),
                            correct=words[-1],
                            incorrect="incorrect",
                            metadata={"label": "pile"},
                        )
                return None

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

            # Format 3: query/prompt + answer
            elif "query" in doc or "prompt" in doc:
                question = str(doc.get("query", doc.get("prompt", ""))).strip()
                # For open-ended questions, use target as correct answer
                correct_answer = str(doc.get("target", doc.get("answer", ""))).strip()
                if correct_answer:
                    metadata = {"label": "pile"}
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
                "label": "pile",
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
