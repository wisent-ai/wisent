"""Extractors for multilingual NLP benchmarks."""
from __future__ import annotations

import random
from typing import Any

from wisent.core.cli.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "PawsXExtractor",
    "MLQAExtractor",
    "DarijaBenchExtractor",
    "EusExamsExtractor",
    "LambadaMultilingualExtractor",
]

log = setup_logger(__name__)


class PawsXExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for PAWS-X - Paraphrase Adversaries from Word Scrambling (Cross-lingual).

    Dataset: google-research-datasets/paws-x on HuggingFace

    PAWS-X is a challenging paraphrase identification dataset that contains
    pairs of sentences with high lexical overlap. Available in 7 languages.
    """

    evaluator_name = "paraphrase_detection"

    def __init__(self, language: str | None = None):
        """
        Initialize PAWS-X extractor.

        Args:
            language: Optional language filter (e.g., 'de', 'fr', 'es', 'zh', 'ja', 'ko')
        """
        super().__init__()
        self.language = language or "en"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from PAWS-X dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            docs = self.load_dataset(
                dataset_name="google-research-datasets/paws-x",
                dataset_config=self.language,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from PAWS-X ({self.language})")
        except Exception as e:
            log.error(f"Failed to load PAWS-X: {e}")
            return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            sentence1 = doc.get("sentence1", "").strip()
            sentence2 = doc.get("sentence2", "").strip()
            label = doc.get("label", 0)

            if not sentence1 or not sentence2:
                return None

            task_prompt = f"""Determine if the following two sentences are paraphrases of each other.

Sentence 1: {sentence1}
Sentence 2: {sentence2}

Are these sentences paraphrases? Answer Yes or No:"""

            # label=1 means paraphrase, label=0 means not paraphrase
            if label == 1:
                correct = "Yes"
                incorrect = "No"
            else:
                correct = "No"
                incorrect = "Yes"

            metadata = {
                "label": "paws_x",
                "source": "google-research-datasets/paws-x",
                "language": self.language,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting PAWS-X pair: {exc}", exc_info=True)
            return None


class MLQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MLQA - Multilingual Question Answering.

    Dataset: facebook/mlqa on HuggingFace

    MLQA is a benchmark for evaluating cross-lingual extractive QA performance.
    It contains QA instances in 7 languages.
    """

    evaluator_name = "extractive_qa"

    def __init__(self, language: str | None = None):
        """
        Initialize MLQA extractor.

        Args:
            language: Optional language filter (e.g., 'en', 'de', 'es', 'ar', 'hi', 'vi', 'zh')
        """
        super().__init__()
        self.language = language or "en"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from MLQA dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []
        all_answers: list[str] = []

        try:
            # MLQA config is like "mlqa.en.en" or "mlqa-translate-train.en"
            config = f"mlqa.{self.language}.{self.language}"
            docs = self.load_dataset(
                dataset_name="facebook/mlqa",
                dataset_config=config,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from MLQA ({self.language})")
        except Exception as e:
            log.error(f"Failed to load MLQA: {e}")
            return []

        # First pass: collect all answers for negative sampling
        for doc in docs:
            answers = doc.get("answers", {}).get("text", [])
            if answers and isinstance(answers, list) and len(answers) > 0:
                all_answers.append(answers[0])

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, all_answers)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], all_answers: list[str]
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            context = doc.get("context", "").strip()
            question = doc.get("question", "").strip()
            answers = doc.get("answers", {}).get("text", [])

            if not context or not question or not answers:
                return None

            correct_answer = answers[0] if answers else ""
            if not correct_answer:
                return None

            task_prompt = f"""Context: {context}

Question: {question}

Answer:"""

            # Get an incorrect answer from other examples
            negative_candidates = [a for a in all_answers if a != correct_answer]
            if negative_candidates:
                incorrect = random.choice(negative_candidates)
            else:
                incorrect = "I don't know."

            metadata = {
                "label": "mlqa",
                "source": "facebook/mlqa",
                "language": self.language,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct_answer,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting MLQA pair: {exc}", exc_info=True)
            return None


class DarijaBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for DarijaBench - Moroccan Arabic (Darija) benchmark.

    Dataset: MBZUAI-Paris/DarijaBench on HuggingFace

    DarijaBench evaluates language models on Moroccan Arabic tasks including
    sentiment analysis, NER, and QA.
    """

    evaluator_name = "multiple_choice"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from DarijaBench dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # DarijaBench has specific splits: doda, madar, seed, flores_plus, etc.
            docs = self.load_dataset(
                dataset_name="MBZUAI-Paris/DarijaBench",
                dataset_config="default",
                split="doda",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from DarijaBench")
        except Exception as e:
            log.warning(f"Failed to load DarijaBench doda split: {e}")
            # Try madar split
            try:
                docs = self.load_dataset(
                    dataset_name="MBZUAI-Paris/DarijaBench",
                    dataset_config="default",
                    split="madar",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from DarijaBench (madar)")
            except Exception as e2:
                log.error(f"Failed to load DarijaBench: {e2}")
                return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair.

        DarijaBench format:
        - dataset: str (e.g., 'doda')
        - direction: str (e.g., 'fr_dr' for French to Darija)
        - messages: list of dicts with 'role' and 'content'
        """
        try:
            messages = doc.get("messages", [])
            direction = doc.get("direction", "")

            # Extract user prompt and assistant response from messages
            user_content = ""
            assistant_content = ""
            for msg in messages:
                if msg.get("role") == "user":
                    user_content = msg.get("content", "").strip()
                elif msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "").strip()

            if not user_content or not assistant_content:
                return None

            task_prompt = user_content
            correct = assistant_content
            # Create incorrect translation by using a generic wrong response
            incorrect = "لا أعرف" if "dr" in direction else "Je ne sais pas"  # "I don't know" in Darija/French

            metadata = {
                "label": "darija_bench",
                "source": "MBZUAI-Paris/DarijaBench",
                "language": "darija",
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting DarijaBench pair: {exc}", exc_info=True)
            return None


class EusExamsExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for EusExams - Basque Exam Questions benchmark.

    Dataset: HiTZ/EusExams on HuggingFace

    EusExams is a multiple-choice QA benchmark in Basque language covering
    various domains from official exams.
    """

    evaluator_name = "multiple_choice"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from EusExams dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # EusExams requires a config (e.g., 'eu_opeosakiadmineu')
            docs = self.load_dataset(
                dataset_name="HiTZ/EusExams",
                dataset_config="eu_opeosakiadmineu",
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from EusExams")
        except Exception as e:
            log.warning(f"Failed to load EusExams test split: {e}")
            # Try train split
            try:
                docs = self.load_dataset(
                    dataset_name="HiTZ/EusExams",
                    dataset_config="eu_opeosakiadmineu",
                    split="train",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from EusExams (train)")
            except Exception as e2:
                # Try validation split
                try:
                    docs = self.load_dataset(
                        dataset_name="HiTZ/EusExams",
                        dataset_config="eu_opeosakiadmineu",
                        split="validation",
                        limit=max_items,
                    )
                    log.info(f"Loaded {len(docs)} examples from EusExams (validation)")
                except Exception as e3:
                    log.error(f"Failed to load EusExams: {e3}")
                    return []

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            question = doc.get("question", doc.get("text", "")).strip()
            choices = doc.get("candidates", doc.get("choices", doc.get("options", [])))
            answer_idx = doc.get("answer", doc.get("label", 0))

            if not question or not choices:
                return None

            choice_letters = ['A', 'B', 'C', 'D']
            choices_text = "\n".join(
                f"{choice_letters[i]}. {c}" for i, c in enumerate(choices[:4])
            )

            task_prompt = f"""Galdera: {question}

{choices_text}

Erantzuna:"""

            if isinstance(answer_idx, int) and answer_idx < len(choices):
                correct = choice_letters[answer_idx]
            elif isinstance(answer_idx, str) and answer_idx.upper() in choice_letters:
                correct = answer_idx.upper()
            else:
                correct = "A"

            # Get incorrect answer
            correct_idx = choice_letters.index(correct) if correct in choice_letters else 0
            wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
            incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"

            metadata = {
                "label": "eus_exams",
                "source": "HiTZ/EusExams",
                "language": "basque",
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting EusExams pair: {exc}", exc_info=True)
            return None


class LambadaMultilingualExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for LAMBADA Multilingual - Word prediction benchmark.

    Dataset: EleutherAI/lambada_openai on HuggingFace

    LAMBADA tests language models on their ability to predict the last word
    of a passage that requires broad context to resolve.
    Multilingual variants available for multiple languages.
    """

    evaluator_name = "word_prediction"

    def __init__(self, language: str | None = None):
        """
        Initialize LAMBADA Multilingual extractor.

        Args:
            language: Optional language filter (e.g., 'de', 'fr', 'it', 'es')
        """
        super().__init__()
        self.language = language or "de"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from LAMBADA Multilingual dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []
        all_targets: list[str] = []

        try:
            # Try the multilingual config
            docs = self.load_dataset(
                dataset_name="EleutherAI/lambada_openai",
                dataset_config=self.language,
                split="test",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from LAMBADA ({self.language})")
        except Exception as e:
            log.error(f"Failed to load LAMBADA multilingual: {e}")
            return []

        # First pass: collect all targets
        for doc in docs:
            text = doc.get("text", "")
            if text:
                words = text.split()
                if words:
                    all_targets.append(words[-1])

        for doc in docs:
            pair = self._extract_pair_from_doc(doc, all_targets)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        return pairs

    def _extract_pair_from_doc(
        self, doc: dict[str, Any], all_targets: list[str]
    ) -> ContrastivePair | None:
        """Convert a single doc into a ContrastivePair."""
        try:
            text = doc.get("text", "").strip()

            if not text:
                return None

            words = text.split()
            if len(words) < 2:
                return None

            # The task is to predict the last word
            context = " ".join(words[:-1])
            target_word = words[-1]

            task_prompt = f"""{context}"""

            # Get an incorrect word
            negative_candidates = [t for t in all_targets if t != target_word]
            if negative_candidates:
                incorrect = random.choice(negative_candidates)
            else:
                incorrect = "unknown"

            metadata = {
                "label": "lambada_multilingual",
                "source": "EleutherAI/lambada_openai",
                "language": self.language,
                "is_multilingual_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=target_word,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting LAMBADA pair: {exc}", exc_info=True)
            return None
