from __future__ import annotations

import random
from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

log = setup_logger(__name__)

class DarijaBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for DarijaBench - Moroccan Arabic (Darija) benchmark.

    Dataset: MBZUAI-Paris/DarijaBench on HuggingFace

    DarijaBench evaluates language models on Moroccan Arabic tasks including
    sentiment analysis, NER, and QA.
    """

    evaluator_name = "darija_bench"

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

    evaluator_name = "eus_exams"

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

