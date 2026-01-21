"""Extractors for medical benchmarks."""
from __future__ import annotations

import random
from typing import Any

from wisent.core.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "MedConceptsQAExtractor",
]

log = setup_logger(__name__)


class MedConceptsQAExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MedConcepts QA - Medical Concepts Question Answering.

    Dataset: TommyChien/MedConceptsQA on HuggingFace (or similar)

    MedConceptsQA evaluates understanding of medical concepts including:
    - Drug interactions
    - Disease symptoms
    - Medical terminology
    - Clinical concepts
    """

    evaluator_name = "medical_qa"

    def __init__(self, subset: str | None = None):
        """
        Initialize MedConceptsQA extractor.

        Args:
            subset: Optional subset (e.g., 'drugs', 'diseases', 'procedures')
        """
        super().__init__()
        self.subset = subset

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from MedConceptsQA dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        # Try multiple possible dataset names
        dataset_candidates = [
            ("TommyChien/MedConceptsQA", self.subset),
            ("medmcqa", None),
            ("medical_qa", None),
        ]

        docs = None
        for dataset_name, config in dataset_candidates:
            try:
                docs = self.load_dataset(
                    dataset_name=dataset_name,
                    dataset_config=config,
                    split="test",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from {dataset_name}")
                break
            except Exception as e:
                log.debug(f"Failed to load {dataset_name}: {e}")
                continue

        if docs is None or len(docs) == 0:
            log.error("Failed to load MedConceptsQA from any source")
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
            # Try different field names for question
            question = doc.get("question", doc.get("text", doc.get("prompt", ""))).strip()

            # Try different field names for choices
            choices = doc.get("choices", doc.get("options", doc.get("opa", None)))

            # Handle different answer formats
            answer = doc.get("answer", doc.get("cop", doc.get("label", None)))

            if not question:
                return None

            # If we have explicit choices
            if choices and isinstance(choices, list):
                choice_letters = ['A', 'B', 'C', 'D', 'E']
                choices_text = "\n".join(
                    f"{choice_letters[i]}. {c}" for i, c in enumerate(choices[:5])
                )

                task_prompt = f"""Medical Question: {question}

{choices_text}

Answer:"""

                if isinstance(answer, int) and answer < len(choices):
                    correct = choice_letters[answer]
                elif isinstance(answer, str):
                    if answer.upper() in choice_letters:
                        correct = answer.upper()
                    elif answer in choices:
                        correct = choice_letters[choices.index(answer)]
                    else:
                        correct = answer
                else:
                    correct = "A"

                # Get incorrect answer
                if correct in choice_letters:
                    correct_idx = choice_letters.index(correct)
                    wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
                    incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"
                else:
                    incorrect = "B"

            # Handle MedMCQA-style format with opa/opb/opc/opd
            elif doc.get("opa") is not None:
                choices = [doc.get("opa"), doc.get("opb"), doc.get("opc"), doc.get("opd")]
                choices = [c for c in choices if c is not None]
                choice_letters = ['A', 'B', 'C', 'D']

                choices_text = "\n".join(
                    f"{choice_letters[i]}. {c}" for i, c in enumerate(choices)
                )

                task_prompt = f"""Medical Question: {question}

{choices_text}

Answer:"""

                cop = doc.get("cop", 1)
                if isinstance(cop, int) and 1 <= cop <= len(choices):
                    correct = choice_letters[cop - 1]  # cop is 1-indexed
                else:
                    correct = "A"

                correct_idx = choice_letters.index(correct)
                wrong_indices = [i for i in range(len(choices)) if i != correct_idx]
                incorrect = choice_letters[random.choice(wrong_indices)] if wrong_indices else "B"

            else:
                # Free-form answer
                task_prompt = f"""Medical Question: {question}

Answer:"""

                correct = str(answer).strip() if answer else "Yes"
                incorrect = "I don't know"

            metadata = {
                "label": "med_concepts_qa",
                "source": "MedConceptsQA",
                "subset": self.subset or "general",
                "is_medical_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting MedConceptsQA pair: {exc}", exc_info=True)
            return None
