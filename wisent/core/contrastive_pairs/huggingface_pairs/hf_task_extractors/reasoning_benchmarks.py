"""Extractors for reasoning benchmarks."""
from __future__ import annotations

import random
from typing import Any

from wisent.core.cli_logger import setup_logger
from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = [
    "InverseScalingExtractor",
]

log = setup_logger(__name__)


class InverseScalingExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for Inverse Scaling benchmark.

    Dataset: inverse-scaling/inverse-scaling on HuggingFace

    Inverse Scaling Prize benchmark contains tasks where larger language models
    perform worse than smaller ones, identifying failure modes in scaling.
    Tasks include:
    - Hindsight neglect
    - Quote repetition
    - Resisting correction
    - Redefine math
    - Memo trap
    - And more
    """

    evaluator_name = "multiple_choice"

    def __init__(self, task: str | None = None):
        """
        Initialize Inverse Scaling extractor.

        Args:
            task: Optional specific task (e.g., 'hindsight_neglect', 'quote_repetition')
        """
        super().__init__()
        self.task = task

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """Extract contrastive pairs from Inverse Scaling dataset."""
        max_items = self._normalize_limit(limit)
        pairs: list[ContrastivePair] = []

        try:
            # Inverse Scaling is split into individual sub-datasets
            # Use NeQA as the primary dataset
            dataset_name = f"inverse-scaling/{self.task}" if self.task else "inverse-scaling/NeQA"

            docs = self.load_dataset(
                dataset_name=dataset_name,
                split="train",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from Inverse Scaling ({dataset_name})")
        except Exception as e:
            log.warning(f"Failed to load {dataset_name}: {e}")
            # Try redefine-math as fallback
            try:
                docs = self.load_dataset(
                    dataset_name="inverse-scaling/redefine-math",
                    split="train",
                    limit=max_items,
                )
                log.info(f"Loaded {len(docs)} examples from Inverse Scaling (redefine-math)")
            except Exception as e2:
                log.error(f"Failed to load Inverse Scaling: {e2}")
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

        Inverse Scaling format:
        - prompt: str (the question with context)
        - classes: list of answer options
        - answer_index: int (index of correct answer)
        """
        try:
            # Inverse Scaling format
            prompt = doc.get("prompt", doc.get("question", doc.get("text", ""))).strip()

            # Get choices/options - support both 'classes' and 'choices'
            choices = doc.get("classes", doc.get("choices", doc.get("options", [])))

            # Get the correct answer index
            answer = doc.get("answer_index", doc.get("answer", doc.get("label", doc.get("target", ""))))

            if not prompt:
                return None

            # Handle different formats
            if choices:
                # Multiple choice format
                choice_letters = ['A', 'B', 'C', 'D']
                choices_text = "\n".join(
                    f"{choice_letters[i]}. {c}" for i, c in enumerate(choices[:4])
                )

                task_prompt = f"""{prompt}

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
                    wrong_choices = [c for c in choices if c != correct]
                    incorrect = random.choice(wrong_choices) if wrong_choices else "incorrect"
            else:
                # Free-form answer format
                task_prompt = f"""{prompt}

Answer:"""

                correct = str(answer).strip() if answer else "yes"
                # Generate plausible incorrect answer
                if correct.lower() in ["yes", "true"]:
                    incorrect = "no"
                elif correct.lower() in ["no", "false"]:
                    incorrect = "yes"
                else:
                    incorrect = "I don't know"

            metadata = {
                "label": "inverse_scaling",
                "source": "inverse-scaling/inverse-scaling",
                "task": doc.get("task", self.task or "unknown"),
                "is_inverse_scaling_benchmark": True,
            }

            return self._build_pair(
                question=task_prompt,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting Inverse Scaling pair: {exc}", exc_info=True)
            return None
