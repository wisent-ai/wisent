from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger
import random

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["MMLUReduxExtractor"]

log = setup_logger(__name__)

# All MMLU-Redux subject configs
MMLU_REDUX_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes",
    "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"
]


class MMLUReduxExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for MMLU-Redux dataset - cleaned version of MMLU with error corrections.

    MMLU-Redux fixes errors in the original MMLU benchmark including:
    - bad_question_clarity
    - bad_options_clarity
    - no_correct_answer
    - multiple_correct_answers
    - wrong_groundtruth

    Schema (edinburgh-dawg/mmlu-redux):
        - question: str (the question text)
        - choices: List[str] (four answer choices)
        - answer: int (correct answer index 0-3)
        - error_type: str (annotation of error type, "ok" means no error)
        - source: str (potential source of the question)
        - correct_answer: str (suggested correct answer if original was wrong)
        - potential_reason: str (annotator notes)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "multiple_choice"

    def __init__(self, subject: str | None = None, only_valid: bool = True):
        """
        Initialize MMLU-Redux extractor.

        Args:
            subject: Specific subject to load (e.g., "philosophy"). If None, loads all.
            only_valid: If True, only include samples with error_type="ok"
        """
        super().__init__()
        self.subject = subject
        self.only_valid = only_valid

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from MMLU-Redux examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        pairs: list[ContrastivePair] = []
        subjects = [self.subject] if self.subject else MMLU_REDUX_SUBJECTS

        for subject in subjects:
            try:
                # Load dataset for this subject
                docs = self.load_dataset(
                    dataset_name="edinburgh-dawg/mmlu-redux",
                    config=subject,
                    split="test",
                    limit=max_items - len(pairs) if max_items else None,
                )

                log.info(f"Processing {len(docs)} examples from MMLU-Redux/{subject}")

                for doc in docs:
                    # Filter to only valid samples if requested
                    if self.only_valid and doc.get("error_type", "") != "ok":
                        continue

                    pair = self._extract_pair_from_doc(doc, subject)
                    if pair is not None:
                        pairs.append(pair)
                        if max_items is not None and len(pairs) >= max_items:
                            break

                if max_items is not None and len(pairs) >= max_items:
                    break

            except Exception as e:
                log.warning(f"Failed to load MMLU-Redux/{subject}: {e}")
                continue

        if not pairs:
            log.warning("No valid MMLU-Redux pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any], subject: str) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            question = doc.get("question", "").strip()
            choices = doc.get("choices", [])
            answer_idx = doc.get("answer")
            error_type = doc.get("error_type", "ok")

            if not question or not choices or answer_idx is None:
                log.debug("Skipping: missing question, choices, or answer")
                return None

            if len(choices) < 2:
                log.debug("Skipping: not enough choices")
                return None

            # Get correct answer
            correct_answer = choices[answer_idx]

            # Use corrected answer if available and error type indicates wrong groundtruth
            if error_type in ["wrong_groundtruth", "no_correct_answer"]:
                corrected = doc.get("correct_answer", "").strip()
                if corrected:
                    correct_answer = corrected

            # Get incorrect answer (random wrong choice)
            wrong_indices = [i for i in range(len(choices)) if i != answer_idx]
            if not wrong_indices:
                return None
            wrong_idx = random.choice(wrong_indices)
            incorrect_answer = choices[wrong_idx]

            # Format the question with choices
            choice_labels = ["A", "B", "C", "D"]
            choices_text = "\n".join([
                f"{choice_labels[i]}. {choice}" for i, choice in enumerate(choices)
            ])

            formatted_question = (
                f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
                f"Question: {question}\n\n"
                f"{choices_text}\n\n"
                f"Answer with the letter of the correct choice."
            )

            # For the response, use the letter + answer text
            correct_letter = choice_labels[answer_idx]
            wrong_letter = choice_labels[wrong_idx]

            metadata = {
                "label": "mmlu_redux",
                "source": "edinburgh-dawg/mmlu-redux",
                "subject": subject,
                "error_type": error_type,
                "correct_idx": answer_idx,
                "incorrect_idx": wrong_idx,
            }

            return self._build_pair(
                question=formatted_question,
                correct=f"{correct_letter}. {correct_answer}",
                incorrect=f"{wrong_letter}. {incorrect_answer}",
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

