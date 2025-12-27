from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

__all__ = ["OlympiadBenchExtractor"]

log = setup_logger(__name__)

# OlympiadBench configurations
OLYMPIAD_CONFIGS = {
    # Math configurations
    "math_en": "OE_TO_maths_en_COMP",
    "math_zh": "OE_TO_maths_zh_COMP",
    "math_en_mm": "OE_MM_maths_en_COMP",
    "math_zh_mm": "OE_MM_maths_zh_COMP",
    # Physics configurations
    "physics_en": "OE_TO_physics_en_COMP",
    "physics_zh": "OE_TO_physics_zh_COMP",
    "physics_en_mm": "OE_MM_physics_en_COMP",
    # Chinese exam configurations
    "math_cee": "OE_TO_maths_zh_CEE",
    "physics_cee": "OE_TO_physics_zh_CEE",
}


class OlympiadBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for OlympiadBench - Olympiad-level Math & Physics Benchmark (ACL 2024).

    OlympiadBench is an Olympiad-level bilingual multimodal scientific benchmark
    featuring 8,476 problems from Olympiad-level mathematics and physics competitions,
    including the Chinese college entrance exam.

    Also serves as IMO-AnswerBench proxy with 400 handpicked Olympiad problems
    across algebra, combinatorics, geometry, and number theory.

    Problem Types:
    - OE: Open-Ended problems
    - TP: Theoretical Problems
    - MM: Multimodal (with images)
    - TO: Text-Only

    Schema (Hothan/OlympiadBench):
        - id: int (unique identifier)
        - question: str (problem statement)
        - solution: list[str] (solution approaches)
        - final_answer: list[str] (correct answers)
        - context: str (additional context)
        - modality: str (MM or TO)
        - difficulty: str (problem difficulty)
        - is_multiple_answer: bool
        - unit: str (answer units)
        - answer_type: str (classification)
        - question_type: str (problem category)
        - subfield: str (subject subdivision)
        - subject: str (Maths/Physics)
        - language: str (en/zh)
    """

    # Evaluator that should be used for this benchmark
    evaluator_name = "math_olympiad"

    def __init__(
        self,
        config: str = "math_en",
        text_only: bool = True,
        subject: str = "maths",
    ):
        """
        Initialize OlympiadBench extractor.

        Args:
            config: Configuration name (e.g., "math_en", "physics_zh")
            text_only: If True, use text-only problems (no images)
            subject: Subject filter ("maths" or "physics")
        """
        super().__init__()
        self.config = OLYMPIAD_CONFIGS.get(config, config)
        self.text_only = text_only
        self.subject = subject

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from OlympiadBench examples.

        For olympiad math problems:
        - Positive (correct) = Correct solution with reasoning
        - Negative (incorrect) = Wrong answer or flawed reasoning

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        try:
            docs = self.load_dataset(
                dataset_name="Hothan/OlympiadBench",
                config=self.config,
                split="train",
                limit=max_items,
            )
            log.info(f"Loaded {len(docs)} examples from OlympiadBench ({self.config})")
        except Exception as e:
            log.error(f"Failed to load Hothan/OlympiadBench: {e}")
            return []

        pairs: list[ContrastivePair] = []

        for doc in docs:
            # Filter by modality if text_only
            if self.text_only:
                modality = doc.get("modality", "TO")
                if modality == "MM":
                    continue

            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning("No valid OlympiadBench pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            doc_id = doc.get("id", "")
            question = doc.get("question", "").strip()
            solutions = doc.get("solution", [])
            final_answers = doc.get("final_answer", [])
            context = doc.get("context", "")
            difficulty = doc.get("difficulty", "")
            subject = doc.get("subject", "")
            subfield = doc.get("subfield", "")
            language = doc.get("language", "en")
            unit = doc.get("unit", "")
            answer_type = doc.get("answer_type", "")

            if not question:
                log.debug("Skipping: missing question")
                return None

            # Build full problem prompt
            prompt = self._build_prompt(question, context)

            # Build correct response from solution and answer
            correct_response = self._build_correct_response(
                solutions, final_answers, unit
            )
            if not correct_response:
                log.debug("Skipping: missing solution or answer")
                return None

            # Create incorrect response
            incorrect_response = self._create_incorrect_response(
                question, final_answers, answer_type
            )

            metadata = {
                "label": "olympiadbench",
                "source": "Hothan/OlympiadBench",
                "id": str(doc_id),
                "config": self.config,
                "difficulty": difficulty,
                "subject": subject,
                "subfield": subfield,
                "language": language,
                "answer_type": answer_type,
                "unit": unit,
                "is_math_benchmark": True,
            }

            return self._build_pair(
                question=prompt,
                correct=correct_response,
                incorrect=incorrect_response,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _build_prompt(self, question: str, context: str) -> str:
        """Build the problem prompt."""
        if context:
            return f"{context}\n\n{question}"
        return question

    def _build_correct_response(
        self,
        solutions: list | str,
        final_answers: list | str,
        unit: str,
    ) -> str:
        """Build the correct response with solution and answer."""
        # Extract first solution
        if isinstance(solutions, list) and solutions:
            solution = solutions[0]
        elif isinstance(solutions, str):
            solution = solutions
        else:
            solution = ""

        # Extract first answer
        if isinstance(final_answers, list) and final_answers:
            answer = final_answers[0]
        elif isinstance(final_answers, str):
            answer = final_answers
        else:
            answer = ""

        if not answer and not solution:
            return ""

        response_parts = []
        if solution:
            response_parts.append(f"Solution:\n{solution}")

        if answer:
            if unit:
                response_parts.append(f"\nFinal Answer: {answer} {unit}")
            else:
                response_parts.append(f"\nFinal Answer: {answer}")

        return "\n".join(response_parts)

    def _create_incorrect_response(
        self,
        question: str,
        final_answers: list | str,
        answer_type: str,
    ) -> str:
        """Create an incorrect response."""
        # Try to create a plausible but wrong answer
        if isinstance(final_answers, list) and final_answers:
            correct = final_answers[0]
        elif isinstance(final_answers, str):
            correct = final_answers
        else:
            correct = "0"

        # Generate wrong answer based on type
        try:
            # Try to modify numeric answer
            if correct.replace(".", "").replace("-", "").isdigit():
                num = float(correct)
                wrong = num + 1 if num < 100 else num * 2
                wrong_answer = str(int(wrong) if wrong == int(wrong) else wrong)
            else:
                wrong_answer = "undefined"
        except (ValueError, TypeError):
            wrong_answer = "cannot be determined"

        return (
            f"Let me solve this problem.\n\n"
            f"After brief consideration, the answer is: {wrong_answer}\n\n"
            "Note: This solution lacks proper mathematical reasoning."
        )

