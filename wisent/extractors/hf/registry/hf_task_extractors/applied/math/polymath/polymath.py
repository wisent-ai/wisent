from __future__ import annotations

from typing import Any
from wisent.core.utils.cli.cli_logger import setup_logger

from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.primitives.contrastive_pairs.core.io.response import NegativeResponse, PositiveResponse
from wisent.extractors.hf.atoms import HuggingFaceBenchmarkExtractor

from sympy.parsing.latex import parse_latex
from sympy import latex

__all__ = ["PolyMathExtractor"]

log = setup_logger(__name__)

task_names = (
    "polymath_ar_top", "polymath_ar_high", "polymath_ar_medium", "polymath_ar_low",
    "polymath_bn_top", "polymath_bn_high", "polymath_bn_medium", "polymath_bn_low",
    "polymath_de_top", "polymath_de_high", "polymath_de_medium", "polymath_de_low",
    "polymath_en_top", "polymath_en_high", "polymath_en_medium", "polymath_en_low",
    "polymath_es_top", "polymath_es_high", "polymath_es_medium", "polymath_es_low",
    "polymath_fr_top", "polymath_fr_high", "polymath_fr_medium", "polymath_fr_low",
    "polymath_id_top", "polymath_id_high", "polymath_id_medium", "polymath_id_low",
    "polymath_it_top", "polymath_it_high", "polymath_it_medium", "polymath_it_low",
    "polymath_ja_top", "polymath_ja_high", "polymath_ja_medium", "polymath_ja_low",
    "polymath_ko_top", "polymath_ko_high", "polymath_ko_medium", "polymath_ko_low",
    "polymath_ms_top", "polymath_ms_high", "polymath_ms_medium", "polymath_ms_low",
    "polymath_pt_top", "polymath_pt_high", "polymath_pt_medium", "polymath_pt_low",
    "polymath_ru_top", "polymath_ru_high", "polymath_ru_medium", "polymath_ru_low",
    "polymath_sw_top", "polymath_sw_high", "polymath_sw_medium", "polymath_sw_low",
    "polymath_te_top", "polymath_te_high", "polymath_te_medium", "polymath_te_low",
    "polymath_th_top", "polymath_th_high", "polymath_th_medium", "polymath_th_low",
    "polymath_vi_top", "polymath_vi_high", "polymath_vi_medium", "polymath_vi_low",
    "polymath_zh_top", "polymath_zh_high", "polymath_zh_medium", "polymath_zh_low",
)

class PolyMathExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for PolyMath dataset (multilingual mathematical reasoning).

    PolyMath schema (Qwen/PolyMath):
        - question: str (math problem statement in Chinese or English)
        - answer: str (final answer)
    """


    evaluator_name = "polymath"

    # These will be overridden by subclasses
    language = "en"
    difficulty = "medium"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from PolyMath examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        # Load PolyMath dataset with appropriate config and split
        # Config is the language (e.g., "en", "zh")
        # Split is the difficulty level (e.g., "high", "medium", "low", "top")
        docs = self.load_dataset(
            dataset_name="Qwen/PolyMath",
            dataset_config=self.language,
            split=self.difficulty,
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} PolyMath ({self.language}_{self.difficulty}) examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning(f"No valid PolyMath ({self.language}_{self.difficulty}) pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single PolyMath doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem = doc.get("question", "").strip()
            correct = doc.get("answer", "").strip()

            if not problem or not correct:
                log.debug("Skipping: missing problem or answer")
                return None

            incorrect_answer = self._create_incorrect_answer(correct)

            # Format the question
            question = f"Question: {problem}\n\nWhat is the answer?"

            metadata = {
                "label": "polymath",
                "source": "Qwen/PolyMath",
                "language": self.language,
                "difficulty": self.difficulty,
            }

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect_answer,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create a meaningful incorrect answer using plausible wrong values."""
        import random
        import re
        random.seed(hash(correct) % (2**32))

        # Try symbolic parsing first
        try:
            parsed_correct = parse_latex(correct)
            transforms = [
                parsed_correct * 2,
                parsed_correct / 2,
                parsed_correct - 1,
                -parsed_correct,
            ]
            wrong = random.choice(transforms)
            return str(latex(wrong))
        except Exception:
            pass

        # Try simple integer
        try:
            clean = correct.replace('$', '').replace(',', '').strip()
            num = int(clean)
            wrong_vals = [num * 2, num // 2 if num > 1 else num * 3, num - 1, -num]
            return str(random.choice(wrong_vals))
        except ValueError:
            pass

        # For fractions
        frac_match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', correct)
        if frac_match:
            n, d = int(frac_match.group(1)), int(frac_match.group(2))
            return random.choice([f"\\frac{{{d}}}{{{n}}}", f"\\frac{{{n*2}}}{{{d}}}"])

        # For pi expressions
        if '\\pi' in correct:
            return correct.replace('\\pi', '2\\pi') if '2\\pi' not in correct else correct.replace('2\\pi', '\\pi')

        # Fallback
        return random.choice(['0', '1', '-1', '2'])


    @staticmethod
    def _build_pair(
        question: str,
        correct: str,
        incorrect: str,
        metadata: dict[str, Any] | None = None,
    ) -> ContrastivePair:
        """Build a ContrastivePair from question and responses."""
        positive_response = PositiveResponse(model_response=correct)
        negative_response = NegativeResponse(model_response=incorrect)
        return ContrastivePair(
            prompt=question,
            positive_response=positive_response,
            negative_response=negative_response,
            label=metadata.get("label") if metadata else None,
            metadata=metadata,
        )


# All 72 language/difficulty specific extractors



# Re-exports from language-specific subclass modules
from .polymath_langs_ar_to_ja import *  # noqa: F401,F403
from .polymath_langs_ko_to_zh import *  # noqa: F401,F403
