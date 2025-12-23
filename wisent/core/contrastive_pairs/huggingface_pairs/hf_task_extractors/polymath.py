from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.core.response import NegativeResponse, PositiveResponse
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

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

# Arabic
class PolyMathARTopExtractor(PolyMathExtractor):
    language = "ar"
    difficulty = "top"

class PolyMathARHighExtractor(PolyMathExtractor):
    language = "ar"
    difficulty = "high"

class PolyMathARMediumExtractor(PolyMathExtractor):
    language = "ar"
    difficulty = "medium"

class PolyMathARLowExtractor(PolyMathExtractor):
    language = "ar"
    difficulty = "low"

# Bengali
class PolyMathBNTopExtractor(PolyMathExtractor):
    language = "bn"
    difficulty = "top"

class PolyMathBNHighExtractor(PolyMathExtractor):
    language = "bn"
    difficulty = "high"

class PolyMathBNMediumExtractor(PolyMathExtractor):
    language = "bn"
    difficulty = "medium"

class PolyMathBNLowExtractor(PolyMathExtractor):
    language = "bn"
    difficulty = "low"

# German
class PolyMathDETopExtractor(PolyMathExtractor):
    language = "de"
    difficulty = "top"

class PolyMathDEHighExtractor(PolyMathExtractor):
    language = "de"
    difficulty = "high"

class PolyMathDEMediumExtractor(PolyMathExtractor):
    language = "de"
    difficulty = "medium"

class PolyMathDELowExtractor(PolyMathExtractor):
    language = "de"
    difficulty = "low"

# English
class PolyMathENTopExtractor(PolyMathExtractor):
    language = "en"
    difficulty = "top"

class PolyMathENHighExtractor(PolyMathExtractor):
    language = "en"
    difficulty = "high"

class PolyMathENMediumExtractor(PolyMathExtractor):
    language = "en"
    difficulty = "medium"

class PolyMathENLowExtractor(PolyMathExtractor):
    language = "en"
    difficulty = "low"

# Spanish
class PolyMathESTopExtractor(PolyMathExtractor):
    language = "es"
    difficulty = "top"

class PolyMathESHighExtractor(PolyMathExtractor):
    language = "es"
    difficulty = "high"

class PolyMathESMediumExtractor(PolyMathExtractor):
    language = "es"
    difficulty = "medium"

class PolyMathESLowExtractor(PolyMathExtractor):
    language = "es"
    difficulty = "low"

# French
class PolyMathFRTopExtractor(PolyMathExtractor):
    language = "fr"
    difficulty = "top"

class PolyMathFRHighExtractor(PolyMathExtractor):
    language = "fr"
    difficulty = "high"

class PolyMathFRMediumExtractor(PolyMathExtractor):
    language = "fr"
    difficulty = "medium"

class PolyMathFRLowExtractor(PolyMathExtractor):
    language = "fr"
    difficulty = "low"

# Indonesian
class PolyMathIDTopExtractor(PolyMathExtractor):
    language = "id"
    difficulty = "top"

class PolyMathIDHighExtractor(PolyMathExtractor):
    language = "id"
    difficulty = "high"

class PolyMathIDMediumExtractor(PolyMathExtractor):
    language = "id"
    difficulty = "medium"

class PolyMathIDLowExtractor(PolyMathExtractor):
    language = "id"
    difficulty = "low"

# Italian
class PolyMathITTopExtractor(PolyMathExtractor):
    language = "it"
    difficulty = "top"

class PolyMathITHighExtractor(PolyMathExtractor):
    language = "it"
    difficulty = "high"

class PolyMathITMediumExtractor(PolyMathExtractor):
    language = "it"
    difficulty = "medium"

class PolyMathITLowExtractor(PolyMathExtractor):
    language = "it"
    difficulty = "low"

# Japanese
class PolyMathJATopExtractor(PolyMathExtractor):
    language = "ja"
    difficulty = "top"

class PolyMathJAHighExtractor(PolyMathExtractor):
    language = "ja"
    difficulty = "high"

class PolyMathJAMediumExtractor(PolyMathExtractor):
    language = "ja"
    difficulty = "medium"

class PolyMathJALowExtractor(PolyMathExtractor):
    language = "ja"
    difficulty = "low"

# Korean
class PolyMathKOTopExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "top"

class PolyMathKOHighExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "high"

class PolyMathKOMediumExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "medium"

class PolyMathKOLowExtractor(PolyMathExtractor):
    language = "ko"
    difficulty = "low"

# Malay
class PolyMathMSTopExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "top"

class PolyMathMSHighExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "high"

class PolyMathMSMediumExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "medium"

class PolyMathMSLowExtractor(PolyMathExtractor):
    language = "ms"
    difficulty = "low"

# Portuguese
class PolyMathPTTopExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "top"

class PolyMathPTHighExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "high"

class PolyMathPTMediumExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "medium"

class PolyMathPTLowExtractor(PolyMathExtractor):
    language = "pt"
    difficulty = "low"

# Russian
class PolyMathRUTopExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "top"

class PolyMathRUHighExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "high"

class PolyMathRUMediumExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "medium"

class PolyMathRULowExtractor(PolyMathExtractor):
    language = "ru"
    difficulty = "low"

# Swahili
class PolyMathSWTopExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "top"

class PolyMathSWHighExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "high"

class PolyMathSWMediumExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "medium"

class PolyMathSWLowExtractor(PolyMathExtractor):
    language = "sw"
    difficulty = "low"

# Telugu
class PolyMathTETopExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "top"

class PolyMathTEHighExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "high"

class PolyMathTEMediumExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "medium"

class PolyMathTELowExtractor(PolyMathExtractor):
    language = "te"
    difficulty = "low"

# Thai
class PolyMathTHTopExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "top"

class PolyMathTHHighExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "high"

class PolyMathTHMediumExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "medium"

class PolyMathTHLowExtractor(PolyMathExtractor):
    language = "th"
    difficulty = "low"

# Vietnamese
class PolyMathVITopExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "top"

class PolyMathVIHighExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "high"

class PolyMathVIMediumExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "medium"

class PolyMathVILowExtractor(PolyMathExtractor):
    language = "vi"
    difficulty = "low"

# Chinese
class PolyMathZHTopExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "top"

class PolyMathZHHighExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "high"

class PolyMathZHMediumExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "medium"

class PolyMathZHLowExtractor(PolyMathExtractor):
    language = "zh"
    difficulty = "low"
