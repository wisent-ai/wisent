from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor

from wisent.core.evaluators.benchmark_specific.math_parsing.scripts import strip_string

from latex2sympy2_extended import latex2sympy
from sympy import latex

__all__ = [
    "LiveMathBenchExtractor",
    # v202412 - December 2024 release
    "LiveMathBenchCnmoEnExtractor",
    "LiveMathBenchCnmoCnExtractor",
    "LiveMathBenchCceeEnExtractor",
    "LiveMathBenchCceeCnExtractor",
    "LiveMathBenchAmcEnExtractor",
    "LiveMathBenchAmcCnExtractor",
    "LiveMathBenchWlpmcEnExtractor",
    "LiveMathBenchWlpmcCnExtractor",
    "LiveMathBenchHardEnExtractor",
    "LiveMathBenchHardCnExtractor",
    # v202505 - May 2025 release
    "LiveMathBenchV202505AllEnExtractor",
    "LiveMathBenchV202505HardEnExtractor",
]

log = setup_logger(__name__)

task_names = (
    # v202412 - December 2024 release
    "livemathbench_cnmo_en", "livemathbench_cnmo_cn",  # China National Mathematical Olympiad
    "livemathbench_ccee_en", "livemathbench_ccee_cn",  # China's College Entrance Examination
    "livemathbench_amc_en", "livemathbench_amc_cn",    # American Mathematics Competition
    "livemathbench_wlpmc_en", "livemathbench_wlpmc_cn", # William Lowell Putnam Mathematical Competition
    "livemathbench_hard_en", "livemathbench_hard_cn",  # Hard problems subset
    # v202505 - May 2025 release
    "livemathbench_v202505_all_en",   # All problems from v202505
    "livemathbench_v202505_hard_en",  # Hard problems from v202505
)


class LiveMathBenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for LiveMathBench dataset (mathematical olympiad problems).

    LiveMathBench schema (opencompass/LiveMathBench):
        - question: str (math problem statement in Chinese or English)
        - answer: str (final answer)
    """

    evaluator_name = "exact_match"

    # Set default dataset_config and config_label, to be override by subclass
    dataset_config: str = "v202412_CNMO_en"
    config_label: str = "cnmo_en"

    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from LiveMathBench examples.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        docs = self.load_dataset(
            dataset_name="opencompass/LiveMathBench",
            dataset_config=self.dataset_config,
            split="test",
            limit=max_items,
        )

        pairs: list[ContrastivePair] = []

        log.info(f"Extracting contrastive pairs from {len(docs)} LiveMathBench ({self.config_label}) examples")

        for doc in docs:
            pair = self._extract_pair_from_doc(doc)
            if pair is not None:
                pairs.append(pair)
                if max_items is not None and len(pairs) >= max_items:
                    break

        if not pairs:
            log.warning(f"No valid LiveMathBench ({self.config_label}) pairs extracted")

        return pairs

    def _extract_pair_from_doc(self, doc: dict[str, Any]) -> ContrastivePair | None:
        """
        Convert a single LiveMathBench doc into a ContrastivePair.

        Returns None when required fields are missing or malformed.
        """
        try:
            problem = doc.get("question", "").strip()
            raw_answer = doc.get("answer", "")

            if not problem or not raw_answer:
                log.debug("Skipping: missing problem or answer")
                return None

            correct = strip_string(raw_answer)
            if not correct:
                correct = raw_answer
            incorrect = self._create_incorrect_answer(correct)

            question = f"Question: {problem}\n\nWhat is the answer?"

            metadata = {
                "label": "livemathbench",
                "source": "opencompass/LiveMathBench",
                "config": self.dataset_config,
            }

            return self._build_pair(
                question=question,
                correct=correct,
                incorrect=incorrect,
                metadata=metadata,
            )

        except Exception as exc:
            log.error(f"Error extracting pair from doc: {exc}", exc_info=True)
            return None

    def _create_incorrect_answer(self, correct: str) -> str:
        """Create an incorrect answer by modifying the correct one (input is already stripped)."""
        try:
            parsed_correct = latex2sympy(correct)
            incorrect = latex(parsed_correct + 1)
            return str(incorrect)
        except Exception:
            return f"{correct} + 1"


# ============================================================================
# v202412 - December 2024 release
# ============================================================================

# China National Mathematical Olympiad (CNMO)
class LiveMathBenchCnmoEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench CNMO English."""
    dataset_config = "v202412_CNMO_en"
    config_label = "cnmo_en"


class LiveMathBenchCnmoCnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench CNMO Chinese."""
    dataset_config = "v202412_CNMO_cn"
    config_label = "cnmo_cn"


# China's College Entrance Examination (CCEE)
class LiveMathBenchCceeEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench CCEE English."""
    dataset_config = "v202412_CCEE_en"
    config_label = "ccee_en"


class LiveMathBenchCceeCnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench CCEE Chinese."""
    dataset_config = "v202412_CCEE_cn"
    config_label = "ccee_cn"


# American Mathematics Competition (AMC)
class LiveMathBenchAmcEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench AMC English."""
    dataset_config = "v202412_AMC_en"
    config_label = "amc_en"


class LiveMathBenchAmcCnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench AMC Chinese."""
    dataset_config = "v202412_AMC_cn"
    config_label = "amc_cn"


# William Lowell Putnam Mathematical Competition (WLPMC)
class LiveMathBenchWlpmcEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench WLPMC English."""
    dataset_config = "v202412_WLPMC_en"
    config_label = "wlpmc_en"


class LiveMathBenchWlpmcCnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench WLPMC Chinese."""
    dataset_config = "v202412_WLPMC_cn"
    config_label = "wlpmc_cn"


# Hard problems subset
class LiveMathBenchHardEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench Hard English."""
    dataset_config = "v202412_hard_en"
    config_label = "hard_en"


class LiveMathBenchHardCnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench Hard Chinese."""
    dataset_config = "v202412_hard_cn"
    config_label = "hard_cn"


# ============================================================================
# v202505 - May 2025 release
# ============================================================================

class LiveMathBenchV202505AllEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench v202505 All English."""
    dataset_config = "v202505_all_en"
    config_label = "v202505_all_en"


class LiveMathBenchV202505HardEnExtractor(LiveMathBenchExtractor):
    """Extractor for LiveMathBench v202505 Hard English."""
    dataset_config = "v202505_hard_en"
    config_label = "v202505_hard_en"
