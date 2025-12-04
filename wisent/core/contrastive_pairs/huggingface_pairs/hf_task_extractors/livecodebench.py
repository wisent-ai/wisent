from __future__ import annotations

from typing import Any
from wisent.core.cli_logger import setup_logger

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.huggingface_pairs.atoms import HuggingFaceBenchmarkExtractor
from wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors.livecodebench_contrastive_pair_generator import (
    generate_livecodebench_pairs,
)

__all__ = ["LivecodebenchExtractor"]

log = setup_logger(__name__)

task_names = ("livecodebench",)

class LivecodebenchExtractor(HuggingFaceBenchmarkExtractor):
    """
    Extractor for livecodebench dataset.

    This extractor loads pre-computed correct and incorrect code solutions
    from the LiveCodeBench dataset's all_outputs.json file.

    Schema (livecodebench/code_generation_samples):
        - question_content: str (question/prompt)
        - all_outputs: dict (pre-computed model outputs with pass/fail status)
    """


    evaluator_name = "coding"
    def extract_contrastive_pairs(
        self,
        limit: int | None = None,
    ) -> list[ContrastivePair]:
        """
        Build contrastive pairs from livecodebench pre-computed outputs.

        This uses existing correct (passing) and incorrect (failing) code
        solutions from various models that were pre-computed on the dataset.

        Args:
            limit: Optional maximum number of pairs to produce.

        Returns:
            A list of ContrastivePair objects.
        """
        max_items = self._normalize_limit(limit)

        log.info(f"Generating livecodebench contrastive pairs (limit={max_items})")

        # Use the contrastive pair generator to load pre-computed solutions
        pairs = generate_livecodebench_pairs(
            limit=max_items,
            cache_dir=None,  # Use default cache
        )

        if not pairs:
            log.warning("No valid livecodebench pairs generated")

        return pairs
