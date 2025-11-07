"""AraDiCE extractor that delegates to base task extractors."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from wisent.core.contrastive_pairs.core.pair import ContrastivePair
from wisent.core.contrastive_pairs.lm_eval_pairs.atoms import LMEvalBenchmarkExtractor
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor
from wisent.core.cli_logger import setup_logger, bind

if TYPE_CHECKING:
    from lm_eval.api.task import ConfigurableTask


__all__ = ["AradiceExtractor"]
_LOG = setup_logger(__name__)


class AradiceExtractor(LMEvalBenchmarkExtractor):
    """Extractor for AraDiCE benchmark that routes subtasks to their base extractors.

    AraDiCE contains 28 subtasks across multiple Arabic dialects:
    - AraDiCE_boolq_{dialect} -> BoolQExtractor
    - AraDiCE_winogrande_{dialect} -> WinograndeExtractor
    - AraDiCE_truthfulqa_mc1_{dialect} -> TruthfulQAMC1Extractor
    - AraDiCE_piqa_{dialect} -> PIQAExtractor
    - AraDiCE_openbookqa_{dialect} -> OpenBookQAExtractor
    - AraDiCE_ArabicMMLU_*_{dialect} -> MMLUExtractor
    """

    def extract_contrastive_pairs(
        self,
        lm_eval_task_data: ConfigurableTask,
        limit: int | None = None,
        preferred_doc: str | None = None,
    ) -> list[ContrastivePair]:
        """Route AraDiCE subtask to appropriate base extractor.

        Args:
            lm_eval_task_data: lm-eval task instance for AraDiCE subtask.
            limit: Optional maximum number of pairs to produce.
            preferred_doc: Optional preferred document source.

        Returns:
            A list of ContrastivePair objects.
        """
        log = bind(_LOG, task=getattr(lm_eval_task_data, "NAME", "unknown"))
        task_name = getattr(lm_eval_task_data.config, "task", "unknown")

        log.info(f"Processing AraDiCE subtask: {task_name}")

        # Parse the subtask name to get the base task
        # Pattern: AraDiCE_{base_task}_{dialect}
        # Examples: AraDiCE_boolq_egy, AraDiCE_ArabicMMLU_high_stem_physics_lev
        parts = task_name.split('_')

        if len(parts) < 3:
            raise ValueError(f"Unexpected AraDiCE task name format: {task_name}")

        # Skip the "AraDiCE" prefix and get the base task
        base_task = parts[1]

        # Special cases
        if base_task == "ArabicMMLU":
            # ArabicMMLU subtasks should use mmlu extractor
            base_task = "mmlu"

        # Map base task names to extractor modules
        extractor_map = {
            "boolq": "boolq:BoolQExtractor",
            "winogrande": "winogrande:WinograndeExtractor",
            "truthfulqa_mc1": "truthfulqa_mc1:TruthfulQAMC1Extractor",
            "piqa": "piqa:PIQAExtractor",
            "openbookqa": "openbookqa:OpenBookQAExtractor",
            "mmlu": "mmlu:MMLUExtractor",
        }

        if base_task not in extractor_map:
            raise ValueError(f"No extractor mapping found for AraDiCE subtask base: {base_task} (from {task_name})")

        log.info(f"Routing {task_name} to {base_task} extractor")
        # Import the extractor directly to avoid recursion
        module_path, class_name = extractor_map[base_task].split(":")
        module = __import__(
            f"wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.{module_path}",
            fromlist=[class_name]
        )
        extractor_class = getattr(module, class_name)
        extractor = extractor_class()
        return extractor.extract_contrastive_pairs(lm_eval_task_data, limit, preferred_doc)
