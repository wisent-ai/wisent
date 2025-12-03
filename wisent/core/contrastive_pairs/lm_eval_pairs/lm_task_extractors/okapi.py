from __future__ import annotations

from typing import TYPE_CHECKING

from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.super_glue import SuperGlueExtractor

if TYPE_CHECKING:
    pass


__all__ = ["OkapiExtractor"]

# All okapi tasks are now managed by their specific extractors:
# - arc_* tasks: OkapiArcMultilingualExtractor
# - hellaswag_* tasks: OkapiHellaswagMultilingualExtractor
# - m_mmlu_* tasks: OkapiMmluMultilingualExtractor
# - truthfulqa_* tasks: OkapiTruthfulqaMultilingualExtractor
task_names = ()
class OkapiExtractor(SuperGlueExtractor):
    """Extractor for Okapi multilingual benchmarks (arc, hellaswag, mmlu, truthfulqa).

    Okapi benchmarks use the same multiple-choice format as SuperGlue, so we inherit
    the extraction logic directly from SuperGlueExtractor.
    """

    evaluator_name = "log_likelihoods"
    pass
