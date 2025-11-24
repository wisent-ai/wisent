"""Okapi arc multilingual group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

OKAPI_ARC_MULTILINGUAL_TASKS = {
    "okapi_arc_multilingual": f"{BASE_IMPORT}okapi_arc_multilingual:OkapiArcMultilingualExtractor",
    "okapi/arc_multilingual": f"{BASE_IMPORT}okapi_arc_multilingual:OkapiArcMultilingualExtractor",
}
