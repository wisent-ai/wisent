"""Xcopa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

XCOPA_TASKS = {
    "xcopa": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_et": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_eu": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_ht": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_id": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_it": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_qu": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_sw": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_ta": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_th": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_tr": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_vi": f"{BASE_IMPORT}xcopa:XcopaExtractor",
    "xcopa_zh": f"{BASE_IMPORT}xcopa:XcopaExtractor",
}
