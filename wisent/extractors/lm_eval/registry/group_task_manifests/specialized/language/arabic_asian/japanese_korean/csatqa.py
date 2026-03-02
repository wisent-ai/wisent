"""Csatqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

CSATQA_TASKS = {
    "csatqa": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
    "csatqa_gr": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
    "csatqa_li": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
    "csatqa_rch": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
    "csatqa_rcs": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
    "csatqa_rcss": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
    "csatqa_wr": f"{BASE_IMPORT}csatqa:CsatqaExtractor",
}
