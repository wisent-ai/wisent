"""Afrobench ntrex group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_NTREX_TASKS = {
    "afrobench_ntrex": f"{BASE_IMPORT}afrobench_ntrex:AfrobenchNtrexExtractor",
}
