"""Afrobench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_TASKS = {
    "adr": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "afrihate": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "afrisenti": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "belebele": f"{BASE_IMPORT}belebele:BelebeleExtractor",
    "african_flores": f"{BASE_IMPORT}flores:FloresExtractor",
    "injongointent": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "mafand": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "masakhaner": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "masakhapos": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "naijarc": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "nollysenti": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "african_ntrex": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "openai_mmlu": f"{BASE_IMPORT}mmlu:MMLUExtractor",
    "salt": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "sib": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "uhura": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
    "xlsum": f"{BASE_IMPORT}xlsum:XlsumExtractor",
    "afrobench": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
}
