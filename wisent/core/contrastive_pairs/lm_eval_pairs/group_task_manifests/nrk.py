"""Nrk group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

NRK_TASKS = {
    "nrk_quiz_qa_nno": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nno_p0": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nno_p1": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nno_p2": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nno_p3": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nno_p4": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nob": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nob_p0": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nob_p1": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nob_p2": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nob_p3": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
    "nrk_quiz_qa_nob_p4": f"{BASE_IMPORT}noreval_mc:NorevalMultipleChoiceExtractor",
}
