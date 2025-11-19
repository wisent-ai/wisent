"""Med group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MED_TASKS = {
    "med_concepts_qa": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_atc_easy": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_atc_medium": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_atc_hard": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd10cm_easy": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd10cm_medium": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd10cm_hard": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd10proc_easy": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd10proc_medium": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd10proc_hard": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd9cm_easy": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd9cm_medium": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd9cm_hard": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd9proc_easy": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd9proc_medium": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
    "med_concepts_qa_icd9proc_hard": f"{BASE_IMPORT}med_concepts_qa:MedConceptsQaExtractor",
}
