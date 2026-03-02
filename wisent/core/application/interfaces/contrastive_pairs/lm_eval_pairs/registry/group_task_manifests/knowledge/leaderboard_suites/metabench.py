"""Metabench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

METABENCH_TASKS = {
    "metabench_arc": f"{BASE_IMPORT}arc:ArcExtractor",
    "metabench_arc_permute": f"{BASE_IMPORT}arc:ArcExtractor",
    "metabench_arc_secondary": f"{BASE_IMPORT}arc:ArcExtractor",
    "metabench_arc_secondary_permute": f"{BASE_IMPORT}arc:ArcExtractor",
    "metabench_arc_subset": f"{BASE_IMPORT}arc:ArcExtractor",
    "metabench_gsm8k": f"{BASE_IMPORT}gsm8k:Gsm8kExtractor",
    "metabench_gsm8k_secondary": f"{BASE_IMPORT}gsm8k:Gsm8kExtractor",
    "metabench_gsm8k_subset": f"{BASE_IMPORT}gsm8k:Gsm8kExtractor",
    "metabench_hellaswag": f"{BASE_IMPORT}hellaswag:HellaswagExtractor",
    "metabench_hellaswag_permute": f"{BASE_IMPORT}hellaswag:HellaswagExtractor",
    "metabench_hellaswag_secondary": f"{BASE_IMPORT}hellaswag:HellaswagExtractor",
    "metabench_hellaswag_secondary_permute": f"{BASE_IMPORT}hellaswag:HellaswagExtractor",
    "metabench_hellaswag_subset": f"{BASE_IMPORT}hellaswag:HellaswagExtractor",
    "metabench_mmlu": f"{BASE_IMPORT}mmlu:MMLUExtractor",
    "metabench_mmlu_permute": f"{BASE_IMPORT}mmlu:MMLUExtractor",
    "metabench_mmlu_secondary": f"{BASE_IMPORT}mmlu:MMLUExtractor",
    "metabench_mmlu_secondary_permute": f"{BASE_IMPORT}mmlu:MMLUExtractor",
    "metabench_mmlu_subset": f"{BASE_IMPORT}mmlu:MMLUExtractor",
    "metabench_truthfulqa": f"{BASE_IMPORT}truthfulqa:TruthfulqaExtractor",
    "metabench_truthfulqa_permute": f"{BASE_IMPORT}truthfulqa:TruthfulqaExtractor",
    "metabench_truthfulqa_secondary": f"{BASE_IMPORT}truthfulqa:TruthfulqaExtractor",
    "metabench_truthfulqa_secondary_permute": f"{BASE_IMPORT}truthfulqa:TruthfulqaExtractor",
    "metabench_truthfulqa_subset": f"{BASE_IMPORT}truthfulqa:TruthfulqaExtractor",
    "metabench_winogrande": f"{BASE_IMPORT}winogrande:WinograndeExtractor",
    "metabench_winogrande_permute": f"{BASE_IMPORT}winogrande:WinograndeExtractor",
    "metabench_winogrande_secondary": f"{BASE_IMPORT}winogrande:WinograndeExtractor",
    "metabench_winogrande_secondary_permute": f"{BASE_IMPORT}winogrande:WinograndeExtractor",
    "metabench_winogrande_subset": f"{BASE_IMPORT}winogrande:WinograndeExtractor",
}
