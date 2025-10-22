__all__ = [
    "EXTRACTORS",
]
base_import: str = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."
EXTRACTORS: dict[str, str] = {
    # key → "module_path:ClassName" (supports dotted attr path after ':')

    # Original
    "winogrande": f"{base_import}winogrande:WinograndeExtractor",

    # Question Answering & Boolean
    "truthfulqa_mc1": f"{base_import}truthfulqa_mc1:TruthfulQAMC1Extractor",
    "truthfulqa_mc2": f"{base_import}truthfulqa_mc1:TruthfulQAMC1Extractor",  # Can reuse MC1
    "boolq": f"{base_import}qa_tasks:BoolQExtractor",
    "record": f"{base_import}qa_tasks:RecordExtractor",

    # Common Reasoning - Multiple Choice
    "hellaswag": f"{base_import}generic_mc:HellaSwagExtractor",
    "copa": f"{base_import}generic_mc:COPAExtractor",
    "piqa": f"{base_import}generic_mc:PIQAExtractor",
    "openbookqa": f"{base_import}generic_mc:OpenBookQAExtractor",
    "swag": f"{base_import}generic_mc:SWAGExtractor",
    "arc_easy": f"{base_import}generic_mc:ARCExtractor",
    "arc_challenge": f"{base_import}generic_mc:ARCExtractor",

    # Math tasks
    "gsm8k": f"{base_import}math_tasks:GSM8KExtractor",
    "math": f"{base_import}math_tasks:MATHExtractor",
    "math500": f"{base_import}math_tasks:MATH500Extractor",
    "hendrycks_math": f"{base_import}math_tasks:MATHExtractor",  # Same as MATH

    # Science tasks (using generic MC)
    "gpqa": f"{base_import}generic_mc:GenericMultipleChoiceExtractor",
    "gpqa_diamond": f"{base_import}generic_mc:GenericMultipleChoiceExtractor",
    "gpqa_extended": f"{base_import}generic_mc:GenericMultipleChoiceExtractor",
    "gpqa_main_zeroshot": f"{base_import}generic_mc:GenericMultipleChoiceExtractor",
    "gpqa_diamond_zeroshot": f"{base_import}generic_mc:GenericMultipleChoiceExtractor",
    "gpqa_extended_zeroshot": f"{base_import}generic_mc:GenericMultipleChoiceExtractor",
}
