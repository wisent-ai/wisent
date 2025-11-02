__all__ = [
    "EXTRACTORS",
]

base_import: str = "wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors."

EXTRACTORS: dict[str, str] = {
    # Math benchmarks
    "aime": f"{base_import}aime:AIMEExtractor",
    "hmmt": f"{base_import}hmmt:HMMTExtractor",
    "livemathbench": f"{base_import}livemathbench:LiveMathBenchExtractor",
    "math": f"{base_import}math:MATHExtractor",
    "math_500": f"{base_import}math:MATHExtractor",
    "polymath": f"{base_import}polymath:PolyMathExtractor",

    # Coding benchmarks
    "humaneval": f"{base_import}humaneval:HumanEvalExtractor",
    "humaneval_plus": f"{base_import}humaneval:HumanEvalExtractor",
    "mbpp": f"{base_import}mbpp:MBPPExtractor",
    "mbpp_plus": f"{base_import}mbpp:MBPPExtractor",
    "instruct_humaneval": f"{base_import}instruct_humaneval:InstructHumanevalExtractor",
    "apps": f"{base_import}apps:AppsExtractor",
    "conala": f"{base_import}conala:ConalaExtractor",
    "concode": f"{base_import}concode:ConcodeExtractor",
    "ds_1000": f"{base_import}ds_1000:Ds1000Extractor",
    "mercury": f"{base_import}mercury:MercuryExtractor",
    "recode": f"{base_import}recode:RecodeExtractor",
    "multipl_e": f"{base_import}multipl_e:MultiplEExtractor",
    "codexglue": f"{base_import}codexglue:CodexglueExtractor",
    "livecodebench": f"{base_import}livecodebench:LivecodebenchExtractor",

    # Reasoning benchmarks
    "super_gpqa": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "hle": f"{base_import}hle:HleExtractor",
}

