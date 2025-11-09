__all__ = [
    "EXTRACTORS",
]

base_import: str = "wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors."

EXTRACTORS: dict[str, str] = {
    # Math benchmarks
    "aime": f"{base_import}aime:AIMEExtractor",
    "aime2024": f"{base_import}aime:AIMEExtractor",
    "aime2025": f"{base_import}aime:AIMEExtractor",
    "hmmt": f"{base_import}hmmt:HMMTExtractor",
    "hmmt_feb_2025": f"{base_import}hmmt:HMMTExtractor",
    "livemathbench": f"{base_import}livemathbench:LiveMathBenchExtractor",
    "livemathbench_cnmo_en": f"{base_import}livemathbench_configs:LiveMathBenchCnmoEnExtractor",
    "livemathbench_cnmo_zh": f"{base_import}livemathbench_configs:LiveMathBenchCnmoZhExtractor",
    "math": f"{base_import}math:MATHExtractor",
    "math_500": f"{base_import}math:MATHExtractor",
    "math500": f"{base_import}math:MATHExtractor",
    "polymath": f"{base_import}polymath:PolyMathExtractor",
    "polymath_en_medium": f"{base_import}polymath_configs:PolyMathEnMediumExtractor",
    "polymath_zh_medium": f"{base_import}polymath_configs:PolyMathZhMediumExtractor",
    "polymath_en_high": f"{base_import}polymath_configs:PolyMathEnHighExtractor",
    "polymath_zh_high": f"{base_import}polymath_configs:PolyMathZhHighExtractor",

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
    "ds1000": f"{base_import}ds_1000:Ds1000Extractor",
    "mercury": f"{base_import}mercury:MercuryExtractor",
    "recode": f"{base_import}recode:RecodeExtractor",
    "multipl_e": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_py": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_js": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_java": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_cpp": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_rs": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_go": f"{base_import}multipl_e:MultiplEExtractor",
    "codexglue": f"{base_import}codexglue:CodexglueExtractor",
    "code_x_glue": f"{base_import}codexglue:CodexglueExtractor",
    "codexglue_code_to_text_python": f"{base_import}codexglue:CodexglueExtractor",
    "codexglue_code_to_text_go": f"{base_import}codexglue:CodexglueExtractor",
    "codexglue_code_to_text_ruby": f"{base_import}codexglue:CodexglueExtractor",
    "codexglue_code_to_text_java": f"{base_import}codexglue:CodexglueExtractor",
    "codexglue_code_to_text_javascript": f"{base_import}codexglue:CodexglueExtractor",
    "codexglue_code_to_text_php": f"{base_import}codexglue:CodexglueExtractor",
    "livecodebench": f"{base_import}livecodebench:LivecodebenchExtractor",

    # Reasoning benchmarks
    "super_gpqa": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa_physics": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa_chemistry": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa_biology": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "hle": f"{base_import}hle:HleExtractor",
    "hle_exact_match": f"{base_import}hle:HleExtractor",
    "hle_multiple_choice": f"{base_import}hle:HleExtractor",
}

