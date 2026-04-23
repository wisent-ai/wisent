"""HuggingFace extractor manifest - Part 1 (math, coding, QA, reasoning)."""

from wisent.core.utils.config_tools.constants import HF_EXTRACTOR_BASE_IMPORT
base_import: str = HF_EXTRACTOR_BASE_IMPORT

EXTRACTORS_MATH_CODING: dict[str, str] = {
    # Math benchmarks
    "aime": f"{base_import}aime:AIMEExtractor",
    "aime2024": f"{base_import}aime2024:AIME2024Extractor",
    "aime2025": f"{base_import}aime2025:AIME2025Extractor",
    "asdiv_cot_llama": f"{base_import}math500:MATH500Extractor",
    "chain": f"{base_import}math500:MATH500Extractor",
    "chain_of_thought": f"{base_import}math500:MATH500Extractor",
    "gsm8k": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_cot": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_cot_llama": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_cot_self_consistency": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_llama": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_platinum": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_platinum_cot": f"{base_import}gsm8k_extractor:GSM8KExtractor",
    "gsm8k_platinum_cot_llama": f"{base_import}math500:MATH500Extractor",
    "gsm8k_platinum_cot_self_consistency": f"{base_import}math500:MATH500Extractor",
    "hmmt": f"{base_import}hmmt:HMMTExtractor",
    "hmmt_feb_2025": f"{base_import}hmmt:HMMTExtractor",
    "livemathbench": f"{base_import}livemathbench:LiveMathBenchExtractor",
    # v202412 - December 2024 release
    "livemathbench_cnmo_en": f"{base_import}livemathbench:LiveMathBenchCnmoEnExtractor",
    "livemathbench_cnmo_cn": f"{base_import}livemathbench:LiveMathBenchCnmoCnExtractor",
    "livemathbench_ccee_en": f"{base_import}livemathbench:LiveMathBenchCceeEnExtractor",
    "livemathbench_ccee_cn": f"{base_import}livemathbench:LiveMathBenchCceeCnExtractor",
    "livemathbench_amc_en": f"{base_import}livemathbench:LiveMathBenchAmcEnExtractor",
    "livemathbench_amc_cn": f"{base_import}livemathbench:LiveMathBenchAmcCnExtractor",
    "livemathbench_wlpmc_en": f"{base_import}livemathbench:LiveMathBenchWlpmcEnExtractor",
    "livemathbench_wlpmc_cn": f"{base_import}livemathbench:LiveMathBenchWlpmcCnExtractor",
    "livemathbench_hard_en": f"{base_import}livemathbench:LiveMathBenchHardEnExtractor",
    "livemathbench_hard_cn": f"{base_import}livemathbench:LiveMathBenchHardCnExtractor",
    # v202505 - May 2025 release
    "livemathbench_v202505_all_en": f"{base_import}livemathbench:LiveMathBenchV202505AllEnExtractor",
    "livemathbench_v202505_hard_en": f"{base_import}livemathbench:LiveMathBenchV202505HardEnExtractor",
    "math": f"{base_import}math_benchmark:MATHExtractor",
    "math_500": f"{base_import}math500:MATH500Extractor",
    "math500": f"{base_import}math500:MATH500Extractor",
    "polymath": f"{base_import}polymath:PolyMathExtractor",
    # Arabic
    "polymath_ar_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ar_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ar_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ar_low": f"{base_import}polymath:PolyMathExtractor",
    # Bengali
    "polymath_bn_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_bn_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_bn_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_bn_low": f"{base_import}polymath:PolyMathExtractor",
    # German
    "polymath_de_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_de_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_de_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_de_low": f"{base_import}polymath:PolyMathExtractor",
    # English
    "polymath_en_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_en_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_en_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_en_low": f"{base_import}polymath:PolyMathExtractor",
    # Spanish
    "polymath_es_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_es_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_es_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_es_low": f"{base_import}polymath:PolyMathExtractor",
    # French
    "polymath_fr_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_fr_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_fr_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_fr_low": f"{base_import}polymath:PolyMathExtractor",
    # Indonesian
    "polymath_id_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_id_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_id_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_id_low": f"{base_import}polymath:PolyMathExtractor",
    # Italian
    "polymath_it_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_it_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_it_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_it_low": f"{base_import}polymath:PolyMathExtractor",
    # Japanese
    "polymath_ja_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ja_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ja_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ja_low": f"{base_import}polymath:PolyMathExtractor",
    # Korean
    "polymath_ko_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ko_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ko_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ko_low": f"{base_import}polymath:PolyMathExtractor",
    # Malay
    "polymath_ms_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ms_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ms_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ms_low": f"{base_import}polymath:PolyMathExtractor",
    # Portuguese
    "polymath_pt_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_pt_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_pt_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_pt_low": f"{base_import}polymath:PolyMathExtractor",
    # Russian
    "polymath_ru_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ru_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ru_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_ru_low": f"{base_import}polymath:PolyMathExtractor",
    # Swahili
    "polymath_sw_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_sw_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_sw_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_sw_low": f"{base_import}polymath:PolyMathExtractor",
    # Telugu
    "polymath_te_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_te_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_te_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_te_low": f"{base_import}polymath:PolyMathExtractor",
    # Thai
    "polymath_th_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_th_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_th_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_th_low": f"{base_import}polymath:PolyMathExtractor",
    # Vietnamese
    "polymath_vi_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_vi_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_vi_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_vi_low": f"{base_import}polymath:PolyMathExtractor",
    # Chinese
    "polymath_zh_top": f"{base_import}polymath:PolyMathExtractor",
    "polymath_zh_high": f"{base_import}polymath:PolyMathExtractor",
    "polymath_zh_medium": f"{base_import}polymath:PolyMathExtractor",
    "polymath_zh_low": f"{base_import}polymath:PolyMathExtractor",

    # Coding benchmarks
    "humaneval": f"{base_import}humaneval:HumanEvalExtractor",
    "humaneval_64": f"{base_import}humaneval:HumanEval64Extractor",
    "humaneval_plus": f"{base_import}humaneval:HumanEvalPlusExtractor",
    "humaneval_instruct": f"{base_import}humaneval:HumanEvalInstructExtractor",
    "humaneval_64_instruct": f"{base_import}humaneval:HumanEval64InstructExtractor",
    "humanevalpack": f"{base_import}humanevalpack:HumanevalpackExtractor",
    "apps": f"{base_import}apps:AppsExtractor",
    "conala": f"{base_import}conala:ConalaExtractor",
    "concode": f"{base_import}concode:ConcodeExtractor",
    "ds_1000": f"{base_import}ds_1000:Ds1000Extractor",
    "ds1000": f"{base_import}ds_1000:Ds1000Extractor",
    "mercury": f"{base_import}mercury:MercuryExtractor",
    "recode": f"{base_import}recode:RecodeExtractor",
    "multipl_e": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_py": f"{base_import}humaneval:HumanEvalExtractor",
    "multiple_js": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_java": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_cpp": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_rs": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_go": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_cs": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_d": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_jl": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_lua": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_php": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_pl": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_r": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_rb": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_rkt": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_scala": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_sh": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_swift": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_ts": f"{base_import}multipl_e:MultiplEExtractor",
    "codexglue": f"{base_import}codexglue:CodexglueExtractor",
    "code2text": f"{base_import}codexglue:Code2TextExtractor",
    "code2text_python": f"{base_import}codexglue:Code2TextExtractor",
    "code2text_go": f"{base_import}codexglue:Code2TextGoExtractor",
    "code2text_java": f"{base_import}codexglue:Code2TextJavaExtractor",
    "code2text_javascript": f"{base_import}codexglue:Code2TextJavascriptExtractor",
    "code2text_php": f"{base_import}codexglue:Code2TextPhpExtractor",
    "code2text_ruby": f"{base_import}codexglue:Code2TextRubyExtractor",
    "code_x_glue": f"{base_import}codexglue:Code2TextExtractor",
    "codexglue_code_to_text_python": f"{base_import}codexglue:Code2TextExtractor",
    "codexglue_code_to_text_go": f"{base_import}codexglue:Code2TextGoExtractor",
    "codexglue_code_to_text_java": f"{base_import}codexglue:Code2TextJavaExtractor",
    "codexglue_code_to_text_javascript": f"{base_import}codexglue:Code2TextJavascriptExtractor",
    "codexglue_code_to_text_php": f"{base_import}codexglue:Code2TextPhpExtractor",
    "codexglue_code_to_text_ruby": f"{base_import}codexglue:Code2TextRubyExtractor",
    "doc": f"{base_import}codexglue:DocNliExtractor",
    "doc_nli": f"{base_import}codexglue:DocNliExtractor",
    "livecodebench": f"{base_import}livecodebench:LivecodebenchExtractor",

    # ACP Bench (planning/agent benchmarks) — group task HF extractors
    # Individual _gen subtasks are handled by per-task entries in the HF manifest.
    # These group extractors aggregate all gen subtasks for the group task names.
    "acp_bench_hard": f"{base_import}acp_bench_hard_hf:AcpBenchHardGroupHFExtractor",
    "acp_bench_hard_with_pddl": f"{base_import}acp_bench_hard_hf:AcpBenchHardWithPddlGroupHFExtractor",
    "acpbench": f"{base_import}acp_bench_hard_hf:AcpBenchGroupHFExtractor",

    # Reasoning benchmarks
    "super_gpqa": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa_physics": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa_chemistry": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "supergpqa_biology": f"{base_import}super_gpqa:SuperGpqaExtractor",
    "hle": f"{base_import}hle:HleExtractor",
    "hle_exact_match": f"{base_import}hle:HleExtractor",
    "hle_multiple_choice": f"{base_import}hle:HleExtractor",

    # Database/Table benchmarks
    "tag": f"{base_import}tag:TagExtractor",
    "Tag": f"{base_import}tag:TagExtractor",

    # Medical benchmarks
    "meddialog": f"{base_import}meddialog:MeddialogExtractor",
    "meddialog_qsumm": f"{base_import}meddialog:MeddialogExtractor",
    "meddialog_qsumm_perplexity": f"{base_import}meddialog:MeddialogExtractor",
    "meddialog_raw_dialogues": f"{base_import}meddialog:MeddialogExtractor",
    "meddialog_raw_perplexity": f"{base_import}meddialog:MeddialogExtractor",

    # MMLU-SR benchmarks (all variants use the same extractor)
    "mmlusr": f"{base_import}mmlusr:MMLUSRExtractor",
    "mmlusr_answer_only": f"{base_import}mmlusr:MMLUSRExtractor",
    "mmlusr_question_only": f"{base_import}mmlusr:MMLUSRExtractor",
    "mmlusr_question_and_answer": f"{base_import}mmlusr:MMLUSRExtractor",

    # Newly moved from lm_eval_pairs
    "atis": f"{base_import}atis:AtisExtractor",
    "babilong": f"{base_import}babilong:BabilongExtractor",
    "bangla_mmlu": f"{base_import}bangla_mmlu:BanglaMmluExtractor",
    # Winogender - gender bias benchmark (HuggingFace oskarvanderwal/winogender)
    "winogender": f"{base_import}winogender:WinogenderHfExtractor",
    "basqueglue": f"{base_import}basqueglue:BasqueglueExtractor",
    "bec2016eu": f"{base_import}bec2016eu:Bec2016euExtractor",
    "boolq_seq2seq": f"{base_import}super_glue_lm_eval_v1_seq2seq:SuperGlueLmEvalV1Seq2seqExtractor",
    "doc_vqa": f"{base_import}doc_vqa:DocVQAExtractor",
    "ds1000": f"{base_import}ds1000:Ds1000Extractor",
    "evalita_mp": f"{base_import}evalita_mp:EvalitaMpExtractor",
    "flores": f"{base_import}flores:FloresExtractor",
    "african_flores": f"{base_import}flores:FloresExtractor",
    "african_flores_tasks": f"{base_import}flores:FloresExtractor",
    "humanevalpack": f"{base_import}humanevalpack:HumanevalpackExtractor",
    "iwslt2017_ar_en": f"{base_import}iwslt2017_ar_en:Iwslt2017ArEnExtractor",
    "iwslt2017_en_ar": f"{base_import}iwslt2017_en_ar:Iwslt2017EnArExtractor",
    # Bare parent — both directions are registered above; route the parent
    # to the ar->en direction so `wisent ... iwslt2017` resolves.
    "iwslt2017": f"{base_import}iwslt2017_ar_en:Iwslt2017ArEnExtractor",
    "llama": f"{base_import}llama:LlamaExtractor",
    "multimedqa": f"{base_import}multimedqa:MultimedqaExtractor",
    "openllm": f"{base_import}openllm:OpenllmExtractor",
    "pythia": f"{base_import}pythia:PythiaExtractor",
    "squad2": f"{base_import}squad_extractor:SQuADv2Extractor",
    "stsb": f"{base_import}stsb:StsbExtractor",
    "super_glue_lm_eval_v1": f"{base_import}super_glue_lm_eval_v1:SuperGlueLmEvalV1Extractor",
    "super_glue_lm_eval_v1_seq2seq": f"{base_import}super_glue_lm_eval_v1_seq2seq:SuperGlueLmEvalV1Seq2seqExtractor",
    "super_glue_t5_prompt": f"{base_import}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "tmlu": f"{base_import}tmlu:TmluExtractor",
    "wiceu": f"{base_import}wiceu:WiceuExtractor",
    "wmt_ro_en_t5_prompt": f"{base_import}wmt_ro_en_t5_prompt:WmtRoEnT5PromptExtractor",

    # Newly created extractors
    # Note: unitxt tasks (xsum, cnn_dailymail, dbpedia_14, ethos_binary, etc.) use lm-eval unitxt extractor
    # Note: babi removed - use lm-eval version instead (HF version has loading issues)
    "bhtc_v2": f"{base_import}bhtc_v2:BhtcV2Extractor",
    "drop": f"{base_import}squad_extractor:DROPExtractor",
    "pubmedqa": f"{base_import}squad_extractor:PubMedQAHFExtractor",
    "sciq": f"{base_import}babi_extractor:SciQExtractor",
    "squadv2": f"{base_import}squad_extractor:SQuADv2Extractor",
    "basque-glue": f"{base_import}basqueglue:BasqueglueExtractor",
    "evalita-sp_sum_task_fp-small_p1": f"{base_import}evalita_sp_sum_task_fp_small_p1:EvalitaSpSumTaskFpSmallP1Extractor",
    "penn_treebank": f"{base_import}penn_treebank:PennTreebankExtractor",
    "ptb": f"{base_import}penn_treebank:PennTreebankExtractor",
    "self_consistency": f"{base_import}self_consistency:SelfConsistencyExtractor",
    "vaxx_stance": f"{base_import}vaxx_stance:VaxxStanceExtractor",
    "wikitext103": f"{base_import}wikitext103:Wikitext103Extractor",

    # NOTE: truthfulqa_gen/truthfulqa_generation now use lm-eval extractor (see group_task_manifests/truthfulqa.py)

    # Factuality benchmarks (NOT lm-eval)
    "simpleqa": f"{base_import}simpleqa:SimpleQAExtractor",
    "simple_qa": f"{base_import}simpleqa:SimpleQAExtractor",

    # MMLU variants (NOT lm-eval)
    "mmlu_redux": f"{base_import}mmlu_redux:MMLUReduxExtractor",
    "mmlu-redux": f"{base_import}mmlu_redux:MMLUReduxExtractor",

    # Multi-hop reasoning benchmarks (NOT lm-eval)
    "frames": f"{base_import}frames:FRAMESExtractor",
    "frames_benchmark": f"{base_import}frames:FRAMESExtractor",

    # Medical/Health benchmarks (NOT lm-eval)
    "healthbench": f"{base_import}healthbench:HealthBenchExtractor",
    "health_bench": f"{base_import}healthbench:HealthBenchExtractor",
}
