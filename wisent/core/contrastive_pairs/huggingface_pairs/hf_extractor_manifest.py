__all__ = [
    "EXTRACTORS",
    "HF_EXTRACTORS",
]

base_import: str = "wisent.core.contrastive_pairs.huggingface_pairs.hf_task_extractors."

EXTRACTORS: dict[str, str] = {
    # Math benchmarks
    "aime": f"{base_import}aime:AIMEExtractor",
    "aime2024": f"{base_import}aime2024:AIME2024Extractor",
    "aime2025": f"{base_import}aime2025:AIME2025Extractor",
    "asdiv_cot_llama": f"{base_import}math500:MATH500Extractor",
    "chain_of_thought": f"{base_import}math500:MATH500Extractor",
    "gsm8k_cot": f"{base_import}math500:MATH500Extractor",
    "gsm8k_cot_llama": f"{base_import}math500:MATH500Extractor",
    "gsm8k_cot_self_consistency": f"{base_import}math500:MATH500Extractor",
    "gsm8k_llama": f"{base_import}math500:MATH500Extractor",
    "gsm8k_platinum_cot": f"{base_import}math500:MATH500Extractor",
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
    "math": f"{base_import}math:MATHExtractor",
    "math_500": f"{base_import}math500:MATH500Extractor",
    "math500": f"{base_import}math500:MATH500Extractor",
    "polymath": f"{base_import}polymath:PolyMathExtractor",
    # Arabic
    "polymath_ar_top": f"{base_import}polymath:PolyMathARTopExtractor",
    "polymath_ar_high": f"{base_import}polymath:PolyMathARHighExtractor",
    "polymath_ar_medium": f"{base_import}polymath:PolyMathARMediumExtractor",
    "polymath_ar_low": f"{base_import}polymath:PolyMathARLowExtractor",
    # Bengali
    "polymath_bn_top": f"{base_import}polymath:PolyMathBNTopExtractor",
    "polymath_bn_high": f"{base_import}polymath:PolyMathBNHighExtractor",
    "polymath_bn_medium": f"{base_import}polymath:PolyMathBNMediumExtractor",
    "polymath_bn_low": f"{base_import}polymath:PolyMathBNLowExtractor",
    # German
    "polymath_de_top": f"{base_import}polymath:PolyMathDETopExtractor",
    "polymath_de_high": f"{base_import}polymath:PolyMathDEHighExtractor",
    "polymath_de_medium": f"{base_import}polymath:PolyMathDEMediumExtractor",
    "polymath_de_low": f"{base_import}polymath:PolyMathDELowExtractor",
    # English
    "polymath_en_top": f"{base_import}polymath:PolyMathENTopExtractor",
    "polymath_en_high": f"{base_import}polymath:PolyMathENHighExtractor",
    "polymath_en_medium": f"{base_import}polymath:PolyMathENMediumExtractor",
    "polymath_en_low": f"{base_import}polymath:PolyMathENLowExtractor",
    # Spanish
    "polymath_es_top": f"{base_import}polymath:PolyMathESTopExtractor",
    "polymath_es_high": f"{base_import}polymath:PolyMathESHighExtractor",
    "polymath_es_medium": f"{base_import}polymath:PolyMathESMediumExtractor",
    "polymath_es_low": f"{base_import}polymath:PolyMathESLowExtractor",
    # French
    "polymath_fr_top": f"{base_import}polymath:PolyMathFRTopExtractor",
    "polymath_fr_high": f"{base_import}polymath:PolyMathFRHighExtractor",
    "polymath_fr_medium": f"{base_import}polymath:PolyMathFRMediumExtractor",
    "polymath_fr_low": f"{base_import}polymath:PolyMathFRLowExtractor",
    # Indonesian
    "polymath_id_top": f"{base_import}polymath:PolyMathIDTopExtractor",
    "polymath_id_high": f"{base_import}polymath:PolyMathIDHighExtractor",
    "polymath_id_medium": f"{base_import}polymath:PolyMathIDMediumExtractor",
    "polymath_id_low": f"{base_import}polymath:PolyMathIDLowExtractor",
    # Italian
    "polymath_it_top": f"{base_import}polymath:PolyMathITTopExtractor",
    "polymath_it_high": f"{base_import}polymath:PolyMathITHighExtractor",
    "polymath_it_medium": f"{base_import}polymath:PolyMathITMediumExtractor",
    "polymath_it_low": f"{base_import}polymath:PolyMathITLowExtractor",
    # Japanese
    "polymath_ja_top": f"{base_import}polymath:PolyMathJATopExtractor",
    "polymath_ja_high": f"{base_import}polymath:PolyMathJAHighExtractor",
    "polymath_ja_medium": f"{base_import}polymath:PolyMathJAMediumExtractor",
    "polymath_ja_low": f"{base_import}polymath:PolyMathJALowExtractor",
    # Korean
    "polymath_ko_top": f"{base_import}polymath:PolyMathKOTopExtractor",
    "polymath_ko_high": f"{base_import}polymath:PolyMathKOHighExtractor",
    "polymath_ko_medium": f"{base_import}polymath:PolyMathKOMediumExtractor",
    "polymath_ko_low": f"{base_import}polymath:PolyMathKOLowExtractor",
    # Malay
    "polymath_ms_top": f"{base_import}polymath:PolyMathMSTopExtractor",
    "polymath_ms_high": f"{base_import}polymath:PolyMathMSHighExtractor",
    "polymath_ms_medium": f"{base_import}polymath:PolyMathMSMediumExtractor",
    "polymath_ms_low": f"{base_import}polymath:PolyMathMSLowExtractor",
    # Portuguese
    "polymath_pt_top": f"{base_import}polymath:PolyMathPTTopExtractor",
    "polymath_pt_high": f"{base_import}polymath:PolyMathPTHighExtractor",
    "polymath_pt_medium": f"{base_import}polymath:PolyMathPTMediumExtractor",
    "polymath_pt_low": f"{base_import}polymath:PolyMathPTLowExtractor",
    # Russian
    "polymath_ru_top": f"{base_import}polymath:PolyMathRUTopExtractor",
    "polymath_ru_high": f"{base_import}polymath:PolyMathRUHighExtractor",
    "polymath_ru_medium": f"{base_import}polymath:PolyMathRUMediumExtractor",
    "polymath_ru_low": f"{base_import}polymath:PolyMathRULowExtractor",
    # Swahili
    "polymath_sw_top": f"{base_import}polymath:PolyMathSWTopExtractor",
    "polymath_sw_high": f"{base_import}polymath:PolyMathSWHighExtractor",
    "polymath_sw_medium": f"{base_import}polymath:PolyMathSWMediumExtractor",
    "polymath_sw_low": f"{base_import}polymath:PolyMathSWLowExtractor",
    # Telugu
    "polymath_te_top": f"{base_import}polymath:PolyMathTETopExtractor",
    "polymath_te_high": f"{base_import}polymath:PolyMathTEHighExtractor",
    "polymath_te_medium": f"{base_import}polymath:PolyMathTEMediumExtractor",
    "polymath_te_low": f"{base_import}polymath:PolyMathTELowExtractor",
    # Thai
    "polymath_th_top": f"{base_import}polymath:PolyMathTHTopExtractor",
    "polymath_th_high": f"{base_import}polymath:PolyMathTHHighExtractor",
    "polymath_th_medium": f"{base_import}polymath:PolyMathTHMediumExtractor",
    "polymath_th_low": f"{base_import}polymath:PolyMathTHLowExtractor",
    # Vietnamese
    "polymath_vi_top": f"{base_import}polymath:PolyMathVITopExtractor",
    "polymath_vi_high": f"{base_import}polymath:PolyMathVIHighExtractor",
    "polymath_vi_medium": f"{base_import}polymath:PolyMathVIMediumExtractor",
    "polymath_vi_low": f"{base_import}polymath:PolyMathVILowExtractor",
    # Chinese
    "polymath_zh_top": f"{base_import}polymath:PolyMathZHTopExtractor",
    "polymath_zh_high": f"{base_import}polymath:PolyMathZHHighExtractor",
    "polymath_zh_medium": f"{base_import}polymath:PolyMathZHMediumExtractor",
    "polymath_zh_low": f"{base_import}polymath:PolyMathZHLowExtractor",

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
    "multiple_py": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_js": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_java": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_cpp": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_rs": f"{base_import}multipl_e:MultiplEExtractor",
    "multiple_go": f"{base_import}multipl_e:MultiplEExtractor",
    "codexglue": f"{base_import}codexglue:CodexglueExtractor",
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
    "basqueglue": f"{base_import}basqueglue:BasqueglueExtractor",
    "bec2016eu": f"{base_import}bec2016eu:Bec2016euExtractor",
    "boolq_seq2seq": f"{base_import}boolq_seq2seq:BoolqSeq2seqExtractor",
    "doc_vqa": f"{base_import}doc_vqa:DocVQAExtractor",
    "ds1000": f"{base_import}ds1000:Ds1000Extractor",
    "evalita_mp": f"{base_import}evalita_mp:EvalitaMpExtractor",
    "flores": f"{base_import}flores:FloresExtractor",
    "humanevalpack": f"{base_import}humanevalpack:HumanevalpackExtractor",
    "iwslt2017_ar_en": f"{base_import}iwslt2017_ar_en:Iwslt2017ArEnExtractor",
    "iwslt2017_en_ar": f"{base_import}iwslt2017_en_ar:Iwslt2017EnArExtractor",
    "llama": f"{base_import}llama:LlamaExtractor",
    "multimedqa": f"{base_import}multimedqa:MultimedqaExtractor",
    "multiple": f"{base_import}multiple:MultipleExtractor",
    "openllm": f"{base_import}openllm:OpenllmExtractor",
    "pythia": f"{base_import}pythia:PythiaExtractor",
    "squad2": f"{base_import}squad2:SQuAD2Extractor",
    "stsb": f"{base_import}stsb:StsbExtractor",
    "super_glue_lm_eval_v1": f"{base_import}super_glue_lm_eval_v1:SuperGlueLmEvalV1Extractor",
    "super_glue_lm_eval_v1_seq2seq": f"{base_import}super_glue_lm_eval_v1_seq2seq:SuperGlueLmEvalV1Seq2seqExtractor",
    "super_glue_t5_prompt": f"{base_import}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "tmlu": f"{base_import}tmlu:TmluExtractor",
    "wiceu": f"{base_import}wiceu:WiceuExtractor",
    "wmt_ro_en_t5_prompt": f"{base_import}wmt_ro_en_t5_prompt:WmtRoEnT5PromptExtractor",

    # Newly created extractors
    # Note: unitxt tasks (xsum, cnn_dailymail, dbpedia_14, ethos_binary, etc.) use lm-eval unitxt extractor
    "bhtc_v2": f"{base_import}bhtc_v2:BhtcV2Extractor",
    "basque-glue": f"{base_import}basqueglue:BasqueglueExtractor",
    "evalita-sp_sum_task_fp-small_p1": f"{base_import}evalita_sp_sum_task_fp_small_p1:EvalitaSpSumTaskFpSmallP1Extractor",
    "flan_held_in": f"{base_import}flan_held_in:FlanHeldInExtractor",
    "gpt3_translation_benchmarks": f"{base_import}gpt3_translation_benchmarks:Gpt3TranslationBenchmarksExtractor",
    "multiple_choice": f"{base_import}multiple_choice:MultipleChoiceExtractor",
    "penn_treebank": f"{base_import}penn_treebank:PennTreebankExtractor",
    "ptb": f"{base_import}penn_treebank:PennTreebankExtractor",
    "self_consistency": f"{base_import}self_consistency:SelfConsistencyExtractor",
    "t0_eval": f"{base_import}t0_eval:T0EvalExtractor",
    "vaxx_stance": f"{base_import}vaxx_stance:VaxxStanceExtractor",
    "wikitext103": f"{base_import}wikitext103:Wikitext103Extractor",

    # TruthfulQA generation (semantic similarity evaluation, NOT lm-eval)
    "truthfulqa_generation": f"{base_import}truthfulqa_generation:TruthfulQAGenerationExtractor",
    "truthfulqa_gen": f"{base_import}truthfulqa_generation:TruthfulQAGenerationExtractor",

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

    # Software Engineering benchmarks (NOT lm-eval)
    "swe_bench_verified": f"{base_import}swe_bench_verified:SWEBenchVerifiedExtractor",
    "swe-bench-verified": f"{base_import}swe_bench_verified:SWEBenchVerifiedExtractor",
    "swebench_verified": f"{base_import}swe_bench_verified:SWEBenchVerifiedExtractor",

    # Scientific Computing benchmarks (NOT lm-eval)
    "scicode": f"{base_import}scicode:SciCodeExtractor",
    "sci_code": f"{base_import}scicode:SciCodeExtractor",

    # Instruction Following benchmarks (NOT lm-eval)
    "alpaca_eval": f"{base_import}alpaca_eval:AlpacaEvalExtractor",
    "alpacaeval": f"{base_import}alpaca_eval:AlpacaEvalExtractor",
    "alpaca_eval_2": f"{base_import}alpaca_eval:AlpacaEvalExtractor",

    # Arena/Chat benchmarks (NOT lm-eval)
    "arena_hard": f"{base_import}arena_hard:ArenaHardExtractor",
    "arena-hard": f"{base_import}arena_hard:ArenaHardExtractor",
    "arenahard": f"{base_import}arena_hard:ArenaHardExtractor",

    # Safety/Refusal benchmarks (NOT lm-eval)
    "sorry_bench": f"{base_import}sorry_bench:SorryBenchExtractor",
    "sorry-bench": f"{base_import}sorry_bench:SorryBenchExtractor",
    "sorrybench": f"{base_import}sorry_bench:SorryBenchExtractor",
    "jailbreakbench": f"{base_import}jailbreakbench:JailbreakBenchExtractor",
    "jailbreak_bench": f"{base_import}jailbreakbench:JailbreakBenchExtractor",
    "jbb": f"{base_import}jailbreakbench:JailbreakBenchExtractor",
    "harmbench": f"{base_import}harmbench:HarmBenchExtractor",
    "harm_bench": f"{base_import}harmbench:HarmBenchExtractor",
    "harm-bench": f"{base_import}harmbench:HarmBenchExtractor",

    # Over-refusal benchmarks (NOT lm-eval)
    "or_bench": f"{base_import}or_bench:ORBenchExtractor",
    "or-bench": f"{base_import}or_bench:ORBenchExtractor",
    "orbench": f"{base_import}or_bench:ORBenchExtractor",
    "or_bench_hard": f"{base_import}or_bench:ORBenchExtractor",
    "or_bench_80k": f"{base_import}or_bench:ORBenchExtractor",

    # Function calling benchmarks (NOT lm-eval)
    "bfcl": f"{base_import}bfcl:BFCLExtractor",
    "berkeley_function_calling": f"{base_import}bfcl:BFCLExtractor",
    "function_calling_leaderboard": f"{base_import}bfcl:BFCLExtractor",

    # Hallucination benchmarks (NOT lm-eval)
    "halueval": f"{base_import}halueval:HaluEvalExtractor",
    "halu_eval": f"{base_import}halueval:HaluEvalExtractor",
    "hallucination_eval": f"{base_import}halueval:HaluEvalExtractor",

    # Agent benchmarks (NOT lm-eval)
    "agentbench": f"{base_import}agentbench:AgentBenchExtractor",
    "agent_bench": f"{base_import}agentbench:AgentBenchExtractor",
    "agentinstruct": f"{base_import}agentbench:AgentBenchExtractor",

    # Math Olympiad benchmarks (NOT lm-eval)
    "olympiadbench": f"{base_import}olympiadbench:OlympiadBenchExtractor",
    "olympiad_bench": f"{base_import}olympiadbench:OlympiadBenchExtractor",
    "imo_answerbench": f"{base_import}olympiadbench:OlympiadBenchExtractor",
    "imo_math": f"{base_import}olympiadbench:OlympiadBenchExtractor",

    # Chinese benchmarks (NOT lm-eval)
    "cluewsc": f"{base_import}cluewsc:CLUEWSCExtractor",
    "cluewsc2020": f"{base_import}cluewsc:CLUEWSCExtractor",
    "clue_wsc": f"{base_import}cluewsc:CLUEWSCExtractor",
    "chinese_simpleqa": f"{base_import}chinese_simpleqa:ChineseSimpleQAExtractor",
    "c_simpleqa": f"{base_import}chinese_simpleqa:ChineseSimpleQAExtractor",
    "csimpleqa": f"{base_import}chinese_simpleqa:ChineseSimpleQAExtractor",

    # Competitive programming benchmarks (NOT lm-eval)
    "codeforces": f"{base_import}codeforces:CodeforcesExtractor",
    "code_forces": f"{base_import}codeforces:CodeforcesExtractor",
    "codeelo": f"{base_import}codeforces:CodeforcesExtractor",

    # Code editing benchmarks (NOT lm-eval)
    "aider_polyglot": f"{base_import}aider_polyglot:AiderPolyglotExtractor",
    "aider-polyglot": f"{base_import}aider_polyglot:AiderPolyglotExtractor",
    "polyglot": f"{base_import}aider_polyglot:AiderPolyglotExtractor",
    "code_exercises": f"{base_import}aider_polyglot:AiderPolyglotExtractor",

    # Agent safety benchmarks (NOT lm-eval)
    "agentharm": f"{base_import}agentharm:AgentHarmExtractor",
    "agent_harm": f"{base_import}agentharm:AgentHarmExtractor",
    "agent-harm": f"{base_import}agentharm:AgentHarmExtractor",

    # Safety refusal benchmarks (NOT lm-eval)
    "donotanswer": f"{base_import}donotanswer:DoNotAnswerExtractor",
    "do_not_answer": f"{base_import}donotanswer:DoNotAnswerExtractor",
    "do-not-answer": f"{base_import}donotanswer:DoNotAnswerExtractor",
    "wildguard": f"{base_import}wildguard:WildGuardExtractor",
    "wildguardmix": f"{base_import}wildguard:WildGuardExtractor",
    "wild_guard": f"{base_import}wildguard:WildGuardExtractor",

    # Hallucination benchmarks (NOT lm-eval)
    "faithbench": f"{base_import}faithbench:FaithBenchExtractor",
    "faith_bench": f"{base_import}faithbench:FaithBenchExtractor",
    "faith-bench": f"{base_import}faithbench:FaithBenchExtractor",

    # Sycophancy benchmarks (NOT lm-eval)
    "sycophancy_eval": f"{base_import}sycophancy_eval:SycophancyEvalExtractor",
    "sycophancy-eval": f"{base_import}sycophancy_eval:SycophancyEvalExtractor",
    "syceval": f"{base_import}sycophancy_eval:SycophancyEvalExtractor",
    "syc_eval": f"{base_import}sycophancy_eval:SycophancyEvalExtractor",

    # Agent task benchmarks (NOT lm-eval)
    "tau_bench": f"{base_import}tau_bench:TauBenchExtractor",
    "tau-bench": f"{base_import}tau_bench:TauBenchExtractor",
    "taubench": f"{base_import}tau_bench:TauBenchExtractor",
    "tau2_bench": f"{base_import}tau_bench:TauBenchExtractor",

    # Agent safety/emulation benchmarks (NOT lm-eval)
    "toolemu": f"{base_import}toolemu:ToolEmuExtractor",
    "tool_emu": f"{base_import}toolemu:ToolEmuExtractor",
    "tool-emu": f"{base_import}toolemu:ToolEmuExtractor",

    # Function calling/API benchmarks (NOT lm-eval)
    "toolbench": f"{base_import}toolbench:ToolBenchExtractor",
    "tool_bench": f"{base_import}toolbench:ToolBenchExtractor",
    "toolllm": f"{base_import}toolbench:ToolBenchExtractor",
    "tool_llm": f"{base_import}toolbench:ToolBenchExtractor",
    "nexus_function_calling": f"{base_import}toolbench:ToolBenchExtractor",

    # Planning benchmarks (NOT lm-eval)
    "travelplanner": f"{base_import}travelplanner:TravelPlannerExtractor",
    "travel_planner": f"{base_import}travelplanner:TravelPlannerExtractor",
    "travel-planner": f"{base_import}travelplanner:TravelPlannerExtractor",

    # Factuality/Grounding benchmarks (NOT lm-eval)
    "facts_grounding": f"{base_import}facts_grounding:FACTSGroundingExtractor",
    "facts-grounding": f"{base_import}facts_grounding:FACTSGroundingExtractor",
    "factsgrounding": f"{base_import}facts_grounding:FACTSGroundingExtractor",

    # Web browsing benchmarks (NOT lm-eval)
    "browsecomp": f"{base_import}browsecomp:BrowseCompExtractor",
    "browse_comp": f"{base_import}browsecomp:BrowseCompExtractor",
    "browse-comp": f"{base_import}browsecomp:BrowseCompExtractor",
    "browsecomp_zh": f"{base_import}browsecomp:BrowseCompExtractor",

    # Selective refusal benchmarks (NOT lm-eval)
    "refusalbench": f"{base_import}refusalbench:RefusalBenchExtractor",
    "refusal_bench": f"{base_import}refusalbench:RefusalBenchExtractor",
    "refusal-bench": f"{base_import}refusalbench:RefusalBenchExtractor",

    # Chinese value alignment benchmarks (NOT lm-eval)
    "flames": f"{base_import}flames:FlamesExtractor",
    "flames_chinese": f"{base_import}flames:FlamesExtractor",

    # Planning/Reasoning benchmarks (NOT lm-eval)
    "planbench": f"{base_import}planbench:PlanBenchExtractor",
    "plan_bench": f"{base_import}planbench:PlanBenchExtractor",
    "planningbench": f"{base_import}planbench:PlanBenchExtractor",

    # SWE-bench variants (NOT lm-eval)
    "swe_verified": f"{base_import}swe_bench:SWEBenchVerifiedExtractor",
    "swe_bench_verified": f"{base_import}swe_bench:SWEBenchVerifiedExtractor",
    "swebench_verified": f"{base_import}swe_bench:SWEBenchVerifiedExtractor",
    "multi_swe_bench": f"{base_import}swe_bench:MultiSWEBenchExtractor",
    "multiswebench": f"{base_import}swe_bench:MultiSWEBenchExtractor",
    "swe_bench_multilingual": f"{base_import}swe_bench:MultiSWEBenchExtractor",

    # LiveCodeBench V6 (NOT lm-eval)
    "livecodebench_v6": f"{base_import}livecodebench_v6:LiveCodeBenchV6Extractor",
    "livecodebench_v5": f"{base_import}livecodebench_v6:LiveCodeBenchV6Extractor",
    "livecodebench_lite": f"{base_import}livecodebench_v6:LiveCodeBenchV6Extractor",

    # Hallucinations Leaderboard (NOT lm-eval)
    "hallucinations_leaderboard": f"{base_import}hallucinations_leaderboard:HallucinationsLeaderboardExtractor",
    "hallucination_leaderboard": f"{base_import}hallucinations_leaderboard:HallucinationsLeaderboardExtractor",
    "hallu_leaderboard": f"{base_import}hallucinations_leaderboard:HallucinationsLeaderboardExtractor",

    # Longform Writing (NOT lm-eval)
    "longform_writing": f"{base_import}longform_writing:LongformWritingExtractor",
    "longform": f"{base_import}longform_writing:LongformWritingExtractor",
    "longform_generation": f"{base_import}longform_writing:LongformWritingExtractor",

    # Agentic Search benchmarks (NOT lm-eval)
    "seal_0": f"{base_import}agentic_search:SealExtractor",
    "seal": f"{base_import}agentic_search:SealExtractor",
    "finsearchcomp": f"{base_import}agentic_search:FinSearchCompExtractor",
    "fin_search_comp": f"{base_import}agentic_search:FinSearchCompExtractor",
    "financial_search": f"{base_import}agentic_search:FinSearchCompExtractor",

    # Coding benchmarks (NOT lm-eval)
    "oj_bench": f"{base_import}coding_benchmarks:OJBenchExtractor",
    "ojbench": f"{base_import}coding_benchmarks:OJBenchExtractor",
    "online_judge_bench": f"{base_import}coding_benchmarks:OJBenchExtractor",
    "terminal_bench": f"{base_import}coding_benchmarks:TerminalBenchExtractor",
    "terminalbench": f"{base_import}coding_benchmarks:TerminalBenchExtractor",
    "terminal": f"{base_import}coding_benchmarks:TerminalBenchExtractor",
    "scicode": f"{base_import}coding_benchmarks:SciCodeExtractor",
    "sci_code": f"{base_import}coding_benchmarks:SciCodeExtractor",

    # Medium priority benchmarks (NOT lm-eval)
    "cnmo_2024": f"{base_import}medium_priority_benchmarks:CNMOExtractor",
    "cnmo": f"{base_import}medium_priority_benchmarks:CNMOExtractor",
    "chinese_math_olympiad": f"{base_import}medium_priority_benchmarks:CNMOExtractor",
    "curate": f"{base_import}medium_priority_benchmarks:CurateExtractor",
    "personalized_alignment": f"{base_import}medium_priority_benchmarks:CurateExtractor",
    "halulens": f"{base_import}medium_priority_benchmarks:HalulensExtractor",
    "hallu_lens": f"{base_import}medium_priority_benchmarks:HalulensExtractor",
    "hallucination_lens": f"{base_import}medium_priority_benchmarks:HalulensExtractor",
    "politicalbias_qa": f"{base_import}medium_priority_benchmarks:PoliticalBiasExtractor",
    "political_bias": f"{base_import}medium_priority_benchmarks:PoliticalBiasExtractor",
    "politicalbias": f"{base_import}medium_priority_benchmarks:PoliticalBiasExtractor",
    "polyglottoxicityprompts": f"{base_import}medium_priority_benchmarks:PolygloToxicityExtractor",
    "polyglot_toxicity": f"{base_import}medium_priority_benchmarks:PolygloToxicityExtractor",
    "multilingual_toxicity": f"{base_import}medium_priority_benchmarks:PolygloToxicityExtractor",
}

# Alias for backwards compatibility
HF_EXTRACTORS = EXTRACTORS

