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
    "asdiv_cot_llama": f"{base_import}math:MATHExtractor",
    "chain_of_thought": f"{base_import}math:MATHExtractor",
    "gsm8k_cot": f"{base_import}math:MATHExtractor",
    "gsm8k_cot_llama": f"{base_import}math:MATHExtractor",
    "gsm8k_cot_self_consistency": f"{base_import}math:MATHExtractor",
    "gsm8k_llama": f"{base_import}math:MATHExtractor",
    "gsm8k_platinum_cot": f"{base_import}math:MATHExtractor",
    "gsm8k_platinum_cot_llama": f"{base_import}math:MATHExtractor",
    "gsm8k_platinum_cot_self_consistency": f"{base_import}math:MATHExtractor",
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
    "humaneval_64_instruct": f"{base_import}instructhumaneval:InstructHumanEvalExtractor",
    "humaneval_instruct": f"{base_import}instructhumaneval:InstructHumanEvalExtractor",
    "humanevalpack": f"{base_import}humaneval:HumanEvalExtractor",
    "instructhumaneval": f"{base_import}instructhumaneval:InstructHumanEvalExtractor",
    "mbpp": f"{base_import}mbpp:MBPPExtractor",
    "mbpp_plus": f"{base_import}mbpp:MBPPExtractor",
    "instruct_humaneval": f"{base_import}instructhumaneval:InstructHumanEvalExtractor",
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
    "freebase": f"{base_import}freebase:FreebaseExtractor",
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
    "non_greedy_robustness_agieval_aqua_rat": f"{base_import}non_greedy_robustness_agieval_aqua_rat:NonGreedyRobustnessAgievalAquaRatExtractor",
    "option_order_robustness_agieval_aqua_rat": f"{base_import}option_order_robustness_agieval_aqua_rat:OptionOrderRobustnessAgievalAquaRatExtractor",
    "penn_treebank": f"{base_import}penn_treebank:PennTreebankExtractor",
    "ptb": f"{base_import}penn_treebank:PennTreebankExtractor",
    "prompt_robustness_agieval_aqua_rat": f"{base_import}prompt_robustness_agieval_aqua_rat:PromptRobustnessAgievalAquaRatExtractor",
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
}

# Alias for backwards compatibility
HF_EXTRACTORS = EXTRACTORS

