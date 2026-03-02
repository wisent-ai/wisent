"""HuggingFace extractor manifest - Part 2 (SE, multilingual, medical, translation)."""

base_import: str = "wisent.extractors.hf.hf_task_extractors."

EXTRACTORS_PART2: dict[str, str] = {
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
    "browsecomp": f"{base_import}agentic_search:BrowseCompExtractor",
    "browse_comp": f"{base_import}agentic_search:BrowseCompExtractor",
    "browse-comp": f"{base_import}agentic_search:BrowseCompExtractor",
    "browsecomp_zh": f"{base_import}agentic_search:BrowseCompExtractor",

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
    # scicode and sci_code are already defined earlier - use scicode.py extractor

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
    # Custom TruthfulQA (loads directly from HF)
    "truthfulqa_custom": f"{base_import}truthfulqa_custom:TruthfulQACustomExtractor",

    # Okapi Multilingual benchmarks
    "okapi_mmlu_multilingual": f"{base_import}okapi_multilingual:OkapiMMLUExtractor",
    "okapi_mmlu": f"{base_import}okapi_multilingual:OkapiMMLUExtractor",
    "okapi_hellaswag_multilingual": f"{base_import}okapi_multilingual:OkapiHellaswagExtractor",
    "okapi_hellaswag": f"{base_import}okapi_multilingual:OkapiHellaswagExtractor",
    "okapi_truthfulqa_multilingual": f"{base_import}okapi_multilingual:OkapiTruthfulQAExtractor",
    "okapi_truthfulqa": f"{base_import}okapi_multilingual:OkapiTruthfulQAExtractor",

    # Multilingual benchmarks
    "paws_x": f"{base_import}multilingual_benchmarks:PawsXExtractor",
    "paws-x": f"{base_import}multilingual_benchmarks:PawsXExtractor",
    "mlqa": f"{base_import}multilingual_benchmarks:MLQAExtractor",
    "darija_bench": f"{base_import}multilingual_benchmarks:DarijaBenchExtractor",
    "eus_exams": f"{base_import}multilingual_benchmarks:EusExamsExtractor",
    "lambada_multilingual_stablelm": f"{base_import}multilingual_benchmarks:LambadaMultilingualExtractor",
    "lambada_multilingual": f"{base_import}multilingual_benchmarks:LambadaMultilingualExtractor",

    # Reasoning benchmarks
    "inverse_scaling": f"{base_import}reasoning_benchmarks:InverseScalingExtractor",

    # Medical benchmarks
    "med_concepts_qa": f"{base_import}medical_benchmarks:MedConceptsQAExtractor",

    # Translation benchmarks
    "translation": f"{base_import}translation_benchmarks:TranslationExtractor",
    "wmt14_en_fr": f"{base_import}translation_benchmarks:WMT14Extractor",
    "wmt14_fr_en": f"{base_import}translation_benchmarks:WMT14Extractor",
    "wmt16_de_en": f"{base_import}translation_benchmarks:WMT16Extractor",
    "wmt16_en_de": f"{base_import}translation_benchmarks:WMT16Extractor",
    "wmt2016": f"{base_import}translation_benchmarks:WMT16Extractor",

    # Multimodal benchmarks
    "mmmu": f"{base_import}multimodal_benchmarks:MMMUExtractor",
}
