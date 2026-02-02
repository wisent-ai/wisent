"""Human-readable benchmark descriptions for matching."""

from __future__ import annotations


__all__ = ["BENCHMARK_DESCRIPTIONS"]


BENCHMARK_DESCRIPTIONS = {
    # Knowledge & QA
    "mmlu": "General knowledge across academic subjects (science, history, literature, etc.)",
    "triviaqa": "Trivia and factual questions from various domains",
    "naturalqs": "Natural questions asking for factual information",
    "webqs": "Web-based questions requiring factual knowledge",
    "arc_easy": "Elementary science questions and reasoning",
    "arc_challenge": "Advanced science questions and reasoning",
    "sciq": "Scientific questions and explanations",
    "social_iqa": "Social situations and common sense reasoning",
    "openbookqa": "Elementary science with open-book style questions",
    "gpqa": "Graduate-level scientific reasoning in biology, physics, and chemistry",
    "gpqa_diamond": "High-quality graduate-level scientific questions (premium subset)",
    "gpqa_extended": "Extended graduate-level scientific reasoning dataset",
    "gpqa_main_zeroshot": "Graduate-level scientific reasoning (main subset, zero-shot)",
    "gpqa_diamond_zeroshot": "Premium graduate-level scientific questions (zero-shot)",
    "gpqa_extended_zeroshot": "Extended graduate-level scientific reasoning (zero-shot)",
    "gpqa_main_cot_zeroshot": "Graduate-level scientific reasoning with chain-of-thought",
    "gpqa_diamond_cot_zeroshot": "Premium graduate-level scientific questions with reasoning",
    "gpqa_extended_cot_zeroshot": "Extended graduate-level scientific reasoning with CoT",
    # SuperGPQA
    "supergpqa": "Large-scale dataset of scientific multiple-choice questions across disciplines",
    "supergpqa_physics": "Large-scale physics multiple-choice questions",
    "supergpqa_chemistry": "Large-scale chemistry multiple-choice questions",
    "supergpqa_biology": "Large-scale biology multiple-choice questions",
    # HLE
    "hle": "Human-Level Evaluation: Multimodal reasoning across multiple domains",
    "hle_exact_match": "Human-Level Evaluation: Exact string matching questions",
    "hle_multiple_choice": "Human-Level Evaluation: Multiple choice questions",
    # Reading Comprehension
    "coqa": "Conversational question answering with context",
    "drop": "Reading comprehension with numerical reasoning",
    "race": "Reading comprehension from English exams",
    "squad2": "Reading comprehension with impossible questions",
    "qasper": "Scientific paper question answering",
    "mutual": "Dialogue understanding and reasoning",
    # Reasoning & Logic
    "hellaswag": "Commonsense reasoning about everyday situations",
    "piqa": "Physical reasoning about objects and actions",
    "winogrande": "Pronoun resolution requiring commonsense",
    "logiqa": "Logical reasoning and inference",
    "wsc273": "Winograd schema challenge for pronoun resolution",
    "swag": "Commonsense reasoning about video situations",
    "boolq": "Yes/no questions requiring reasoning",
    # Mathematics
    "gsm8k": "Grade school math word problems",
    "math": "Mathematical reasoning problems requiring multi-step solutions",
    "math500": "500-problem subset of MATH benchmark for mathematical reasoning",
    "hendrycks_math": "Competition-level mathematics problems from Hendrycks et al.",
    "aime": "High-difficulty AIME contest problems (latest: 2025)",
    "aime2025": "High-difficulty AIME contest problems from 2025 (MathArena)",
    "aime2024": "High-difficulty AIME contest problems from 2024",
    "hmmt": "High-difficulty HMMT contest problems (latest: February 2025)",
    "hmmt_feb_2025": "High-difficulty HMMT February 2025 contest problems",
    "polymath": "PolyMath multilingual mathematical reasoning (default: English medium)",
    "polymath_en_medium": "PolyMath medium-difficulty mathematical problems in English",
    "polymath_zh_medium": "PolyMath medium-difficulty mathematical problems in Chinese",
    "polymath_en_high": "PolyMath high-difficulty mathematical problems in English",
    "polymath_zh_high": "PolyMath high-difficulty mathematical problems in Chinese",
    "livemathbench": "LiveMathBench CNMO 2024 mathematical olympiad problems",
    "livemathbench_cnmo_en": "LiveMathBench CNMO 2024 problems in English",
    "livemathbench_cnmo_zh": "LiveMathBench CNMO 2024 problems in Chinese",
    "math_qa": "Mathematical reasoning and problem solving",
    "arithmetic": "Basic arithmetic operations and calculations",
    "asdiv": "Arithmetic story problems for children",
    "mgsm": "Multilingual grade school math problems",
    # Coding
    "humaneval": "Python code generation and programming",
    "mbpp": "Python programming problems and solutions",
    # Language & Linguistics
    "blimp": "Grammatical acceptability and linguistic knowledge",
    "lambada": "Language modeling and word prediction",
    "lambada_cloze": "Cloze test for language understanding",
    "lambada_multilingual": "Multilingual language modeling",
    "wikitext": "Language modeling on Wikipedia text",
    "unscramble": "Word unscrambling and letter manipulation",
    # Multilingual
    "xnli": "Cross-lingual natural language inference",
    "xcopa": "Cross-lingual commonsense reasoning",
    "xstorycloze": "Cross-lingual story completion",
    "xwinograd": "Cross-lingual pronoun resolution",
    "paws_x": "Cross-lingual paraphrase detection",
    "belebele": "Multilingual reading comprehension",
    # Bias & Safety
    "toxigen": "Toxicity detection and harmful content",
    "crows_pairs": "Bias measurement in language models",
    "hendrycks_ethics": "Ethical reasoning and moral judgments",
    "truthfulqa_mc1": "Truthfulness and factual accuracy",
    "truthfulqa_mc2": "Truthfulness with multiple correct answers",
    "truthfulqa_gen": "Truthful text generation",
    # Medical
    "medqa": "Medical knowledge and clinical reasoning",
    "pubmedqa": "Biomedical literature question answering",
    "headqa": "Medical and healthcare knowledge",
    # Temporal & Event
    "prost": "Temporal reasoning in procedural text",
    # Adversarial
    "anli": "Adversarial natural language inference",
    # Benchmark Suites
    "glue": "General language understanding tasks",
    "superglue": "Advanced language understanding tasks",
    "big_bench": "Diverse challenging tasks for large models",
}
