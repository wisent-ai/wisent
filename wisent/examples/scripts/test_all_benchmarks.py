"""Test all benchmarks to verify extractor and evaluator work."""

import sys
import signal
from contextlib import contextmanager
from wisent.examples.scripts.test_one_benchmark import test_benchmark


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


@contextmanager
def timeout(seconds):
    """Context manager for timing out operations."""
    def signal_handler(signum, frame):
        raise TimeoutError(f"Test timed out after {seconds} seconds")

    # Set the signal handler and alarm
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


BENCHMARKS = [
    # Log-likelihood tasks
    "boolq", "winogrande", "piqa", "copa", "cb",
    "hellaswag", "swag", "openbookqa", "race",
    "arc_easy", "arc_challenge", "mmlu", "gpqa",
    "truthfulqa_mc1", "truthfulqa_mc2",
    # Math tasks
    "gsm8k", "asdiv", "arithmetic",
    "math", "math500", "hendrycks_math",
    "aime", "aime2024", "aime2025",
    "hmmt", "hmmt_feb_2025",
    "polymath_en_medium", "polymath_zh_medium", "polymath_en_high", "polymath_zh_high",
    "livemathbench_cnmo_en", "livemathbench_cnmo_zh",
    # QA tasks
    "drop", "triviaqa", "record", "squadv2", "squad2",
    "webqs", "nq_open", "coqa",
    # Perplexity tasks
    "wikitext", "wikitext103", "ptb", "penn_treebank",
    "lambada_openai", "lambada_standard",
    # Coding tasks
    "livecodebench",
    "humaneval", "humaneval_plus", "instruct_humaneval",
    "mbpp", "mbpp_plus",
    "apps", "ds1000", "conala", "concode", "mercury", "recode",
    "codexglue_code_to_text_python", "codexglue_code_to_text_go", "codexglue_code_to_text_ruby",
    "codexglue_code_to_text_java", "codexglue_code_to_text_javascript", "codexglue_code_to_text_php",
    # Newly added extractors
    # Reasoning
    "bbh", "commonsense_qa", "siqa", "ifeval",
    # Arabic benchmarks
    "AraDICE", "ArabCulture", "arabicmmlu", "arabic_leaderboard_complete", "arabic_leaderboard_light",
    "egyhellaswag", "egymmlu", "darijahellaswag", "darijammlu", "darija_bench",
    # Asian language benchmarks
    "ceval", "cmmlu", "kmmlu", "turkishmmlu", "bangla_mmlu",
    "japanese_leaderboard", "kobest", "kormedmcqa", "haerae",
    # European language benchmarks
    "basque_bench", "basqueglue", "eus_exams", "eus_proficiency", "eus_reading", "eus_trivia",
    "catalan_bench", "french_bench", "galician_bench", "portuguese_bench", "spanish_bench",
    # Multilingual variants
    "global_mmlu", "belebele", "mlqa", "xquad", "xcopa",
    "okapi/arc_multilingual", "okapi/hellaswag_multilingual", "okapi/mmlu_multilingual", "okapi/truthfulqa_multilingual",
    # Safety/Ethics
    "hendrycks_ethics", "toxigen", "bbq", "crows_pairs", "moral_stories", "realtoxicityprompts",
    "simple_cooccurrence_bias", "winogender",
    # Medical
    "medmcqa", "med_concepts_qa", "meddialog", "mediqa_qa2019", "medtext", "meqsum", "mimic_repsum",
    # Math variants
    "minerva_math", "mgsm", "hrm8k", "agieval",
    # Long context
    "babi", "babilong", "ruler", "scrolls",
    # Other reasoning
    "inverse_scaling", "storycloze", "histoires_morales", "groundcocoa",
    # Language understanding
    "blimp", "multiblimp",
    "lambada_cloze", "lambada_multilingual", "lambada_multilingual_stablelm",
    # Paraphrase/Translation
    "paws-x", "translation", "wmt2016",
    # Code understanding
    "code_x_glue",
    # QA variants
    "aclue", "bertaqa", "careqa", "copal_id", "csatqa",
    # Specialized benchmarks
    "acp_bench", "acp_bench_hard", "aexams", "benchmarks",
    "c4", "chartqa", "eq_bench",
    "evalita_LLM", "fda", "fld", "jsonschema_bench", "kbl", "leaderboard",
    "libra", "lingoly", "mastermind", "metabench",
    "mmlusr", "mmmu", "model_written_evals", "mts_dialog", "multiblimp",
    "noreval", "olaph", "paloma", "pile", "pile_10k", "polemo2",
    "score", "squad_completion", "super_glue", "swde", "tinyBenchmarks",
    "tmmluplus", "truthfulqa-multi", "unitxt", "unscramble", "wmdp", "wsc273",
    # NEW: Missing 115 task families now added
    "global_mmlu_ar", "arabic_exams", "persona", "afrixnli_en_direct_amh",
    "evalita_mp", "truthfulqa", "eus_exams_es", "flores", "afrimgsm_direct_amh",
    "ceval_valid", "advanced_ai_risk", "tmlu", "arc_ar", "afrimmlu_direct_amh",
    "m_mmlu", "non_greedy_robustness_agieval_aqua_rat", "prompt_robustness_agieval_aqua_rat",
    "inverse_scaling_hindsight_neglect_10shot", "mela", "paws_ca",
    "ja_leaderboard_jaqket_v2", "super_glue-boolq-t5-prompt", "multiple_choice",
    "option_order_robustness_agieval_aqua_rat", "phrases_ca-va", "code2text_go",
    "ethics_cm", "cabreu", "sycophancy", "evalita_sp_sum_task_fp-small_p1",
    "glianorex", "flan_held_in", "assin_entailment", "gsm_plus", "mnli",
    "tinyTruthfulQA", "multimedqa", "openllm", "pythia", "t0_eval", "Tag",
    "basque-glue", "chain_of_thought", "freebase", "gpt3_translation_benchmarks",
    "iwslt2017", "llama", "self_consistency", "super-glue-lm-eval-v1",
    "super-glue-lm-eval-v1-seq2seq", "super-glue-t5-prompt", "wmt14", "wmt14_en_fr",
    "wmt14_fr_en", "wmt16_de_en", "wmt16_en_de", "wmt16_en_ro", "wmt16_ro_en",
    "20_newsgroups", "ag_news", "anagrams1", "anagrams2", "argument_topic",
    "atis", "banking77", "bec2016eu", "bhtc_v2", "boolq-seq2seq", "catalanqa",
    "catcola", "claim_stance_topic", "cnn_dailymail", "cocoteros_es", "coedit_gec",
    "cola", "coqcat", "cycle_letters", "dbpedia_14", "doc_vqa", "epec_koref_bin",
    "escola", "ethos_binary", "financial_tweets", "galcola", "iwslt2017-ar-en",
    "iwslt2017-en-ar", "law_stack_exchange", "ledgar", "logieval", "medical_abstracts",
    "noticia", "parafraseja", "parafrases_gl", "qnlieu", "random_insertion",
    "reversed_words", "sglue_rte", "stsb", "summarization_gl", "teca", "tinyArc",
    "tinyGSM8k", "tinyHellaswag", "tinyMMLU", "tinyWinogrande", "unfair_tos",
    "vaxx_stance", "wiceu", "wmt-ro-en-t5-prompt", "xlsum_es", "xsum",
    "yahoo_answers_topics", "instructhumaneval", "humanevalpack"
]


def test_all_benchmarks(model_name: str = "meta-llama/Llama-3.1-8B-Instruct", output_dir: str = "."):
    """Test all benchmarks.

    Args:
        model_name: Model to use for testing
        output_dir: Directory to save results

    Returns:
        Dictionary with results for each benchmark
    """
    results = {
        "model": model_name,
        "total": len(BENCHMARKS),
        "passed": 0,
        "failed": 0,
        "benchmarks": {}
    }

    print(f"\n{'='*70}")
    print(f"Testing {len(BENCHMARKS)} benchmarks with {model_name}")
    print(f"{'='*70}\n")

    for i, benchmark in enumerate(BENCHMARKS, 1):
        print(f"[{i}/{len(BENCHMARKS)}] Testing {benchmark}...")

        try:
            with timeout(1200):
                success = test_benchmark(benchmark, model_name, output_dir)
            results["benchmarks"][benchmark] = {
                "status": "passed" if success else "failed",
                "success": success
            }

            if success:
                results["passed"] += 1
                print(f"   PASSED\n")
            else:
                results["failed"] += 1
                print(f"   FAILED\n")

        except TimeoutError as e:
            results["benchmarks"][benchmark] = {
                "status": "timeout",
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"   TIMEOUT: {e}\n")

        except Exception as e:
            results["benchmarks"][benchmark] = {
                "status": "error",
                "success": False,
                "error": str(e)
            }
            results["failed"] += 1
            print(f"   ERROR: {e}\n")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['passed']/results['total']*100:.1f}%")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Llama-3.1-8B-Instruct"
    # Default to results directory in scripts folder
    from pathlib import Path
    default_output = Path(__file__).parent / "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else str(default_output)

    results = test_all_benchmarks(model, output_dir)

    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)
