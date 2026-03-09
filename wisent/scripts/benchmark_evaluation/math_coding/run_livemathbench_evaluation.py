"""
Run LiveMathBenchEvaluator on LiveMathBench benchmark.

This script:
1. Loads opencompass/LiveMathBench dataset
2. Prompts an LLM to solve problems (greedy + sampling modes)
3. Uses LiveMathBenchEvaluator to compare model answer with ground truth
4. Computes G-Pass@k metrics (greedy accuracy, Pass@k, G-Pass@k, mG-Pass@k)
5. Saves results to JSON file

Metrics computed following the LiveMathBench paper (arxiv.org/abs/2412.13147):
- Greedy accuracy: Single-shot accuracy with temperature=0
- Pass@k: Probability of at least 1 correct in k samples
- G-Pass@k(tau): Probability of at least tau*k correct in k samples
- mG-Pass@k: Mean G-Pass@k integrated over tau in [0.5, 1.0]

Default parameters:
- n (num_samples): 48 - total samples generated per problem for G-Pass@k computation
- k_values: [4, 8, 16] - number of samples to select for metric computation
- tau_values: LIVEMATHBENCH_TAU_VALUES - threshold fractions for G-Pass@k
  - tau=0.25, k=16 means at least 4 of 16 samples must be correct
  - tau=0.5, k=16 means at least 8 of 16 samples must be correct
  - tau=1.0, k=16 means all 16 samples must be correct
- judge_model: Qwen/Qwen2.5-72B-Instruct (~144GB FP16, ~72GB INT8, ~36GB INT4)

Total answers generated per problem: 1 (greedy) + 48 (sampling) = 49
"""

import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.primitives.models.config import get_generate_kwargs
from wisent.core.utils.config_tools.constants import (
    LIVEMATHBENCH_K_VALUES,
    LIVEMATHBENCH_TAU_VALUES,
)
from wisent.core.reading.evaluators.benchmark_specific.livemathbench_evaluator import (
    LiveMathBenchEvaluator,
    compute_all_metrics,
)

# Re-export from helpers
from wisent.scripts.benchmark_evaluation._helpers.livemathbench_run_helpers import (
    evaluate_sampling,
    main,
)


# Dataset configs
DATASET_CONFIGS = {
    # v202412 - December 2024 release
    "cnmo_en": "v202412_CNMO_en",
    "cnmo_cn": "v202412_CNMO_cn",
    "ccee_en": "v202412_CCEE_en",
    "ccee_cn": "v202412_CCEE_cn",
    "amc_en": "v202412_AMC_en",
    "amc_cn": "v202412_AMC_cn",
    "wlpmc_en": "v202412_WLPMC_en",
    "wlpmc_cn": "v202412_WLPMC_cn",
    "hard_en": "v202412_hard_en",
    "hard_cn": "v202412_hard_cn",
    # v202505 - May 2025 release
    "v202505_all_en": "v202505_all_en",
    "v202505_hard_en": "v202505_hard_en",
}

# Generation configs following LiveMathBench paper
# For greedy decoding (temperature=0)
GREEDY_CONFIG = get_generate_kwargs(temperature=0.0, do_sample=False)

# For sampling (temperature=1.0)
# Paper: temperature=1.0, top_p=0.8, top_k=50, repetition_penalty=1.0
SAMPLING_CONFIG = get_generate_kwargs()

# Reasoning model config (longer context)
REASONING_CONFIG = get_generate_kwargs()


def get_language(config: str) -> str:
    """Get language code from config name."""
    if config.endswith("_cn"):
        return "cn"
    return "en"


def evaluate_greedy(
    model: WisentModel,
    evaluator: LiveMathBenchEvaluator,
    dataset_config: str,
    eval_mode: str,
    limit: Optional[int] = None,
    judge_model: Optional[WisentModel] = None,
) -> tuple[float, list[dict]]:
    """Evaluate model with greedy decoding (temperature=0).

    Args:
        model: The model to evaluate
        evaluator: The evaluator instance
        dataset_config: Dataset config name
        limit: Optional limit on number of problems
        eval_mode: "math" for answer extraction or "llm_judge" for LLM-as-a-judge
        judge_model: Model to use for LLM judge (required if eval_mode="llm_judge")

    Returns:
        Tuple of (accuracy, results_list)
    """
    ds = load_dataset("opencompass/LiveMathBench", dataset_config, split="test")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    language = get_language(dataset_config)
    correct = 0
    results = []

    for example in tqdm(ds, desc=f"Greedy eval ({dataset_config})"):
        question = example.get("question", "")
        answer = example.get("answer", "")

        prompt = LiveMathBenchEvaluator.get_prompt(question, language)

        responses = model.generate(
            inputs=prompt,
            **GREEDY_CONFIG,
            prompt_is_formatted=True,
        )

        response = responses[0] if responses else ""

        # Evaluate using selected mode
        eval_result = evaluator.evaluate(
            response,
            answer,
            mode=eval_mode,
            judge_model=judge_model,
            question=question,
            language=language,
        )

        if eval_result.ground_truth == "TRUTHFUL":
            correct += 1

        results.append({
            "question": question,
            "true_answer": answer,
            "model_output": response,
            "ground_truth": eval_result.ground_truth,
            "confidence": eval_result.confidence,
            "details": eval_result.details,
        })

    accuracy = correct / len(ds) if len(ds) > 0 else 0.0
    return accuracy, results


if __name__ == "__main__":
    # Example: Run on CNMO English with limited samples for testing
    main(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_config="amc_en",
        limit=None,
        num_samples=16,  # Reduced samples for testing
        k_values=list(LIVEMATHBENCH_K_VALUES),
        tau_values=list(LIVEMATHBENCH_TAU_VALUES),
        skip_sampling=False,
        eval_mode="llm_judge",
        judge_model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Same model as judge for testing
        # eval_mode="math",  # Or use math extraction for faster evaluation
    )
