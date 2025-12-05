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
- G-Pass@k(τ): Probability of at least τ*k correct in k samples
- mG-Pass@k: Mean G-Pass@k integrated over τ ∈ [0.5, 1.0]

Default parameters:
- n (num_samples): 48 - total samples generated per problem for G-Pass@k computation
- k_values: [4, 8, 16] - number of samples to select for metric computation
- tau_values: [0.25, 0.5, 0.75, 1.0] - threshold fractions for G-Pass@k
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

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.benchmark_specific.livemathbench_evaluator import (
    LiveMathBenchEvaluator,
    compute_all_metrics,
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
GREEDY_CONFIG = {
    "max_new_tokens": 8192,
    "temperature": 0.0,
    "do_sample": False,
}

# For sampling (temperature=1.0)
# Paper: temperature=1.0, top_p=0.8, top_k=50, repetition_penalty=1.0
SAMPLING_CONFIG = {
    "max_new_tokens": 8192,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "do_sample": True,
}

# Reasoning model config (longer context)
REASONING_CONFIG = {
    "max_new_tokens": 32768,
    "temperature": 1.0,
    "top_p": 0.8,
    "top_k": 50,
    "repetition_penalty": 1.0,
    "do_sample": True,
}


def get_language(config: str) -> str:
    """Get language code from config name."""
    if config.endswith("_cn"):
        return "cn"
    return "en"


def evaluate_greedy(
    model: WisentModel,
    evaluator: LiveMathBenchEvaluator,
    dataset_config: str,
    limit: Optional[int] = None,
    eval_mode: str = "math",
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


def evaluate_sampling(
    model: WisentModel,
    evaluator: LiveMathBenchEvaluator,
    dataset_config: str,
    num_samples: int = 48,
    limit: Optional[int] = None,
    is_reasoning_model: bool = False,
    eval_mode: str = "math",
    judge_model: Optional[WisentModel] = None,
) -> tuple[list[int], list[dict]]:
    """Evaluate model with sampling (temperature=1.0).

    Generates num_samples responses per problem for G-Pass@k computation.

    Args:
        model: The model to evaluate
        evaluator: The evaluator instance
        dataset_config: Dataset config name
        num_samples: Number of samples per problem (default 48 = 16 * 3)
        limit: Optional limit on number of problems
        is_reasoning_model: If True, use longer context config
        eval_mode: "math" for answer extraction or "llm_judge" for LLM-as-a-judge
        judge_model: Model to use for LLM judge (required if eval_mode="llm_judge")

    Returns:
        Tuple of (correct_counts_per_problem, results_list)
    """
    ds = load_dataset("opencompass/LiveMathBench", dataset_config, split="test")

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    language = get_language(dataset_config)
    config = REASONING_CONFIG if is_reasoning_model else SAMPLING_CONFIG

    correct_counts = []
    results = []

    for example in tqdm(ds, desc=f"Sampling eval ({dataset_config})"):
        question = example.get("question", "")
        answer = example.get("answer", "")

        prompt = LiveMathBenchEvaluator.get_prompt(question, language)

        # Generate multiple samples
        sample_responses = []
        sample_correct = 0

        for _ in range(num_samples):
            responses = model.generate(
                inputs=prompt,
                **config,
                prompt_is_formatted=True,
            )
            response = responses[0] if responses else ""
            sample_responses.append(response)

            eval_result = evaluator.evaluate(
                response,
                answer,
                mode=eval_mode,
                judge_model=judge_model,
                question=question,
                language=language,
            )
            if eval_result.ground_truth == "TRUTHFUL":
                sample_correct += 1

        correct_counts.append(sample_correct)

        results.append({
            "question": question,
            "true_answer": answer,
            "num_samples": num_samples,
            "correct_count": sample_correct,
            "sample_responses": sample_responses,  
        })

    return correct_counts, results


def main(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    dataset_config: str = "amc_en",
    limit: Optional[int] = None,
    num_samples: int = 48,
    k_values: list[int] = [4, 8, 16],
    tau_values: list[float] = [0.25, 0.5, 0.75, 1.0],
    skip_sampling: bool = False,
    is_reasoning_model: bool = False,
    eval_mode: str = "math",
    judge_model_name: Optional[str] = None,  # Use "Qwen/Qwen2.5-7B-Instruct" for llm_judge mode (paper uses 72B)
):
    """Run full LiveMathBench evaluation with G-Pass@k metrics.

    Args:
        model_name: HuggingFace model name for the model to evaluate
        dataset_config: Dataset config key (e.g., "cnmo_en", "amc_zh")
        limit: Optional limit on number of problems
        num_samples: Number of samples per problem for G-Pass@k
        k_values: k values for G-Pass@k metrics
        tau_values: Threshold values for G-Pass@k
        skip_sampling: If True, only compute greedy accuracy
        is_reasoning_model: If True, use longer context config
        eval_mode: "math" for answer extraction or "llm_judge" for LLM-as-a-judge (default)
        judge_model_name: Model name for LLM judge (defaults to model_name if not specified)
    """
    print(f"Loading model: {model_name}")
    model = WisentModel(model_name=model_name)
    evaluator = LiveMathBenchEvaluator()

    # Load judge model if using LLM judge mode
    judge_model = None
    if eval_mode == "llm_judge":
        judge_name = judge_model_name or model_name
        print(f"Loading judge model: {judge_name}")
        if judge_name == model_name:
            judge_model = model  # Reuse the same model
        else:
            judge_model = WisentModel(model_name=judge_name)

    hf_config = DATASET_CONFIGS.get(dataset_config, dataset_config)
    print(f"Dataset config: {hf_config}")
    print(f"Evaluation mode: {eval_mode}")
    print("=" * 60)

    # Greedy evaluation
    print("\n--- GREEDY EVALUATION ---")
    greedy_accuracy, greedy_results = evaluate_greedy(
        model, evaluator, hf_config, limit,
        eval_mode=eval_mode,
        judge_model=judge_model,
    )
    print(f"Greedy Accuracy: {greedy_accuracy * 100:.2f}%")

    # Sampling evaluation for G-Pass@k
    metrics = {"greedy_accuracy": greedy_accuracy * 100}
    sampling_results = []

    if not skip_sampling:
        print(f"\n--- SAMPLING EVALUATION (n={num_samples}) ---")
        correct_counts, sampling_results = evaluate_sampling(
            model, evaluator, hf_config, num_samples, limit, is_reasoning_model,
            eval_mode=eval_mode,
            judge_model=judge_model,
        )

        # Compute G-Pass@k metrics
        metrics.update(
            compute_all_metrics(
                correct_counts,
                total_samples=num_samples,
                k_values=k_values,
                tau_values=tau_values,
            )
        )

        # Convert to percentages
        for key in metrics:
            if key != "greedy_accuracy":
                metrics[key] = metrics[key] * 100

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: opencompass/LiveMathBench ({hf_config})")
    print(f"Greedy Accuracy: {metrics['greedy_accuracy']:.2f}%")

    if not skip_sampling:
        for k in k_values:
            print(f"\n--- k={k} ---")
            for tau in tau_values:
                key = f"G-Pass@{k}_{tau}"
                if key in metrics:
                    print(f"  G-Pass@{k}(τ={tau}): {metrics[key]:.2f}%")
            mg_key = f"mG-Pass@{k}"
            if mg_key in metrics:
                print(f"  mG-Pass@{k}: {metrics[mg_key]:.2f}%")

    # Save results
    output_dir = Path(__file__).parent / "results_test_evaluator"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"livemathbench_results_{dataset_config}_{eval_mode}.json"

    output_data = {
        "model_name": model_name,
        "dataset": "opencompass/LiveMathBench",
        "dataset_config": hf_config,
        "eval_mode": eval_mode,
        "judge_model_name": judge_model_name or model_name if eval_mode == "llm_judge" else None,
        "num_samples": num_samples,
        "k_values": k_values,
        "tau_values": tau_values,
        "metrics": metrics,
        "greedy_results": greedy_results,
        "sampling_results": sampling_results,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return metrics


if __name__ == "__main__":
    # Example: Run on CNMO English with limited samples for testing
    main(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_config="amc_en",
        limit=3,  # Limit to 3 problems for testing
        num_samples=16,  # Reduced samples for testing
        k_values=[2, 4, 8],
        tau_values=[0.25, 0.5, 0.75, 1.0],
        skip_sampling=False,
        eval_mode="llm_judge",
        judge_model_name="Qwen/Qwen2.5-1.5B-Instruct",  # Same model as judge for testing
        # eval_mode="math",  # Or use math extraction for faster evaluation
    )
