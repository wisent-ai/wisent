"""Helpers for run_livemathbench_evaluation.py.

Contains evaluate_sampling and main functions.
Extracted to keep run_livemathbench_evaluation.py under 300 lines.
"""

import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.utils.config_tools.constants import (
    LIVEMATHBENCH_K_VALUES,
    LIVEMATHBENCH_NUM_SAMPLES,
    LIVEMATHBENCH_TAU_VALUES,
    SEPARATOR_WIDTH_STANDARD, JSON_INDENT,
)
from wisent.core.reading.evaluators.benchmark_specific.livemathbench_evaluator import (
    LiveMathBenchEvaluator,
    compute_all_metrics,
)


def evaluate_sampling(
    model: WisentModel,
    evaluator: LiveMathBenchEvaluator,
    dataset_config: str,
    eval_mode: str,
    num_samples: int = LIVEMATHBENCH_NUM_SAMPLES,
    limit: Optional[int] = None,
    is_reasoning_model: bool = False,
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
    from datasets import load_dataset
    from wisent.scripts.benchmark_evaluation.run_livemathbench_evaluation import (
        REASONING_CONFIG, SAMPLING_CONFIG, get_language,
    )

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
    dataset_config: str,
    math_timeout: int,
    eval_mode: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    limit: Optional[int] = None,
    num_samples: int = LIVEMATHBENCH_NUM_SAMPLES,
    k_values: list[int] = list(LIVEMATHBENCH_K_VALUES),
    tau_values: list[float] = list(LIVEMATHBENCH_TAU_VALUES),
    skip_sampling: bool = False,
    is_reasoning_model: bool = False,
    judge_model_name: Optional[str] = None,
):
    """Run full LiveMathBench evaluation with G-Pass@k metrics."""
    from wisent.scripts.benchmark_evaluation.run_livemathbench_evaluation import (
        DATASET_CONFIGS, evaluate_greedy,
    )

    print(f"Loading model: {model_name}")
    model = WisentModel(model_name=model_name)
    evaluator = LiveMathBenchEvaluator(math_timeout=math_timeout)

    # Load judge model if using LLM judge mode
    judge_model = None
    if eval_mode == "llm_judge":
        judge_name = judge_model_name or model_name
        print(f"Loading judge model: {judge_name}")
        if judge_name == model_name:
            judge_model = model
        else:
            judge_model = WisentModel(model_name=judge_name)

    hf_config = DATASET_CONFIGS.get(dataset_config, dataset_config)
    print(f"Dataset config: {hf_config}")
    print(f"Evaluation mode: {eval_mode}")
    print("=" * SEPARATOR_WIDTH_STANDARD)

    # Greedy evaluation
    print("\n--- GREEDY EVALUATION ---")
    greedy_accuracy, greedy_results = evaluate_greedy(
        model, evaluator, hf_config,
        eval_mode=eval_mode,
        limit=limit,
        judge_model=judge_model,
    )
    print(f"Greedy Accuracy: {greedy_accuracy * 100:.2f}%")

    # Sampling evaluation for G-Pass@k
    metrics = {"greedy_accuracy": greedy_accuracy * 100}
    sampling_results = []

    if not skip_sampling:
        print(f"\n--- SAMPLING EVALUATION (n={num_samples}) ---")
        correct_counts, sampling_results = evaluate_sampling(
            model, evaluator, hf_config,
            eval_mode=eval_mode,
            num_samples=num_samples,
            limit=limit,
            is_reasoning_model=is_reasoning_model,
            judge_model=judge_model,
        )

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
    print("\n" + "=" * SEPARATOR_WIDTH_STANDARD)
    print("SUMMARY")
    print("=" * SEPARATOR_WIDTH_STANDARD)
    print(f"Model: {model_name}")
    print(f"Dataset: opencompass/LiveMathBench ({hf_config})")
    print(f"Greedy Accuracy: {metrics['greedy_accuracy']:.2f}%")

    if not skip_sampling:
        for k in k_values:
            print(f"\n--- k={k} ---")
            for tau in tau_values:
                key = f"G-Pass@{k}_{tau}"
                if key in metrics:
                    print(f"  G-Pass@{k}(tau={tau}): {metrics[key]:.2f}%")
            mg_key = f"mG-Pass@{k}"
            if mg_key in metrics:
                print(f"  mG-Pass@{k}: {metrics[mg_key]:.2f}%")

    # Save results
    output_dir = Path(__file__).parent.parent / "results_test_evaluator"
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
        json.dump(output_data, f, indent=JSON_INDENT, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")
    return metrics
