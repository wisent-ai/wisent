"""
Evaluate a trained LoRA adapter on benchmark test set.

Uses the same test split as other steering methods for fair comparison.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict

from wisent.core.models.wisent_model import WisentModel
from wisent.core.utils.dataset_splits import get_test_docs
from wisent.comparison.lora import apply_lora_to_model, remove_lora
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor


def create_test_only_task(task_name: str, train_ratio: float = 0.8) -> dict:
    """
    Create a task that evaluates only on our test split.
    """
    task_dict = get_task_dict([task_name])
    task = task_dict[task_name]

    test_docs = get_test_docs(task, benchmark_name=task_name, train_ratio=train_ratio)
    test_pct = round((1 - train_ratio) * 100)

    print(f"Test split size: {len(test_docs)} docs ({test_pct}% of pooled data)")

    task.test_docs = lambda: test_docs
    task.has_test_docs = lambda: True
    task._eval_docs = test_docs

    return {task_name: task}


def run_lm_eval_evaluation(
    wisent_model: WisentModel,
    task_dict: dict,
    task_name: str,
    batch_size: int = 1,
    max_batch_size: int = 8,
    limit: int | None = None,
) -> dict:
    """Run evaluation using lm-eval-harness."""
    lm = HFLM(
        pretrained=wisent_model.hf_model,
        tokenizer=wisent_model.tokenizer,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
    )

    return results


def run_ll_evaluation(
    wisent_model: WisentModel,
    task_dict: dict,
    task_name: str,
    limit: int | None = None,
) -> dict:
    """Run evaluation using log-likelihood evaluator."""
    from wisent.core.evaluators.benchmark_specific.log_likelihoods_evaluator import LogLikelihoodsEvaluator

    ll_evaluator = LogLikelihoodsEvaluator()
    extractor = get_extractor(task_name)

    task = task_dict[task_name]
    test_docs = list(task.test_docs())

    if limit:
        test_docs = test_docs[:limit]

    print(f"Evaluating {len(test_docs)} examples with LogLikelihoodsEvaluator")

    results = []
    correct = 0

    for i, doc in enumerate(test_docs):
        question = task.doc_to_text(doc)
        choices, expected = extractor.extract_choices_and_answer(task, doc)

        result = ll_evaluator.evaluate(
            response="",
            expected=expected,
            model=wisent_model,
            question=question,
            choices=choices,
            task_name=task_name,
        )

        is_correct = result.ground_truth == "TRUTHFUL"
        results.append({
            "question": question[:100] + "..." if len(question) > 100 else question,
            "predicted": result.meta["predicted"],
            "expected": expected,
            "correct": is_correct,
        })

        if is_correct:
            correct += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(test_docs)}, acc: {correct/(i+1):.4f}")

    accuracy = correct / len(test_docs) if test_docs else 0.0

    return {
        "accuracy": accuracy,
        "num_correct": correct,
        "num_total": len(test_docs),
    }


def extract_accuracy(results: dict, task: str) -> float:
    """Extract accuracy from lm-eval results."""
    task_results = results.get("results", {}).get(task, {})
    for key in ["acc", "acc,none", "accuracy", "acc_norm", "acc_norm,none"]:
        if key in task_results:
            return task_results[key]
    return 0.0


def run_evaluation(
    model_name: str,
    lora_path: str | Path,
    task: str,
    train_ratio: float = 0.8,
    device: str = "cuda:0",
    batch_size: int = 1,
    max_batch_size: int = 8,
    limit: int | None = None,
    output_dir: str | Path = None,
) -> dict:
    """
    Run full evaluation comparing base model vs LoRA-adapted model.

    Args:
        model_name: HuggingFace model name
        lora_path: Path to trained LoRA adapter
        task: lm-eval task name
        train_ratio: Train/test split ratio (to get same test set)
        device: Device to run on
        batch_size: Batch size for evaluation
        max_batch_size: Max batch size
        limit: Limit number of eval examples
        output_dir: Where to save results

    Returns:
        Dict with evaluation results
    """
    lora_path = Path(lora_path)

    # Step 1: Create test task
    print(f"\n{'='*60}")
    print(f"Creating test task for: {task}")
    print(f"{'='*60}")
    task_dict = create_test_only_task(task, train_ratio=train_ratio)

    # Step 2: Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    wisent_model = WisentModel(model_name=model_name, device=device)

    # Step 3: Run BASE evaluation (no LoRA)
    print(f"\n{'='*60}")
    print(f"Running BASE evaluation (no LoRA)")
    print(f"{'='*60}")

    base_lm_eval_results = run_lm_eval_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=limit,
    )
    base_acc_lm_eval = extract_accuracy(base_lm_eval_results, task)
    print(f"Base accuracy (lm-eval): {base_acc_lm_eval:.4f}")

    base_ll_results = run_ll_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        limit=limit,
    )
    base_acc_ll = base_ll_results["accuracy"]
    print(f"Base accuracy (LL): {base_acc_ll:.4f}")

    # Step 4: Apply LoRA and run evaluation
    print(f"\n{'='*60}")
    print(f"Applying LoRA adapter from: {lora_path}")
    print(f"{'='*60}")
    apply_lora_to_model(wisent_model, lora_path)

    print(f"\n{'='*60}")
    print(f"Running LORA evaluation")
    print(f"{'='*60}")

    lora_lm_eval_results = run_lm_eval_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        limit=limit,
    )
    lora_acc_lm_eval = extract_accuracy(lora_lm_eval_results, task)
    print(f"LoRA accuracy (lm-eval): {lora_acc_lm_eval:.4f}")

    lora_ll_results = run_ll_evaluation(
        wisent_model=wisent_model,
        task_dict=task_dict,
        task_name=task,
        limit=limit,
    )
    lora_acc_ll = lora_ll_results["accuracy"]
    print(f"LoRA accuracy (LL): {lora_acc_ll:.4f}")

    # Remove LoRA
    remove_lora(wisent_model)

    # Cleanup
    del wisent_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Compile results
    results = {
        "task": task,
        "model": model_name,
        "lora_path": str(lora_path),
        "train_ratio": train_ratio,
        "base_accuracy_lm_eval": base_acc_lm_eval,
        "base_accuracy_ll": base_acc_ll,
        "lora_accuracy_lm_eval": lora_acc_lm_eval,
        "lora_accuracy_ll": lora_acc_ll,
        "difference_lm_eval": lora_acc_lm_eval - base_acc_lm_eval,
        "difference_ll": lora_acc_ll - base_acc_ll,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Model: {model_name}")
    print(f"LoRA: {lora_path}")
    print(f"{'-'*60}")
    print(f"{'Metric':<20} {'Base':<12} {'LoRA':<12} {'Diff':<12}")
    print(f"{'-'*60}")
    print(f"{'lm-eval accuracy':<20} {base_acc_lm_eval:<12.4f} {lora_acc_lm_eval:<12.4f} {lora_acc_lm_eval - base_acc_lm_eval:+.4f}")
    print(f"{'LL accuracy':<20} {base_acc_ll:<12.4f} {lora_acc_ll:<12.4f} {lora_acc_ll - base_acc_ll:+.4f}")
    print(f"{'='*60}")

    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / f"{task}_lora_eval_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA adapter on benchmark")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--lora-path", required=True, help="Path to trained LoRA adapter")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--max-batch-size", type=int, default=8, help="Max batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")

    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        lora_path=args.lora_path,
        task=args.task,
        train_ratio=args.train_ratio,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
