"""
Test comparing log likelihood evaluation methods.

Compares:
1. lm-eval-harness (official library)
2. wisent LogLikelihoodsEvaluator (original)
3. wisent LogLikelihoodsEvaluatorBC (lm-eval compatible)
"""

import torch
from lm_eval.tasks import get_task_dict
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from wisent.core.models.wisent_model import WisentModel
from wisent.core.evaluators.benchmark_specific.log_likelihoods_evaluator import LogLikelihoodsEvaluator
from wisent.core.evaluators.benchmark_specific.log_likelihoods_evaluator_bc import LogLikelihoodsEvaluatorBC
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_extractor_registry import get_extractor


MODEL_NAME = "meta-llama/Llama-3.2-1B"
TASK_NAME = "boolq"
EVAL_LIMIT = 1500


def run_lm_eval_evaluation(wisent_model, task_dict, task_name, limit):
    """Run official lm-eval evaluation."""
    lm = HFLM(
        pretrained=wisent_model.hf_model,
        tokenizer=wisent_model.tokenizer,
        batch_size=1,
        max_batch_size=8,
    )

    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
    )

    return results


def run_ll_evaluation(wisent_model, task, extractor, docs, evaluator_class, task_name, return_details=False):
    """Run evaluation with specified LogLikelihoodsEvaluator class."""
    ll_evaluator = evaluator_class()

    correct = 0
    results_list = []
    for i, doc in enumerate(docs):
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

        if result.ground_truth == "TRUTHFUL":
            correct += 1

        if return_details:
            results_list.append({
                "idx": i,
                "question": question,
                "choices": choices,
                "expected": expected,
                "predicted": result.meta["predicted"],
                "log_probs": result.meta["log_probs"],
                "correct": result.ground_truth == "TRUTHFUL",
            })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(docs)}")

    accuracy = correct / len(docs) if docs else 0.0
    if return_details:
        return accuracy, results_list
    return accuracy


def main():
    print(f"{'='*70}")
    print(f"Log Likelihood Evaluation Comparison Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Task: {TASK_NAME}")
    print(f"Limit: {EVAL_LIMIT}")
    print(f"{'='*70}\n")

    # Load model
    print("Loading model...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wisent_model = WisentModel(model_name=MODEL_NAME, device=device)

    # Load task
    print("Loading task...")
    task_dict = get_task_dict([TASK_NAME])
    task = task_dict[TASK_NAME]
    extractor = get_extractor(TASK_NAME)

    # Get docs
    if task.has_validation_docs():
        docs = list(task.validation_docs())
    elif task.has_test_docs():
        docs = list(task.test_docs())
    else:
        docs = list(task.training_docs())

    docs = docs[:EVAL_LIMIT]
    print(f"Evaluating {len(docs)} examples\n")

    # 1. Run lm-eval-harness
    print("1. Running lm-eval-harness...")
    lm_eval_results = run_lm_eval_evaluation(wisent_model, task_dict, TASK_NAME, EVAL_LIMIT)
    lm_eval_acc = lm_eval_results["results"][TASK_NAME].get("acc,none", 0.0)
    print(f"   Accuracy: {lm_eval_acc:.4f}\n")

    # 2. Run original LogLikelihoodsEvaluator (with normalization)
    print("2. Running LogLikelihoodsEvaluator (original, with normalization)...")
    original_acc, original_details = run_ll_evaluation(wisent_model, task, extractor, docs, LogLikelihoodsEvaluator, TASK_NAME, return_details=True)
    print(f"   Accuracy: {original_acc:.4f}\n")

    # 3. Run LogLikelihoodsEvaluatorBC (lm-eval compatible, no normalization)
    print("3. Running LogLikelihoodsEvaluatorBC (lm-eval compatible, no normalization)...")
    bc_acc, bc_details = run_ll_evaluation(wisent_model, task, extractor, docs, LogLikelihoodsEvaluatorBC, TASK_NAME, return_details=True)
    print(f"   Accuracy: {bc_acc:.4f}\n")

    # Print summary
    print(f"{'='*70}")
    print(f"ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"lm-eval-harness:                    {lm_eval_acc:.4f}")
    print(f"LogLikelihoodsEvaluator (original): {original_acc:.4f}  (diff: {original_acc - lm_eval_acc:+.4f})")
    print(f"LogLikelihoodsEvaluatorBC:          {bc_acc:.4f}  (diff: {bc_acc - lm_eval_acc:+.4f})")
    print(f"{'='*70}\n")

    # Find and print differences
    print(f"{'='*70}")
    print(f"DIFFERENCES (Original vs BC/lm-eval)")
    print(f"{'='*70}")
    diff_count = 0
    for orig, bc in zip(original_details, bc_details):
        if orig["predicted"] != bc["predicted"]:
            diff_count += 1
            print(f"\n--- Example {orig['idx']} ---")
            print(f"Question: {orig['question'][:200]}...")
            print(f"Choices: {orig['choices']}")
            print(f"Expected: {orig['expected']}")
            print(f"")
            print(f"Original (normalized) predicted: {orig['predicted']}")
            print(f"  Log probs: {orig['log_probs']}")
            print(f"")
            print(f"BC (no norm) predicted: {bc['predicted']}")
            print(f"  Log probs: {bc['log_probs']}")
            print(f"")
            print(f"Original correct: {orig['correct']}, BC correct: {bc['correct']}")

    print(f"\n{'='*70}")
    print(f"Total differences: {diff_count} / {len(original_details)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
