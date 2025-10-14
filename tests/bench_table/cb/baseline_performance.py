from __future__ import annotations

import re
from lm_eval import tasks
from typing import List, Dict
import json
from pathlib import Path

from wisent_guard.core.models.wisent_model import WisentModel


def load_cb_questions(limit, preferred_doc: str = "test"):
    """Load first limit CB questions from specified document source.

    Args:
        limit: Number of questions to load for evaluation
        preferred_doc: Preferred document source ("validation", "test", "training", "fewshot")
                      Default is "test" for test set evaluation.
    """
    print(f"Loading CB task from {preferred_doc} docs...")

    task_dict = tasks.get_task_dict(["cb"])
    cb_task = task_dict["cb"]

    # Map preferred_doc to method names
    doc_source_map = {
        "validation": ("has_validation_docs", "validation_docs"),
        "test": ("has_test_docs", "test_docs"),
        "training": ("has_training_docs", "training_docs"),
        "fewshot": ("has_fewshot_docs", "fewshot_docs"),
    }

    # Build preferred order with preferred_doc first
    default_order = [
        ("has_validation_docs", "validation_docs"),
        ("has_test_docs", "test_docs"),
        ("has_training_docs", "training_docs"),
        ("has_fewshot_docs", "fewshot_docs"),
    ]

    if preferred_doc in doc_source_map:
        preferred_source = doc_source_map[preferred_doc]
        other_sources = [s for s in default_order if s != preferred_source]
        preferred_sources = [preferred_source] + other_sources
    else:
        preferred_sources = default_order

    docs = []
    for has_method, docs_method in preferred_sources:

        if hasattr(cb_task, has_method):
            has_attr = getattr(cb_task, has_method)
            has_docs = has_attr() if callable(has_attr) else has_attr

            if has_docs and hasattr(cb_task, docs_method):
                docs_func = getattr(cb_task, docs_method)
                if callable(docs_func):
                    docs = list(docs_func())
                    if docs:
                        print(f"Loaded from {docs_method}")
                        break

    if not docs:
        raise RuntimeError("Could not load CB documents from lm-eval task")

    docs = docs[:limit]

    print(f"Successfully loaded {len(docs)} questions (indices 0-{limit - 1})")

    # CB uses numerical labels: 0 = True, 1 = False, 2 = Neither
    label_map = {0: "True", 1: "False", 2: "Neither"}

    questions = []
    for doc in docs:
        premise = doc.get("premise", "")
        hypothesis = doc.get("hypothesis", "")
        label_idx = doc.get("label")

        # Map numerical label to text
        label = label_map.get(label_idx, "Neither")

        questions.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
        })

    return questions

def extract_answer(text: str) -> str | None:
    """
    Extract True/False/Neither answer from model response for CB.

    Tries multiple extraction strategies:
    1. JSON format: {"final_answer": "True"} etc.
    2. Common patterns: "The answer is True", "Answer: False"
    3. First occurrence of "True", "False", or "Neither" in text

    Args:
        text: Model's response text

    Returns:
        Extracted answer as "True", "False", or "Neither", or None if not found
    """
    if not text:
        return None

    text_lower = text.lower()

    # Strategy 1: Try to extract from JSON format
    try:
        # Look for JSON object with final_answer key
        json_match = re.search(r'\{[^}]*"final_answer"[^}]*:\s*"?([^,}"\s]+)"?[^}]*\}', text_lower, re.IGNORECASE)
        if json_match:
            answer = json_match.group(1).strip().strip('"').strip("'").lower()
            if answer == "true":
                return "True"
            elif answer == "false":
                return "False"
            elif answer == "neither":
                return "Neither"
    except Exception:
        pass

    # Strategy 2: Look for common CB answer patterns
    # Pattern: "The answer is True/False/Neither"
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(true|false|neither)',
        r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(true|false|neither)',
        r'answer\s*:\s*(true|false|neither)',
        r'(?:the\s+)?(?:relationship|label)\s+is\s*:?\s*(true|false|neither)',
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            return answer.capitalize()  # "true" -> "True", "false" -> "False", "neither" -> "Neither"

    # Strategy 3: First occurrence (fallback)
    # Look for True/False/Neither as standalone words
    true_match = re.search(r'\b(true)\b', text_lower)
    false_match = re.search(r'\b(false)\b', text_lower)
    neither_match = re.search(r'\b(neither)\b', text_lower)

    # Return whichever appears first
    matches = []
    if true_match:
        matches.append((true_match.start(), "True"))
    if false_match:
        matches.append((false_match.start(), "False"))
    if neither_match:
        matches.append((neither_match.start(), "Neither"))

    if matches:
        return min(matches, key=lambda x: x[0])[1]

    return None

def evaluate_baseline_performance(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    num_questions: int = 50,
    max_new_tokens: int = 700,
    output_path: str | None = None,
) -> Dict:
    """
    Evaluate baseline model performance on CB.

    Args:
        model_name: HuggingFace model name or path
        num_questions: Number of questions to evaluate
        max_new_tokens: Maximum tokens to generate
        output_path: Optional path to save results

    Returns:
        Dict containing evaluation results and statistics
    """

    print(f"Evaluating baseline performance for {model_name}")
    print(f"Questions: {num_questions}")
    print(f"Generation params: max_new_tokens={max_new_tokens}")
    print("=" * 80)

    questions = load_cb_questions(limit=num_questions, preferred_doc="test")

    if len(questions) < num_questions:
        print(f"Warning: Only loaded {len(questions)} questions, expected {num_questions}")
        num_questions = len(questions)

    print(f"\nLoading model: {model_name}")
    model = WisentModel(model_name=model_name, layers={})
    print(f"Model loaded. Hidden size: {model.hidden_size}, Layers: {model.num_layers}")

    results: List[Dict] = []
    correct_count = 0

    print(f"\nEvaluating {len(questions)} questions...")
    print("=" * 80)

    for idx, item in enumerate(questions):
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        correct_label = item["label"]

        # Format similar to CB extractor: {premise}\nQuestion: {hypothesis}. True, False, or Neither?
        prompt = f"{premise}\nQuestion: {hypothesis}. True, False, or Neither?\n\nAfter your reasoning, provide the final answer in this exact JSON format:\n{{\n  \"final_answer\": \"<True, False, or Neither>\"\n}}\n\nAnswer:"

        messages = [[{"role": "user", "content": prompt}]]

        try:
            responses = model.generate(
                inputs=messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_steering=False,
            )
            response = responses[0]

            predicted_label = extract_answer(response)

            is_correct = False
            if predicted_label is not None:
                try:
                    is_correct = predicted_label.strip() == correct_label.strip()
                except (ValueError, TypeError):
                    pass

            if is_correct:
                correct_count += 1

            result = {
                "question_id": idx,
                "premise": premise,
                "hypothesis": hypothesis,
                "correct_label": correct_label,
                "model_response": response,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
            }
            results.append(result)

            if (idx + 1) % 10 == 0 or idx == 0:
                current_accuracy = (correct_count / (idx + 1)) * 100
                status = "✓" if is_correct else "✗"
                print(f"{status} Question {idx + 1}/{len(questions)}: {correct_count}/{idx + 1} correct ({current_accuracy:.1f}%)")

        except Exception as e:
            print(f"✗ Error processing question {idx}: {e}")
            result = {
                "question_id": idx,
                "premise": premise,
                "hypothesis": hypothesis,
                "correct_label": correct_label,
                "model_response": None,
                "predicted_label": None,
                "is_correct": False,
                "error": str(e),
            }
            results.append(result)

        total_questions = len(results)
        accuracy = (correct_count / total_questions * 100) if total_questions > 0 else 0.0

    summary = {
        "model_name": model_name,
        "total_questions": total_questions,
        "correct_answers": correct_count,
        "incorrect_answers": total_questions - correct_count,
        "accuracy_percent": accuracy
    }

    evaluation_result = {
    "summary": summary,
    "results": results,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving results to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(evaluation_result, f, indent=2)
        print("Results saved successfully!")

    print("\n" + "=" * 80)
    print("BASELINE PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Model: {summary['model_name']}")
    print(f"Total questions: {summary['total_questions']}")
    print(f"Correct answers: {summary['correct_answers']}")
    print(f"Incorrect answers: {summary['incorrect_answers']}")
    print(f"Accuracy: {summary['accuracy_percent']:.2f}%")
    print("=" * 80)

    print("\nSample Results:")
    print("-" * 80)
    correct_examples = [r for r in results if r["is_correct"]][:2]
    incorrect_examples = [r for r in results if not r["is_correct"]][:2]

    if correct_examples:
        print("\n✓ CORRECT EXAMPLES:")
        for i, result in enumerate(correct_examples):
            print(f"\nExample {i+1}:")
            print(f"  Premise: {result['premise']}")
            print(f"  Hypothesis: {result['hypothesis']}")
            print(f"  Correct label: {result['correct_label']}")
            print(f"  Model predicted: {result['predicted_label']}")

    if incorrect_examples:
        print("\n✗ INCORRECT EXAMPLES:")
        for i, result in enumerate(incorrect_examples):
            print(f"\nExample {i+1}:")
            print(f"  Premise: {result['premise']}")
            print(f"  Hypothesis: {result['hypothesis']}")
            print(f"  Correct label: {result['correct_label']}")
            print(f"  Model predicted: {result['predicted_label']}")
            if result["model_response"]:
                print(f"  Full response: {result['model_response'][:300]}...")

    return evaluation_result

if __name__ == "__main__":

    results = evaluate_baseline_performance(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_questions=50,
        max_new_tokens=700,
        output_path="tests/bench_table/cb/baseline_results.json",
    )

    print("\nDone! Baseline evaluation complete.")
    print("You can load the results later with:")
    print("  import json")
    print("  with open('tests/bench_table/cb/baseline_results.json', 'r') as f:")
    print("      results = json.load(f)")