from __future__ import annotations

import sys
import os
# Add project root to Python path for wisent_guard imports
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import re
from lm_eval import tasks
from typing import List, Dict
import json
import os

from wisent_guard.core.models.wisent_model import WisentModel

def load_gsm8k_questions(limit, preferred_doc: str = "test"):
    """Load first limit GSM8K questions from specified document source.

    Args:
        limit: Number of questions to load for evaluation
        preferred_doc: Document source ("validation", "test", "training", "fewshot")
                      Default is "test". Only this source will be used, no fallback.
    """
    print(f"Loading GSM8K task from {preferred_doc} docs...")

    task_dict = tasks.get_task_dict(["gsm8k"])
    gsm8k_task = task_dict["gsm8k"]

    # Map preferred_doc to method names
    doc_source_map = {
        "validation": ("has_validation_docs", "validation_docs"),
        "test": ("has_test_docs", "test_docs"),
        "training": ("has_training_docs", "training_docs"),
        "fewshot": ("has_fewshot_docs", "fewshot_docs"),
    }

    if preferred_doc not in doc_source_map:
        raise ValueError(f"Invalid doc source: {preferred_doc}. Must be one of {list(doc_source_map.keys())}")

    has_method, docs_method = doc_source_map[preferred_doc]
    docs = []

    if hasattr(gsm8k_task, has_method):
        has_attr = getattr(gsm8k_task, has_method)
        has_docs = has_attr() if callable(has_attr) else has_attr

        if has_docs and hasattr(gsm8k_task, docs_method):
            docs_func = getattr(gsm8k_task, docs_method)
            if callable(docs_func):
                docs = list(docs_func())
                if docs:
                    print(f"Loaded from {docs_method}")

    if not docs:
        raise RuntimeError(f"Could not load GSM8K documents from {preferred_doc} source")

    docs = docs[:limit]

    print(f"Successfully loaded {len(docs)} questions (indices 0-{limit - 1})")

    questions = []
    for doc in docs:
        question_text = doc.get("question", "")
        answer_text = doc.get("answer", "")

        if "####" in answer_text:
            numerical_answer = answer_text.split("####")[-1].strip()
        else:
            numerical_answer = answer_text.strip()

        questions.append({
            "question": question_text,
            "answer": numerical_answer,
        })

    return questions

def extract_numerical_answer(text: str) -> str | None:
    """
    Extract numerical answer from model response.

    Tries multiple extraction strategies:
    1. JSON format: {"final_answer": 123}
    2. Common patterns: "The answer is 123", "#### 123"
    3. Last number in text

    Args:
        text: Model's response text

    Returns:
        Extracted numerical answer as string, or None if not found
    """
    if not text:
        return None

    # Strategy 1: Try to extract from JSON format
    try:
        # Look for JSON object with final_answer key
        json_match = re.search(r'\{[^}]*"final_answer"[^}]*:\s*([^,}\s]+)[^}]*\}', text, re.IGNORECASE)
        if json_match:
            answer = json_match.group(1).strip().strip('"').strip("'")
            # Clean up and validate it's a number
            answer = re.sub(r'[^\d.\-]', '', answer)
            if answer and answer.replace('.', '').replace('-', '').isdigit():
                return answer
    except Exception:
        pass

    # Strategy 2: Look for common GSM8K answer patterns
    # Pattern: "#### 123" (standard GSM8K format)
    hash_match = re.search(r'####\s*([\d,\.\-]+)', text)
    if hash_match:
        answer = hash_match.group(1).replace(',', '')
        return answer

    # Pattern: "The answer is 123" or "The final answer is 123"
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,\.\-]+)',
        r'(?:therefore|thus|so),?\s+(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([\d,\.\-]+)',
        r'=\s*([\d,\.\-]+)\s*$',  # Ends with = 123
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            answer = match.group(1).replace(',', '')
            return answer

    # Strategy 3: Last number in the text (fallback)
    # Find all numbers in the text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    if numbers:
        # Return the last number found
        return numbers[-1]

    return None


def evaluate_baseline_performance(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    num_questions: int = 250,
    max_new_tokens: int = 700,
    output_path: str | None = None,
) -> Dict:
    """
    Evaluate baseline model performance on GSM8K.

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

    questions = load_gsm8k_questions(limit=num_questions, preferred_doc="test")

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
        question = item["question"]
        correct_answer = item["answer"]

        # Format question as a clear prompt
        prompt = f"Question: {question}\n\nSolve this step by step. After your reasoning, provide the final numerical answer in this exact JSON format:\n{{\n  \"final_answer\": <your_number_here>\n}}\n\nAnswer:"

        messages = [[{"role": "user", "content": prompt}]]

        try:
            responses = model.generate(
                inputs=messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_steering=False,
            )
            response = responses[0]

            predicted_answer = extract_numerical_answer(response)

            is_correct = False
            if predicted_answer is not None:
                try:
                    pred_float = float(predicted_answer)
                    correct_float = float(correct_answer)
                    # Allow small floating point differences
                    is_correct = abs(pred_float - correct_float) < 1e-6
                except (ValueError, TypeError):
                    # Fall back to string comparison
                    is_correct = predicted_answer.strip() == correct_answer.strip()

            if is_correct:
                correct_count += 1

            result = {
                "question_id": idx,
                "question": question,
                "correct_answer": correct_answer,
                "model_response": response,
                "predicted_answer": predicted_answer,
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
                "question": question,
                "correct_answer": correct_answer,
                "model_response": None,
                "predicted_answer": None,
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"\nSaving results to {output_path}...")
        with open(output_path, "w") as f:
            json.dump(evaluation_result, f, indent=2)
        print("Results saved successfully!")

        # Save metadata
        metadata_path = os.path.join(os.path.dirname(output_path), "gsm8k_baseline_metadata.json")
        correct_questions = [r for r in results if r.get("is_correct")]
        incorrect_questions = [r for r in results if not r.get("is_correct")]

        metadata = {
            "summary": {
                "total_questions": summary["total_questions"],
                "correct_count": summary["correct_answers"],
                "incorrect_count": summary["incorrect_answers"],
                "accuracy_percent": summary["accuracy_percent"],
                "model_name": summary["model_name"],
            },
            "correct_questions": [
                {
                    "id": r["question_id"],
                    "question": r["question"],
                    "correct_answer": r["correct_answer"],
                    "predicted_answer": r["predicted_answer"],
                }
                for r in correct_questions
            ],
            "incorrect_questions": [
                {
                    "id": r["question_id"],
                    "question": r["question"],
                    "correct_answer": r["correct_answer"],
                    "predicted_answer": r["predicted_answer"],
                    "model_response": r.get("model_response", "")[:500],  # Truncate long responses
                }
                for r in incorrect_questions
            ],
        }

        print(f"Saving metadata to {metadata_path}...")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print("Metadata saved successfully!")

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
            print(f"  Q: {result['question'][:150]}...")
            print(f"  Correct answer: {result['correct_answer']}")
            print(f"  Model predicted: {result['predicted_answer']}")

    if incorrect_examples:
        print("\n✗ INCORRECT EXAMPLES:")
        for i, result in enumerate(incorrect_examples):
            print(f"\nExample {i+1}:")
            print(f"  Q: {result['question'][:150]}...")
            print(f"  Correct answer: {result['correct_answer']}")
            print(f"  Model predicted: {result['predicted_answer']}")
            if result["model_response"]:
                print(f"  Full response: {result['model_response'][:300]}...")

    return evaluation_result

if __name__ == "__main__":

    results = evaluate_baseline_performance(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        num_questions=150,
        max_new_tokens=700,
        output_path="/workspace/results/gsm8k/gsm8k_baseline_results.json",
    )

    print("\nDone! Baseline evaluation complete.")
    print("You can load the results later with:")
    print("  import json")
    print("  with open('/workspace/results/gsm8k/gsm8k_baseline_results.json', 'r') as f:")
    print("      results = json.load(f)")
