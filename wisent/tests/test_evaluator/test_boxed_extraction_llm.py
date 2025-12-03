"""
Test boxed answer extraction with LLM-generated responses.

This script:
1. Loads competition_math dataset
2. Prompts a small LLM to solve problems and put answer in \boxed{}
3. Extracts answer from \boxed{} in LLM response
4. Compares with ground truth answer extracted from dataset solution
"""

import re
from datasets import load_dataset
from math_equivalence import is_equiv

from wisent.core.models.wisent_model import WisentModel


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the LAST \\boxed{} answer from text (final answer convention).

    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}}).

    Args:
        text: The text containing \\boxed{answer}

    Returns:
        The extracted answer from the last \\boxed{} or None if not found
    """
    # Find all \boxed{ occurrences
    start_pattern = r'\\boxed\{'
    matches = list(re.finditer(start_pattern, text))

    if not matches:
        return None

    # Process the LAST match (final answer convention)
    last_match = matches[-1]

    # Start after \boxed{
    start_idx = last_match.end()
    brace_count = 1
    idx = start_idx

    # Find the matching closing brace
    while idx < len(text) and brace_count > 0:
        if text[idx] == '{':
            brace_count += 1
        elif text[idx] == '}':
            brace_count -= 1
        idx += 1

    if brace_count == 0:
        # Extract content between the braces
        return text[start_idx:idx-1].strip()

    return None


def answers_are_equivalent(answer1: str | None, answer2: str | None) -> bool:
    """Check if two math answers are equivalent using math_equivalence library."""
    if answer1 is None or answer2 is None:
        return False
    return is_equiv(answer1, answer2)


def create_prompt(problem: str) -> str:
    """Create instruction prompt for LLM to solve math problem."""
    return f"""Solve the following math problem step by step. At the end, put your final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""


def main():
    # Load model using WisentModel
    print("Loading model...")
    model_name = "Qwen/Qwen2.5-Math-7B-Instruct"
    model = WisentModel(model_name=model_name)

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset('qwedsacf/competition_math', split='train')

    # Test on first N examples
    num_examples = 1

    correct = 0
    incorrect = 0
    no_boxed_in_response = 0
    no_boxed_in_ground_truth = 0

    results = []

    print(f"\nTesting on {num_examples} examples...")
    print("=" * 60)

    for i, example in enumerate(ds.select(range(num_examples))):
        problem = example.get('problem', '')
        solution = example.get('solution', '')

        # Extract ground truth from dataset solution
        ground_truth = extract_boxed_answer(solution)

        if ground_truth is None:
            no_boxed_in_ground_truth += 1
            print(f"\n{i+1}. SKIP - No boxed answer in ground truth")
            continue

        # Create prompt and generate response
        prompt = create_prompt(problem)

        # Generate using WisentModel
        responses = model.generate(
            inputs=prompt,
            max_new_tokens=1000,
            temperature=0.7,
            do_sample=True,
            prompt_is_formatted=True
        )

        response = responses[0] if responses else ""

        # Extract answer from LLM response
        llm_answer = extract_boxed_answer(response)

        if llm_answer is None:
            no_boxed_in_response += 1
            print(f"\n{i+1}. NO BOXED - LLM didn't use \\boxed{{}}")
            print(f"  Problem: {problem}...")
            print(f"  Ground truth: {ground_truth}")
            print(f"  LLM response: {response}...")
            continue

        # Compare answers using math equivalence
        is_correct = answers_are_equivalent(llm_answer, ground_truth)

        if is_correct:
            correct += 1
            status = "✓ CORRECT"
        else:
            incorrect += 1
            status = "✗ INCORRECT"

        print(f"\n{i+1}. {status}")
        print(f"  Problem: {problem[:80]}...")
        print(f"  Ground truth (boxed): {ground_truth}")
        print(f"  LLM answer (boxed): {llm_answer}")
        print(f"  LLM full response: {response}...")

        results.append({
            'problem': problem,
            'ground_truth': ground_truth,
            'llm_answer': llm_answer,
            'is_correct': is_correct,
            'llm_response': response
        })

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total examples: {num_examples}")
    print(f"No boxed in ground truth: {no_boxed_in_ground_truth}")
    print(f"No boxed in LLM response: {no_boxed_in_response}")
    print(f"Correct answers: {correct}")
    print(f"Incorrect answers: {incorrect}")

    evaluated = correct + incorrect
    if evaluated > 0:
        print(f"Accuracy: {100 * correct / evaluated:.2f}%")


if __name__ == "__main__":
    main()
