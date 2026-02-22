import re
from datasets import load_dataset
from evaluate import load
from wisent.core.evaluators.benchmark_specific.math_parsing.is_equiv import is_equiv


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the final answer from \\boxed{} notation in competition math solutions.

    Handles nested braces correctly (e.g., \\boxed{\\frac{1}{2}}).

    Args:
        text: The text containing \\boxed{answer}

    Returns:
        The extracted answer or None if not found
    """
    # Find \boxed{ and then match balanced braces
    start_pattern = r'\\boxed\{'
    match = re.search(start_pattern, text)
    if not match:
        return None

    # Start after \boxed{
    start_idx = match.end()
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


# Test the extract_boxed_answer function
test_cases = [
    (r"The answer is \boxed{42}.", "42"),
    (r"Therefore, $x = \boxed{\frac{1}{2}}$.", r"\frac{1}{2}"),
    (r"\boxed{x^2 + y^2}", "x^2 + y^2"),
    (r"We get \boxed{\sqrt{2}}", r"\sqrt{2}"),
    (r"\boxed{{a, b, c}}", "{a, b, c}"),
    ("No boxed answer here", None),
]

print("Testing extract_boxed_answer:")
for text, expected in test_cases:
    result = extract_boxed_answer(text)
    status = "✓" if result == expected else "✗"
    print(f"  {status} Input: {text[:50]}... -> Got: {result}, Expected: {expected}")

print()

math_metric = load("competition_math")

references = ["2"]

predictions = ["2"]


results = math_metric.compute(
    references=references,
    predictions=predictions,
)

print(results)

print(is_equiv("2", "4/2"))

# Compare extraction methods on competition_math dataset
print("\n" + "="*60)
print("Comparing extraction methods on competition_math dataset:")
print("="*60)

from wisent.core.evaluators.benchmark_specific.generation_evaluator import GenerationEvaluator

evaluator = GenerationEvaluator()
ds = load_dataset('qwedsacf/competition_math', split='train')

# Count differences
total = 0
different = 0
boxed_none = 0
gen_none = 0
examples_different = []

for i, example in enumerate(ds):
    solution = example.get('solution', '')

    # Method 1: Boxed extraction
    boxed_answer = extract_boxed_answer(solution)

    # Method 2: Generation evaluator extraction
    gen_answer = evaluator._extract_numerical_answer(solution)

    total += 1

    if boxed_answer is None:
        boxed_none += 1
    if gen_answer is None:
        gen_none += 1

    # Compare: convert boxed to float if possible for fair comparison
    boxed_as_num = None
    if boxed_answer is not None:
        try:
            boxed_as_num = float(boxed_answer)
        except ValueError:
            pass

    if boxed_as_num is not None and gen_answer is not None:
        if abs(boxed_as_num - gen_answer) > 1e-6:
            different += 1
            if len(examples_different) < 10:
                examples_different.append({
                    'idx': i,
                    'problem': example.get('problem', '')[:100],
                    'boxed': boxed_answer,
                    'boxed_num': boxed_as_num,
                    'gen': gen_answer,
                    'solution_end': solution[-150:]
                })
    elif boxed_answer is not None or gen_answer is not None:
        # One is None, the other is not
        different += 1
        if len(examples_different) < 10:
            examples_different.append({
                'idx': i,
                'problem': example.get('problem', '')[:100],
                'boxed': boxed_answer,
                'boxed_num': boxed_as_num,
                'gen': gen_answer,
                'solution_end': solution[-150:]
            })

print(f"\nTotal examples: {total}")
print(f"Boxed extraction returned None: {boxed_none}")
print(f"Generation extraction returned None: {gen_none}")
print(f"Different answers: {different}")
print(f"Same answers: {total - different}")
print(f"Percentage different: {100 * different / total:.2f}%")

print("\n" + "-"*60)
print("First 10 examples where methods differ:")
print("-"*60)
for ex in examples_different:
    print(f"\nExample {ex['idx']}:")
    print(f"  Problem: {ex['problem']}...")
    print(f"  Boxed answer: {ex['boxed']} (as num: {ex['boxed_num']})")
    print(f"  Generation answer: {ex['gen']}")
    print(f"  Solution end: ...{ex['solution_end']}")
