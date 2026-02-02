"""Benchmark testing functions."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Dict, List, Tuple

from wisent.core.errors import BenchmarkLoadError

from ..sample_extraction import get_task_samples_for_analysis, get_task_samples_with_subtasks
from ..sample_helpers import try_alternative_task_names
from ..readme_parsing import update_benchmark_from_readme


__all__ = [
    "test_single_benchmark_direct",
    "test_benchmark_creation",
    "extract_contrastive_pairs_from_output",
    "test_readme_updates",
    "test_benchmark_matching",
]


def test_single_benchmark_direct(benchmark_name: str, benchmark_config: dict) -> bool:
    """Test a single benchmark directly using wisent CLI."""
    task_name = benchmark_config["task"]
    tags = benchmark_config["tags"]
    trust_remote_code = benchmark_config.get("trust_remote_code", False)
    use_subtasks = benchmark_config.get("use_subtasks", False)
    limit_subtasks = benchmark_config.get("limit_subtasks", None)

    print(f"\n{'='*60}")
    print(f"Testing: {benchmark_name} ({task_name})")
    print(f"Tags: {', '.join(tags)}")
    if trust_remote_code:
        print("Trust remote code: ENABLED")
    if use_subtasks:
        print("Use subtasks: ENABLED")
        if limit_subtasks:
            print(f"Limit subtasks: {limit_subtasks}")
    print("=" * 60)

    try:
        output_dir = f"test_results/{benchmark_name}"
        os.makedirs(output_dir, exist_ok=True)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))

        cmd = [
            "python", "-m", "wisent.cli",
            "tasks", task_name,
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--layer", "15",
            "--limit", "5",
            "--classifier-type", "logistic",
            "--verbose",
        ]

        print(f"Running command:")
        print(f"   {' '.join(cmd)}")

        env = os.environ.copy()
        if trust_remote_code:
            env["HF_ALLOW_CODE_EVAL"] = "1"
            env["TRUST_REMOTE_CODE"] = "1"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=project_root,
            env=env,
        )

        output_file = os.path.join(output_dir, "output.txt")
        with open(output_file, "w") as f:
            f.write(f"COMMAND: {' '.join(cmd)}\n")
            f.write(f"RETURN CODE: {result.returncode}\n")
            f.write(f"TRUST_REMOTE_CODE: {trust_remote_code}\n")
            f.write(f"USE_SUBTASKS: {use_subtasks}\n")
            if limit_subtasks:
                f.write(f"LIMIT_SUBTASKS: {limit_subtasks}\n")
            f.write(f"STDOUT:\n{result.stdout}\n")
            f.write(f"STDERR:\n{result.stderr}\n")

        if result.returncode == 0:
            print(f"Successfully tested {benchmark_name}")
            print(f"Output preview: {result.stdout[:300]}...")

            contrastive_pairs = extract_contrastive_pairs_from_output(result.stdout)
            if contrastive_pairs:
                print(f"Found {len(contrastive_pairs)} contrastive pairs")
                pairs_file = os.path.join(output_dir, "contrastive_pairs.json")
                with open(pairs_file, "w") as f:
                    json.dump(contrastive_pairs, f, indent=2)
                print(f"Contrastive pairs saved to: {pairs_file}")

            return True
        else:
            print(f"Failed to test {benchmark_name} - WILL CAUSE SCRIPT TO EXIT")
            print(f"Return code: {result.returncode}")
            print(f"Error: {result.stderr}")
            print(f"Full output saved to: {output_file}")

            if result.stderr and "trust_remote_code" in result.stderr.lower() and not trust_remote_code:
                print("Detected trust_remote_code error, but benchmark not configured for it")
                print("Consider adding 'trust_remote_code': True to benchmark config")

            return False

    except subprocess.TimeoutExpired:
        print(f"Timeout testing {benchmark_name} - WILL CAUSE SCRIPT TO EXIT")
        return False
    except Exception as e:
        print(f"Exception testing {benchmark_name}: {e} - WILL CAUSE SCRIPT TO EXIT")
        return False


def extract_contrastive_pairs_from_output(output: str) -> List[Dict]:
    """Extract contrastive pairs from CLI output."""
    pairs = []
    lines = output.split("\n")

    for i, line in enumerate(lines):
        if "Question:" in line or "Prompt:" in line:
            question = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            correct_answer = None
            incorrect_answer = None

            for j in range(i + 1, min(i + 10, len(lines))):
                next_line = lines[j].strip()
                if "Correct:" in next_line or "Good:" in next_line:
                    correct_answer = next_line.split(":", 1)[1].strip() if ":" in next_line else next_line
                elif "Incorrect:" in next_line or "Bad:" in next_line:
                    incorrect_answer = next_line.split(":", 1)[1].strip() if ":" in next_line else next_line
                elif "---" in next_line or next_line == "":
                    break

            if question and correct_answer and incorrect_answer:
                pairs.append({
                    "question": question,
                    "correct_answer": correct_answer,
                    "incorrect_answer": incorrect_answer,
                })

    return pairs


def test_benchmark_creation(
    benchmark_name: str, benchmark_config: dict
) -> Tuple[bool, List[str]]:
    """Test creating a dataset for a benchmark."""
    task_name = benchmark_config["task"]
    tags = benchmark_config["tags"]
    trust_remote_code = benchmark_config.get("trust_remote_code", False)
    use_subtasks = benchmark_config.get("use_subtasks", False)
    limit_subtasks = benchmark_config.get("limit_subtasks", None)

    print(f"\nTesting dataset creation for {benchmark_name} ({task_name})...")
    if tags:
        print(f"Predefined tags: {', '.join(tags)}")
    else:
        print("Tags will be auto-determined from README")

    if trust_remote_code:
        print("Trust remote code: ENABLED")
    if use_subtasks:
        print("Use subtasks: ENABLED")
        if limit_subtasks:
            print(f"Limit subtasks: {limit_subtasks}")

    try:
        if use_subtasks:
            result = get_task_samples_with_subtasks(
                task_name,
                num_samples=5,
                trust_remote_code=trust_remote_code,
                limit_subtasks=limit_subtasks,
            )
        else:
            result = get_task_samples_for_analysis(
                task_name, num_samples=5, trust_remote_code=trust_remote_code
            )

        if "error" in result:
            error_msg = result["error"]
            print(f"Error retrieving samples: {error_msg}")

            if "trust_remote_code" in error_msg and not trust_remote_code:
                print("Retrying with trust_remote_code=True...")
                result = get_task_samples_for_analysis(
                    task_name, num_samples=5, trust_remote_code=True
                )
                if "error" not in result:
                    print("Success with trust_remote_code=True")
                else:
                    print("Still failed with trust_remote_code=True")
                    return False, tags
            elif "not found" in error_msg.lower():
                alternative_results = try_alternative_task_names(
                    benchmark_name, task_name, num_samples=5, trust_remote_code=trust_remote_code
                )
                if alternative_results:
                    result = alternative_results
                    print("Success with alternative task name")
                else:
                    print("No alternative task names worked")
                    return False, tags
            else:
                print("Unhandled error type")
                return False, tags

        if not result.get("samples"):
            print(f"No samples found for {task_name}")

            if not use_subtasks:
                print("Trying subtask approach...")
                result = get_task_samples_with_subtasks(
                    task_name, num_samples=5, trust_remote_code=trust_remote_code
                )
                if result.get("samples"):
                    print("Success with subtask approach")
                else:
                    print("No samples found with subtask approach")
                    return False, tags
            else:
                print("No samples found even with subtask approach")
                return False, tags

        print(f"Successfully retrieved {len(result['samples'])} samples")

        actual_tags = tags
        if "readme_tags" in result and result["readme_tags"]:
            actual_tags = result["readme_tags"]
            print(f"README-determined tags: {', '.join(actual_tags)}")

        if result["samples"]:
            sample = result["samples"][0]
            print(f"Sample question: {sample.get('question', '')[:100]}...")
            print(f"Correct answer: {sample.get('correct_answer', '')}")
            if sample.get("choices"):
                print(f"Choices: {len(sample['choices'])} options")

        return True, actual_tags

    except Exception as e:
        print(f"Exception testing {benchmark_name}: {e}")
        raise BenchmarkLoadError(benchmark_name=benchmark_name, cause=e)


def test_readme_updates(core_benchmarks: Dict) -> None:
    """Test the README update functionality on a few benchmarks."""
    print("Testing README updates...")

    test_benchmarks = ["glue", "superglue", "truthfulqa_mc1", "mmlu", "hellaswag"]

    for benchmark_name in test_benchmarks:
        if benchmark_name in core_benchmarks:
            print(f"\nTesting {benchmark_name}...")
            config = core_benchmarks[benchmark_name]
            updated_config = update_benchmark_from_readme(benchmark_name, config)
            print(f"   Original: {config}")
            print(f"   Updated:  {updated_config}")


def test_benchmark_matching(find_relevant_fn, top_k: int = 3) -> None:
    """Test the benchmark matching function with various prompts."""
    test_prompts = [
        "What is the capital of France?",
        "Write a Python function to calculate factorial",
        "Solve this math problem: 5 + 3 * 2",
        "Is this sentence grammatically correct?",
        "Translate 'hello world' to Spanish",
        "What are the symptoms of diabetes?",
        "Why do objects fall down?",
        "Write a story about a robot",
        "Is artificial intelligence dangerous?",
        "How do you debug a program?",
    ]

    print("Testing benchmark matching...")

    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        matches = find_relevant_fn(prompt, top_k=top_k)

        for i, match in enumerate(matches, 1):
            print(f"  {i}. {match['benchmark']} (score: {match['score']})")
            print(f"     Description: {match['description']}")
            print(f"     Reasons: {', '.join(match['reasons'])}")
            print(f"     Tags: {match['tags']}")
            print(f"     Priority: {match['priority']}")
