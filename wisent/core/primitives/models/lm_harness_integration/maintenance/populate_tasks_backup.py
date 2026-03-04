#!/usr/bin/env python3
"""
Script to populate tasks.json with real information from lm_eval tasks.
This script fetches actual examples, descriptions, and evaluation methods.

Implementation split across _populate_backup/ sub-package for the 300-line limit.
"""

import json
import os
import sys
import subprocess

from wisent.core.utils.config_tools.constants import DISPLAY_TRUNCATION_MEDIUM, DISPLAY_TRUNCATION_LONG, BENCH_TEST_SAMPLE_SIZE, JSON_INDENT, SEPARATOR_WIDTH_MEDIUM

# Re-export all public API from the split parts
from wisent.core.primitives.models.lm_harness_integration._populate_backup import (
    find_working_task_from_group,
    get_benchmark_tags_with_llama,
    get_benchmark_groups_from_readme,
    get_samples_from_group_task,
    get_task_samples_for_analysis,
    load_lm_eval,
    expand_group_task,
    get_task_info,
    process_individual_task,
    process_group_task,
    extract_examples_from_task,
    get_evaluation_method,
    get_category,
    get_relevant_benchmarks_for_prompt,
)


def test_prompt_benchmark_matching(test_prompt: str = "I like food"):
    """Test the prompt-to-benchmark matching function."""
    print(f"Testing prompt-to-benchmark matching")
    print(f"Test prompt: '{test_prompt}'")
    print("=" * SEPARATOR_WIDTH_MEDIUM)
    results = get_relevant_benchmarks_for_prompt(test_prompt)
    print("\nResults:")
    for i, result in enumerate(results, 1):
        print(f"{i}. **{result['benchmark']}**")
        print(f"   Explanation: {result['explanation']}")
        print(f"   Relevance Score: {result['relevance_score']}")
        print()
    return results


def test_sample_retrieval(task_name: str, timeout: int):
    """Test function to demonstrate the get_task_samples_for_analysis function."""
    print(f"\n=== Testing Sample Retrieval for '{task_name}' ===")
    result = get_task_samples_for_analysis(task_name, timeout=timeout, num_samples=BENCH_TEST_SAMPLE_SIZE)
    if "error" in result:
        print(f"Error: {result['error']}")
        return False
    print(f"Successfully retrieved samples from '{result['task_name']}'")
    description = result.get('description') or "No description available"
    print(f"Description: {description[:DISPLAY_TRUNCATION_MEDIUM]}...")
    print(f"Total documents: {result['total_docs']}")
    print(f"Sampled documents: {result['sampled_docs']}")
    print(f"Output type: {result['output_type']}")
    print(f"\n--- Sample Questions ---")
    for i, sample in enumerate(result['samples']):
        print(f"\nSample {sample['sample_id']}:")
        question = sample.get('question', 'No question available')
        print(f"Question: {question[:DISPLAY_TRUNCATION_LONG]}...")
        answer = sample.get('correct_answer', 'No answer available')
        print(f"Correct Answer: {answer}")
        format_type = sample.get('format', 'unknown')
        print(f"Format: {format_type}")
        if sample.get('choices'):
            print(f"Choices:")
            for j, choice in enumerate(sample['choices']):
                marker = ">" if j == sample.get('correct_choice_index') else " "
                print(f"  {marker} {j}: {choice}")
        if sample.get('additional_info'):
            print(f"Additional info: {list(sample['additional_info'].keys())}")
    print(f"\n=== Analysis Summary ===")
    print("Based on these samples, an AI could analyze:")
    print("- Question format and complexity")
    print("- Type of reasoning required")
    print("- Domain knowledge needed")
    print("- Cognitive abilities being tested")
    return True


def test_specific_task(timeout: int):
    """Test the specific problematic task for error handling."""
    print("\nTesting specific problematic tasks...")
    test_tasks = [
        'evalita-mp_ner_fic_group',
        'flan_held_in'
    ]
    for task_name in test_tasks:
        print(f"\nTesting task: {task_name}")
        try:
            result = get_task_samples_for_analysis(task_name, timeout=timeout, num_samples=BENCH_TEST_SAMPLE_SIZE)
            print(f"Result keys: {list(result.keys())}")
            if 'error' in result:
                print(f"Error: {result['error']}")
                if 'skip_reason' in result:
                    print(f"Skip reason: {result['skip_reason']}")
                    print("Error was handled gracefully with skip reason!")
                else:
                    print("No skip reason provided")
            else:
                print(f"Success: Retrieved "
                      f"{len(result.get('samples', []))} samples")
        except Exception as e:
            print(f"Exception raised: {e}")
            import traceback
            traceback.print_exc()


def main(timeout: int):
    """Main function to populate tasks.json."""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        task_name = sys.argv[2] if len(sys.argv) > 2 else "truthfulqa_mc1"
        print(f"Running in TEST MODE")
        success = test_sample_retrieval(task_name, timeout=timeout)
        if success:
            print(f"\nTest completed successfully!")
        else:
            print(f"\nTest failed!")
        return
    print("Loading lm_eval library...")
    get_task_dict, task_registry = load_lm_eval()
    print("Getting all available tasks from lm_eval...")
    try:
        print("Using subprocess to get task list...")
        result = subprocess.run(
            ['lm_eval', '--tasks', 'list'],
            capture_output=True, text=True)
        all_tasks_output = result.stdout
        task_names = []
        for line in all_tasks_output.split('\n'):
            if ('|' in line and not line.startswith('|---') and
                    'Group' not in line and 'Config Location' not in line):
                parts = line.split('|')
                if len(parts) >= 2:
                    task_name = parts[1].strip()
                    if (task_name and not task_name.startswith('-') and
                            task_name != 'Group'):
                        task_names.append(task_name)
        available_tasks = task_names
        print(f"Found {len(available_tasks)} total tasks via subprocess")
    except Exception as e:
        print(f"Error getting tasks from lm_eval: {e}")
        available_tasks = ["truthfulqa_mc1", "hellaswag"]
    tasks_file = "wisent/core/tasks.json"
    print(f"\nPhase 1: Saving {len(available_tasks)} task names to {tasks_file}")
    initial_data = {
        "tasks": {tn: {} for tn in available_tasks},
        "task_list": available_tasks
    }
    with open(tasks_file, 'w') as f:
        json.dump(initial_data, f, indent=JSON_INDENT, ensure_ascii=False)
    print(f"Saved {len(available_tasks)} task names to {tasks_file}")
    print(f"\nPhase 2: Populating detailed information for 2 test tasks...")
    tasks_to_populate = ["truthfulqa_mc1", "hellaswag"]
    print(f"Will populate: {tasks_to_populate}")
    with open(tasks_file, 'r') as f:
        current_data = json.load(f)
    processed = 0
    for i, task_name in enumerate(tasks_to_populate):
        print(f"Processing {i+1}/{len(tasks_to_populate)}: {task_name}")
        task_info = get_task_info(task_name, get_task_dict, task_registry)
        if task_info:
            current_data["tasks"][task_name] = task_info
            processed += 1
            with open(tasks_file, 'w') as f:
                json.dump(current_data, f, indent=JSON_INDENT, ensure_ascii=False)
            print(f"  Updated {task_name}")
        else:
            print(f"  Failed to process {task_name}")
    print(f"\nCompleted processing. Successfully populated "
          f"{processed}/{len(tasks_to_populate)} tasks")
    print(f"Total tasks in file: {len(current_data['task_list'])}")
    print(f"Tasks with details: "
          f"{sum(1 for task in current_data['tasks'].values() if task)}")
    _print_summary_statistics(current_data)


def _print_summary_statistics(current_data):
    """Print category and evaluation method distribution."""
    categories = {}
    eval_methods = {}
    for task_info in current_data["tasks"].values():
        if task_info:
            cat = task_info.get('category', 'unknown')
            eval_method = task_info.get('evaluation_method', 'unknown')
            if isinstance(cat, list):
                for c in cat:
                    categories[c] = categories.get(c, 0) + 1
            else:
                categories[cat] = categories.get(cat, 0) + 1
            if isinstance(eval_method, list):
                for method in eval_method:
                    eval_methods[method] = eval_methods.get(method, 0) + 1
            else:
                eval_methods[eval_method] = eval_methods.get(eval_method, 0) + 1
    if categories:
        print("\nCategory distribution (populated tasks):")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
    if eval_methods:
        print("\nEvaluation method distribution (populated tasks):")
        for method, count in sorted(eval_methods.items()):
            print(f"  {method}: {count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, required=True, help="HTTP timeout in seconds")
    parser.add_argument("mode", nargs="?", default="run")
    args = parser.parse_args()
    if args.mode == "test":
        test_specific_task(timeout=args.timeout)
    else:
        main(timeout=args.timeout)
