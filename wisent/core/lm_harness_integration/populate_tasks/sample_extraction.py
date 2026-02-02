"""Sample extraction functions for populate_tasks."""

from typing import Dict, Any, List, Optional, Tuple

from wisent.core.utils import get_all_docs_from_task, create_deterministic_split


def get_evaluation_method(task) -> str:
    """Get evaluation method from a task."""
    eval_method = "Unknown evaluation method"

    if hasattr(task, 'process_results'):
        try:
            if hasattr(task, 'OUTPUT_TYPE'):
                output_type = task.OUTPUT_TYPE
                if output_type == "multiple_choice":
                    eval_method = "Multiple choice accuracy (argmax of log-likelihoods vs gold labels)"
                elif output_type == "generate_until":
                    eval_method = "Text generation (exact match, F1, BLEU, ROUGE depending on task)"
                elif output_type == "loglikelihood":
                    eval_method = "Log-likelihood evaluation (perplexity, accuracy)"
                elif output_type == "loglikelihood_rolling":
                    eval_method = "Rolling log-likelihood (word/byte perplexity)"
                else:
                    eval_method = f"Unknown output type: {output_type}"
            else:
                eval_method = "Has process_results but no OUTPUT_TYPE found"
        except Exception as e:
            eval_method = f"Has process_results but couldn't inspect: {e}"
    else:
        eval_method = "No process_results method found"

    return eval_method


def get_category(task) -> str:
    """Get category from a task."""
    if hasattr(task, 'OUTPUT_TYPE'):
        output_type = task.OUTPUT_TYPE
        if output_type == "multiple_choice":
            return "multiple_choice"
        elif output_type == "generate_until":
            return "open_ended_generation"
        elif output_type == "loglikelihood":
            return "log_likelihood"
        elif output_type == "loglikelihood_rolling":
            return "rolling_log_likelihood"
        else:
            return f"other_{output_type}"
    return "no_output_type"


def extract_examples_from_task(task_name: str, task) -> Dict[str, str]:
    """Extract example prompts and responses from a task."""
    examples = {}

    try:
        all_docs, split_counts = get_all_docs_from_task(task)
        if not all_docs:
            return {"error": "No documents found"}

        doc = all_docs[0]

        if hasattr(task, 'doc_to_text'):
            examples["example_prompt"] = str(task.doc_to_text(doc))[:500]
        elif 'question' in doc:
            examples["example_prompt"] = str(doc['question'])[:500]
        elif 'text' in doc:
            examples["example_prompt"] = str(doc['text'])[:500]

        if hasattr(task, 'doc_to_target'):
            examples["example_target"] = str(task.doc_to_target(doc))[:200]
        elif 'answer' in doc:
            examples["example_target"] = str(doc['answer'])[:200]
        elif 'target' in doc:
            examples["example_target"] = str(doc['target'])[:200]

        if 'choices' in doc and isinstance(doc['choices'], list):
            examples["example_choices"] = [str(c)[:100] for c in doc['choices'][:6]]
            examples["format"] = "multiple_choice"
        else:
            examples["format"] = "open_ended"

    except Exception as e:
        examples["error"] = str(e)

    return examples


def get_task_samples_for_analysis(task_name: str, num_samples: int = 5) -> Dict[str, Any]:
    """Retrieve sample questions and answers from a benchmark task for AI analysis."""
    from lm_eval import evaluator
    from .group_handling import find_working_task_from_group

    print(f"\nGetting samples from task: {task_name}")

    try:
        task_dict = evaluator.get_task_dict([task_name])
    except Exception as e:
        return {"task_name": task_name, "error": f"Failed to load task: {e}", "samples": []}

    if task_name not in task_dict:
        return {"task_name": task_name, "error": f"Task '{task_name}' not found in lm-eval", "samples": []}

    task = task_dict[task_name]

    if hasattr(task, 'items') and callable(task.items):
        print(f"   '{task_name}' is a group task, finding working subtask...")
        task = find_working_task_from_group(task)
        if task is None:
            return {"task_name": task_name, "error": "No working subtask found in group", "samples": []}

    try:
        all_docs, split_counts = get_all_docs_from_task(task)
    except Exception as e:
        return {"task_name": task_name, "error": f"Failed to get documents: {e}", "samples": []}

    if not all_docs:
        return {"task_name": task_name, "error": "No documents found in task", "samples": []}

    samples = []
    for i, doc in enumerate(all_docs[:num_samples]):
        sample = _extract_sample_from_doc(task, doc, i + 1)
        samples.append(sample)

    description = _get_task_description(task, task_name)

    return {
        "task_name": task_name,
        "description": description,
        "samples": samples,
        "total_docs": len(all_docs),
        "evaluation_method": get_evaluation_method(task),
        "category": get_category(task),
    }


def _extract_sample_from_doc(task, doc, sample_id: int) -> Dict[str, Any]:
    """Extract a sample from a document."""
    sample = {"sample_id": sample_id}

    try:
        if hasattr(task, 'doc_to_text'):
            sample["question"] = str(task.doc_to_text(doc))
        elif 'question' in doc:
            sample["question"] = str(doc['question'])
        elif 'text' in doc:
            sample["question"] = str(doc['text'])
        elif 'prompt' in doc:
            sample["question"] = str(doc['prompt'])
        else:
            sample["question"] = "Question format not recognized"
    except Exception as e:
        sample["question"] = f"Error extracting question: {e}"

    try:
        if hasattr(task, 'doc_to_target'):
            target = task.doc_to_target(doc)
            sample["correct_answer"] = str(target)
        elif 'answer' in doc:
            sample["correct_answer"] = str(doc['answer'])
        elif 'target' in doc:
            sample["correct_answer"] = str(doc['target'])
        else:
            sample["correct_answer"] = "Answer format not recognized"
    except Exception as e:
        sample["correct_answer"] = f"Error extracting answer: {e}"

    sample["choices"] = []
    sample["format"] = "open_ended"
    sample["correct_choice_index"] = None

    try:
        if 'choices' in doc and isinstance(doc['choices'], list):
            sample["choices"] = [str(choice) for choice in doc['choices']]
            sample["format"] = "multiple_choice"
            gold = doc.get('gold', doc.get('label', None))
            if isinstance(gold, list) and len(gold) > 0:
                sample["correct_choice_index"] = gold[0]
            elif isinstance(gold, int):
                sample["correct_choice_index"] = gold
    except Exception:
        pass

    return sample


def _get_task_description(task, task_name: str) -> str:
    """Get task description from various sources."""
    if hasattr(task, 'DESCRIPTION'):
        return str(task.DESCRIPTION)
    if hasattr(task, 'description'):
        return str(task.description)
    if hasattr(task, 'config') and hasattr(task.config, 'description'):
        return str(task.config.description)
    return f"Benchmark task: {task_name}"
