"""Helper functions for sample extraction."""

from __future__ import annotations

import os
from typing import Dict, List, Optional


__all__ = [
    "try_alternative_task_names",
    "get_task_samples_direct",
    "try_datasets_direct_load",
]


def try_alternative_task_names(
    benchmark_name: str,
    original_task_name: str,
    num_samples: int = 5,
    trust_remote_code: bool = False,
) -> Optional[dict]:
    """Try alternative task names for benchmarks with different naming conventions."""
    # Import here to avoid circular imports
    from .sample_extraction import get_task_samples_for_analysis

    alternative_names = {
        "squad2": ["squadv2", "squad_v2", "squad2.0"],
        "math_qa": ["mathqa", "math_qa_python", "math_algebra"],
        "paws_x": ["pawsx", "paws-x", "paws_en", "paws_de", "paws_es"],
        "big_bench": ["bigbench", "big_bench_lite", "bbh"],
        "mmmlu": ["m_mmlu", "mmmlu_direct", "mmmlu_dev"],
        "pawsx": ["paws_en", "paws_de", "paws_es", "paws_fr", "paws_ja", "paws_ko"],
        "xnli": ["xnli_en", "xnli_de", "xnli_es", "xnli_fr", "xnli_ru"],
        "xcopa": ["xcopa_en", "xcopa_et", "xcopa_ht", "xcopa_id", "xcopa_it"],
        "mmlu": ["mmlu_abstract_algebra", "mmlu_anatomy", "mmlu_astronomy"],
        "crows_pairs": ["crows_pairs_english", "crows_pairs_french"],
        "bigbench": [
            "bigbench_causal_judgement",
            "bigbench_date_understanding",
            "bigbench_disambiguation_qa",
        ],
    }

    alternatives = alternative_names.get(benchmark_name, [])
    alternatives.extend(alternative_names.get(original_task_name, []))
    alternatives = list(set(alternatives))
    if original_task_name in alternatives:
        alternatives.remove(original_task_name)

    if not alternatives:
        return None

    print(f"Trying {len(alternatives)} alternative names: {alternatives}")

    for alt_name in alternatives:
        print(f"   Trying alternative: {alt_name}")
        try:
            result = get_task_samples_for_analysis(
                alt_name, num_samples=num_samples, trust_remote_code=trust_remote_code
            )
            if result.get("samples"):
                print(f"   Success with alternative: {alt_name}")
                return result
        except Exception as e:
            print(f"   Alternative {alt_name} failed: {e}")
            continue

    return None


def get_task_samples_direct(task, num_samples: int = 5) -> dict:
    """Get samples directly from a task object."""
    try:
        if hasattr(task, "eval_docs") and callable(task.eval_docs):
            docs = list(task.eval_docs(task.dataset))
        elif hasattr(task, "test_docs") and callable(task.test_docs):
            docs = list(task.test_docs())
        elif hasattr(task, "validation_docs") and callable(task.validation_docs):
            docs = list(task.validation_docs())
        elif hasattr(task, "train_docs") and callable(task.train_docs):
            docs = list(task.train_docs())
        else:
            return {"error": "No documents found for this task"}

        if not docs:
            return {"error": "No documents found for this task"}

        docs = docs[:num_samples]
        samples: List[Dict] = []

        for doc in docs:
            try:
                question = (
                    task.doc_to_text(doc) if hasattr(task, "doc_to_text") else str(doc)
                )
                target = (
                    task.doc_to_target(doc) if hasattr(task, "doc_to_target") else ""
                )

                choices: List = []
                if hasattr(task, "doc_to_choice") and callable(task.doc_to_choice):
                    try:
                        choices = task.doc_to_choice(doc)
                    except Exception:
                        choices = []

                sample = {
                    "question": question,
                    "correct_answer": target,
                    "choices": choices,
                    "metadata": {
                        "task": task.config.task if hasattr(task, "config") else "unknown",
                        "source": "direct",
                    },
                }
                samples.append(sample)

            except Exception as e:
                print(f"   Error processing document: {e}")
                continue

        if not samples:
            return {"error": "No samples could be processed from documents"}

        return {
            "samples": samples,
            "task": task.config.task if hasattr(task, "config") else "unknown",
            "total_samples": len(samples),
        }

    except Exception as e:
        return {"error": f"Exception in get_task_samples_direct: {e}"}


def try_datasets_direct_load(
    task_name: str, num_samples: int = 5, trust_remote_code: bool = False
) -> dict:
    """Try loading dataset directly with datasets library."""
    try:
        import datasets

        os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"

        dataset_patterns = [
            task_name,
            f"hf-internal-testing/{task_name}",
            f"EleutherAI/{task_name}",
            f"bigscience/{task_name}",
            f"allenai/{task_name}",
        ]

        for pattern in dataset_patterns:
            try:
                if trust_remote_code:
                    dataset = datasets.load_dataset(pattern, trust_remote_code=True)
                else:
                    dataset = datasets.load_dataset(pattern)

                samples = []
                split_name = None

                for split in ["test", "validation", "train"]:
                    if split in dataset:
                        split_name = split
                        break

                if split_name:
                    split_data = dataset[split_name]
                    for i, example in enumerate(split_data):
                        if i >= num_samples:
                            break
                        samples.append(example)

                    return {
                        "samples": samples,
                        "task": task_name,
                        "total_samples": len(samples),
                        "split": split_name,
                        "source": "datasets_direct",
                    }

            except Exception as e:
                print(f"Failed to load {pattern}: {e}")
                continue

        return {"error": "No dataset patterns worked"}

    except Exception as e:
        return {"error": f"Exception in datasets direct load: {e}"}
