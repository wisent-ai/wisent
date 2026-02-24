#!/usr/bin/env python3
"""
Benchmark matching: LLM-based prompt-to-benchmark relevance analysis.
Split from populate_tasks_backup.py to meet 300-line limit.
"""

import os
import sys
import re
from typing import Dict, Any, List, Optional

from wisent.core.constants import (DEFAULT_LAYER, LLAMA_PAD_TOKEN_ID, MAX_BENCHMARKS_SINGLE,
    TAG_ANALYSIS_MAX_NEW_TOKENS, TAG_GEN_MAX_NEW_TOKENS, TAG_GEN_TEMPERATURE)
from wisent.core.utils import preferred_dtype, resolve_default_device, resolve_device


def get_relevant_benchmarks_for_prompt(prompt: str, max_benchmarks: int = MAX_BENCHMARKS_SINGLE,
                                        existing_model=None) -> List[Dict[str, Any]]:
    """
    Use Llama-3.1B-Instruct to determine the most relevant benchmarks
    for testing a given prompt.

    Args:
        prompt: The prompt to analyze (e.g., "I like food")
        max_benchmarks: Maximum number of benchmarks to return (default: 1)
        existing_model: Optional pre-loaded model for generation

    Returns:
        List of dicts containing benchmark names and relevance explanations
    """
    available_benchmarks, benchmark_descriptions = _load_benchmark_list()
    print(f"Analyzing prompt to find most relevant benchmarks: '{prompt}'")
    print(f"Available benchmarks: {len(available_benchmarks)}")
    try:
        from transformers import pipeline
        import torch
        print(f"   Loading Llama-3.1-8B-Instruct pipeline...")
        device_kind = resolve_default_device()
        device_obj = resolve_device(device_kind)
        if device_kind == "cuda" and torch.cuda.is_available():
            print("   Using CUDA device")
        elif device_kind == "mps":
            print("   Using MPS device")
        else:
            print("   Using CPU device")
        torch_dtype = preferred_dtype(device_kind)
        device_map = "auto" if device_kind == "cuda" else None
        if device_kind == "cuda":
            pipeline_device = 0
        elif device_kind == "mps":
            pipeline_device = device_obj
        else:
            pipeline_device = -1
        generator = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch_dtype,
            device_map=device_map,
            device=pipeline_device,
            max_new_tokens=TAG_GEN_MAX_NEW_TOKENS,
            temperature=TAG_GEN_TEMPERATURE,
            do_sample=True,
            pad_token_id=LLAMA_PAD_TOKEN_ID
        )
        print(f"   Successfully loaded Llama-3.1-8B-Instruct pipeline")
        benchmark_list = "\n".join([
            f"- {name}: {desc}"
            for name, desc in benchmark_descriptions.items()
        ])
        user_prompt = f"""Analyze this prompt and determine which benchmarks would be most relevant for testing it.

Prompt to analyze: "{prompt}"

Available benchmarks:
{benchmark_list}

Instructions:
1. Analyze what cognitive abilities, knowledge, or skills this prompt would test
2. Match the prompt's requirements to the most relevant benchmarks
3. Choose the top {max_benchmarks} benchmarks that would best evaluate this type of prompt
4. Provide a brief explanation for each choice

Format your response as:
1. [benchmark_name]: [explanation]
2. [benchmark_name]: [explanation]
3. [benchmark_name]: [explanation]

Top {max_benchmarks} most relevant benchmarks:"""
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are an expert in AI evaluation benchmarks. Your task is to "
            f"analyze prompts and determine which benchmarks would be most "
            f"relevant for testing them."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        print("   Analyzing with Llama...")
        if existing_model is not None:
            response, _ = existing_model.generate(
                formatted_prompt, layer_index=DEFAULT_LAYER, max_new_tokens=TAG_ANALYSIS_MAX_NEW_TOKENS)
            generated_text = response.strip()
        else:
            response = generator(
                formatted_prompt, max_new_tokens=TAG_ANALYSIS_MAX_NEW_TOKENS, temperature=TAG_GEN_TEMPERATURE)
            full_response = response[0]['generated_text']
            generated_text = full_response.split(
                "<|start_header_id|>assistant<|end_header_id|>"
            )[-1].strip()
        print(f"   LLM Response: {generated_text}")
        relevant_benchmarks = _parse_benchmark_response(
            generated_text, available_benchmarks, max_benchmarks)
        if len(relevant_benchmarks) < max_benchmarks:
            relevant_benchmarks = _fill_with_general_benchmarks(
                relevant_benchmarks, available_benchmarks, max_benchmarks)
        print(f"   Found {len(relevant_benchmarks)} relevant benchmarks")
        for i, rb in enumerate(relevant_benchmarks, 1):
            print(f"   {i}. {rb['benchmark']}: {rb['explanation']}")
        return relevant_benchmarks[:max_benchmarks]
    except Exception as e:
        print(f"   Error using LLM: {e}")
        print(f"   Using basic analysis instead...")
        return _basic_prompt_matching(
            prompt, available_benchmarks, max_benchmarks)


def _load_benchmark_list():
    """Load the benchmark list from only_benchmarks or use a basic one."""
    try:
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        try:
            from only_benchmarks import BENCHMARKS
            available = list(BENCHMARKS.keys())
            descriptions = {
                name: f"{name}: {', '.join(info['tags'])}"
                for name, info in BENCHMARKS.items()
            }
        except ImportError:
            available = [
                "mmlu", "truthfulqa_mc1", "hellaswag", "arc_easy",
                "arc_challenge", "winogrande", "piqa", "boolq", "copa",
                "rte", "wsc", "wic", "multirc", "record", "drop", "squad",
                "coqa", "humaneval", "mbpp", "math", "gsm8k", "toxigen",
                "winobias", "stereoset"
            ]
            descriptions = {
                name: f"{name}: benchmark for language model evaluation"
                for name in available
            }
    except Exception:
        available = [
            "mmlu", "truthfulqa_mc1", "hellaswag", "arc_easy",
            "arc_challenge", "winogrande", "piqa", "boolq", "copa", "rte"
        ]
        descriptions = {
            name: f"{name}: benchmark for language model evaluation"
            for name in available
        }
    return available, descriptions


def _parse_benchmark_response(generated_text, available_benchmarks,
                               max_benchmarks):
    """Parse LLM response to extract benchmark recommendations."""
    relevant_benchmarks = []
    lines = generated_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or not any(char.isdigit() for char in line[:3]):
            continue
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) >= 2:
                benchmark_part = parts[0].strip()
                explanation = parts[1].strip()
                benchmark_part = re.sub(r'^\d+\.?\s*', '', benchmark_part)
                benchmark_part = re.sub(r'[\[\]]', '', benchmark_part)
                benchmark_name = benchmark_part.strip()
                matched = None
                for available in available_benchmarks:
                    if available.lower() == benchmark_name.lower():
                        matched = available
                        break
                    elif (benchmark_name.lower() in available.lower() or
                          available.lower() in benchmark_name.lower()):
                        matched = available
                        break
                if matched:
                    relevant_benchmarks.append({
                        'benchmark': matched,
                        'explanation': explanation,
                        'relevance_score': len(relevant_benchmarks) + 1
                    })
                    if len(relevant_benchmarks) >= max_benchmarks:
                        break
    return relevant_benchmarks


def _fill_with_general_benchmarks(existing, available_benchmarks,
                                    max_benchmarks):
    """Fill remaining slots with general-purpose benchmarks."""
    general = ["mmlu", "truthfulqa_mc1", "hellaswag"]
    result = list(existing)
    for gb in general:
        if (gb in available_benchmarks and
                not any(rb['benchmark'] == gb for rb in result)):
            result.append({
                'benchmark': gb,
                'explanation': 'General purpose benchmark for various prompts',
                'relevance_score': len(result) + 1
            })
            if len(result) >= max_benchmarks:
                break
    return result


def _basic_prompt_matching(prompt, available_benchmarks, max_benchmarks):
    """Basic content-based prompt matching when LLM is unavailable."""
    prompt_lower = prompt.lower()
    results = []
    if any(w in prompt_lower for w in ["food", "eat", "cook", "recipe", "restaurant"]):
        results.append({
            'benchmark': 'mmlu',
            'explanation': 'General knowledge benchmark including food-related questions',
            'relevance_score': 1
        })
    if any(w in prompt_lower for w in ["math", "calculate", "number", "equation"]):
        bm = 'math' if 'math' in available_benchmarks else 'mmlu'
        results.append({
            'benchmark': bm,
            'explanation': 'Mathematical reasoning benchmark',
            'relevance_score': 1
        })
    if any(w in prompt_lower for w in ["code", "program", "python", "function"]):
        bm = 'humaneval' if 'humaneval' in available_benchmarks else 'mmlu'
        results.append({
            'benchmark': bm,
            'explanation': 'Code generation benchmark',
            'relevance_score': 1
        })
    general = ["mmlu", "truthfulqa_mc1", "hellaswag"]
    for gb in general:
        if (gb in available_benchmarks and
                not any(fr['benchmark'] == gb for fr in results)):
            results.append({
                'benchmark': gb,
                'explanation': 'General purpose benchmark for language understanding',
                'relevance_score': len(results) + 1
            })
            if len(results) >= max_benchmarks:
                break
    return results[:max_benchmarks]
