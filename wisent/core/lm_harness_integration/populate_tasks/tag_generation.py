"""LLM-based tag generation for benchmark tasks."""

from typing import Dict, Any, List, Optional

from wisent.core.utils import preferred_dtype, resolve_default_device, resolve_device


APPROVED_SKILLS = [
    "coding", "mathematics", "long context", "creative writing",
    "general knowledge", "medical", "law", "science", "history",
    "tool use", "multilingual", "reasoning"
]

APPROVED_RISKS = [
    "harmfulness", "toxicity", "bias", "hallucination", "violence",
    "adversarial robustness", "sycophancy", "deception"
]


def get_benchmark_tags_with_llama(task_name: str, readme_content: str = "") -> List[str]:
    """Use Llama-3.1B-Instruct to determine appropriate tags for a benchmark."""
    print(f"   Using Llama-3.1B-Instruct to determine tags for '{task_name}'...")

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
            max_new_tokens=1000,
            temperature=0.3,
            do_sample=True,
            pad_token_id=50256
        )

        print(f"   Successfully loaded Llama-3.1-8B-Instruct pipeline")

        description = readme_content[:1500] if readme_content else f"A benchmark called '{task_name}' for evaluating language models."

        user_prompt = f"""Analyze the benchmark and determine exactly 3 tags.

Benchmark: {task_name}
Description: {description}

Available tags:
Skills: {', '.join(APPROVED_SKILLS)}
Risks: {', '.join(APPROVED_RISKS)}

Instructions:
1. Analyze what this benchmark actually tests
2. Choose EXACTLY 3 tags that best describe what is being evaluated
3. Focus on the primary capabilities/risks being measured
4. Output only the 3 tags, one per line, no explanations

Tags:"""

        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert in AI evaluation benchmarks analyzing benchmark tasks to determine what specific cognitive abilities they test.<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        print("   Analyzing with Llama...")
        response = generator(formatted_prompt, max_new_tokens=800, temperature=0.3)

        full_response = response[0]['generated_text']
        generated_text = full_response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()

        print(f"   LLM Response: {generated_text}")

        all_approved_tags = APPROVED_SKILLS + APPROVED_RISKS
        lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
        determined_tags = []

        for line in lines[:5]:
            clean_line = line.strip('- *123456789.').strip()
            for tag in all_approved_tags:
                if tag.lower() == clean_line.lower() or clean_line.lower() in tag.lower():
                    if tag not in determined_tags:
                        determined_tags.append(tag)
                        break

        if len(determined_tags) < 3:
            default_tags = ["reasoning", "general knowledge", "science"]
            for default in default_tags:
                if default not in determined_tags:
                    determined_tags.append(default)
                if len(determined_tags) >= 3:
                    break

        determined_tags = determined_tags[:3]
        print(f"   Final LLM-determined tags: {determined_tags}")
        return determined_tags

    except Exception as e:
        print(f"   Error using LLM: {e}")
        print(f"   Using basic analysis...")
        return _basic_tag_analysis(readme_content)


def _basic_tag_analysis(readme_content: str) -> List[str]:
    """Basic content analysis for tag determination when LLM is unavailable."""
    if readme_content:
        content_lower = readme_content.lower()
        determined_tags = []

        if any(word in content_lower for word in ["math", "arithmetic", "calculation"]):
            determined_tags.append("mathematics")
        if any(word in content_lower for word in ["code", "programming", "python"]):
            determined_tags.append("coding")
        if any(word in content_lower for word in ["medical", "health", "clinical"]):
            determined_tags.append("medical")
        if any(word in content_lower for word in ["adversarial", "robust", "challenging"]):
            determined_tags.append("adversarial robustness")
        if any(word in content_lower for word in ["bias", "fairness", "stereotype"]):
            determined_tags.append("bias")
        if any(word in content_lower for word in ["truthful", "hallucination", "factual"]):
            determined_tags.append("hallucination")
        if any(word in content_lower for word in ["multilingual", "cross-lingual"]):
            determined_tags.append("multilingual")

        if "reasoning" not in determined_tags:
            determined_tags.append("reasoning")
        if len(determined_tags) < 3 and "general knowledge" not in determined_tags:
            determined_tags.append("general knowledge")
        if len(determined_tags) < 3:
            determined_tags.append("science")

        return determined_tags[:3]

    return ["reasoning", "general knowledge", "science"]


def get_benchmark_groups_from_readme(task_name: str) -> Dict[str, Any]:
    """Read README from lm-eval-harness repository and use LLM for tags."""
    import requests

    task_dir_map = {
        'superglue': 'super_glue', 'super_glue': 'super_glue', 'super-glue': 'super_glue',
        'glue': 'glue', 'mmlu': 'mmlu', 'truthfulqa': 'truthfulqa',
        'hellaswag': 'hellaswag', 'arc': 'arc', 'winogrande': 'winogrande'
    }

    task_dir = task_dir_map.get(task_name.lower(), task_name.lower())
    readme_url = f"https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/{task_dir}/README.md"

    try:
        print(f"   Fetching README from: {readme_url}")
        response = requests.get(readme_url, timeout=10)
        response.raise_for_status()
        readme_content = response.text

        determined_tags = get_benchmark_tags_with_llama(task_name, readme_content)
        groups = _extract_groups_from_readme(readme_content)

        print(f"   Extracted {len(groups)} groups from README: {groups}")
        print(f"   Determined tags from README content: {determined_tags}")

        return {'groups': groups, 'tags': determined_tags}

    except Exception as e:
        print(f"   Failed to fetch README: {e}")
        return {'groups': [], 'tags': []}


def _extract_groups_from_readme(readme_content: str) -> List[str]:
    """Extract group names from README content."""
    groups = []
    lines = readme_content.split('\n')
    in_target_section = False

    for line in lines:
        line = line.strip()

        if (line.lower().startswith('groups') or line.lower().startswith('## groups') or
            line.lower().startswith('tags') or line.lower().startswith('## tags') or
            line.lower().startswith('#### groups') or line.lower().startswith('#### tags')):
            in_target_section = True
            continue

        if in_target_section and line.startswith('##') and not any(x in line.lower() for x in ['groups', 'tags']):
            break

        if in_target_section and line:
            if line.startswith('* `') and '`' in line and ':' in line:
                start = line.find('`') + 1
                end = line.find('`', start)
                if start > 0 and end > start:
                    group_name = line[start:end].strip()
                    if group_name and '-' in group_name:
                        groups.append(group_name)
            elif ':' in line and not line.startswith('#'):
                group_name = line.split(':')[0].strip()
                group_name = group_name.lstrip('*- `').rstrip('`')
                if group_name and '-' in group_name:
                    groups.append(group_name)

    groups = list(set(groups))
    valid_groups = [g for g in groups if g and len(g) > 3]
    return valid_groups
