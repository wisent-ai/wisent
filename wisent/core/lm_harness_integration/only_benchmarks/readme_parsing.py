"""README parsing functions for benchmark configuration."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from .constants import LM_EVAL_TASKS_PATH


__all__ = [
    "extract_readme_info",
    "determine_skill_risk_tags",
    "update_benchmark_from_readme",
    "update_all_benchmarks_from_readme",
]


def extract_readme_info(benchmark_name: str) -> Dict[str, Any]:
    """Extract groups and tags from README.md file for a benchmark."""
    readme_path = Path(LM_EVAL_TASKS_PATH) / benchmark_name / "README.md"

    result: Dict[str, Any] = {"groups": [], "tags": [], "tasks": []}

    if not readme_path.exists():
        print(f"   No README.md found for {benchmark_name}")
        return result

    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract Groups section
        groups_match = re.search(r"#### Groups\s*\n(.*?)(?=####|$)", content, re.DOTALL)
        if groups_match:
            groups_text = groups_match.group(1).strip()
            if groups_text.lower() not in ("none.", "none"):
                for line in groups_text.split("\n"):
                    line = line.strip()
                    if line and not line.startswith("*") and not line.startswith("-"):
                        continue
                    group_name = re.sub(r"[*-]\s*`?([^`]+)`?.*", r"\1", line)
                    if group_name and group_name != line:
                        result["groups"].append(group_name.strip())

        # Extract Tags section
        tags_match = re.search(r"#### Tags\s*\n(.*?)(?=####|$)", content, re.DOTALL)
        if tags_match:
            tags_text = tags_match.group(1).strip()
            for line in tags_text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                tag_match = re.search(r"[*-]\s*`([^`]+)`", line)
                if tag_match:
                    result["tags"].append(tag_match.group(1))

        # Extract Tasks section
        tasks_match = re.search(r"#### Tasks\s*\n(.*?)(?=####|$)", content, re.DOTALL)
        if tasks_match:
            tasks_text = tasks_match.group(1).strip()
            for line in tasks_text.split("\n"):
                line = line.strip()
                if line.startswith("*") and not line.startswith("* `"):
                    task_name = re.sub(r"\*\s*`?([^`\s\[\]]+)`?.*", r"\1", line)
                    if task_name and task_name != line and len(task_name) > 1:
                        result["tasks"].append(task_name.strip())
                elif line.startswith("* `") and "`" in line:
                    task_match = re.search(r"\* `([^`]+)`", line)
                    if task_match:
                        task_name = task_match.group(1)
                        if task_name and len(task_name) > 1:
                            result["tasks"].append(task_name.strip())

        tasks_preview = result["tasks"][:3]
        tasks_str = f"{tasks_preview}{'...' if len(result['tasks']) > 3 else ''}"
        print(
            f"   {benchmark_name}: Groups={result['groups']}, "
            f"Tags={result['tags']}, Tasks={tasks_str}"
        )

    except Exception as e:
        print(f"   Error reading README for {benchmark_name}: {e}")

    return result


def determine_skill_risk_tags(benchmark_name: str, readme_content: str = "") -> List[str]:
    """Determine appropriate skill and risk tags based on benchmark name."""
    name_lower = benchmark_name.lower()
    determined_tags: List[str] = []

    # Skill detection
    if any(word in name_lower for word in ["math", "arithmetic", "gsm"]):
        determined_tags.append("mathematics")
    elif any(word in name_lower for word in ["code", "human", "mbpp"]):
        determined_tags.append("coding")
    elif any(word in name_lower for word in ["med", "pubmed", "head"]):
        determined_tags.append("medical")
    elif any(word in name_lower for word in ["law", "legal"]):
        determined_tags.append("law")
    elif any(word in name_lower for word in ["science", "sci", "arc"]):
        determined_tags.append("science")
    elif any(word in name_lower for word in ["history", "historical"]):
        determined_tags.append("history")
    elif any(word in name_lower for word in ["multi", "xnli", "xlang"]):
        determined_tags.append("multilingual")
    elif any(word in name_lower for word in ["long", "context", "scroll"]):
        determined_tags.append("long context")
    elif any(word in name_lower for word in ["creative", "story"]):
        determined_tags.append("creative writing")
    elif any(word in name_lower for word in ["tool", "use"]):
        determined_tags.append("tool use")

    if "reasoning" not in determined_tags:
        determined_tags.append("reasoning")

    if any(word in name_lower for word in ["mmlu", "trivia", "qa", "question"]):
        if "general knowledge" not in determined_tags:
            determined_tags.append("general knowledge")

    # Risk detection
    if any(word in name_lower for word in ["truth", "truthful"]):
        determined_tags.append("hallucination")
    elif any(word in name_lower for word in ["toxigen", "toxic"]):
        determined_tags.append("toxicity")
    elif any(word in name_lower for word in ["bias", "crows"]):
        determined_tags.append("bias")
    elif any(word in name_lower for word in ["adversarial", "anli"]):
        determined_tags.append("adversarial robustness")
    elif any(word in name_lower for word in ["harm", "ethics"]):
        determined_tags.append("harmfulness")
    elif any(word in name_lower for word in ["violence", "violent"]):
        determined_tags.append("violence")
    elif any(word in name_lower for word in ["deception", "deceive"]):
        determined_tags.append("deception")
    elif any(word in name_lower for word in ["sycophancy"]):
        determined_tags.append("sycophancy")

    return determined_tags[:3]


def update_benchmark_from_readme(benchmark_name: str, current_config: Dict) -> Dict:
    """Update a benchmark configuration based on its README file."""
    benchmark_dir = None
    potential_dirs = [
        benchmark_name,
        benchmark_name.replace("_", ""),
        benchmark_name.replace("_", "-"),
        benchmark_name.replace("-", "_"),
        current_config.get("task", benchmark_name),
        "super_glue" if benchmark_name == "superglue" else benchmark_name,
        "truthfulqa" if benchmark_name.startswith("truthfulqa") else benchmark_name,
        "arc" if benchmark_name.startswith("arc_") else benchmark_name,
        "ai2_arc" if benchmark_name.startswith("ai2_arc") else benchmark_name,
        benchmark_name.split("_")[0] if "_" in benchmark_name else benchmark_name,
    ]

    for potential_dir in potential_dirs:
        if (Path(LM_EVAL_TASKS_PATH) / potential_dir).exists():
            benchmark_dir = potential_dir
            break

    if not benchmark_dir:
        print(f"   Directory not found for {benchmark_name}, using auto-generated tags")
        tags = determine_skill_risk_tags(benchmark_name)
        return {
            "task": current_config.get("task", benchmark_name),
            "tags": tags,
            "groups": [],
            "readme_tasks": [],
        }

    readme_info = extract_readme_info(benchmark_dir)

    if readme_info["tags"]:
        return {
            "task": current_config.get("task", benchmark_name),
            "tags": determine_skill_risk_tags(benchmark_name),
            "groups": readme_info["tags"],
            "readme_tasks": readme_info["tasks"],
        }

    tags = determine_skill_risk_tags(benchmark_name)
    return {
        "task": current_config.get("task", benchmark_name),
        "tags": tags,
        "groups": readme_info["groups"],
        "readme_tasks": readme_info["tasks"],
    }


def update_all_benchmarks_from_readme(core_benchmarks: Dict) -> Dict:
    """Update all benchmarks with information from their README files."""
    print("Updating all benchmarks with README information...")

    updated_benchmarks = {}
    for benchmark_name, config in core_benchmarks.items():
        print(f"Processing {benchmark_name}...")
        updated_config = update_benchmark_from_readme(benchmark_name, config)
        updated_benchmarks[benchmark_name] = updated_config

    print(f"Updated {len(updated_benchmarks)} benchmarks")
    return updated_benchmarks
