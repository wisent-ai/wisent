"""Benchmark relevance matching functions."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .descriptions import BENCHMARK_DESCRIPTIONS
from .filtering import apply_priority_filtering


__all__ = ["find_most_relevant_benchmarks"]


def _create_llm_prompt(prompt: str, benchmark_list: List[Dict]) -> str:
    """Create the prompt for LLM-based benchmark matching."""
    return f"""You are an expert AI researcher analyzing which benchmarks are most relevant for evaluating a given prompt.

USER PROMPT TO ANALYZE: "{prompt}"

AVAILABLE BENCHMARKS:
{json.dumps(benchmark_list, indent=2)}

TASK: Analyze the user prompt and identify the 3 most relevant benchmarks for evaluating this type of query. Consider:
1. What cognitive skills does the prompt require?
2. What domain knowledge is needed?
3. What type of reasoning or capabilities are being tested?

RESPONSE FORMAT (JSON only):
{{
    "analysis": "Brief analysis of what the prompt requires",
    "recommendations": [
        {{"benchmark": "benchmark_name", "relevance_score": 0.95, "reasoning": "Why this benchmark is relevant"}},
        {{"benchmark": "benchmark_name", "relevance_score": 0.85, "reasoning": "Why this benchmark is relevant"}},
        {{"benchmark": "benchmark_name", "relevance_score": 0.75, "reasoning": "Why this benchmark is relevant"}}
    ]
}}

Respond with JSON only, no additional text."""


def _fallback_semantic_matching(
    prompt: str,
    benchmarks: Dict[str, Dict],
    prefer_fast: bool,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Fallback to semantic keyword matching when LLM is unavailable."""
    benchmark_scores = []
    prompt_lower = prompt.lower()

    for benchmark_name, benchmark_config in benchmarks.items():
        score = 0
        reasons = []
        description = BENCHMARK_DESCRIPTIONS.get(benchmark_name, "")
        tags = benchmark_config.get("tags", [])
        description_lower = description.lower()

        # Content matching
        if any(word in prompt_lower for word in ["what is", "who is", "where is", "capital"]):
            if any(word in description_lower for word in ["knowledge", "factual", "trivia"]):
                score += 3
                reasons.append("factual knowledge match")

        if any(word in prompt_lower for word in ["code", "program", "python", "function"]):
            if any(word in description_lower for word in ["code", "programming", "python"]):
                score += 3
                reasons.append("programming match")

        if any(word in prompt_lower for word in ["math", "calculate", "solve", "number"]):
            if any(word in description_lower for word in ["math", "arithmetic", "calculation"]):
                score += 3
                reasons.append("mathematics match")

        if any(word in prompt_lower for word in ["doctor", "medicine", "health", "symptom"]):
            if any(word in description_lower for word in ["medical", "health", "clinical"]):
                score += 3
                reasons.append("medical match")

        # Tag matching
        for tag in tags:
            if tag == "general knowledge" and any(w in prompt_lower for w in ["what", "who", "where"]):
                score += 2
                reasons.append("general knowledge tag")
            elif tag == "coding" and any(w in prompt_lower for w in ["code", "program"]):
                score += 2
                reasons.append("coding tag")
            elif tag == "mathematics" and any(w in prompt_lower for w in ["math", "calculate"]):
                score += 2
                reasons.append("mathematics tag")
            elif tag == "medical" and any(w in prompt_lower for w in ["health", "medicine"]):
                score += 2
                reasons.append("medical tag")

        # Popular benchmark bonus
        popular = ["mmlu", "truthfulqa_mc1", "gsm8k", "humaneval", "hellaswag", "gpqa", "gpqa_main_zeroshot"]
        if benchmark_name in popular:
            score += 0.5
            reasons.append("popular benchmark")

        # Priority bonuses
        benchmark_priority = benchmark_config.get("priority", "unknown")
        if prefer_fast and benchmark_priority == "high":
            score += 1.0
            reasons.append("fast benchmark bonus")
        elif benchmark_priority == "high":
            score += 0.3
            reasons.append("high priority")
        elif benchmark_priority == "medium":
            score += 0.1
            reasons.append("medium priority")

        if score > 0:
            benchmark_scores.append({
                "benchmark": benchmark_name,
                "score": score,
                "reasons": reasons,
                "description": description,
                "tags": tags,
                "task": benchmark_config.get("task", benchmark_name),
                "groups": benchmark_config.get("groups", []),
                "priority": benchmark_config.get("priority", "unknown"),
                "loading_time": benchmark_config.get("loading_time", 60.0),
            })

    if prefer_fast:
        benchmark_scores.sort(key=lambda x: (x["score"], -x["loading_time"]), reverse=True)
    else:
        benchmark_scores.sort(key=lambda x: x["score"], reverse=True)

    return benchmark_scores[:top_k]


def find_most_relevant_benchmarks(
    prompt: str,
    benchmarks: Dict[str, Dict],
    top_k: int = 1,
    priority: str = "all",
    fast_only: bool = False,
    time_budget_minutes: float = None,
    prefer_fast: bool = False,
) -> List[Dict[str, Any]]:
    """Find the most relevant benchmarks for a given prompt using LLM analysis."""
    # Apply priority filtering
    if priority != "all" or fast_only or time_budget_minutes is not None:
        benchmarks = apply_priority_filtering(benchmarks, priority, fast_only, time_budget_minutes)

    # Build benchmark list for LLM
    benchmark_list = []
    for benchmark_name, benchmark_config in benchmarks.items():
        description = BENCHMARK_DESCRIPTIONS.get(benchmark_name, "")
        tags = benchmark_config.get("tags", [])
        benchmark_list.append({
            "name": benchmark_name,
            "description": description,
            "tags": tags,
        })

    llm_prompt = _create_llm_prompt(prompt, benchmark_list)

    try:
        from transformers import pipeline

        print("Initializing LLM for benchmark analysis...")
        generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            max_length=512,
            do_sample=True,
            temperature=0.3,
        )

        response = generator(llm_prompt, max_new_tokens=200, return_full_text=False)
        response_text = response[0]["generated_text"].strip()

        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                llm_response = json.loads(json_str)
            else:
                llm_response = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            print(f"Raw response: {response_text}")
            return []

        benchmark_results = []
        for rec in llm_response.get("recommendations", []):
            benchmark_name = rec.get("benchmark")
            if benchmark_name in benchmarks:
                benchmark_config = benchmarks[benchmark_name]
                description = BENCHMARK_DESCRIPTIONS.get(benchmark_name, "")

                base_score = rec.get("relevance_score", 0.0)
                priority_bonus = 0.0

                benchmark_priority = benchmark_config.get("priority", "unknown")
                if prefer_fast and benchmark_priority == "high":
                    priority_bonus += 0.15
                elif benchmark_priority == "high":
                    priority_bonus += 0.05
                elif benchmark_priority == "medium":
                    priority_bonus += 0.02

                adjusted_score = base_score + priority_bonus

                benchmark_results.append({
                    "benchmark": benchmark_name,
                    "score": adjusted_score,
                    "reasons": [rec.get("reasoning", "LLM recommendation")],
                    "description": description,
                    "tags": benchmark_config.get("tags", []),
                    "task": benchmark_config.get("task", benchmark_name),
                    "groups": benchmark_config.get("groups", []),
                    "priority": benchmark_config.get("priority", "unknown"),
                    "loading_time": benchmark_config.get("loading_time", 60.0),
                })

        if prefer_fast:
            benchmark_results.sort(key=lambda x: (x["score"], -x["loading_time"]), reverse=True)
        else:
            benchmark_results.sort(key=lambda x: x["score"], reverse=True)

        return benchmark_results[:top_k]

    except Exception as e:
        print(f"Error calling LLM: {e}")
        print("Falling back to semantic matching...")
        return _fallback_semantic_matching(prompt, benchmarks, prefer_fast, top_k)
