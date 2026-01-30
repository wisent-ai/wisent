"""LLM-based concept naming using wisent inference pipeline."""

import json
from typing import Dict, List, Any, Optional

# Global cache for Wisent instance
_wisent_cache = {}


def format_naming_prompt(
    prompts: List[str],
    positives: List[str],
    negatives: List[str],
    n_samples: int = 5,
) -> str:
    """Format a prompt for the LLM to name this concept cluster."""

    samples = []
    for i in range(min(n_samples, len(prompts))):
        samples.append(f"""Example {i+1}:
Question: {prompts[i]}
Truthful answer: {positives[i]}
False answer: {negatives[i]}""")

    samples_text = "\n\n".join(samples)

    return f"""You are analyzing a cluster of question-answer pairs from a truthfulness dataset.

These pairs all belong to the same "concept" - they share some common pattern in how truthful vs false answers differ.

Here are sample pairs from this cluster:

{samples_text}

Based on these examples, provide:
1. A short name (2-4 words, snake_case) that captures what makes this cluster distinctive
2. A one-sentence description of what pattern these pairs share

Respond in this exact JSON format:
{{"name": "your_name_here", "description": "Your description here"}}

Only output the JSON, nothing else. /no_think"""


def get_wisent_model(model_name: str):
    """Get or create cached Wisent instance."""
    if model_name in _wisent_cache:
        return _wisent_cache[model_name]

    from wisent.core.wisent import Wisent

    print(f"  Loading {model_name}...", flush=True)
    wisent = Wisent.for_text(model_name)
    _wisent_cache[model_name] = wisent
    return wisent


def call_local_llm(prompt: str, model_name: str = "Qwen/Qwen3-8B") -> str:
    """Generate text using wisent inference pipeline."""
    wisent = get_wisent_model(model_name)
    response = wisent.generate(prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()


def parse_llm_response(response: str) -> Dict[str, str]:
    """Parse the JSON response from the LLM."""
    import re

    # Strip thinking tags (Qwen3 format)
    response = response.strip()

    # Remove <think>...</think> blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    response = response.strip()

    # Look for JSON pattern
    start = response.find("{")
    end = response.rfind("}") + 1

    if start >= 0 and end > start:
        json_str = response[start:end]
        try:
            parsed = json.loads(json_str)
            name = parsed.get("name", "unknown_concept")
            # Clean up name - convert to snake_case if needed
            name = name.replace(" ", "_").replace("-", "_").lower()
            return {
                "name": name,
                "description": parsed.get("description", "No description available"),
            }
        except json.JSONDecodeError:
            pass

    # Fallback: try to extract name from response text
    # Look for patterns like "name": "something" or name: something
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', response)
    if name_match:
        name = name_match.group(1).replace(" ", "_").replace("-", "_").lower()
        desc_match = re.search(r'"description"\s*:\s*"([^"]+)"', response)
        desc = desc_match.group(1) if desc_match else response[:200]
        return {"name": name, "description": desc}

    return {
        "name": "unknown_concept",
        "description": response[:200] if response else "No description available",
    }


def name_concept_with_llm(
    prompts: List[str],
    positives: List[str],
    negatives: List[str],
    model: str = "Qwen/Qwen3-8B",
    debug: bool = False,
) -> Dict[str, str]:
    """Use LLM to generate a meaningful name for a concept cluster."""

    if not prompts:
        return {"name": "empty_cluster", "description": "No samples available"}

    prompt = format_naming_prompt(prompts, positives, negatives)

    try:
        response = call_local_llm(prompt, model)
        if debug:
            print(f"    LLM response: {response[:500]}...")
        return parse_llm_response(response)
    except Exception as e:
        return {
            "name": "naming_failed",
            "description": f"LLM naming failed: {str(e)}",
        }


def name_all_concepts_with_llm(
    concepts: List[Dict[str, Any]],
    pair_texts: Dict[int, Dict[str, str]],
    cluster_labels,
    model: str = "Qwen/Qwen3-8B",
) -> List[Dict[str, Any]]:
    """Name all concepts using LLM."""
    import numpy as np

    # Group pairs by concept
    pair_ids = sorted(pair_texts.keys())
    concept_pairs = {i: [] for i in range(len(concepts))}

    for idx, pair_id in enumerate(pair_ids):
        if idx < len(cluster_labels):
            cluster_id = cluster_labels[idx]
            if pair_id in pair_texts:
                concept_pairs[cluster_id].append(pair_texts[pair_id])

    # Name each concept
    for i, concept in enumerate(concepts):
        pairs = concept_pairs.get(i, [])

        if pairs:
            prompts = [p.get("prompt", "") for p in pairs]
            positives = [p.get("positive", "") for p in pairs]
            negatives = [p.get("negative", "") for p in pairs]

            print(f"  Naming concept {i+1} ({len(pairs)} pairs)...", flush=True)
            naming = name_concept_with_llm(prompts, positives, negatives, model)

            concept["name"] = naming["name"]
            concept["description"] = naming["description"]
            concept["llm_named"] = True
        else:
            concept["name"] = f"empty_concept_{i+1}"
            concept["description"] = "Empty concept cluster"
            concept["llm_named"] = False

    return concepts
