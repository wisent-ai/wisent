#!/usr/bin/env python3
"""Generate capability pairs using gpt-oss-20b via Together AI API."""

import json
import os
import time
import requests

CAPABILITIES = {
    "coding": "writing correct, complete, and well-structured code that solves programming problems accurately",
    "mathematics": "solving mathematical problems with accurate numerical reasoning and step-by-step derivations",
    "reasoning_logic": "performing logical deduction, causal reasoning, and multi-step planning",
    "hallucination_factuality": "providing factually accurate information without making up false claims",
    "safety_bias": "avoiding harmful content, stereotypes, and biased responses",
    "multilingual": "understanding and generating text accurately across multiple languages",
    "knowledge_qa": "answering questions accurately using world knowledge and factual recall",
    "reading_comprehension": "extracting and inferring information from provided text passages",
    "commonsense_reasoning": "applying everyday knowledge about physical, social, and temporal reasoning",
    "science_medical": "providing accurate scientific and medical domain knowledge",
    "instruction_following": "following complex instructions and adhering to specified constraints",
    "tool_use_agents": "selecting and using tools appropriately to complete tasks",
    "language_understanding": "demonstrating syntactic, semantic, and pragmatic language competence",
    "translation": "translating text accurately between languages while preserving meaning",
    "ethics_values": "reasoning about ethical dilemmas and demonstrating value alignment",
}

TARGET_COUNT = 100
MODEL_NAME = "openai/gpt-oss-20b"
FILENAME = "gpt_oss_20b_pairs.json"
API_URL = "https://api.together.xyz/v1/chat/completions"


def call_api(api_key: str, messages: list, max_tokens: int = 2000, temperature: float = 0.7) -> str:
    """Call Together AI API.

    Note: gpt-oss-20b is a reasoning model that uses chain-of-thought in a 'reasoning' field.
    It needs high max_tokens to complete reasoning before generating content.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"].strip()
    # If content is empty but there's reasoning, the model ran out of tokens
    if not content:
        reasoning = result["choices"][0]["message"].get("reasoning", "")
        if reasoning:
            # Return empty to signal retry needed
            return ""
    return content


def generate_opposite_trait(api_key: str, trait: str) -> str:
    """Generate the opposite of a trait."""
    messages = [{
        "role": "user",
        "content": f"What is the OPPOSITE personality trait of: {trait}?\n\nDescribe the opposite in one sentence, be specific about what words/style/tone to use."
    }]
    return call_api(api_key, messages, max_tokens=2000)


def generate_pair(api_key: str, trait: str, opposite_trait: str) -> dict | None:
    """Generate a single contrastive pair."""
    # Generate prompt
    prompt_messages = [{
        "role": "user",
        "content": "Write one short question a user might ask. Example: 'What is your favorite hobby?' Just the question, nothing else."
    }]
    prompt = call_api(api_key, prompt_messages, max_tokens=2000, temperature=0.9)

    if not prompt:
        return None

    # Generate positive response
    positive_messages = [{
        "role": "user",
        "content": f"Question: {prompt}\n\nAnswer the question AS IF you have this personality: {trait}\n\nWrite 1-2 sentences showing this personality clearly. Just the answer."
    }]
    positive = call_api(api_key, positive_messages)

    if not positive:
        return None

    # Generate negative response
    negative_messages = [{
        "role": "user",
        "content": f"Question: {prompt}\n\nAnswer the question AS IF you have this personality: {opposite_trait}\n\nWrite 1-2 sentences showing this personality clearly. Just the answer."
    }]
    negative = call_api(api_key, negative_messages)

    if not negative:
        return None

    return {
        "prompt": prompt,
        "positive_response": {
            "model_response": positive,
            "layers_activations": None,
            "label": None
        },
        "negative_response": {
            "model_response": negative,
            "layers_activations": None,
            "label": None
        },
        "label": trait[:50],
        "trait_description": trait,
        "metadata": None
    }


def generate_for_capability(api_key: str, capability: str, trait: str, base_dir: str):
    """Generate pairs for a single capability."""
    filepath = os.path.join(base_dir, capability, FILENAME)

    # Check if already exists
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        if len(data.get('pairs', [])) >= TARGET_COUNT:
            print(f"✓ {capability}: already has {len(data['pairs'])} pairs, skipping")
            return

    print(f"\n{'='*60}")
    print(f"Generating {capability}")
    print(f"Trait: {trait[:60]}...")
    print(f"{'='*60}")

    # Generate opposite trait
    print("Generating opposite trait...")
    opposite_trait = generate_opposite_trait(api_key, trait)
    print(f"Opposite: {opposite_trait[:60]}...")

    # Generate pairs
    pairs = []
    attempts = 0
    max_attempts = TARGET_COUNT * 5  # Higher multiplier for reasoning model

    while len(pairs) < TARGET_COUNT and attempts < max_attempts:
        attempts += 1
        print(f"\rGenerating pair {len(pairs)+1}/{TARGET_COUNT} (attempt {attempts})...", end="", flush=True)

        try:
            pair = generate_pair(api_key, trait, opposite_trait)
            if pair:
                pairs.append(pair)
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(2)  # Rate limit backoff
            continue

        # Small delay to avoid rate limits
        time.sleep(0.2)

    print(f"\nGenerated {len(pairs)} pairs")

    # Save
    save_data = {
        'model': 'gpt-oss-20b',
        'trait_description': trait,
        'trait_label': trait[:50],
        'num_pairs': len(pairs),
        'requested': TARGET_COUNT,
        'kept_after_dedupe': len(pairs),
        'pairs': pairs
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"✓ Saved to {filepath}")


def main():
    # Initialize Together AI client
    api_key = os.environ.get('TOGETHER_API_KEY')
    if not api_key:
        print("ERROR: TOGETHER_API_KEY not set")
        print("Please set it with: export TOGETHER_API_KEY=your_key")
        print("Get a key at: https://api.together.xyz/")
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Generating pairs for {len(CAPABILITIES)} capabilities using gpt-oss-20b via Together AI")
    print(f"Target: {TARGET_COUNT} pairs each")

    for capability, trait in CAPABILITIES.items():
        generate_for_capability(api_key, capability, trait, base_dir)

    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
