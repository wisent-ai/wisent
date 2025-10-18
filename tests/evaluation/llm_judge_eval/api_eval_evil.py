# Load results from results.json
# For each (layer, scale, aggregation) triple, read prompt, baseline_response, steered_response
# Insert them into the evaluation prompt template
# Save to prompt_instruction variable

import json

# Load the evaluation prompt template
with open("tests/llm_judge_eval/prompt_eval_evil.py") as f: #change
    prompt_template = f.read().strip('"""\n')

# Load results from results.json
with open("tests/llm_judge_eval/gen_results_evil.json") as f: #change
    results = json.load(f)

# Process each result
evaluation_prompts = []
for result in results:
    layer = result["layer"]
    scale = result["scale"]
    aggregation = result["aggregation method"]
    prompt = result["prompt"]
    baseline_response = result.get("baseline_response", result.get("unsteered_response", ""))
    steered_response = result["steered_response"]

    # Fill in the template
    prompt_instruction = prompt_template.format(
        question=prompt,
        baseline=baseline_response,
        steered=steered_response,
        trait="evil"
    )

    evaluation_prompts.append({
        "layer": layer,
        "scale": scale,
        "aggregation_method": aggregation,
        "prompt": prompt,
        "baseline_response": baseline_response,
        "steered_response": steered_response,
        "prompt_instruction": prompt_instruction
    })

# Use Anthropic API as judge
from anthropic import Anthropic
import re

client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment
print("Using Claude API as judge model")

# Run evaluation on each prompt
evaluation_results = []
for eval_prompt in evaluation_prompts:
    print(f"Evaluating: Layer={eval_prompt['layer']}, Scale={eval_prompt['scale']}, Aggregation={eval_prompt['aggregation_method']}")

    # Generate response from judge using Claude API
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": eval_prompt["prompt_instruction"]}]
        )
        judge_response = message.content[0].text
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        judge_response = ""

    # Try to extract JSON from the response
    try:
        # Find JSON object in response
        json_match = re.search(r'\{[^}]*"differentiation_score"[^}]*\}', judge_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            scores = json.loads(json_str)
            differentiation_score = scores.get("differentiation_score")
            coherence_score = scores.get("coherence_score")
            trait_alignment_score = scores.get("trait_alignment_score")
        else:
            # Fallback: extract numbers
            numbers = re.findall(r'\b\d+\.?\d*\b', judge_response)
            if len(numbers) >= 3:
                differentiation_score = float(numbers[0])
                coherence_score = float(numbers[1])
                trait_alignment_score = float(numbers[2])
            else:
                raise ValueError(f"Could not extract scores")
    except Exception as e:
        print(f"Warning: Error extracting scores: {e}")
        print(f"Response: {judge_response}")
        differentiation_score = None
        coherence_score = None
        trait_alignment_score = None

    overall_score = None
    if all([differentiation_score is not None, coherence_score is not None, trait_alignment_score is not None]):
        overall_score = 0.2 * differentiation_score + 0.3 * coherence_score + 0.5 * trait_alignment_score

    evaluation_results.append({
        "layer": eval_prompt["layer"],
        "scale": eval_prompt["scale"],
        "aggregation_method": eval_prompt["aggregation_method"],
        "prompt instruction": eval_prompt["prompt_instruction"],
        "baseline_response": eval_prompt["baseline_response"],
        "steered_response": eval_prompt["steered_response"],
        "differentiation_score": differentiation_score,
        "coherence_score": coherence_score,
        "trait_alignment_score": trait_alignment_score,
        "overall_score": overall_score,
        "judge_response": judge_response
    })

# Sort by overall_score in descending order (None values at the end)
evaluation_results.sort(key=lambda x: x["overall_score"] if x["overall_score"] is not None else -1, reverse=True)

# Save evaluation results
output_file = "tests/llm_judge_eval/api_evaluation_results_evil.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

print(f"Completed {len(evaluation_results)} evaluations and saved to {output_file}")