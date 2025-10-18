"""
Function which takes trait, prompt format, question, baseline response, and steered response
and return differentiation score, coherence score, trait alignment score, 3 traits which best describe steered response, choice
which response is steered, explanations of scoring, and overall score.
"""

from anthropic import Anthropic
import json
import re

def evaluate(trait, prompt_format, question, baseline_response, steered_response):

    # Build prompt
    metric_configs = {
            "differentiation": {
                "question": question,
                "response_A": baseline_response,
                "response_B": steered_response
            },
            "coherence": {
                "question": question,
                "response": steered_response
            },
            "trait_alignment": {
                "question": question,
                "response": steered_response
            },
            "open": {
                "question": question,
                "response": steered_response
            },
            "choose": {
                "question": question,
                "response_A": baseline_response,
                "response_B": steered_response
            }
        }
    
    format_extensions = {
    "txt": ".txt",
    "markdown": "_markdown.txt",
    "json": "_json.txt"
  }

    # Build prompts for all metrics
    prompts = {}
    for metric_name, kwargs in metric_configs.items():

        prompt_format_path = f"tests/one_eval/instruction_prompts/{trait if metric_name in ['trait_alignment', 'choose'] else 'general'}/{metric_name}{format_extensions[prompt_format]}"
        
        with open(prompt_format_path, 'r') as f:
            template = f.read().strip()

        result = template
        for key, value in kwargs.items():
              result = result.replace(f"<<<{key}>>>", str(value))

        prompts[metric_name] = result

    client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment
    print("Using Claude API as judge model")

    judge_responses = {}
    for metric, prompt in prompts.items():

        try:
            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            judge_responses[metric] = message.content[0].text
        except Exception as e:
            print(f"Error calling Claude API for {metric}: {e}")
            judge_responses[metric] = ""

    # Extract all results
    results = {}
    explanations = {}

    for metric_name, response in judge_responses.items():
        results[metric_name] = None
        explanations[metric_name] = None

        json_match = re.search(r'\{.*?\}', response, re.DOTALL)
        if json_match:
            try:
                json_obj = json.loads(json_match.group(0))

                # Extract score/value based on metric type
                if metric_name == "differentiation" and "differentiation_score" in json_obj:
                    results[metric_name] = float(json_obj["differentiation_score"])
                elif metric_name == "coherence" and "coherence_score" in json_obj:
                    results[metric_name] = float(json_obj["coherence_score"])
                elif metric_name == "trait_alignment" and "trait_alignment_score" in json_obj:
                    results[metric_name] = float(json_obj["trait_alignment_score"])
                elif metric_name == "open" and "traits" in json_obj:
                    results[metric_name] = json_obj["traits"]
                elif metric_name == "choose" and "choice" in json_obj:
                    results[metric_name] = str(json_obj["choice"]).strip().upper()

                # Extract explanation
                if "explanation" in json_obj:
                    explanations[metric_name] = json_obj["explanation"]
            except:
                pass

    # Unpack results
    differentiation_score = results.get("differentiation")
    coherence_score = results.get("coherence")
    trait_alignment_score = results.get("trait_alignment")
    open_traits = results.get("open")
    choose_result = results.get("choose")

    # Calculate overall score (weighted average)
    # Weights: differentiation=0.2, coherence=0.3, trait_alignment=0.5
    if all([differentiation_score is not None, coherence_score is not None, trait_alignment_score is not None]):
        overall_score = (0.2 * differentiation_score + 0.3 * coherence_score + 0.5 * trait_alignment_score)
    else:
        overall_score = None 

    # Return results
    return {
        "differentiation_score": differentiation_score,
        "coherence_score": coherence_score,
        "trait_alignment_score": trait_alignment_score,
        "overall_score": overall_score,
        "open_traits": open_traits,
        "choose_result": choose_result,
        "explanations": explanations
    }

