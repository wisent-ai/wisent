import json
from anthropic import Anthropic
from utils.json_extraction import extract_score, extract_traits, extract_choice
from utils.build_prompt import build_prompt_from_template

config = {
    "happy": {
        "instruction_prompt": {
            "differentiation": "tests/EVAL/instruction_prompts/general/differentiation",
            "coherence": "tests/EVAL/instruction_prompts/general/coherence",
            "trait_alignment": "tests/EVAL/instruction_prompts/happy/trait_alignment",
            "open": "tests/EVAL/instruction_prompts/general/open",
            "choose": "tests/EVAL/instruction_prompts/happy/choose"
        },
        "results": "tests/EVAL/output/happy_output.json",
        "output_best_answers": "tests/EVAL/output/happy_scores.json",
        "output_best_lsa": "tests/EVAL/output/happy_stats.json"
    },
    "evil": {
        "instruction_prompt": {
            "differentiation": "tests/EVAL/instruction_prompts/general/differentiation",
            "coherence": "tests/EVAL/instruction_prompts/general/coherence",
            "trait_alignment": "tests/EVAL/instruction_prompts/evil/trait_alignment",
            "open": "tests/EVAL/instruction_prompts/general/open",
            "choose": "tests/EVAL/instruction_prompts/evil/choose"
        },
        "results": "tests/EVAL/output/evil_output.json",
        "output_best_answers": "tests/EVAL/output/evil_scores.json",
        "output_best_lsa": "tests/EVAL/output/evil_stats.json"
    }
}

formats = ["txt", "markdown", "json"]
format_extensions = {"txt": ".txt", "markdown": "_markdown.txt", "json": "_json.txt"}

client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment
print("Using Claude API as judge model")

for trait_name, trait in config.items():

    with open(trait["results"]) as f:
        results = json.load(f)

    for format_type in formats:
        print(f"\n{'='*60}")
        print(f"Processing {trait_name} with format: {format_type}")
        print(f"{'='*60}\n")

        # Load the evaluation prompt templates with the current format
        prompt_templates = {}
        for metric_name, prompt_path in trait["instruction_prompt"].items():
            # Add format extension to the prompt path
            prompt_path_formatted = f"{prompt_path}{format_extensions[format_type]}"
            with open(prompt_path_formatted) as f:
                # All formats are now plain text files
                prompt_templates[metric_name] = f.read().strip()

        # Process each result
        evaluation_prompts = []
        for result in results:
            layer = result["layer"]
            strength = result["strength"]
            aggregation = result["aggregation method"]
            question = result["question"]
            baseline_response = result.get("baseline_response", result.get("unsteered_response", ""))
            steered_response = result["steered_response"]

            # Fill in each template with appropriate placeholders
            differentiation_instruction = build_prompt_from_template(
                prompt_templates["differentiation"],
                format_type,
                question=question,
                response_a=baseline_response,
                response_b=steered_response
            )

            coherence_instruction = build_prompt_from_template(
                prompt_templates["coherence"],
                format_type,
                question=question,
                response=steered_response
            )

            trait_alignment_instruction = build_prompt_from_template(
                prompt_templates["trait_alignment"],
                format_type,
                question=question,
                response=steered_response
            )

            open_instruction = build_prompt_from_template(
                prompt_templates["open"],
                format_type,
                question=question,
                response=steered_response
            )

            choose_instruction = build_prompt_from_template(
                prompt_templates["choose"],
                format_type,
                question=question,
                response_a=baseline_response,
                response_b=steered_response
            )

            evaluation_prompts.append({
                "layer": layer,
                "strength": strength,
                "aggregation_method": aggregation,
                "question": question,
                "baseline_response": baseline_response,
                "steered_response": steered_response,
                "differentiation_instruction": differentiation_instruction,
                "coherence_instruction": coherence_instruction,
                "trait_alignment_instruction": trait_alignment_instruction,
                "open_instruction": open_instruction,
                "choose_instruction": choose_instruction
            })

        # Run evaluation on each prompt
        evaluation_results = []
        for eval_prompt in evaluation_prompts:
            print(f"Evaluating: Layer={eval_prompt['layer']}, Strength={eval_prompt['strength']}, Aggregation={eval_prompt['aggregation_method']}")

            # Generate responses from judge using Claude API for each metric
            judge_responses = {}
            for metric in ["differentiation", "coherence", "trait_alignment", "open", "choose"]:
                try:
                    message = client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=512,
                        messages=[{"role": "user", "content": eval_prompt[f"{metric}_instruction"]}]
                    )
                    judge_responses[metric] = message.content[0].text
                except Exception as e:
                    print(f"Error calling Claude API for {metric}: {e}")
                    judge_responses[metric] = ""


            # Extract scores from each metric response
            differentiation_score = extract_score(judge_responses["differentiation"], "differentiation_score")
            coherence_score = extract_score(judge_responses["coherence"], "coherence_score")
            trait_alignment_score = extract_score(judge_responses["trait_alignment"], "trait_alignment_score")
            open_traits = extract_traits(judge_responses["open"])
            choose_result = extract_choice(judge_responses["choose"])

            # Print warnings for failed extractions
            if differentiation_score is None:
                print(f"Warning: Could not extract differentiation score")
            if coherence_score is None:
                print(f"Warning: Could not extract coherence score")
            if trait_alignment_score is None:
                print(f"Warning: Could not extract trait alignment score")
            if open_traits is None:
                print(f"Warning: Could not extract open traits")
            if choose_result is None:
                print(f"Warning: Could not extract choose result")

            overall_score = None
            if all([differentiation_score is not None, coherence_score is not None, trait_alignment_score is not None]):
                overall_score = 0.2 * differentiation_score + 0.3 * coherence_score + 0.5 * trait_alignment_score

            evaluation_results.append({
                "layer": eval_prompt["layer"],
                "strength": eval_prompt["strength"],
                "aggregation_method": eval_prompt["aggregation_method"],
                "question": eval_prompt["question"],
                "baseline_response": eval_prompt["baseline_response"],
                "steered_response": eval_prompt["steered_response"],
                "differentiation_score": differentiation_score,
                "coherence_score": coherence_score,
                "trait_alignment_score": trait_alignment_score,
                "open_traits": open_traits,
                "choose_result": choose_result,
                "overall_score": overall_score,
                "judge_responses": judge_responses
            })

        # Sort evaluation results by overall_score in descending order (highest first)
        evaluation_results.sort(key=lambda x: x["overall_score"] if x["overall_score"] is not None else -float('inf'), reverse=True)

        # Calculate average overall score for each (layer, strength, aggregation) combination
        from collections import defaultdict

        stats_by_config = defaultdict(lambda: {
            "scores": [],
            "differentiation_scores": [],
            "coherence_scores": [],
            "trait_alignment_scores": [],
            "choose_correct": 0,
            "choose_incorrect": 0,
            "choose_equal": 0,
            "choose_total": 0
        })

        for result in evaluation_results:
            key = (result["layer"], result["strength"], result["aggregation_method"])

            if result["overall_score"] is not None:
                stats_by_config[key]["scores"].append(result["overall_score"])
            if result["differentiation_score"] is not None:
                stats_by_config[key]["differentiation_scores"].append(result["differentiation_score"])
            if result["coherence_score"] is not None:
                stats_by_config[key]["coherence_scores"].append(result["coherence_score"])
            if result["trait_alignment_score"] is not None:
                stats_by_config[key]["trait_alignment_scores"].append(result["trait_alignment_score"])

            # Track choose results (B = steered, A = baseline)
            # Correct means the model chose B (steered response embodies trait more)
            if result["choose_result"] is not None:
                stats_by_config[key]["choose_total"] += 1
                if result["choose_result"] == "B":
                    stats_by_config[key]["choose_correct"] += 1
                elif result["choose_result"] == "A":
                    stats_by_config[key]["choose_incorrect"] += 1
                elif result["choose_result"] == "EQUAL":
                    stats_by_config[key]["choose_equal"] += 1

        # Calculate averages and format statistics
        config_statistics = []
        for (layer, strength, aggregation), stats in stats_by_config.items():
            avg_overall = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None
            avg_diff = sum(stats["differentiation_scores"]) / len(stats["differentiation_scores"]) if stats["differentiation_scores"] else None
            avg_coh = sum(stats["coherence_scores"]) / len(stats["coherence_scores"]) if stats["coherence_scores"] else None
            avg_trait = sum(stats["trait_alignment_scores"]) / len(stats["trait_alignment_scores"]) if stats["trait_alignment_scores"] else None

            config_statistics.append({
                "layer": layer,
                "strength": strength,
                "aggregation_method": aggregation,
                "avg_overall_score": avg_overall,
                "avg_differentiation_score": avg_diff,
                "avg_coherence_score": avg_coh,
                "avg_trait_alignment_score": avg_trait,
                "choose_correct": stats["choose_correct"],
                "choose_incorrect": stats["choose_incorrect"],
                "choose_equal": stats["choose_equal"],
                "choose_total": stats["choose_total"]
            })

        # Sort config statistics by avg_overall_score
        config_statistics.sort(key=lambda x: x["avg_overall_score"] if x["avg_overall_score"] is not None else -float('inf'), reverse=True)

        # Save evaluation results to output file with format in filename
        output_scores = trait["output_best_answers"].replace('.json', f'_{format_type}.json')
        with open(output_scores, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        # Save config statistics to separate file with format in filename
        output_stats = trait["output_best_lsa"].replace('.json', f'_{format_type}.json')
        with open(output_stats, "w", encoding="utf-8") as f:
            json.dump(config_statistics, f, indent=2, ensure_ascii=False)

        print(f"\nEvaluation complete for {trait_name} with {format_type} format.")
        print(f"Results saved to {output_scores} and {output_stats}")