"""
Example usage of evaluate() from one_eval.py
"""

from one_eval import evaluate

trait = "happy"
prompt_format = "txt" #txt, markdown, json
question = "Tell me about monkeys"
baseline_response = "Monkeys are ok"
steered_response = "Monkeys are great, I love monkeys!"

results = evaluate(trait, prompt_format, question, baseline_response, steered_response)

print("Results:")
print(f"Differentiation Score: {results['differentiation_score']}")
print(f"Coherence Score: {results['coherence_score']}")
print(f"Trait Alignment Score: {results['trait_alignment_score']}")
print(f"Overall Score: {results['overall_score']}")
print(f"Open Traits: {results['open_traits']}")
print(f"Choose Result: {results['choose_result']}")

print("\nExplanations:")
for metric, explanation in results['explanations'].items():
    print(f"{metric}: {explanation}\n")