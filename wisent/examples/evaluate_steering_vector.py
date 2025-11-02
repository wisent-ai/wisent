"""
Example: Evaluate a steering control vector for personalization.

This script demonstrates how to use PersonalizationEvaluator to assess
the effectiveness of a steering control vector.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wisent.core.evaluators.personalization_evaluator import PersonalizationEvaluator


def main():
    """Evaluate a control vector for steering."""

    # 1. Load model and tokenizer
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Or your preferred model
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. Load or create a control vector
    # In practice, this would be loaded from your trained control vector
    # For demo purposes, we'll create a random vector
    hidden_size = model.config.hidden_size
    control_vector = torch.randn(hidden_size, dtype=torch.float16)

    # 3. Define the trait to evaluate
    trait_name = "British"
    trait_description = "Uses British English spelling, expressions, and cultural references"

    # 4. Create test prompts
    test_prompts = [
        "Tell me about your favorite food.",
        "What do you think about the weather today?",
        "Describe your morning routine.",
        "What's your opinion on public transportation?",
        "Tell me about your weekend plans.",
    ]

    # 5. Initialize evaluator
    print("Initializing PersonalizationEvaluator...")
    evaluator = PersonalizationEvaluator(
        model=model,
        tokenizer=tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 6. Evaluate the steering vector
    print(f"\nEvaluating steering for trait: {trait_name}")
    result = evaluator.evaluate_steering(
        control_vector=control_vector,
        trait_name=trait_name,
        trait_description=trait_description,
        test_prompts=test_prompts,
        steering_strength=1.0,
        max_new_tokens=100,
    )

    # 7. Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Trait: {result.trait_name}")
    print(f"\nScores:")
    print(f"  Difference:  {result.difference_score:.3f} (how different from baseline)")
    print(f"  Quality:     {result.quality_score:.3f} (coherence, not lobotomized)")
    print(f"  Alignment:   {result.alignment_score:.3f} (matches trait direction)")
    print(f"  Overall:     {result.overall_score:.3f} (weighted average)")

    print(f"\n\nSample Baseline Response:")
    print("-" * 60)
    print(result.baseline_response[:300] + "..." if len(result.baseline_response) > 300 else result.baseline_response)

    print(f"\n\nSample Steered Response:")
    print("-" * 60)
    print(result.steered_response[:300] + "..." if len(result.steered_response) > 300 else result.steered_response)

    # 8. Interpret results
    print("\n\nInterpretation:")
    print("-" * 60)

    if result.difference_score < 0.2:
        print("⚠️  Steering has minimal effect (responses too similar to baseline)")
    elif result.difference_score > 0.8:
        print("✓ Steering creates significant difference from baseline")
    else:
        print("✓ Steering creates moderate difference from baseline")

    if result.quality_score < 0.5:
        print("❌ Response quality is poor (may be lobotomized or repetitive)")
    elif result.quality_score > 0.8:
        print("✓ Response quality is good (coherent and non-repetitive)")
    else:
        print("⚠️  Response quality is acceptable but could be improved")

    if result.alignment_score < 0.4:
        print("❌ Steering is not aligned with the trait (wrong direction?)")
    elif result.alignment_score > 0.7:
        print("✓ Steering is well-aligned with the trait")
    else:
        print("⚠️  Steering shows some trait characteristics but not strong")

    if result.overall_score > 0.7:
        print("\n✅ Overall: This control vector is effective!")
    elif result.overall_score > 0.5:
        print("\n⚠️  Overall: This control vector is moderately effective")
    else:
        print("\n❌ Overall: This control vector needs improvement")


if __name__ == "__main__":
    main()
