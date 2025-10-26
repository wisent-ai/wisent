"""Generate responses command execution logic."""

import json
import os
import sys


def execute_generate_responses(args):
    """
    Execute the generate-responses command.

    Generates model responses to questions from a task and saves them to a file.
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader

    print(f"\n{'='*80}")
    print(f"🎯 GENERATING RESPONSES FROM TASK")
    print(f"{'='*80}")
    print(f"   Task: {args.task}")
    print(f"   Model: {args.model}")
    print(f"   Num questions: {args.num_questions}")
    print(f"   Device: {args.device or 'auto'}")
    print(f"{'='*80}\n")

    # Load model
    print(f"📦 Loading model...")
    model = WisentModel(args.model, device=args.device)
    print(f"   ✓ Model loaded\n")

    # Load task data
    print(f"📊 Loading task data...")
    loader = LMEvalDataLoader()
    try:
        result = loader._load_one_task(
            task_name=args.task,
            split_ratio=0.8,
            seed=42,
            limit=args.num_questions,
            training_limit=None,
            testing_limit=None
        )

        # Use test pairs for generation
        pairs = result['test_qa_pairs'].pairs[:args.num_questions]
        print(f"   ✓ Loaded {len(pairs)} question pairs\n")

    except Exception as e:
        print(f"   ❌ Failed to load task: {e}")
        sys.exit(1)

    # Generate responses
    print(f"🤖 Generating responses...\n")
    results = []

    for idx, pair in enumerate(pairs, 1):
        if args.verbose:
            print(f"Question {idx}/{len(pairs)}:")
            print(f"   Prompt: {pair.prompt[:100]}...")

        try:
            # Convert prompt to chat format
            messages = [
                {"role": "user", "content": pair.prompt}
            ]

            # Generate response
            responses = model.generate(
                inputs=[messages],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                use_steering=args.use_steering,
            )

            generated_text = responses[0] if responses else ""

            if args.verbose:
                print(f"   Generated: {generated_text[:100]}...")
                print()

            results.append({
                "question_id": idx,
                "prompt": pair.prompt,
                "generated_response": generated_text,
                "positive_reference": pair.positive_response.model_response,
                "negative_reference": pair.negative_response.model_response
            })

        except Exception as e:
            print(f"   ❌ Error generating response for question {idx}: {e}")
            results.append({
                "question_id": idx,
                "prompt": pair.prompt,
                "generated_response": None,
                "error": str(e)
            })

    # Save results
    print(f"\n💾 Saving results...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "task": args.task,
        "model": args.model,
        "num_questions": len(pairs),
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_steering": args.use_steering
        },
        "responses": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   ✓ Results saved to: {args.output}\n")

    # Print summary
    print(f"{'='*80}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Total questions: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"   Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"{'='*80}\n")
