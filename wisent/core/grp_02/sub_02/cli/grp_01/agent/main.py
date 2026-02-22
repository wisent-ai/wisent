"""Agent command execution logic."""

import sys
import time

from wisent.core.errors import UnknownTypeError


def execute_agent(args):
    """Execute the agent command - autonomous agent with configurable strategy."""
    print(f"\n🤖 Starting autonomous agent")
    print(f"   Strategy: {args.agent_strategy}")
    print(f"   Model: {args.model}")
    print(f"   Prompt: {args.prompt}")

    try:
        if args.agent_strategy == "synthetic_pairs_classifier_steering":
            execute_synthetic_pairs_classifier_steering_strategy(args)
        else:
            raise UnknownTypeError(entity_type="agent_strategy", value=args.agent_strategy)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def execute_synthetic_pairs_classifier_steering_strategy(args):
    """
    Execute the synthetic_pairs_classifier_steering strategy.

    Steps:
    1. Agent creates contrastive pairs synthetically for the desired representation
    2. Agent trains classifiers and chooses the best one using evaluation
    3. Agent generates an unsteered response and uses classifier to check correctness
    4. If incorrect: uses steering to train control vector, creates new response, evaluates until success
    5. If correct: returns the response
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.cli.agent import (
        generate_synthetic_pairs,
        train_classifier_on_pairs,
        evaluate_response_with_classifier,
        apply_steering_and_evaluate
    )

    start_time = time.time()

    # Load model
    model = WisentModel(model_name=args.model)

    # Determine target layer: use --layer if provided, otherwise middle layer
    if hasattr(args, 'layer') and args.layer is not None:
        target_layer = int(args.layer)
        print(f"   Using specified layer: {target_layer}")
    else:
        target_layer = model.num_layers // 2
        print(f"   Using middle layer: {target_layer}/{model.num_layers}")

    # Step 1: Generate synthetic contrastive pairs
    pair_set, report = generate_synthetic_pairs(
        model=model,
        prompt=args.prompt,
        time_budget=args.time_budget,
        verbose=args.verbose
    )

    # Step 2: Train classifier on the pairs
    classifier, layer_key, collector = train_classifier_on_pairs(
        model=model,
        pair_set=pair_set,
        target_layer=target_layer,
        verbose=args.verbose,
        classifier_epochs=getattr(args, 'classifier_epochs', 50),
        classifier_lr=getattr(args, 'classifier_lr', 1e-3),
        classifier_batch_size=getattr(args, 'classifier_batch_size', None),
        token_aggregation=getattr(args, 'token_aggregation', 'average'),
        prompt_strategy=getattr(args, 'prompt_strategy', 'chat_template'),
        normalize_layers=getattr(args, 'normalize_layers', False),
        return_full_sequence=getattr(args, 'return_full_sequence', False),
        classifier_type=getattr(args, 'classifier_type', 'logistic')
    )

    # Step 3: Generate and evaluate unsteered response
    unsteered_response, quality_score = evaluate_response_with_classifier(
        model=model,
        prompt=args.prompt,
        classifier=classifier,
        collector=collector,
        layer_key=layer_key,
        quality_threshold=args.quality_threshold,
        token_aggregation=getattr(args, 'token_aggregation', 'average'),
        prompt_strategy=getattr(args, 'prompt_strategy', 'chat_template'),
        normalize_layers=getattr(args, 'normalize_layers', False),
        return_full_sequence=getattr(args, 'return_full_sequence', False)
    )

    if quality_score >= args.quality_threshold:
        print(f"\n✅ Response meets quality threshold!")
        print(f"\n🎉 Final response:")
        print(f"{unsteered_response}")
        elapsed = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed:.2f}s")
        print(f"\n✅ Agent execution completed!\n")
        return

    # Step 4: Apply steering to improve response
    steered_text, steered_quality = apply_steering_and_evaluate(
        model=model,
        prompt=args.prompt,
        pair_set=pair_set,
        classifier=classifier,
        collector=collector,
        layer_key=layer_key,
        quality_threshold=args.quality_threshold,
        steering_strength=getattr(args, 'steering_strength', 1.0),
        steering_normalize=getattr(args, 'normalize_mode', True),
        verbose=args.verbose,
        token_aggregation=getattr(args, 'token_aggregation', 'average'),
        prompt_strategy=getattr(args, 'prompt_strategy', 'chat_template'),
        normalize_layers=getattr(args, 'normalize_layers', False),
        return_full_sequence=getattr(args, 'return_full_sequence', False)
    )

    if steered_quality >= args.quality_threshold:
        print(f"\n✅ Steered response meets quality threshold!")
        print(f"\n🎉 Final response:")
        print(f"{steered_text}")
    else:
        print(f"\n⚠️  Steered response still below threshold")
        print(f"   Would retry with different parameters (max attempts: {args.max_attempts})")
        print(f"\n🎉 Best response so far:")
        print(f"{steered_text}")

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"\n✅ Agent execution completed!\n")
