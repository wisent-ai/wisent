"""Apply steering vectors to improve response."""

from wisent.core.models.inference_config import get_generate_kwargs


def _map_token_aggregation(aggregation_str: str):
    """Map string token aggregation to ExtractionStrategy."""
    from wisent.core.activations.extraction_strategy import ExtractionStrategy

    mapping = {
        "average": ExtractionStrategy.CHAT_MEAN,
        "final": ExtractionStrategy.CHAT_LAST,
        "first": ExtractionStrategy.CHAT_FIRST,
        "max": ExtractionStrategy.CHAT_MAX_NORM,
        "min": ExtractionStrategy.CHAT_MEAN,
    }
    return mapping.get(aggregation_str, ExtractionStrategy.CHAT_MEAN)


def _map_prompt_strategy(strategy_str: str):
    """Map string prompt strategy to ExtractionStrategy."""
    

    mapping = {
        "chat_template": ExtractionStrategy.CHAT_LAST,
        "direct_completion": ExtractionStrategy.CHAT_LAST,
        "instruction_following": ExtractionStrategy.CHAT_LAST,
        "multiple_choice": ExtractionStrategy.MC_BALANCED,
        "role_playing": ExtractionStrategy.ROLE_PLAY,
    }
    return mapping.get(strategy_str, ExtractionStrategy.CHAT_LAST)


def apply_steering_and_evaluate(
    model,
    prompt: str,
    pair_set,
    classifier,
    collector,
    layer_key: str,
    quality_threshold: float,
    steering_strength: float = 1.0,
    steering_normalize: bool = True,
    verbose: bool = False,
    token_aggregation: str = "average",
    prompt_strategy: str = "chat_template",
    normalize_layers: bool = False,
    return_full_sequence: bool = False
) -> tuple[str, float]:
    """
    Apply steering vectors and generate improved response.

    arguments:
        model:
            WisentModel instance
        prompt:
            User prompt
        pair_set:
            ContrastivePairSet for creating steering vectors
        classifier:
            Trained classifier
        collector:
            ActivationCollector instance
        layer_key:
            Layer to use
        quality_threshold:
            Minimum quality score
        steering_strength:
            Strength of steering application
        steering_normalize:
            Whether to normalize steering vectors
        verbose:
            Enable verbose output
        token_aggregation:
            Token aggregation strategy (average, final, first, max, min)
        prompt_strategy:
            Prompt construction strategy
        normalize_layers:
            Whether to normalize layer activations
        return_full_sequence:
            Whether to return full sequence or aggregated activations

    returns:
        Tuple of (steered_response, quality_score)
    """
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.models.core.atoms import SteeringPlan
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse

    print(f"\n🎯 Step 4: Response below threshold, using steering to improve")
    print(f"   Collecting activations from pairs...")
    print(f"   Token aggregation: {token_aggregation}")
    print(f"   Prompt strategy: {prompt_strategy}")
    print(f"   Normalize layers: {normalize_layers}")
    print(f"   Return full sequence: {return_full_sequence}")

    # Map string parameters to enums
    aggregation_strategy = _map_token_aggregation(token_aggregation)
    prompt_construction_strategy = _map_prompt_strategy(prompt_strategy)

    # Use specified layer for steering
    target_layers = [layer_key]
    print(f"   Target layers: {target_layers}")

    enriched_pairs = []
    for i, pair in enumerate(pair_set.pairs):  # Use ALL pairs
        if verbose:
            print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...")

        updated_pair = collector.collect(
            pair, strategy=aggregation_strategy,
            return_full_sequence=return_full_sequence,
            normalize_layers=normalize_layers,
            prompt_strategy=prompt_construction_strategy
        )
        enriched_pairs.append(updated_pair)

    print(f"   ✓ Collected activations for {len(enriched_pairs)} pairs")

    # Create steering vector using CAA
    print(f"\n   Creating steering vector with CAA method (normalize={steering_normalize})...")

    enriched_pair_set = ContrastivePairSet(
        name=pair_set.name,
        task_type=pair_set.task_type
    )
    for pair in enriched_pairs:
        enriched_pair_set.add(pair)

    caa = CAAMethod(normalize=steering_normalize)
    steering_vectors = caa.train(enriched_pair_set)

    print(f"   ✓ Created steering vectors for {len(steering_vectors)} layers")

    # Apply steering and generate new response
    print(f"\n   Applying steering and generating new response...")

    steering_plan_data = {}
    for layer_str, vector in steering_vectors.items():
        steering_plan_data[layer_str] = vector

    steering_plan = SteeringPlan.from_raw(
        raw=steering_plan_data,
        weights=[steering_strength],
        layers_description=None
    )

    model.apply_steering(plan=steering_plan)

    # Generate steered response
    messages = [[{"role": "user", "content": prompt}]]
    steered_responses = model.generate(
        inputs=messages,
        **get_generate_kwargs(),
    )

    model.detach()  # Remove steering

    steered_text = steered_responses[0] if steered_responses else ""
    print(f"\n   Steered response:")
    print(f"   {steered_text[:200]}{'...' if len(steered_text) > 200 else ''}")

    # Evaluate steered response with classifier
    print(f"\n   Evaluating steered response with classifier...")
    steered_dummy_pair = ContrastivePair(
        prompt=prompt,
        positive_response=PositiveResponse(model_response=steered_text),
        negative_response=NegativeResponse(model_response="placeholder"),
        label="evaluation",
        trait_description="evaluation"
    )

    steered_evaluated_pair = collector.collect(
        steered_dummy_pair, strategy=aggregation_strategy,
        return_full_sequence=return_full_sequence,
        normalize_layers=normalize_layers,
        prompt_strategy=prompt_construction_strategy
    )

    steered_quality = 0.0
    if steered_evaluated_pair.positive_response.layers_activations and layer_key in steered_evaluated_pair.positive_response.layers_activations:
        steered_act = steered_evaluated_pair.positive_response.layers_activations[layer_key].cpu().numpy()
        steered_quality = classifier.predict_proba(steered_act)
        print(f"   Classifier quality score: {steered_quality:.3f}")

    return steered_text, steered_quality
