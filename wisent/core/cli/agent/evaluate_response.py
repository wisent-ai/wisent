"""Generate and evaluate response with classifier."""

from wisent.core.models.inference_config import get_generate_kwargs


def _map_token_aggregation(aggregation_str: str):
    """Map string token aggregation to ActivationAggregationStrategy enum."""
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy

    mapping = {
        "average": ActivationAggregationStrategy.MEAN_POOLING,
        "final": ActivationAggregationStrategy.LAST_TOKEN,
        "first": ActivationAggregationStrategy.FIRST_TOKEN,
        "max": ActivationAggregationStrategy.MAX_POOLING,
        "min": ActivationAggregationStrategy.MAX_POOLING,  # Note: MIN_POOLING not in enum, using MAX_POOLING
    }
    return mapping.get(aggregation_str, ActivationAggregationStrategy.MEAN_POOLING)


def _map_prompt_strategy(strategy_str: str):
    """Map string prompt strategy to PromptConstructionStrategy enum."""
    from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy

    mapping = {
        "chat_template": PromptConstructionStrategy.CHAT_TEMPLATE,
        "direct_completion": PromptConstructionStrategy.DIRECT_COMPLETION,
        "instruction_following": PromptConstructionStrategy.INSTRUCTION_FOLLOWING,
        "multiple_choice": PromptConstructionStrategy.MULTIPLE_CHOICE,
        "role_playing": PromptConstructionStrategy.ROLE_PLAYING,
    }
    return mapping.get(strategy_str, PromptConstructionStrategy.CHAT_TEMPLATE)


def evaluate_response_with_classifier(
    model,
    prompt: str,
    classifier,
    collector,
    layer_key: str,
    quality_threshold: float,
    token_aggregation: str = "average",
    prompt_strategy: str = "chat_template",
    normalize_layers: bool = False,
    return_full_sequence: bool = False
) -> tuple[str, float]:
    """
    Generate a response and evaluate it with the classifier.

    arguments:
        model:
            WisentModel instance
        prompt:
            User prompt to respond to
        classifier:
            Trained classifier
        collector:
            ActivationCollector instance
        layer_key:
            Layer to collect activations from
        quality_threshold:
            Minimum quality score required
        token_aggregation:
            Token aggregation strategy (average, final, first, max, min)
        prompt_strategy:
            Prompt construction strategy
        normalize_layers:
            Whether to normalize layer activations
        return_full_sequence:
            Whether to return full sequence or aggregated activations

    returns:
        Tuple of (response_text, quality_score)
    """
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse

    print(f"\n💬 Step 3: Generating unsteered response")

    # Generate response without steering
    messages = [[{"role": "user", "content": prompt}]]

    print(f"   Generating response...")
    responses = model.generate(
        inputs=messages,
        **get_generate_kwargs(),
    )

    unsteered_response = responses[0] if responses else ""
    print(f"\n   Unsteered response:")
    print(f"   {unsteered_response[:200]}{'...' if len(unsteered_response) > 200 else ''}")

    # Evaluate with classifier
    print(f"\n✅ Step 3b: Evaluating response with classifier")
    print(f"   Quality threshold: {quality_threshold}")
    print(f"   Token aggregation: {token_aggregation}")
    print(f"   Prompt strategy: {prompt_strategy}")

    # Map string parameters to enums
    aggregation_strategy = _map_token_aggregation(token_aggregation)
    prompt_construction_strategy = _map_prompt_strategy(prompt_strategy)

    # Collect activations from the unsteered response
    print(f"   Collecting activations from response...")
    dummy_pair = ContrastivePair(
        prompt=prompt,
        positive_response=PositiveResponse(model_response=unsteered_response),
        negative_response=NegativeResponse(model_response="placeholder"),
        label="evaluation",
        trait_description="evaluation"
    )

    evaluated_pair = collector.collect_for_pair(
        dummy_pair,
        layers=[layer_key],
        aggregation=aggregation_strategy,
        return_full_sequence=return_full_sequence,
        normalize_layers=normalize_layers,
        prompt_strategy=prompt_construction_strategy
    )

    # Get activation and predict quality
    quality_score = 0.0
    if evaluated_pair.positive_response.layers_activations and layer_key in evaluated_pair.positive_response.layers_activations:
        response_act = evaluated_pair.positive_response.layers_activations[layer_key].cpu().numpy()
        quality_score = classifier.predict_proba(response_act)
        print(f"   Classifier quality score: {quality_score:.3f}")

    return unsteered_response, quality_score
