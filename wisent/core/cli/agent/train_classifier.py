"""Train classifier on contrastive pairs for agent."""

import numpy as np
import torch
from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainReport
from wisent.core.errors import UnknownTypeError
from wisent.core.utils.device import preferred_dtype


def _torch_dtype_to_numpy(torch_dtype: torch.dtype):
    """Convert torch dtype to numpy dtype."""
    mapping = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,  # numpy doesn't support bfloat16, use float32
    }
    return mapping.get(torch_dtype, np.float32)


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


def train_classifier_on_pairs(
    model,
    pair_set,
    target_layer: int,
    verbose: bool = False,
    classifier_epochs: int = 50,
    classifier_lr: float = 1e-3,
    classifier_batch_size: int = None,
    token_aggregation: str = "average",
    prompt_strategy: str = "chat_template",
    normalize_layers: bool = False,
    return_full_sequence: bool = False,
    classifier_type: str = "logistic"
):
    """
    Train a classifier on activations from contrastive pairs.

    arguments:
        model:
            WisentModel instance
        pair_set:
            ContrastivePairSet with pairs to train on
        target_layer:
            Which layer to use for activations
        verbose:
            Enable verbose output
        classifier_epochs:
            Number of epochs for classifier training
        classifier_lr:
            Learning rate for classifier training
        classifier_batch_size:
            Batch size for classifier training
        token_aggregation:
            Token aggregation strategy (average, final, first, max, min)
        prompt_strategy:
            Prompt construction strategy
        normalize_layers:
            Whether to normalize layer activations
        return_full_sequence:
            Whether to return full sequence or aggregated activations
        classifier_type:
            Type of classifier to use (logistic or mlp)

    returns:
        Tuple of (trained_classifier, layer_key, collector)
    """
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
    from wisent.core.classifiers.classifiers.models.mlp import MLPClassifier
    from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig

    print(f"\n🧠 Step 2: Training classifier on contrastive pairs")
    print(f"   Collecting activations for classifier training...")
    print(f"   Token aggregation: {token_aggregation}")
    print(f"   Prompt strategy: {prompt_strategy}")
    print(f"   Normalize layers: {normalize_layers}")
    print(f"   Return full sequence: {return_full_sequence}")
    print(f"   Classifier type: {classifier_type}")

    # Map string parameters to enums
    aggregation_strategy = _map_token_aggregation(token_aggregation)
    prompt_construction_strategy = _map_prompt_strategy(prompt_strategy)

    # Collect activations for all pairs
    collector = ActivationCollector(model=model)
    target_layers = [str(target_layer)]
    layer_key = target_layers[0]

    enriched_training_pairs = []
    for i, pair in enumerate(pair_set.pairs):
        if verbose:
            print(f"   Processing training pair {i+1}/{len(pair_set.pairs)}...")

        updated_pair = collector.collect(
            pair, strategy=aggregation_strategy,
            return_full_sequence=return_full_sequence,
            normalize_layers=normalize_layers,
            prompt_strategy=prompt_construction_strategy
        )
        enriched_training_pairs.append(updated_pair)

    print(f"   ✓ Collected activations for {len(enriched_training_pairs)} pairs")

    # Prepare training data: positive activations = label 1, negative activations = label 0
    X_list = []
    y_list = []

    for pair in enriched_training_pairs:
        # Positive activation -> label 1
        if pair.positive_response.layers_activations and layer_key in pair.positive_response.layers_activations:
            pos_act = pair.positive_response.layers_activations[layer_key]
            X_list.append(pos_act.cpu().numpy())
            y_list.append(1.0)

        # Negative activation -> label 0
        if pair.negative_response.layers_activations and layer_key in pair.negative_response.layers_activations:
            neg_act = pair.negative_response.layers_activations[layer_key]
            X_list.append(neg_act.cpu().numpy())
            y_list.append(0.0)

    np_dtype = _torch_dtype_to_numpy(preferred_dtype())
    X_train = np.array(X_list, dtype=np_dtype)
    y_train = np.array(y_list, dtype=np_dtype)

    print(f"   Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Instantiate classifier based on type
    if classifier_type == "logistic":
        classifier = LogisticClassifier(threshold=0.5)
        print(f"   Training logistic classifier...")
    elif classifier_type == "mlp":
        classifier = MLPClassifier(threshold=0.5, hidden_dim=128)
        print(f"   Training MLP classifier...")
    else:
        raise UnknownTypeError(entity_type="classifier_type", value=classifier_type, valid_values=["logistic", "mlp"])

    # Determine batch size: use provided value or adaptive default
    if classifier_batch_size is None:
        batch_size = min(32, len(X_train) // 2)
    else:
        batch_size = classifier_batch_size

    train_config = ClassifierTrainConfig(
        test_size=0.2,
        num_epochs=classifier_epochs,
        batch_size=batch_size,
        learning_rate=classifier_lr,
    )

    report = classifier.fit(X_train, y_train, config=train_config)

    print(f"   ✓ Training complete!")
    print(f"     Accuracy: {report.final.accuracy:.3f}")
    print(f"     F1 Score: {report.final.f1:.3f}")
    print(f"     Best epoch: {report.best_epoch}/{report.epochs_ran}")

    return classifier, layer_key, collector
