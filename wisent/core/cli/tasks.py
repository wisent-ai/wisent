"""Tasks command execution logic."""

import sys
import os
import json
import numpy as np


def execute_tasks(args):
    """Execute the tasks command - train classifier on benchmark tasks."""
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations.core.atoms import ActivationAggregationStrategy
    from wisent.core.classifiers.classifiers.models.logistic import LogisticClassifier
    from wisent.core.classifiers.classifiers.models.mlp import MLPClassifier
    from wisent.core.classifiers.classifiers.core.atoms import ClassifierTrainConfig
    from wisent.core.model_persistence import ModelPersistence, create_classifier_metadata

    print(f"\n🎯 Starting classifier training on task: {args.task_names}")
    print(f"   Model: {args.model}")
    print(f"   Layer: {args.layer}")
    print(f"   Classifier type: {args.classifier_type}")

    try:
        # 1. Load task data using LMEvalDataLoader
        print(f"\n📊 Loading task '{args.task_names}'...")
        loader = LMEvalDataLoader()
        result = loader._load_one_task(
            task_name=args.task_names,
            split_ratio=args.split_ratio,
            seed=args.seed,
            limit=args.limit,
            training_limit=args.training_limit,
            testing_limit=args.testing_limit
        )

        # Use training pairs for classifier training
        pair_set = result.train_pairs
        print(f"   ✓ Loaded {len(pair_set.pairs)} training pairs")

        # 3. Load model
        print(f"\n🤖 Loading model '{args.model}'...")
        model = WisentModel(args.model, device=args.device)
        print(f"   ✓ Model loaded with {model.num_layers} layers")

        # 4. Parse layer specification
        layer = int(args.layer) if isinstance(args.layer, str) else args.layer
        print(f"\n🧠 Extracting activations from layer {layer}...")

        # 5. Collect activations for all pairs
        collector = ActivationCollector(model=model, store_device="cpu")
        aggregation_strategy = ActivationAggregationStrategy[args.token_aggregation.upper()]

        positive_activations = []
        negative_activations = []

        for i, pair in enumerate(pair_set.pairs):
            if i % 10 == 0:
                print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...", end='\r')

            # Collect for positive (correct) response
            updated_pair = collector.collect_for_pair(
                pair,
                layers=[layer],
                aggregation=aggregation_strategy,
                return_full_sequence=False,
                normalize_layers=False
            )

            # Extract activations
            if updated_pair.positive_activations and layer in updated_pair.positive_activations.raw_map:
                positive_activations.append(updated_pair.positive_activations.raw_map[layer].cpu().numpy())
            if updated_pair.negative_activations and layer in updated_pair.negative_activations.raw_map:
                negative_activations.append(updated_pair.negative_activations.raw_map[layer].cpu().numpy())

        print(f"\n   ✓ Collected {len(positive_activations)} positive and {len(negative_activations)} negative activations")

        # 6. Prepare training data
        print(f"\n🎯 Preparing training data...")
        X_positive = np.array(positive_activations)
        X_negative = np.array(negative_activations)
        X = np.vstack([X_positive, X_negative])
        y = np.array([1] * len(positive_activations) + [0] * len(negative_activations))

        print(f"   Training set: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Positive samples: {sum(y == 1)}, Negative samples: {sum(y == 0)}")

        # 7. Create and train classifier
        print(f"\n🏋️  Training {args.classifier_type} classifier...")
        if args.classifier_type == 'logistic':
            classifier = LogisticClassifier(threshold=args.detection_threshold, device=args.device)
        elif args.classifier_type == 'mlp':
            classifier = MLPClassifier(threshold=args.detection_threshold, device=args.device)
        else:
            raise ValueError(f"Unknown classifier type: {args.classifier_type}")

        # Training configuration
        train_config = ClassifierTrainConfig(
            test_size=1.0 - args.split_ratio,
            num_epochs=50,
            batch_size=32,
            learning_rate=1e-3,
            monitor='f1',
            random_state=args.seed
        )

        # Train the classifier
        report = classifier.fit(X, y, config=train_config)

        # 8. Print results
        print(f"\n📈 Training completed!")
        print(f"   Best epoch: {report.best_epoch}/{report.epochs_ran}")
        print(f"   Final metrics:")
        print(f"     • Accuracy:  {report.final.accuracy:.4f}")
        print(f"     • Precision: {report.final.precision:.4f}")
        print(f"     • Recall:    {report.final.recall:.4f}")
        print(f"     • F1 Score:  {report.final.f1:.4f}")
        print(f"     • AUC:       {report.final.auc:.4f}")

        # 9. Save classifier if requested
        if args.save_classifier:
            print(f"\n💾 Saving classifier to '{args.save_classifier}'...")

            # Create metadata
            metadata = create_classifier_metadata(
                model_name=args.model,
                task_name=args.task_names,
                layer=layer,
                classifier_type=args.classifier_type,
                training_accuracy=report.final.accuracy,
                training_samples=len(X),
                token_aggregation=args.token_aggregation,
                detection_threshold=args.detection_threshold
            )

            # Save using model persistence
            save_path = ModelPersistence.save_classifier(
                classifier=classifier,
                layer=layer,
                save_path=args.save_classifier,
                metadata=metadata
            )
            print(f"   ✓ Classifier saved to: {save_path}")

        # 10. Save output artifacts if requested
        if args.output:
            print(f"\n📁 Saving artifacts to '{args.output}'...")
            os.makedirs(args.output, exist_ok=True)

            # Save training report
            report_path = os.path.join(args.output, 'training_report.json')
            with open(report_path, 'w') as f:
                json.dump(report.asdict(), f, indent=2)
            print(f"   ✓ Training report saved to: {report_path}")

        print(f"\n✅ Task completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
