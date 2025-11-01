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

    # Check if this is inference-only mode with steering vector
    if args.inference_only and args.load_steering_vector:
        import torch
        print(f"\n🎯 Starting inference with steering vector")
        print(f"   Loading vector from: {args.load_steering_vector}")

        # Load steering vector
        vector_data = torch.load(args.load_steering_vector)
        steering_vector = vector_data['vector']
        layer = vector_data['layer']

        print(f"   ✓ Loaded steering vector for layer {layer}")
        print(f"   Model: {vector_data.get('model', 'unknown')}")
        print(f"   Method: {vector_data.get('method', 'unknown')}")

        # For now, just load and validate - actual inference would require more implementation
        print(f"\n✅ Steering vector loaded successfully!\n")
        print(f"Note: Inference with steering vector requires additional implementation")

        # Return results for programmatic access
        return {
            "steering_vector_loaded": True,
            "vector_path": args.load_steering_vector,
            "layer": layer,
            "method": vector_data.get('method', 'unknown'),
            "test_accuracy": None,
            "test_f1_score": None,
            "training_time": 0.0,
            "evaluation_results": {}
        }

    print(f"\n🎯 Starting classifier training on task: {args.task_names}")
    print(f"   Model: {args.model}")
    print(f"   Layer: {args.layer}")
    print(f"   Classifier type: {args.classifier_type}")

    try:
        # 1. Load task data using LMEvalDataLoader
        task_name = args.task_names[0] if isinstance(args.task_names, list) else args.task_names
        print(f"\n📊 Loading task '{task_name}'...")
        loader = LMEvalDataLoader()
        result = loader._load_one_task(
            task_name=task_name,
            split_ratio=args.split_ratio,
            seed=args.seed,
            limit=args.limit,
            training_limit=args.training_limit,
            testing_limit=args.testing_limit
        )

        # Use training pairs for classifier training
        pair_set = result['train_qa_pairs']
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

        # Map parser values to enum members
        aggregation_map = {
            'average': 'MEAN_POOLING',
            'final': 'LAST_TOKEN',
            'first': 'FIRST_TOKEN',
            'max': 'MAX_POOLING',
            'min': 'MAX_POOLING',  # Fallback to MAX_POOLING for min
        }
        aggregation_key = aggregation_map.get(args.token_aggregation.lower(), 'MEAN_POOLING')
        aggregation_strategy = ActivationAggregationStrategy[aggregation_key]

        positive_activations = []
        negative_activations = []

        # Convert layer int to string for activation collection
        layer_str = str(layer)

        for i, pair in enumerate(pair_set.pairs):
            if i % 10 == 0:
                print(f"   Processing pair {i+1}/{len(pair_set.pairs)}...", end='\r')

            # Collect for positive (correct) response
            updated_pair = collector.collect_for_pair(
                pair,
                layers=[layer_str],
                aggregation=aggregation_strategy,
                return_full_sequence=False,
                normalize_layers=False
            )

            # Extract activations from positive and negative responses
            if updated_pair.positive_response.layers_activations and layer_str in updated_pair.positive_response.layers_activations:
                act = updated_pair.positive_response.layers_activations[layer_str]
                if act is not None:
                    positive_activations.append(act.cpu().numpy())

            if updated_pair.negative_response.layers_activations and layer_str in updated_pair.negative_response.layers_activations:
                act = updated_pair.negative_response.layers_activations[layer_str]
                if act is not None:
                    negative_activations.append(act.cpu().numpy())

        print(f"\n   ✓ Collected {len(positive_activations)} positive and {len(negative_activations)} negative activations")

        # Check if steering vector mode is requested
        if args.save_steering_vector and args.train_only:
            import torch
            from wisent.core.steering_methods.methods.caa import CAAMethod

            print(f"\n🎯 Training steering vector using {args.steering_method} method...")

            # Convert activations to tensors
            pos_tensors = [torch.from_numpy(act).float() for act in positive_activations]
            neg_tensors = [torch.from_numpy(act).float() for act in negative_activations]

            # Create steering method
            steering_method = CAAMethod(normalize=True)

            # Train steering vector
            steering_vector = steering_method.train_for_layer(pos_tensors, neg_tensors)

            # Save steering vector
            print(f"\n💾 Saving steering vector to '{args.save_steering_vector}'...")
            os.makedirs(os.path.dirname(args.save_steering_vector) or '.', exist_ok=True)
            torch.save({
                'steering_vector': steering_vector,
                'layer_index': layer,
                'method': args.steering_method,
                'model': args.model,
                'task': args.task_names,
                # Legacy keys for backward compatibility
                'vector': steering_vector,
                'layer': layer,
            }, args.save_steering_vector)
            print(f"   ✓ Steering vector saved to: {args.save_steering_vector}")

            # Save output artifacts if requested
            if args.output:
                print(f"\n📁 Saving artifacts to '{args.output}'...")
                os.makedirs(args.output, exist_ok=True)
                report_path = os.path.join(args.output, 'training_report.json')
                with open(report_path, 'w') as f:
                    json.dump({
                        'method': args.steering_method,
                        'layer': layer,
                        'num_positive': len(positive_activations),
                        'num_negative': len(negative_activations),
                        'vector_shape': list(steering_vector.shape)
                    }, f, indent=2)
                print(f"   ✓ Training report saved to: {report_path}")

            print(f"\n✅ Steering vector training completed successfully!\n")

            # Return results for programmatic access
            return {
                "steering_vector_saved": True,
                "vector_path": args.save_steering_vector,
                "layer": layer,
                "method": args.steering_method,
                "num_positive": len(positive_activations),
                "num_negative": len(negative_activations),
                "vector_shape": list(steering_vector.shape),
                "test_accuracy": None,
                "test_f1_score": None,
                "training_time": 0.0,
                "evaluation_results": {}
            }

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

        # Return results for programmatic access
        return {
            "test_accuracy": float(report.final.accuracy),
            "test_f1_score": float(report.final.f1),
            "test_precision": float(report.final.precision),
            "test_recall": float(report.final.recall),
            "test_auc": float(report.final.auc),
            "best_epoch": report.best_epoch,
            "epochs_ran": report.epochs_ran,
            "training_time": 0.0,  # TODO: track actual training time
            "evaluation_results": report.asdict() if hasattr(report, 'asdict') else {}
        }

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
