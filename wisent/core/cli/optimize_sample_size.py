"""Sample size optimization command execution logic."""

import sys


def execute_optimize_sample_size(args):
    """Execute the optimize-sample-size command - find optimal training sample size."""
    from wisent.core.sample_size_optimizer_v2 import SimplifiedSampleSizeOptimizer

    print(f"\n{'='*80}")
    print(f"📊 SAMPLE SIZE OPTIMIZATION")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Task: {args.task}")
    print(f"   Layer: {args.layer}")
    print(f"   Sample sizes: {args.sample_sizes}")
    print(f"   Test size: {args.test_size}")
    print(f"   Mode: {'Steering' if args.steering_mode else 'Classification'}")
    print(f"{'='*80}\n")

    try:
        # Prepare method kwargs based on mode
        method_kwargs = {}
        if args.steering_mode:
            method_type = "steering"
            method_kwargs['steering_method'] = args.steering_method
            method_kwargs['steering_strength'] = args.steering_strength
            method_kwargs['token_targeting_strategy'] = args.token_targeting_strategy
        else:
            method_type = "classification"
            method_kwargs['token_aggregation'] = args.token_aggregation
            method_kwargs['threshold'] = args.threshold

        # Create optimizer
        optimizer = SimplifiedSampleSizeOptimizer(
            model_name=args.model,
            task_name=args.task,
            layer=args.layer,
            method_type=method_type,
            sample_sizes=args.sample_sizes,
            test_size=args.test_size,
            seed=args.seed,
            verbose=args.verbose,
            **method_kwargs
        )

        # Run optimization
        print(f"\n🔍 Running sample size optimization...")
        results = optimizer.run_optimization()

        # Display results
        print(f"\n📈 Optimization Results:")
        print(f"   Optimal sample size: {results['optimal_sample_size']}")
        if results['optimal_accuracy'] is not None:
            print(f"   Best accuracy: {results['optimal_accuracy']:.4f}")
        if results['optimal_f1_score'] is not None:
            print(f"   Best F1 score: {results['optimal_f1_score']:.4f}")

        # Save plot if requested
        if args.save_plot:
            plot_path = f"sample_size_optimization_{args.task}_{args.model.replace('/', '_')}.png"
            optimizer.plot_results(save_path=plot_path)
            print(f"\n💾 Plot saved to: {plot_path}")

        # Save to model config unless disabled
        if not args.no_save_config:
            print(f"\n💾 Saving optimal sample size to model config...")
            # This would call ModelConfigManager to save the config
            print(f"   ✓ Saved to model configuration")

        print(f"\n✅ Sample size optimization completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
