"""Cross-benchmark evaluation logic for tasks command."""

import sys


def load_cross_benchmark_data(args, LMEvalDataLoader):
    """Load training and evaluation data for cross-benchmark mode."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet

    train_task_name = args.train_task
    eval_task_name = args.eval_task

    print(f"\n🎯 Cross-benchmark evaluation mode")
    print(f"   Training on: {train_task_name}")
    print(f"   Evaluating on: {eval_task_name}")
    print(f"   Model: {args.model}")
    print(f"   Layer: {args.layer}")
    print(f"   Classifier type: {args.classifier_type}")

    try:
        # Load training data
        print(f"\n📊 Loading training data from '{train_task_name}'...")
        loader = LMEvalDataLoader()
        train_task_obj = loader.load_lm_eval_task(train_task_name)

        if isinstance(train_task_obj, dict):
            all_train_pairs = []
            training_limit_per_task = getattr(args, 'training_limit', None)
            if training_limit_per_task:
                training_limit_per_task = training_limit_per_task // len(train_task_obj)

            for subname, task_obj in train_task_obj.items():
                subtask_pairs = lm_build_contrastive_pairs(
                    task_name=subname,
                    lm_eval_task=task_obj,
                    limit=training_limit_per_task,
                )
                all_train_pairs.extend(subtask_pairs)
            train_pairs = all_train_pairs
        else:
            train_pairs = lm_build_contrastive_pairs(
                task_name=train_task_name,
                lm_eval_task=train_task_obj,
                limit=getattr(args, 'training_limit', None),
            )

        train_pair_set = ContrastivePairSet("train", train_pairs, task_type=train_task_name)
        print(f"   ✓ Loaded {len(train_pair_set.pairs)} training pairs from {train_task_name}")

        # Load evaluation data
        print(f"\n📊 Loading evaluation data from '{eval_task_name}'...")
        eval_task_obj = loader.load_lm_eval_task(eval_task_name)

        if isinstance(eval_task_obj, dict):
            all_eval_pairs = []
            testing_limit_per_task = getattr(args, 'testing_limit', None)
            if testing_limit_per_task:
                testing_limit_per_task = testing_limit_per_task // len(eval_task_obj)

            for subname, task_obj in eval_task_obj.items():
                subtask_pairs = lm_build_contrastive_pairs(
                    task_name=subname,
                    lm_eval_task=task_obj,
                    limit=testing_limit_per_task,
                )
                all_eval_pairs.extend(subtask_pairs)
            eval_pairs = all_eval_pairs
        else:
            eval_pairs = lm_build_contrastive_pairs(
                task_name=eval_task_name,
                lm_eval_task=eval_task_obj,
                limit=getattr(args, 'testing_limit', None),
            )

        test_pair_set = ContrastivePairSet("test", eval_pairs, task_type=eval_task_name)
        print(f"   ✓ Loaded {len(test_pair_set.pairs)} test pairs from {eval_task_name}")

        return {
            'train_pair_set': train_pair_set,
            'test_pair_set': test_pair_set,
            'task_name': train_task_name,
            'eval_task_name': eval_task_name
        }

    except Exception as e:
        print(f"\n❌ Error loading cross-benchmark data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_single_task_data(args, LMEvalDataLoader):
    """Load data for single-task mode."""
    import sys

    print(f"\n🎯 Starting classifier training on task: {args.task_names}")
    print(f"   Model: {args.model}")
    print(f"   Layer: {args.layer}")
    print(f"   Classifier type: {args.classifier_type}")

    try:
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

        train_pair_set = result['train_qa_pairs']
        test_pair_set = result['test_qa_pairs']
        print(f"   ✓ Loaded {len(train_pair_set.pairs)} training pairs")

        return {
            'train_pair_set': train_pair_set,
            'test_pair_set': test_pair_set,
            'task_name': task_name,
            'eval_task_name': None
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
