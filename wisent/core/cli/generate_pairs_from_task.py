"""Generate pairs from task command execution logic."""

import sys
import json
import os


def _load_custom_task(task_name: str, limit: int | None):
    """Load custom tasks that aren't in lm-eval."""
    if task_name == "livecodebench":
        from wisent.core.tasks.livecodebench_task import LiveCodeBenchTask
        return LiveCodeBenchTask(release_version="release_v1", limit=limit)
    else:
        raise ValueError(
            f"Task '{task_name}' not found in lm-eval or custom tasks. "
            f"Available custom tasks: livecodebench"
        )


def _build_pairs_from_custom_task(task, limit: int | None):
    """Build contrastive pairs from custom TaskInterface tasks."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors.livecodebench import (
        LiveCodeBenchExtractor as LiveCodeBenchPairExtractor
    )

    task_name = task.task_name

    if task_name == "livecodebench":
        # Use the contrastive pair extractor for LiveCodeBench
        extractor = LiveCodeBenchPairExtractor()
        # Extract pairs using the task's test_docs interface
        return extractor.extract_contrastive_pairs(task, limit=limit)
    else:
        raise ValueError(f"No contrastive pair extractor configured for custom task: {task_name}")


def execute_generate_pairs_from_task(args):
    """Execute the generate-pairs-from-task command - load and save contrastive pairs from a task."""
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader

    print(f"\n📊 Generating contrastive pairs from task: {args.task_name}")

    if args.limit:
        print(f"   Limit: {args.limit} pairs")

    try:
        # 1. Load task data using LMEvalDataLoader
        print(f"\n🔄 Loading task '{args.task_name}'...")

        # Try to load from lm-eval first
        loader = LMEvalDataLoader()
        try:
            # Use load_lm_eval_task to get the task object
            task_obj = loader.load_lm_eval_task(args.task_name)
        except KeyError:
            # Task not in lm-eval, try our custom tasks
            print(f"   ℹ️  Task not found in lm-eval, trying custom tasks...")
            task_obj = _load_custom_task(args.task_name, args.limit)

        # Import the pair generation function
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
            lm_build_contrastive_pairs,
        )
        from wisent.core.task_interface import TaskInterface

        # Handle both lm-eval tasks (dict or ConfigurableTask) and custom tasks (TaskInterface)
        if isinstance(task_obj, dict):
            # lm-eval task group with subtasks
            if len(task_obj) != 1:
                keys = ", ".join(sorted(task_obj.keys()))
                raise ValueError(
                    f"Task '{args.task_name}' returned {len(task_obj)} subtasks ({keys}). "
                    "Specify an explicit subtask, e.g. 'benchmark/subtask'."
                )
            (subname, task), = task_obj.items()
            pairs_task_name = subname

            # 2. Generate contrastive pairs using lm-eval interface
            print(f"   🔨 Building contrastive pairs...")
            pairs = lm_build_contrastive_pairs(
                task_name=pairs_task_name,
                lm_eval_task=task,
                limit=args.limit,
            )
        elif isinstance(task_obj, TaskInterface):
            # Custom task (TaskInterface) - only livecodebench for now
            task = task_obj
            pairs_task_name = args.task_name

            # 2. Generate contrastive pairs using custom task interface
            print(f"   🔨 Building contrastive pairs...")
            pairs = _build_pairs_from_custom_task(task, args.limit)
        else:
            # Single lm-eval task (ConfigurableTask), not wrapped in dict
            task = task_obj
            pairs_task_name = args.task_name

            # 2. Generate contrastive pairs using lm-eval interface
            print(f"   🔨 Building contrastive pairs...")
            pairs = lm_build_contrastive_pairs(
                task_name=pairs_task_name,
                lm_eval_task=task,
                limit=args.limit,
            )

        print(f"   ✓ Generated {len(pairs)} contrastive pairs")

        # 3. Convert pairs to dict format for JSON serialization
        print(f"\n💾 Saving pairs to '{args.output}'...")
        pairs_data = []
        for pair in pairs:
            pair_dict = pair.to_dict()
            pairs_data.append(pair_dict)

        # 4. Save to JSON file
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'task_name': pairs_task_name,
                'num_pairs': len(pairs),
                'pairs': pairs_data
            }, f, indent=2)

        print(f"   ✓ Saved {len(pairs)} pairs to: {args.output}")
        print(f"\n✅ Contrastive pairs generation completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
