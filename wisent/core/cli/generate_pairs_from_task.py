"""Generate pairs from task command execution logic."""

import sys
import json
import os


def execute_generate_pairs_from_task(args):
    """Execute the generate-pairs-from-task command - load and save contrastive pairs from a task."""
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader

    print(f"\n📊 Generating contrastive pairs from task: {args.task_name}")

    if args.limit:
        print(f"   Limit: {args.limit} pairs")

    try:
        # 1. Load task data using LMEvalDataLoader
        print(f"\n🔄 Loading task '{args.task_name}'...")
        loader = LMEvalDataLoader()

        # Use load_lm_eval_task to get the task object
        task_obj = loader.load_lm_eval_task(args.task_name)

        # Import the pair generation function
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
            lm_build_contrastive_pairs,
        )

        # Handle both single task and dict of subtasks
        if isinstance(task_obj, dict):
            if len(task_obj) != 1:
                keys = ", ".join(sorted(task_obj.keys()))
                raise ValueError(
                    f"Task '{args.task_name}' returned {len(task_obj)} subtasks ({keys}). "
                    "Specify an explicit subtask, e.g. 'benchmark/subtask'."
                )
            (subname, task), = task_obj.items()
            pairs_task_name = subname
        else:
            task = task_obj
            pairs_task_name = args.task_name

        # 2. Generate contrastive pairs
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
