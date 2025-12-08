"""Generate pairs from task command execution logic."""

import sys
import json
import os

from wisent.core.errors import InvalidDataFormatError


def execute_generate_pairs_from_task(args):
    """Execute the generate-pairs-from-task command - load and save contrastive pairs from a task."""
    # Expand task if it's a skill or risk name
    from wisent.core.task_selector import expand_task_if_skill_or_risk
    if hasattr(args, 'task_name') and args.task_name:
        args.task_name = expand_task_if_skill_or_risk(args.task_name)
    
    from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import HF_EXTRACTORS
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
        lm_build_contrastive_pairs,
    )

    print(f"\n📊 Generating contrastive pairs from task: {args.task_name}")

    if args.limit:
        print(f"   Limit: {args.limit} pairs")

    try:
        print(f"\n🔄 Loading task '{args.task_name}'...")

        # Check if task is in HuggingFace manifest (doesn't need lm-eval loading)
        task_name_lower = args.task_name.lower()
        is_hf_task = task_name_lower in {k.lower() for k in HF_EXTRACTORS.keys()}

        if is_hf_task:
            # HuggingFace task - skip lm-eval loading, go directly to extractor
            print(f"   Found in HuggingFace manifest, using HF extractor...")
            print(f"   🔨 Building contrastive pairs...")
            pairs = lm_build_contrastive_pairs(
                task_name=args.task_name,
                lm_eval_task=None,  # HF extractors don't need lm_eval_task
                limit=args.limit,
            )
            pairs_task_name = args.task_name
        else:
            # lm-eval task - load via LMEvalDataLoader
            from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
            loader = LMEvalDataLoader()
            task_obj = loader.load_lm_eval_task(args.task_name)

            # Handle both lm-eval tasks (dict or ConfigurableTask)
            if isinstance(task_obj, dict):
                # lm-eval task group with subtasks
                if len(task_obj) != 1:
                    keys = ", ".join(sorted(task_obj.keys()))
                    raise InvalidDataFormatError(
                        reason=f"Task '{args.task_name}' returned {len(task_obj)} subtasks ({keys}). "
                               "Specify an explicit subtask, e.g. 'benchmark/subtask'."
                    )
                (subname, task), = task_obj.items()
                pairs_task_name = subname

                # Generate contrastive pairs using lm-eval interface
                print(f"   🔨 Building contrastive pairs...")
                pairs = lm_build_contrastive_pairs(
                    task_name=pairs_task_name,
                    lm_eval_task=task,
                    limit=args.limit,
                )
            else:
                # Single lm-eval task (ConfigurableTask), not wrapped in dict
                task = task_obj
                pairs_task_name = args.task_name

                # Generate contrastive pairs using lm-eval interface
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
