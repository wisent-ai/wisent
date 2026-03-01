"""Generate pairs from task command execution logic."""

import sys
import json
import os

from wisent.core.constants import JSON_INDENT


def execute_generate_pairs_from_task(args):
    """Execute the generate-pairs-from-task command - load and save contrastive pairs from a task."""
    # Expand task if it's a skill or risk name
    from wisent.core.tasks.base.task_selector import expand_task_if_skill_or_risk
    if hasattr(args, 'task_name') and args.task_name:
        args.task_name = expand_task_if_skill_or_risk(args.task_name)
    
    from wisent.extractors.lm_eval._registry.lm_task_pairs_generation import (
        build_contrastive_pairs,
    )

    print(f"\n📊 Generating contrastive pairs from task: {args.task_name}")

    if args.limit:
        print(f"   Limit: {args.limit} pairs")

    try:
        print(f"\n🔄 Loading task '{args.task_name}'...")
        print(f"   🔨 Building contrastive pairs...")
        
        # Use unified loader - handles HF, lm-eval, and group tasks automatically
        pairs = build_contrastive_pairs(
            task_name=args.task_name,
            limit=args.limit,
        )
        pairs_task_name = args.task_name

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
            }, f, indent=JSON_INDENT)

        print(f"   ✓ Saved {len(pairs)} pairs to: {args.output}")
        print(f"\n✅ Contrastive pairs generation completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        raise
