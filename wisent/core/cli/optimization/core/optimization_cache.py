"""Optimization cache management command execution logic."""

import sys
import json
from wisent.core.config_manager import (
    get_cache,
    get_cached_optimization,
    OptimizationCache,
    OptimizationResult,
)


def execute_optimization_cache(args):
    """
    Execute the optimization-cache command.

    Supports multiple subcommands:
    - list: List cached optimization results
    - show: Show details of a cached result
    - delete: Delete a cached result
    - clear: Clear all cached results
    - set-default: Set a cached result as the default
    - get-default: Get the default optimization result
    - export: Export cache to a JSON file
    - import: Import cache from a JSON file
    """
    # Check which subcommand was called
    if not hasattr(args, 'cache_action') or args.cache_action is None:
        print("\n  No cache action specified")
        print("Available actions: list, show, delete, clear, set-default, get-default, export, import")
        sys.exit(1)

    cache = get_cache()

    if args.cache_action == 'list':
        return execute_list(args, cache)
    elif args.cache_action == 'show':
        return execute_show(args, cache)
    elif args.cache_action == 'delete':
        return execute_delete(args, cache)
    elif args.cache_action == 'clear':
        return execute_clear(args, cache)
    elif args.cache_action == 'set-default':
        return execute_set_default(args, cache)
    elif args.cache_action == 'get-default':
        return execute_get_default(args, cache)
    elif args.cache_action == 'export':
        return execute_export(args, cache)
    elif args.cache_action == 'import':
        return execute_import(args, cache)
    else:
        print(f"\n  Unknown cache action: {args.cache_action}")
        sys.exit(1)


def execute_list(args, cache: OptimizationCache):
    """List cached optimization results."""
    print(f"\n{'='*80}")
    print(f" OPTIMIZATION CACHE")
    print(f"{'='*80}")

    results = cache.list_cached(model=args.model, task=args.task)

    if not results:
        print("\nNo cached optimization results found.")
        if args.model:
            print(f"   Filter: model={args.model}")
        if args.task:
            print(f"   Filter: task={args.task}")
        print()
        return {"action": "list", "count": 0, "results": []}

    print(f"\nFound {len(results)} cached optimization result(s):\n")
    print(f"{'Model':<40} {'Task':<20} {'Method':<8} {'Layer':<6} {'Strength':<10} {'Score':<8}")
    print("-" * 100)

    for result in results:
        model_display = result.model[:37] + "..." if len(result.model) > 40 else result.model
        task_display = result.task[:17] + "..." if len(result.task) > 20 else result.task
        print(f"{model_display:<40} {task_display:<20} {result.method:<8} {result.layer:<6} {result.strength:<10.2f} {result.score:<8.3f}")

    print("-" * 100)
    print()

    return {
        "action": "list",
        "count": len(results),
        "results": [r.to_dict() for r in results]
    }


def execute_show(args, cache: OptimizationCache):
    """Show details of a cached optimization result."""
    result = cache.get(args.model, args.task, args.method)

    if not result:
        print(f"\nNo cached result found for:")
        print(f"   Model: {args.model}")
        print(f"   Task: {args.task}")
        print(f"   Method: {args.method}")
        print()
        return {"action": "show", "found": False}

    print(f"\n{'='*80}")
    print(f" CACHED OPTIMIZATION RESULT")
    print(f"{'='*80}")
    print(f"\n   Model: {result.model}")
    print(f"   Task: {result.task}")
    print(f"   Method: {result.method}")
    print(f"\n   Optimal Parameters:")
    print(f"     Layer: {result.layer}")
    print(f"     Strength: {result.strength}")
    print(f"     Token Aggregation: {result.token_aggregation}")
    print(f"     Prompt Strategy: {result.prompt_strategy}")
    print(f"\n   Performance:")
    print(f"     Score: {result.score:.4f}")
    print(f"     Metric: {result.metric}")
    print(f"\n   Timestamp: {result.timestamp}")

    if result.metadata:
        print(f"\n   Metadata:")
        for key, value in result.metadata.items():
            print(f"     {key}: {value}")

    print()

    return {
        "action": "show",
        "found": True,
        "result": result.to_dict()
    }


def execute_delete(args, cache: OptimizationCache):
    """Delete a cached optimization result."""
    if cache.delete(args.model, args.task, args.method):
        print(f"\n   Deleted cached result:")
        print(f"   Model: {args.model}")
        print(f"   Task: {args.task}")
        print(f"   Method: {args.method}")
        print()
        return {"action": "delete", "success": True}
    else:
        print(f"\nNo cached result found to delete:")
        print(f"   Model: {args.model}")
        print(f"   Task: {args.task}")
        print(f"   Method: {args.method}")
        print()
        return {"action": "delete", "success": False}


def execute_clear(args, cache: OptimizationCache):
    """Clear all cached optimization results."""
    if not args.confirm:
        print("\n   Warning: This will delete ALL cached optimization results.")
        print("   To confirm, run with --confirm flag.")
        print()
        return {"action": "clear", "success": False, "reason": "not_confirmed"}

    count = cache.clear()
    print(f"\n   Cleared {count} cached optimization result(s).")
    print()

    return {"action": "clear", "success": True, "count": count}


def execute_set_default(args, cache: OptimizationCache):
    """Set a cached result as the default for a model/task."""
    if cache.set_default(args.model, args.task, args.method):
        print(f"\n   Set default for {args.model}/{args.task}:")
        print(f"   Method: {args.method}")
        print()
        return {"action": "set-default", "success": True}
    else:
        print(f"\nNo cached result found to set as default:")
        print(f"   Model: {args.model}")
        print(f"   Task: {args.task}")
        print(f"   Method: {args.method}")
        print()
        return {"action": "set-default", "success": False}


def execute_get_default(args, cache: OptimizationCache):
    """Get the default optimization result for a model/task."""
    result = cache.get_default(args.model, args.task)

    if not result:
        print(f"\nNo default optimization result found for:")
        print(f"   Model: {args.model}")
        print(f"   Task: {args.task}")
        print()
        return {"action": "get-default", "found": False}

    print(f"\n{'='*80}")
    print(f" DEFAULT OPTIMIZATION RESULT")
    print(f"{'='*80}")
    print(f"\n   Model: {result.model}")
    print(f"   Task: {result.task}")
    print(f"   Method: {result.method}")
    print(f"\n   Optimal Parameters:")
    print(f"     Layer: {result.layer}")
    print(f"     Strength: {result.strength}")
    print(f"     Token Aggregation: {result.token_aggregation}")
    print(f"     Prompt Strategy: {result.prompt_strategy}")
    print(f"\n   Performance:")
    print(f"     Score: {result.score:.4f}")
    print(f"     Metric: {result.metric}")
    print()

    return {
        "action": "get-default",
        "found": True,
        "result": result.to_dict()
    }


def execute_export(args, cache: OptimizationCache):
    """Export cache to a JSON file."""
    results = cache.list_cached()

    export_data = {
        "version": "1.0",
        "results": [r.to_dict() for r in results],
        "defaults": cache._defaults
    }

    with open(args.output, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"\n   Exported {len(results)} cached result(s) to: {args.output}")
    print()

    return {
        "action": "export",
        "success": True,
        "count": len(results),
        "output_file": args.output
    }


def execute_import(args, cache: OptimizationCache):
    """Import cache from a JSON file."""
    # OptimizationResult already imported at top of file

    try:
        with open(args.input, 'r') as f:
            import_data = json.load(f)
    except FileNotFoundError:
        print(f"\nFile not found: {args.input}")
        return {"action": "import", "success": False, "reason": "file_not_found"}
    except json.JSONDecodeError as e:
        print(f"\nInvalid JSON in file: {args.input}")
        print(f"   Error: {e}")
        return {"action": "import", "success": False, "reason": "invalid_json"}

    if not args.merge:
        cache.clear()
        print(f"   Cleared existing cache.")

    results = import_data.get("results", [])
    defaults = import_data.get("defaults", {})

    imported_count = 0
    for result_data in results:
        try:
            cache.store(
                model=result_data["model"],
                task=result_data["task"],
                layer=result_data["layer"],
                strength=result_data["strength"],
                method=result_data.get("method", "CAA"),
                token_aggregation=result_data.get("token_aggregation", "average"),
                prompt_strategy=result_data.get("prompt_strategy", "question_only"),
                score=result_data.get("score", 0.0),
                metric=result_data.get("metric", "accuracy"),
                metadata=result_data.get("metadata", {}),
                set_as_default=False
            )
            imported_count += 1
        except Exception as e:
            print(f"   Warning: Failed to import result: {e}")

    # Restore defaults
    for key, value in defaults.items():
        cache._defaults[key] = value
    cache._save()

    print(f"\n   Imported {imported_count} cached result(s) from: {args.input}")
    if args.merge:
        print(f"   (merged with existing cache)")
    print()

    return {
        "action": "import",
        "success": True,
        "count": imported_count,
        "input_file": args.input
    }
