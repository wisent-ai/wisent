#!/usr/bin/env python3
"""
Main entry point for OOP-refactored evaluation pipeline.

Usage:
    # Run full pipeline for all traits
    python run_oop.py

    # Run only generation
    python run_oop.py --generation-only

    # Run only evaluation
    python run_oop.py --evaluation-only

    # Run specific trait
    python run_oop.py --trait happy

    # Run with custom config
    python run_oop.py --config path/to/config.json
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.EVALOOP.core.config import ConfigManager
from tests.EVALOOP.core.pipeline import EvaluationPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OOP Evaluation Pipeline")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config JSON file"
    )

    parser.add_argument(
        "--trait",
        type=str,
        default=None,
        help="Run pipeline for specific trait only (e.g., 'happy', 'evil')"
    )

    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Run only the generation phase"
    )

    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Run only the evaluation phase (requires generation output)"
    )

    parser.add_argument(
        "--list-traits",
        action="store_true",
        help="List available traits and exit"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config_path = Path(args.config) if args.config else None
    config_manager = ConfigManager(config_path)

    # List traits if requested
    if args.list_traits:
        print("Available traits:")
        for trait in config_manager.list_traits():
            print(f"  - {trait}")
        return

    # Create pipeline
    pipeline = EvaluationPipeline(config_manager)

    # Determine which traits to process
    if args.trait:
        traits = [args.trait]
        if args.trait not in config_manager.list_traits():
            print(f"Error: Trait '{args.trait}' not found")
            print(f"Available traits: {', '.join(config_manager.list_traits())}")
            sys.exit(1)
    else:
        traits = config_manager.list_traits()

    # Run pipeline
    try:
        for trait_name in traits:
            if args.generation_only:
                pipeline.run_generation_phase(trait_name)
            elif args.evaluation_only:
                pipeline.run_evaluation_phase(trait_name)
            else:
                pipeline.run_full_pipeline(trait_name)

        print(f"\n{'#'*80}")
        print("# PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'#'*80}\n")

    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
