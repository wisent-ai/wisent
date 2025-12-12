#!/usr/bin/env python3
"""
General dataset inspector for Hugging Face datasets.

Usage:
    python inspect_hf_dataset.py <dataset_name> [config_name]

Example:
    python inspect_hf_dataset.py ptb_text_only
    python inspect_hf_dataset.py FALcon6/ptb_text_only
    python inspect_hf_dataset.py super_glue boolq
    python inspect_hf_dataset.py glue mrpc
"""

import argparse
import sys
from datasets import load_dataset, get_dataset_config_names


def inspect_hf_dataset(dataset_name: str, config_name: str | None = None, num_examples: int = 5) -> None:
    """Load a dataset from Hugging Face and display its structure and examples."""

    print(f"\n{'='*80}")
    print(f"INSPECTING HF DATASET: {dataset_name}")
    if config_name:
        print(f"CONFIG: {config_name}")
    print(f"{'='*80}\n")

    # Try to get available configs
    try:
        configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
        if configs:
            print(f"AVAILABLE CONFIGS: {configs}")
            if not config_name and len(configs) > 1:
                print(f"\nNote: Multiple configs available. Using first config: '{configs[0]}'")
                print("Specify config with: python inspect_hf_dataset.py <dataset> <config>\n")
                config_name = configs[0]
    except Exception as e:
        print(f"Could not get config names: {e}")
        configs = []

    # Try all common splits
    split_names = ['train', 'test', 'validation', 'dev']

    print("\nAVAILABLE SPLITS:")
    print("-" * 40)

    available_splits = {}
    for split_name in split_names:
        try:
            if config_name:
                ds = load_dataset(dataset_name, config_name, split=split_name, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
            available_splits[split_name] = ds
            print(f"  {split_name}: {len(ds)} samples")
        except ValueError as e:
            if "Unknown split" in str(e) or "should be one of" in str(e):
                print(f"  {split_name}: not available")
            else:
                print(f"  {split_name}: error - {e}")
        except Exception as e:
            print(f"  {split_name}: error - {e}")

    if not available_splits:
        print("\nNo data found in any split!")
        # Try loading without split to see what's available
        try:
            if config_name:
                ds = load_dataset(dataset_name, config_name, trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, trust_remote_code=True)
            print(f"\nDataset structure: {ds}")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
        sys.exit(1)

    # For each available split, show examples
    for split_name, ds in available_splits.items():
        print(f"\n{'='*80}")
        print(f"SPLIT: {split_name.upper()} ({len(ds)} samples)")
        print(f"{'='*80}")

        # Show structure
        print(f"\nColumn names: {ds.column_names}")
        print(f"\nFeatures:")
        for name, feature in ds.features.items():
            print(f"  {name}: {feature}")

        # Show examples
        print(f"\n{'-'*40}")
        print(f"FIRST {min(num_examples, len(ds))} EXAMPLES:")
        print(f"{'-'*40}")

        for i in range(min(num_examples, len(ds))):
            doc = ds[i]
            print(f"\n--- EXAMPLE {i+1} ---")
            for key, value in doc.items():
                if isinstance(value, str):
                    print(f"{key}: {value}")
                elif isinstance(value, list):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value} (type: {type(value).__name__})")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Hugging Face dataset format and data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python inspect_hf_dataset.py ptb_text_only
    python inspect_hf_dataset.py FALcon6/ptb_text_only
    python inspect_hf_dataset.py super_glue boolq
    python inspect_hf_dataset.py glue mrpc
    python inspect_hf_dataset.py --examples 10 squad
        """
    )
    parser.add_argument(
        "dataset",
        help="Name of the Hugging Face dataset to inspect (e.g., 'glue', 'squad', 'user/dataset')"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Optional config/subset name (e.g., 'mrpc' for glue)"
    )
    parser.add_argument(
        "--examples", "-n",
        type=int,
        default=5,
        help="Number of examples to display per split (default: 5)"
    )

    args = parser.parse_args()
    inspect_hf_dataset(args.dataset, args.config, args.examples)


if __name__ == "__main__":
    main()
