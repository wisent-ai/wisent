"""
CLI for viewing and updating inference configuration.

Usage:
    wisent inference-config show
    wisent inference-config set --temperature 0.7 --top_p 0.9
    wisent inference-config reset
"""

import argparse
import json
from wisent.core.models.inference_config import (
    get_config,
    update_config,
    reset_config,
    CONFIG_FILE,
)


def main():
    parser = argparse.ArgumentParser(
        description="View and update inference configuration"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show current config")

    # Set command
    set_parser = subparsers.add_parser("set", help="Update config values")
    set_parser.add_argument("--do-sample", type=lambda x: x.lower() == "true", help="Enable sampling (true/false)")
    set_parser.add_argument("--temperature", type=float, help="Sampling temperature")
    set_parser.add_argument("--top-p", type=float, help="Top-p (nucleus) sampling")
    set_parser.add_argument("--top-k", type=int, help="Top-k sampling")
    set_parser.add_argument("--max-new-tokens", type=int, help="Max new tokens to generate")
    set_parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty")
    set_parser.add_argument("--no-repeat-ngram-size", type=int, help="No repeat n-gram size")
    set_parser.add_argument("--enable-thinking", type=lambda x: x.lower() == "true", help="Enable thinking mode (true/false)")

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset config to defaults")

    args = parser.parse_args()

    if args.command == "show" or args.command is None:
        config = get_config()
        print(f"Inference Config (stored at {CONFIG_FILE}):")
        print("-" * 50)
        print(json.dumps(config.to_dict(), indent=2))

    elif args.command == "set":
        updates = {}
        if args.do_sample is not None:
            updates["do_sample"] = args.do_sample
        if args.temperature is not None:
            updates["temperature"] = args.temperature
        if args.top_p is not None:
            updates["top_p"] = args.top_p
        if args.top_k is not None:
            updates["top_k"] = args.top_k
        if args.max_new_tokens is not None:
            updates["max_new_tokens"] = args.max_new_tokens
        if args.repetition_penalty is not None:
            updates["repetition_penalty"] = args.repetition_penalty
        if args.no_repeat_ngram_size is not None:
            updates["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        if args.enable_thinking is not None:
            updates["enable_thinking"] = args.enable_thinking

        if updates:
            config = update_config(**updates)
            print("Updated config:")
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print("No updates specified. Use --help to see options.")

    elif args.command == "reset":
        config = reset_config()
        print("Config reset to defaults:")
        print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()
