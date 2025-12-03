"""Parser for inference config command."""

import argparse


def setup_inference_config_parser(parser: argparse.ArgumentParser) -> None:
    """Set up the inference-config command parser."""
    subparsers = parser.add_subparsers(dest="subcommand", help="Config subcommands")

    # Show command
    subparsers.add_parser("show", help="Show current inference config")

    # Set command
    set_parser = subparsers.add_parser("set", help="Update inference config values")
    set_parser.add_argument(
        "--do-sample",
        type=lambda x: x.lower() == "true",
        metavar="BOOL",
        help="Enable sampling (true/false)"
    )
    set_parser.add_argument(
        "--temperature",
        type=float,
        metavar="FLOAT",
        help="Sampling temperature (e.g., 0.7)"
    )
    set_parser.add_argument(
        "--top-p",
        type=float,
        metavar="FLOAT",
        help="Top-p (nucleus) sampling (e.g., 0.9)"
    )
    set_parser.add_argument(
        "--top-k",
        type=int,
        metavar="INT",
        help="Top-k sampling (e.g., 50)"
    )
    set_parser.add_argument(
        "--max-new-tokens",
        type=int,
        metavar="INT",
        help="Max new tokens to generate (e.g., 512)"
    )
    set_parser.add_argument(
        "--repetition-penalty",
        type=float,
        metavar="FLOAT",
        help="Repetition penalty (e.g., 1.0)"
    )
    set_parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        metavar="INT",
        help="No repeat n-gram size (e.g., 0)"
    )
    set_parser.add_argument(
        "--enable-thinking",
        type=lambda x: x.lower() == "true",
        metavar="BOOL",
        help="Enable thinking mode for Qwen3 models (true/false)"
    )

    # Reset command
    subparsers.add_parser("reset", help="Reset inference config to defaults")
