"""CLI execution for inference config command."""

import json
from wisent.core.models.inference_config import (
    get_config,
    update_config,
    reset_config,
    CONFIG_FILE,
)


def execute_inference_config(args):
    """Execute the inference-config command."""
    subcommand = getattr(args, "subcommand", None)

    if subcommand == "show" or subcommand is None:
        config = get_config()
        print(f"Inference Config (stored at {CONFIG_FILE}):")
        print("-" * 50)
        print(json.dumps(config.to_dict(), indent=2))

    elif subcommand == "set":
        updates = {}

        # Map CLI args (with dashes) to config attrs (with underscores)
        arg_mapping = {
            "do_sample": "do_sample",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "max_new_tokens": "max_new_tokens",
            "repetition_penalty": "repetition_penalty",
            "no_repeat_ngram_size": "no_repeat_ngram_size",
            "enable_thinking": "enable_thinking",
        }

        for arg_name, config_name in arg_mapping.items():
            value = getattr(args, arg_name, None)
            if value is not None:
                updates[config_name] = value

        if updates:
            config = update_config(**updates)
            print("Updated inference config:")
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print("No updates specified. Use --help to see available options.")
            print("\nCurrent config:")
            print(json.dumps(get_config().to_dict(), indent=2))

    elif subcommand == "reset":
        config = reset_config()
        print("Inference config reset to defaults:")
        print(json.dumps(config.to_dict(), indent=2))
