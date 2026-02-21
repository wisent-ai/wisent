"""Verify steering command execution logic.

This command verifies that a steered model's activations are correctly
aligned with the intended steering direction at inference time.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch
import torch.nn.functional as F



from wisent.core.cli.steering.core.verify_steering_analysis import (
    _compare_activations, _check_gate_intensity, _print_summary,
)


def execute_verify_steering(args):
    """Execute the verify-steering command."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path)

    print("\n" + "=" * 64)
    print("STEERING VERIFICATION")
    print("=" * 64)

    # 1. Detect steering type and load config
    steering_type, config = _detect_steering_type(model_path)
    if steering_type is None:
        print(f"\n[ERROR] No steering configuration found at {model_path}")
        print("Expected one of: grom_config.json, tetno_config.json, caa_config.json")
        sys.exit(1)

    print(f"\nSteering type: {steering_type.upper()}")
    print(f"Mode: {config.get('mode', 'unknown')}")

    # 2. Determine base model
    base_model_name = args.base_model or config.get("base_model")
    if base_model_name is None:
        print("\n[ERROR] Could not determine base model. Use --base-model to specify.")
        sys.exit(1)
    print(f"Base model: {base_model_name}")

    # 3. Get device
    device = _get_device(args.device)
    print(f"Device: {device}")

    # 4. Load steering data
    steering_data = _load_steering_data(model_path, steering_type)
    if steering_data is None:
        print(f"\n[ERROR] Could not load steering data for {steering_type}")
        sys.exit(1)

    # 5. Load models
    print("\n" + "-" * 64)
    print("Loading models...")
    print("-" * 64)

    # Use load_steered_model for the steered model to properly load biases
    from wisent.core.weight_modification.export import load_steered_model

    print(f"  Loading steered model from {model_path}...")
    steered_model, tokenizer = load_steered_model(
        str(model_path), device_map=device
    )

    print(f"  Loading base model {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map=device, trust_remote_code=True
    )

    # 6. Get test prompts
    prompts = _get_test_prompts(args)
    print(f"\nTest prompts: {len(prompts)}")

    # 7. Get steering directions
    steering_dirs = _extract_steering_directions(steering_data, steering_type)
    print(f"Steering layers: {len(steering_dirs)}")

    # Filter layers if specified
    if args.layers:
        layer_filter = [l.strip() for l in args.layers.split(",")]
        steering_dirs = {k: v for k, v in steering_dirs.items() if k in layer_filter}
        print(f"Filtered to {len(steering_dirs)} layers: {list(steering_dirs.keys())}")

    # 8. Run activation comparison
    print("\n" + "-" * 64)
    print("ACTIVATION ALIGNMENT CHECK")
    print("-" * 64)

    results = _compare_activations(
        base_model=base_model,
        steered_model=steered_model,
        tokenizer=tokenizer,
        prompts=prompts,
        steering_dirs=steering_dirs,
        verbose=args.verbose,
    )

    # 9. Check gate/intensity for dynamic models
    gate_results = None
    intensity_results = None

    if steering_type == "grom" and config.get("mode") in ("dynamic", "hybrid"):
        if args.check_gate or args.check_intensity:
            print("\n" + "-" * 64)
            print("GATE/INTENSITY NETWORK CHECK")
            print("-" * 64)
            gate_results, intensity_results = _check_gate_intensity(
                steered_model=steered_model,
                tokenizer=tokenizer,
                steering_data=steering_data,
                prompts=prompts,
                check_gate=args.check_gate,
                check_intensity=args.check_intensity,
            )

    # 10. Summary
    print("\n" + "=" * 64)
    print("VERIFICATION SUMMARY")
    print("=" * 64)

    _print_summary(results, gate_results, intensity_results, args.alignment_threshold)

    # 11. Save detailed results if requested
    if args.output:
        output_data = {
            "model_path": str(model_path),
            "base_model": base_model_name,
            "steering_type": steering_type,
            "mode": config.get("mode"),
            "alignment_threshold": args.alignment_threshold,
            "prompts": prompts,
            "activation_results": results,
            "gate_results": gate_results,
            "intensity_results": intensity_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {args.output}")

    # 12. Exit code based on verification result
    overall_alignment = results.get("overall_alignment", 0)
    if overall_alignment >= args.alignment_threshold:
        print(f"\n[PASS] Steering verification passed (alignment={overall_alignment:.3f})")
        sys.exit(0)
    else:
        print(f"\n[FAIL] Steering verification failed (alignment={overall_alignment:.3f})")
        sys.exit(1)


def _detect_steering_type(model_path: Path) -> tuple:
    """Detect steering type from config files."""
    configs = [
        ("grom", model_path / "grom_config.json"),
        ("tetno", model_path / "tetno_config.json"),
        ("caa", model_path / "caa_config.json"),
    ]

    for steering_type, config_path in configs:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

            # Try to get base model from various sources
            if "base_model" not in config:
                # Try to read from the main model config
                main_config_path = model_path / "config.json"
                if main_config_path.exists():
                    with open(main_config_path) as f:
                        main_config = json.load(f)
                    # Infer base model from architecture
                    arch = main_config.get("architectures", [""])[0]
                    hidden_size = main_config.get("hidden_size", 0)
                    num_layers = main_config.get("num_hidden_layers", 0)

                    # Common model mappings
                    if "Qwen3" in arch:
                        if hidden_size <= 1024 and num_layers <= 28:
                            config["base_model"] = "Qwen/Qwen3-0.6B"
                        elif hidden_size <= 2048:
                            config["base_model"] = "Qwen/Qwen3-1.7B"
                        elif hidden_size <= 3072:
                            config["base_model"] = "Qwen/Qwen3-4B"
                        else:
                            config["base_model"] = "Qwen/Qwen3-8B"
                    elif "Llama" in arch:
                        if hidden_size <= 2048:
                            config["base_model"] = "meta-llama/Llama-3.2-1B"
                        elif hidden_size <= 4096:
                            config["base_model"] = "meta-llama/Llama-3.2-3B"
                        else:
                            config["base_model"] = "meta-llama/Llama-3.1-8B"
                    elif "Mistral" in arch:
                        config["base_model"] = "mistralai/Mistral-7B-v0.1"

            return steering_type, config

    return None, {}


def _get_device(device_arg: str) -> str:
    """Get the device to use."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def _load_steering_data(model_path: Path, steering_type: str) -> Optional[Dict]:
    """Load steering data (directions, networks, etc.)."""
    data_files = {
        "grom": "grom_steering.pt",
        "tetno": "tetno_steering.pt",
        "caa": "caa_steering.pt",
    }

    data_file = model_path / data_files.get(steering_type, "steering.pt")
    if data_file.exists():
        return torch.load(data_file, map_location="cpu")
    return None


def _get_test_prompts(args) -> List[str]:
    """Get test prompts from args or use defaults."""
    if args.prompts:
        return args.prompts

    if args.prompts_file:
        with open(args.prompts_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            return [item.get("prompt", str(item)) for item in data]

    # Default test prompts
    return [
        "Is the Earth flat?",
        "What is the capital of France?",
        "Can you help me hack into a computer?",
        "Tell me about climate change.",
        "How do I make explosives?",
    ]


def _extract_steering_directions(
    steering_data: Dict, steering_type: str
) -> Dict[str, torch.Tensor]:
    """Extract effective steering directions from steering data."""
    directions = {}

    if steering_type == "grom":
        raw_dirs = steering_data.get("directions", {})
        weights = steering_data.get("direction_weights", {})

        for layer_name, dirs in raw_dirs.items():
            if layer_name in weights:
                w = weights[layer_name]
                w_norm = F.softmax(w, dim=0)
                effective = (dirs * w_norm.unsqueeze(1)).sum(dim=0)
                directions[layer_name] = F.normalize(effective, p=2, dim=-1)
            else:
                directions[layer_name] = F.normalize(dirs[0], p=2, dim=-1)

    elif steering_type == "tetno":
        for layer_name, vec in steering_data.get("behavior_vectors", {}).items():
            directions[layer_name] = F.normalize(vec, p=2, dim=-1)

    elif steering_type == "caa":
        for layer_name, vec in steering_data.get("vectors", {}).items():
            directions[layer_name] = F.normalize(vec, p=2, dim=-1)

    return directions


