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
        print("Expected one of: titan_config.json, pulse_config.json, caa_config.json")
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

    if steering_type == "titan" and config.get("mode") in ("dynamic", "hybrid"):
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
        ("titan", model_path / "titan_config.json"),
        ("pulse", model_path / "pulse_config.json"),
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
        "titan": "titan_steering.pt",
        "pulse": "pulse_steering.pt",
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

    if steering_type == "titan":
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

    elif steering_type == "pulse":
        for layer_name, vec in steering_data.get("behavior_vectors", {}).items():
            directions[layer_name] = F.normalize(vec, p=2, dim=-1)

    elif steering_type == "caa":
        for layer_name, vec in steering_data.get("vectors", {}).items():
            directions[layer_name] = F.normalize(vec, p=2, dim=-1)

    return directions


def _compare_activations(
    base_model,
    steered_model,
    tokenizer,
    prompts: List[str],
    steering_dirs: Dict[str, torch.Tensor],
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare activations between base and steered models."""
    # Get layer modules
    if hasattr(base_model, "model"):
        base_layers = base_model.model.layers
        steered_layers = steered_model.model.layers
    elif hasattr(base_model, "transformer"):
        base_layers = base_model.transformer.h
        steered_layers = steered_model.transformer.h
    else:
        base_layers = getattr(base_model, "layers", [])
        steered_layers = getattr(steered_model, "layers", [])

    # Map layer names to indices
    layer_name_to_idx = {}
    for layer_name in steering_dirs.keys():
        try:
            idx = int(str(layer_name).split("_")[-1]) if "_" in str(layer_name) else int(layer_name)
            layer_name_to_idx[layer_name] = idx
        except (ValueError, IndexError):
            pass

    all_results = []

    for prompt in prompts:
        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt")
        inputs_base = {k: v.to(base_model.device) for k, v in inputs.items()}
        inputs_steered = {k: v.to(steered_model.device) for k, v in inputs.items()}

        # Capture activations
        base_acts = {}
        steered_acts = {}

        def make_hook(storage, layer_name):
            def hook(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                storage[layer_name] = hidden[:, -1, :].detach().clone()
                return output
            return hook

        # Install hooks
        hooks = []
        for layer_name, layer_idx in layer_name_to_idx.items():
            if layer_idx < len(base_layers):
                h1 = base_layers[layer_idx].register_forward_hook(make_hook(base_acts, layer_name))
                h2 = steered_layers[layer_idx].register_forward_hook(make_hook(steered_acts, layer_name))
                hooks.extend([h1, h2])

        # Forward pass
        with torch.no_grad():
            _ = base_model(**inputs_base)
            _ = steered_model(**inputs_steered)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Compute alignments
        prompt_result = {"prompt": prompt[:50], "layers": {}}

        for layer_name, direction in steering_dirs.items():
            if layer_name in base_acts and layer_name in steered_acts:
                base_act = base_acts[layer_name][0].float()
                steered_act = steered_acts[layer_name][0].float().to(base_act.device)
                direction = direction.to(base_act.device).float()

                shift = steered_act - base_act
                shift_norm = shift.norm().item()

                if shift_norm > 1e-6:
                    alignment = F.cosine_similarity(
                        shift.unsqueeze(0), direction.unsqueeze(0)
                    ).item()
                else:
                    alignment = 0.0

                prompt_result["layers"][layer_name] = {
                    "shift_norm": shift_norm,
                    "alignment": alignment,
                    "base_norm": base_act.norm().item(),
                    "steered_norm": steered_act.norm().item(),
                }

        all_results.append(prompt_result)

    # Compute overall metrics
    all_alignments = []
    layer_alignments = {}

    for result in all_results:
        for layer_name, layer_data in result["layers"].items():
            all_alignments.append(layer_data["alignment"])
            if layer_name not in layer_alignments:
                layer_alignments[layer_name] = []
            layer_alignments[layer_name].append(layer_data["alignment"])

    overall_alignment = sum(all_alignments) / len(all_alignments) if all_alignments else 0

    # Print results
    print(f"\n{'Prompt':<40} {'Avg Alignment':<15} {'Status'}")
    print("-" * 70)

    for result in all_results:
        alignments = [v["alignment"] for v in result["layers"].values()]
        avg = sum(alignments) / len(alignments) if alignments else 0
        status = "ALIGNED" if avg > 0.3 else ("WEAK" if avg > 0 else "MISALIGNED")
        print(f"{result['prompt']:<40} {avg:>12.4f}    {status}")

    if verbose:
        print("\n" + "-" * 70)
        print("Per-Layer Alignment:")
        print(f"{'Layer':<10} {'Avg Alignment':<15} {'Min':<10} {'Max':<10} {'Status'}")
        print("-" * 55)

        for layer_name in sorted(layer_alignments.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            aligns = layer_alignments[layer_name]
            avg = sum(aligns) / len(aligns)
            min_a = min(aligns)
            max_a = max(aligns)
            status = "OK" if avg > 0.3 else ("WEAK" if avg > 0 else "BAD")
            print(f"{layer_name:<10} {avg:>12.4f}    {min_a:>8.4f}  {max_a:>8.4f}  {status}")

    return {
        "per_prompt": all_results,
        "per_layer": {k: sum(v) / len(v) for k, v in layer_alignments.items()},
        "overall_alignment": overall_alignment,
    }


def _check_gate_intensity(
    steered_model,
    tokenizer,
    steering_data: Dict,
    prompts: List[str],
    check_gate: bool,
    check_intensity: bool,
) -> tuple:
    """Check gate and intensity network predictions."""
    from wisent.core.weight_modification.standalone_loader import TITANHooks

    hooks = TITANHooks(steered_model, steering_data)
    hooks.install()

    gate_results = []
    intensity_results = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted = prompt

        inputs = tokenizer(formatted, return_tensors="pt").to(steered_model.device)

        with torch.no_grad():
            _ = steered_model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if check_gate:
            gate = hooks.get_current_gate()
            gate_results.append({"prompt": prompt[:50], "gate": gate})

        if check_intensity:
            intensities = hooks.get_current_intensities()
            intensity_results.append({"prompt": prompt[:50], "intensities": intensities})

    hooks.remove()

    # Print gate results
    if gate_results:
        print(f"\nGate Values:")
        print(f"{'Prompt':<40} {'Gate':<10} {'Status'}")
        print("-" * 60)
        for r in gate_results:
            gate = r["gate"]
            status = "ACTIVE" if gate > 0.7 else ("PARTIAL" if gate > 0.3 else "INACTIVE")
            print(f"{r['prompt']:<40} {gate:>8.4f}  {status}")

        # Check discrimination
        gates = [r["gate"] for r in gate_results]
        gate_std = torch.tensor(gates).std().item()
        if gate_std < 0.1:
            print(f"\n[WARNING] Gate not discriminating (std={gate_std:.4f})")
            print("  All prompts get similar gate values - gating network may need more training")

    # Print intensity summary
    if intensity_results:
        print(f"\nIntensity Summary:")
        for r in intensity_results:
            if r["intensities"]:
                avg = sum(r["intensities"].values()) / len(r["intensities"])
                print(f"  {r['prompt']}: avg={avg:.4f}")

    return gate_results, intensity_results


def _print_summary(
    results: Dict,
    gate_results: Optional[List],
    intensity_results: Optional[List],
    threshold: float,
):
    """Print verification summary."""
    overall = results.get("overall_alignment", 0)
    per_layer = results.get("per_layer", {})

    # Count layers by status
    good_layers = sum(1 for v in per_layer.values() if v > 0.5)
    weak_layers = sum(1 for v in per_layer.values() if 0 < v <= 0.5)
    bad_layers = sum(1 for v in per_layer.values() if v <= 0)

    print(f"\nOverall Alignment: {overall:.4f}")
    print(f"  - Threshold: {threshold}")
    print(f"  - Status: {'PASS' if overall >= threshold else 'FAIL'}")

    print(f"\nLayer Breakdown:")
    print(f"  - Well-aligned (>0.5): {good_layers}")
    print(f"  - Weak alignment (0-0.5): {weak_layers}")
    print(f"  - Misaligned (<0): {bad_layers}")

    if bad_layers > 0:
        print("\n[WARNING] Some layers have negative alignment!")
        print("  The steering direction is OPPOSITE to the activation shift.")
        print("  This will likely cause steering to fail or have opposite effect.")
        bad_layer_names = [k for k, v in per_layer.items() if v <= 0]
        print(f"  Problematic layers: {bad_layer_names}")

    if gate_results:
        gates = [r["gate"] for r in gate_results]
        avg_gate = sum(gates) / len(gates)
        print(f"\nGate Network:")
        print(f"  - Average gate: {avg_gate:.4f}")
        print(f"  - Std dev: {torch.tensor(gates).std().item():.4f}")

    print("\nInterpretation Guide:")
    print("  - Alignment > 0.5: Steering working correctly")
    print("  - Alignment 0-0.5: Steering is weak")
    print("  - Alignment < 0: Steering going WRONG direction")
    print("  - Gate ~0.5 for all: Gate not discriminating (needs more data)")
