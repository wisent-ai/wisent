"""Activation analysis helpers for verify-steering."""
import torch
from typing import Any, Dict, List, Optional
from wisent.core.utils.config_tools.constants import (
    COMPARE_TOL,
    DISPLAY_TRUNCATION_SHORT,
    MAX_NEW_TOKENS_VERIFY_SINGLE,
    SEPARATOR_WIDTH_MEDIUM_PLUS,
    SEPARATOR_WIDTH_STANDARD,
    SEPARATOR_WIDTH_WIDE,
)


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
        prompt_result = {"prompt": prompt[:DISPLAY_TRUNCATION_SHORT], "layers": {}}

        for layer_name, direction in steering_dirs.items():
            if layer_name in base_acts and layer_name in steered_acts:
                base_act = base_acts[layer_name][0].float()
                steered_act = steered_acts[layer_name][0].float().to(base_act.device)
                direction = direction.to(base_act.device).float()

                shift = steered_act - base_act
                shift_norm = shift.norm().item()

                if shift_norm > COMPARE_TOL:
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
    print("-" * SEPARATOR_WIDTH_WIDE)

    for result in all_results:
        alignments = [v["alignment"] for v in result["layers"].values()]
        avg = sum(alignments) / len(alignments) if alignments else 0
        status = "ALIGNED" if avg > 0.3 else ("WEAK" if avg > 0 else "MISALIGNED")
        print(f"{result['prompt']:<40} {avg:>12.4f}    {status}")

    if verbose:
        print("\n" + "-" * SEPARATOR_WIDTH_WIDE)
        print("Per-Layer Alignment:")
        print(f"{'Layer':<10} {'Avg Alignment':<15} {'Min':<10} {'Max':<10} {'Status'}")
        print("-" * SEPARATOR_WIDTH_MEDIUM_PLUS)

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
    from wisent.core.weight_modification.standalone_loader import GROMHooks

    hooks = GROMHooks(steered_model, steering_data)
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
                max_new_tokens=MAX_NEW_TOKENS_VERIFY_SINGLE,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if check_gate:
            gate = hooks.get_current_gate()
            gate_results.append({"prompt": prompt[:DISPLAY_TRUNCATION_SHORT], "gate": gate})

        if check_intensity:
            intensities = hooks.get_current_intensities()
            intensity_results.append({"prompt": prompt[:DISPLAY_TRUNCATION_SHORT], "intensities": intensities})

    hooks.remove()

    # Print gate results
    if gate_results:
        print(f"\nGate Values:")
        print(f"{'Prompt':<40} {'Gate':<10} {'Status'}")
        print("-" * SEPARATOR_WIDTH_STANDARD)
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
