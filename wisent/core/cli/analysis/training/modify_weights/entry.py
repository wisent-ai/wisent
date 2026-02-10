"""Main entry point for weight modification command."""
from __future__ import annotations

import json, sys, time
from typing import Dict, Optional

import torch
from wisent.core.cli.cli_logger import setup_logger, bind
from wisent.core.models.wisent_model import WisentModel
from wisent.core.utils import resolve_default_device
from .method_training import get_all_layers, auto_select_steering_method
from .vector_loading import (
    load_vectors_from_file, generate_personalization_vectors,
    generate_multi_benchmark_vectors, generate_task_vectors,
)
from .executor import (
    execute_standard_modification, execute_titan_mode, execute_pulse_mode,
    execute_prism_mode, execute_guided_modification, execute_multi_concept_modification,
    execute_null_space_modification,
)

_LOG = setup_logger(__name__)


def execute_modify_weights(args):
    """Execute weight modification command."""
    from wisent.core.tasks.base.task_selector import expand_task_if_skill_or_risk
    if getattr(args, 'task', None):
        args.task = expand_task_if_skill_or_risk(args.task)
    log = bind(_LOG)
    start_time = time.time()
    needs_auto_selection = (args.method == "auto" or getattr(args, 'steering_method', 'auto') == "auto")

    if args.verbose:
        print("\n" + "=" * 80)
        print("WEIGHT MODIFICATION")
        print("=" * 80)
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print("=" * 80 + "\n")

    if args.method == "null-space":
        _execute_null_space_from_activations(args, start_time)
        return

    steering_vectors = _load_or_generate_vectors(args, needs_auto_selection)
    harmless_vectors = _load_harmless_vectors(args)
    wisent_model, model, tokenizer = _load_model(args)

    if needs_auto_selection and args.task:
        steering_vectors = _run_auto_selection(args, wisent_model, steering_vectors)
    if getattr(args, 'guided', False):
        execute_guided_modification(args, wisent_model, model, tokenizer)
        return
    if getattr(args, 'concepts', None):
        execute_multi_concept_modification(args, wisent_model, model, tokenizer, steering_vectors)
        return

    pairs = None
    if args.method in ("titan", "pulse", "prism"):
        pairs = _generate_pairs(args, wisent_model)
        if not pairs:
            print(f"Error: Could not generate pairs for {args.method.upper()}")
            sys.exit(1)

    for method_name, executor_fn in [("titan", execute_titan_mode), ("pulse", execute_pulse_mode), ("prism", execute_prism_mode)]:
        if args.method == method_name:
            if getattr(args, 'steering_method', 'caa') != method_name:
                args.method = "directional"
            else:
                executor_fn(args, model, tokenizer, wisent_model, pairs)
                _print_timing(args, start_time)
                return

    stats = execute_standard_modification(args, model, tokenizer, steering_vectors, harmless_vectors)
    _print_timing(args, start_time)
    _print_summary(args, stats)


def _execute_null_space_from_activations(args, start_time: float):
    """Execute null-space weight modification from get-activations JSON."""
    from wisent.core.geometry.repscan.repscan_with_concepts import load_activations_from_json
    from wisent.core.weight_modification.directional.null_space.projector import PreservedKeyMatrix

    activations_json = getattr(args, 'activations_json', None)
    if not activations_json:
        print("Error: --activations-json is required for --method null-space")
        sys.exit(1)
    if args.verbose:
        print(f"Loading activations from {activations_json}...")

    activations_by_layer, _ = load_activations_from_json(activations_json)
    steering_vectors = {layer: pos.mean(0) - neg.mean(0) for layer, (pos, neg) in activations_by_layer.items()}
    preserved_keys = PreservedKeyMatrix()
    preserved_keys.accumulate({layer: pos for layer, (pos, neg) in activations_by_layer.items()})

    if args.verbose:
        print(f"Computed {len(steering_vectors)} steering vectors and preserved keys")

    _, model, tokenizer = _load_model(args)
    stats = execute_null_space_modification(args, model, tokenizer, steering_vectors, preserved_keys)
    _print_timing(args, start_time)
    _print_summary(args, stats)


def _load_or_generate_vectors(args, needs_auto_selection: bool) -> Optional[Dict[int, torch.Tensor]]:
    """Load or generate steering vectors based on arguments."""
    if needs_auto_selection and not args.steering_vectors:
        if args.verbose:
            print("Skipping initial vector generation (will generate after auto-selection)\n")
        return None
    if args.steering_vectors:
        if args.verbose:
            print(f"Loading steering vectors from {args.steering_vectors}...")
        return load_vectors_from_file(args.steering_vectors, args.verbose)
    if args.task:
        task_lower = args.task.lower()
        if task_lower in ("personalization", "refusal", "custom"):
            if task_lower == "refusal":
                args.trait = "refusal"
            if args.verbose and args.trait:
                print(f"Generating vectors from trait '{args.trait}'...")
            return generate_personalization_vectors(args, args.verbose)
        if "," in args.task:
            return generate_multi_benchmark_vectors(args, args.verbose)
        return generate_task_vectors(args, args.verbose)
    return None


def _load_harmless_vectors(args) -> Optional[Dict[int, torch.Tensor]]:
    """Load harmless vectors for biprojection if provided."""
    if args.method != "directional" or getattr(args, 'no_biprojection', False):
        return None
    if not hasattr(args, 'harmless_vectors') or not args.harmless_vectors:
        return None
    if args.verbose:
        print(f"Loading harmless vectors from {args.harmless_vectors}...")
    with open(args.harmless_vectors, 'r') as f:
        harmless_data = json.load(f)
    harmless_dict = harmless_data.get("steering_vectors") or harmless_data.get("vectors", {})
    harmless_vectors = {int(layer) - 1: torch.tensor(vector) for layer, vector in harmless_dict.items()}
    if args.verbose:
        print(f"Loaded {len(harmless_vectors)} harmless vectors for biprojection\n")
    return harmless_vectors


def _load_model(args):
    """Load the model for weight modification."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    if args.verbose:
        print(f"Loading model '{args.model}'...")
    enable_bias = (args.method == "additive" and getattr(args, 'additive_method', 'bias') == 'bias')
    if enable_bias:
        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        if hasattr(config, 'attention_bias'):
            config.attention_bias = True
        if hasattr(config, 'mlp_bias'):
            config.mlp_bias = True
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, torch_dtype=torch.bfloat16,
            device_map='auto', trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

        class ModelInfo:
            def __init__(self, m):
                if hasattr(m, 'model') and hasattr(m.model, 'layers'):
                    self.num_layers = len(m.model.layers)
                elif hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
                    self.num_layers = len(m.transformer.h)
                else:
                    self.num_layers = 32
        wisent_model = ModelInfo(model)
    else:
        wisent_model = WisentModel(args.model, device=getattr(args, 'device', None))
        model = wisent_model.hf_model
        tokenizer = wisent_model.tokenizer
    if args.verbose:
        print(f"Model loaded with {wisent_model.num_layers} layers\n")
    return wisent_model, model, tokenizer


def _run_auto_selection(args, wisent_model, steering_vectors):
    """Run auto-selection and generate CAA vectors if needed."""
    from wisent.core.activations.activations_collector import ActivationCollector
    from wisent.core.activations import ExtractionStrategy
    from wisent.core.steering_methods.methods.caa import CAAMethod
    from wisent.core.contrastive_pairs.core.set import ContrastivePairSet
    pairs = _generate_pairs(args, wisent_model)
    if pairs and len(pairs) >= 10:
        steering_method, modification_method, _ = auto_select_steering_method(pairs, wisent_model, args.verbose)
        args.steering_method = steering_method
        args.method = modification_method
        if modification_method == "directional" and steering_vectors is None:
            collector = ActivationCollector(model=wisent_model)
            all_layers = get_all_layers(wisent_model)
            enriched_pairs = [collector.collect(p, strategy=ExtractionStrategy.default(), layers=all_layers) for p in pairs]
            pair_set = ContrastivePairSet(pairs=enriched_pairs, name="auto_caa")
            caa_method = CAAMethod()
            caa_result = caa_method.train(pair_set)
            steering_vectors = {}
            for layer_name, vector in caa_result.directions.items():
                layer_idx = int(layer_name.replace("layer_", "")) if "layer_" in str(layer_name) else int(layer_name)
                steering_vectors[layer_idx] = vector
    else:
        args.steering_method = "titan"
        args.method = "titan"
    return steering_vectors


def _generate_pairs(args, wisent_model):
    """Generate contrastive pairs for training."""
    from wisent.core.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    trait = getattr(args, 'trait', None)
    use_synthetic = trait and (not args.task or args.task.lower() == "personalization")
    if args.task and not use_synthetic:
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
        return list(build_contrastive_pairs(task_name=args.task, limit=args.num_pairs))
    if trait:
        from wisent.core.contrastive_pairs.generators.llm_synthetic import LLMSyntheticGenerator
        generator = LLMSyntheticGenerator(model_id=args.model)
        raw_pairs = generator.generate_pairs(trait=trait, num_pairs=args.num_pairs)
        return [ContrastivePair(
            prompt=rp.get('prompt', ''),
            positive_response=PositiveResponse(model_response=rp.get('positive_response', '')),
            negative_response=NegativeResponse(model_response=rp.get('negative_response', '')),
            label=trait,
        ) for rp in raw_pairs]
    return []


def _print_timing(args, start_time: float):
    """Print timing information if requested."""
    if args.timing:
        print(f"\nTotal time: {time.time() - start_time:.2f}s")


def _print_summary(args, stats: Dict):
    """Print final summary."""
    if args.verbose:
        print("\n" + "=" * 80)
        print("WEIGHT MODIFICATION COMPLETE")
        print("=" * 80)
        print(f"Modified model: {args.output_dir}")
        print(f"Method: {args.method}")
        print(f"Layers modified: {stats['layers_modified']}")
        print(f"Parameters modified: {stats['total_parameters_modified']:,}")
        print("=" * 80 + "\n")
