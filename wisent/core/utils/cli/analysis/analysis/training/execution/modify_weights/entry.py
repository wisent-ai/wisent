"""Main entry point for weight modification command."""
from __future__ import annotations
import json
import sys
import time
from typing import Dict, Optional
import torch
from wisent.core.utils.cli.cli_logger import setup_logger, bind
from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.core.utils import resolve_default_device
from .method_training import get_all_layers
from .vector_loading import (
    load_vectors_from_file, generate_personalization_vectors,
    generate_multi_benchmark_vectors, generate_task_vectors,
)
from .executor import (
    execute_standard_modification, execute_grom_mode, execute_tetno_mode,
    execute_tecza_mode, execute_guided_modification, execute_multi_concept_modification,
    execute_szlak_mode, execute_wicher_mode,
)
from wisent.core.weight_modification.utils import get_default_components_for_extraction
from wisent.core import constants as _C
_LOG = setup_logger(__name__)

def execute_modify_weights(args):
    """Execute weight modification command."""
    from wisent.core.control.tasks.base.task_selector import expand_task_if_skill_or_risk
    if getattr(args, 'task', None):
        args.task = expand_task_if_skill_or_risk(args.task)
    log = bind(_LOG)
    start_time = time.time()
    if args.method == "auto" or getattr(args, 'steering_method', 'auto') == "auto":
        raise ValueError("Auto method selection has been removed. Specify --method and --steering-method explicitly.")
    # Auto-select weight-modification components from extraction component
    extraction_component = getattr(args, 'extraction_component', 'residual_stream')
    if getattr(args, 'components', None) is None and extraction_component != 'residual_stream':
        args.components = get_default_components_for_extraction(extraction_component)
        if getattr(args, 'verbose', False):
            print(f"Auto-selected --components {args.components} from --extraction-component {extraction_component}")
    if args.verbose:
        print("\n" + "=" * _C.SEPARATOR_WIDTH_REPORT)
        print("WEIGHT MODIFICATION")
        print("=" * _C.SEPARATOR_WIDTH_REPORT)
        print(f"Model: {args.model}")
        print(f"Output: {args.output_dir}")
        print("=" * _C.SEPARATOR_WIDTH_REPORT + "\n")
    steering_vectors = _load_or_generate_vectors(args)
    harmless_vectors = _load_harmless_vectors(args)
    wisent_model, model, tokenizer = _load_model(args)
    if getattr(args, 'guided', False):
        execute_guided_modification(args, wisent_model, model, tokenizer)
        return
    if getattr(args, 'concepts', None):
        execute_multi_concept_modification(args, wisent_model, model, tokenizer, steering_vectors)
        return
    pairs = None
    if args.method in ("grom", "tetno", "tecza", "nurt", "szlak", "wicher"):
        pairs = _generate_pairs(args, wisent_model)
        if not pairs:
            print(f"Error: Could not generate pairs for {args.method.upper()}")
            sys.exit(1)

    if args.method == "grom":
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method != "grom":
            args.method = "directional"
        else:
            execute_grom_mode(args, model, tokenizer, wisent_model, pairs)
            _print_timing(args, start_time)
            return

    if args.method == "tetno":
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method != "tetno":
            args.method = "directional"
        else:
            execute_tetno_mode(args, model, tokenizer, wisent_model, pairs)
            _print_timing(args, start_time)
            return

    if args.method == "tecza":
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method != "tecza":
            args.method = "directional"
        else:
            execute_tecza_mode(args, model, tokenizer, wisent_model, pairs)
            _print_timing(args, start_time)
            return

    if args.method == "nurt":
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method == "nurt":
            from .method_training import train_nurt_for_task
            from wisent.core.weight_modification.export import export_nurt_model
            cf_steering = train_nurt_for_task(args, wisent_model, pairs)
            export_nurt_model(
                model=model, nurt_steering=cf_steering,
                save_path=args.output_dir, tokenizer=tokenizer,
                base_strength=args.strength, push_to_hub=args.push_to_hub,
                repo_id=args.repo_id if args.push_to_hub else None,
                commit_message=args.commit_message,
            )
            _print_timing(args, start_time)
            return
        else:
            args.method = "directional"

    if args.method == "szlak":
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method == "szlak":
            execute_szlak_mode(args, model, tokenizer, wisent_model, pairs)
            _print_timing(args, start_time)
            return
        else:
            args.method = "directional"

    if args.method == "wicher":
        steering_method = getattr(args, 'steering_method', 'caa')
        if steering_method == "wicher":
            execute_wicher_mode(args, model, tokenizer, wisent_model, pairs)
            _print_timing(args, start_time)
            return
        else:
            args.method = "directional"

    stats = execute_standard_modification(args, model, tokenizer, steering_vectors, harmless_vectors)
    _print_timing(args, start_time)
    _print_summary(args, stats)

def _load_or_generate_vectors(args) -> Optional[Dict[int, torch.Tensor]]:
    """Load or generate steering vectors based on arguments."""
    if args.steering_vectors:
        if args.verbose:
            print(f"Loading steering vectors from {args.steering_vectors}...")
        return load_vectors_from_file(args.steering_vectors, args.verbose)
    if args.task:
        task_lower = args.task.lower()
        if task_lower == "personalization" and args.trait:
            if args.verbose:
                print(f"Generating vectors from trait '{args.trait}'...")
            return generate_personalization_vectors(args, args.verbose)
        if task_lower == "refusal":
            args.trait = "refusal"
            return generate_personalization_vectors(args, args.verbose)
        if task_lower == "custom" and args.trait:
            return generate_personalization_vectors(args, args.verbose)
        if "," in args.task:
            return generate_multi_benchmark_vectors(args, args.verbose)
        return generate_task_vectors(args, args.verbose)
    return None

def _load_harmless_vectors(args) -> Optional[Dict[int, torch.Tensor]]:
    """Load harmless vectors for biprojection if provided."""
    use_biprojection = not getattr(args, 'no_biprojection', False)

    if args.method != "directional" or not use_biprojection:
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
                    raise ValueError("num_layers must be specified: cannot detect layers from model architecture")
        wisent_model = ModelInfo(model)
    else:
        wisent_model = WisentModel(args.model, device=getattr(args, 'device', None))
        model = wisent_model.hf_model
        tokenizer = wisent_model.tokenizer
    if args.verbose:
        print(f"Model loaded with {wisent_model.num_layers} layers\n")
    return wisent_model, model, tokenizer

def _generate_pairs(args, wisent_model):
    """Generate contrastive pairs for training."""
    from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
    from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
    pairs = []
    if args.task:
        task_lower = args.task.lower()
        if task_lower == "personalization" and args.trait:
            from wisent.core.primitives.contrastive_pairs.generators.llm_synthetic import LLMSyntheticGenerator
            generator = LLMSyntheticGenerator(model_id=args.model)
            raw_pairs = generator.generate_pairs(trait=args.trait, num_pairs=args.num_pairs)
            for rp in raw_pairs:
                pairs.append(ContrastivePair(
                    prompt=rp.get('prompt', ''),
                    positive_response=PositiveResponse(model_response=rp.get('positive_response', '')),
                    negative_response=NegativeResponse(model_response=rp.get('negative_response', '')),
                    label=args.trait,
                ))
        else:
            from wisent.extractors.lm_eval.lm_task_pairs_generation import build_contrastive_pairs
            raw_pairs = build_contrastive_pairs(task_name=args.task, limit=args.num_pairs, train_ratio=args.train_ratio)
            pairs.extend(raw_pairs)
    elif args.trait:
        from wisent.core.primitives.contrastive_pairs.generators.llm_synthetic import LLMSyntheticGenerator
        generator = LLMSyntheticGenerator(model_id=args.model)
        raw_pairs = generator.generate_pairs(trait=args.trait, num_pairs=args.num_pairs)
        for rp in raw_pairs:
            pairs.append(ContrastivePair(
                prompt=rp.get('prompt', ''),
                positive_response=PositiveResponse(model_response=rp.get('positive_response', '')),
                negative_response=NegativeResponse(model_response=rp.get('negative_response', '')),
                label=args.trait,
            ))
    return pairs

def _print_timing(args, start_time: float):
    """Print timing information if requested."""
    if args.timing:
        print(f"\nTotal time: {time.time() - start_time:.2f}s")

def _print_summary(args, stats: Dict):
    """Print final summary."""
    if args.verbose:
        print("\n" + "=" * _C.SEPARATOR_WIDTH_REPORT)
        print("WEIGHT MODIFICATION COMPLETE")
        print("=" * _C.SEPARATOR_WIDTH_REPORT)
        print(f"Modified model: {args.output_dir}")
        print(f"Method: {args.method}")
        print(f"Layers modified: {stats['layers_modified']}")
        print(f"Parameters modified: {stats['total_parameters_modified']:,}")
        print("=" * _C.SEPARATOR_WIDTH_REPORT + "\n")
