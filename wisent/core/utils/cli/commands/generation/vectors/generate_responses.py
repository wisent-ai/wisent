"""Generate responses command execution logic."""

import json
import os

from wisent.core.primitives.models import get_generate_kwargs
from wisent.core.utils.config_tools.constants import DEFAULT_RANDOM_SEED, JSON_INDENT, DISPLAY_TRUNCATION_COMPACT
from wisent.core.utils.cli.commands.generation.vectors.generation_helpers import generate_batched, generate_sequential


def execute_generate_responses(args):
    """
    Execute the generate-responses command.

    Generates model responses to questions from a task and saves them to a file.
    """
    from wisent.core.primitives.models.wisent_model import WisentModel
    from wisent.core.utils.infra_tools.data.loaders.task_interface_loader import TaskInterfaceDataLoader
    from wisent.core.utils.infra_tools.data.loaders.lm_eval.lm_loader import LMEvalDataLoader
    from wisent.core.control.steering_methods.steering_object import load_steering_object

    # Validate arguments - need either task or input_file
    input_file = getattr(args, 'input_file', None)
    if not args.task and not input_file:
        raise ValueError("Either --task or --input-file must be provided")

    task_name = args.task or "custom"

    print(f"\n{'='*80}")
    print(f"🎯 GENERATING RESPONSES FROM TASK")
    print(f"{'='*80}")
    print(f"   Task: {task_name}")
    print(f"   Model: {args.model}")
    print(f"   Num questions: {args.num_questions}")
    print(f"   Device: {args.device or 'auto'}")
    if args.steering_object:
        print(f"   Steering object: {args.steering_object}")
        print(f"   Steering strength: {args.steering_strength}")
        steering_strategy = getattr(args, 'steering_strategy', 'constant')
        print(f"   Steering strategy: {steering_strategy}")
    print(f"{'='*80}\n")

    # Load model (or use cached)
    cached_model = getattr(args, 'cached_model', None)
    if cached_model is not None:
        model = cached_model
        print(f"📦 Using cached model")
    else:
        print(f"📦 Loading model...")
        model = WisentModel(args.model, device=args.device)
        print(f"   ✓ Model loaded\n")

    # Load steering object if provided
    steering_object = None
    if args.steering_object:
        print(f"📦 Loading steering object...")
        steering_object = load_steering_object(args.steering_object)
        print(f"   ✓ Loaded {steering_object.metadata.method} steering object\n")

    # Load task data - from file or from task
    print(f"📊 Loading task data...")
    pairs = None

    # If input file is provided, load from file
    if input_file and os.path.exists(input_file):
        from wisent.core.primitives.contrastive_pairs.core.pair import ContrastivePair
        from wisent.core.primitives.contrastive_pairs.core.io.response import PositiveResponse, NegativeResponse
        
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        pairs_list = data.get('pairs', [])
        pairs = []
        for pair_data in pairs_list[:args.num_questions]:
            pair = ContrastivePair(
                prompt=pair_data['prompt'],
                positive_response=PositiveResponse(
                    model_response=pair_data['positive_response']['model_response']
                ),
                negative_response=NegativeResponse(
                    model_response=pair_data['negative_response']['model_response']
                ),
                label=pair_data.get('label', ''),
                trait_description=pair_data.get('trait_description', ''),
                metadata=pair_data.get('metadata', {}),
            )
            pairs.append(pair)
        print(f"   ✓ Loaded {len(pairs)} question pairs from file\n")
    else:
        from wisent.core.utils.infra_tools.data.loaders.huggingface_loader import HuggingFaceDataLoader
        
        load_limit = max(args.num_questions * args.max_incorrect_per_correct, args.min_load_limit_questions)
        load_kwargs = dict(
            split_ratio=0.8,
            seed=DEFAULT_RANDOM_SEED,
            limit=load_limit,
            training_limit=None,
            testing_limit=None
        )
        
        # Try loaders in order: HuggingFace (for custom extractors) -> TaskInterface -> LMEval
        errors = {}
        
        # 1. Try HuggingFaceDataLoader (for truthfulqa_custom, etc.)
        try:
            hf_loader = HuggingFaceDataLoader()
            result = hf_loader.load(task_name=args.task, **load_kwargs)
            # Result is a TypedDict, access via key
            pairs = result['train_qa_pairs'].pairs[:args.num_questions]
            print(f"   ✓ Loaded {len(pairs)} question pairs via HuggingFace extractor\n")
        except Exception as hf_err:
            errors['HuggingFace'] = hf_err
            
            # 2. Try TaskInterfaceDataLoader (for livecodebench, gsm8k, etc.)
            try:
                task_loader = TaskInterfaceDataLoader()
                result = task_loader.load(task=args.task, **load_kwargs)
                pairs = result.train_qa_pairs.pairs[:args.num_questions]
                print(f"   ✓ Loaded {len(pairs)} question pairs via TaskInterface\n")
            except Exception as task_err:
                errors['TaskInterface'] = task_err
                
                # 3. Try LMEvalDataLoader
                try:
                    loader = LMEvalDataLoader()
                    result = loader._load_one_task(task_name=args.task, **load_kwargs)
                    pairs = result['train_qa_pairs'].pairs[:args.num_questions]
                    print(f"   ✓ Loaded {len(pairs)} question pairs via LMEval\n")
                except Exception as lm_err:
                    errors['LMEval'] = lm_err
                    print(f"   ❌ Failed to load task '{args.task}'")
                    for loader_name, err in errors.items():
                        print(f"      {loader_name} error: {err}")
                    raise ValueError(f"Failed to load task '{args.task}': {errors}")

    # Generate responses
    print(f"🤖 Generating responses...\n")
    results = []

    extract_activations = getattr(args, 'extract_activations', False)

    # Shared generation kwargs
    gen_kwargs = get_generate_kwargs(max_new_tokens=args.max_new_tokens)
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    steering_strategy = getattr(args, 'steering_strategy', 'constant')

    # Fall back to sequential if activation extraction is needed (requires per-prompt handling)
    if extract_activations:
        results = generate_sequential(
            pairs, model, gen_kwargs, args, steering_object, steering_strategy, extract_activations,
        )
    else:
        results = generate_batched(
            pairs, model, gen_kwargs, args, steering_object, steering_strategy,
        )

    # Save results
    print(f"\n💾 Saving results...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "task": task_name,
        "model": args.model,
        "num_questions": len(pairs),
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "use_steering": args.use_steering,
            "steering_object": args.steering_object,
            "steering_strength": args.steering_strength if args.steering_object else None,
            "steering_strategy": getattr(args, 'steering_strategy', 'constant'),
            "steering_method": steering_object.metadata.method if steering_object else None
        },
        "activation_params": {
            "extract_activations": getattr(args, 'extract_activations', False),
            "extraction_strategy": getattr(args, 'extraction_strategy', None),
            "layers": getattr(args, 'layers', None),
        } if getattr(args, 'extract_activations', False) else None,
        "responses": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=JSON_INDENT)

    print(f"   ✓ Results saved to: {args.output}\n")

    # Print summary
    print(f"{'='*80}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Total questions: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"   Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"{'='*80}\n")
