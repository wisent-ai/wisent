"""Generate responses command execution logic."""

import json
import os
import sys
import torch

from wisent.core.models.inference_config import get_generate_kwargs


def execute_generate_responses(args):
    """
    Execute the generate-responses command.

    Generates model responses to questions from a task and saves them to a file.
    """
    from wisent.core.models.wisent_model import WisentModel
    from wisent.core.data_loaders.loaders.task_interface_loader import TaskInterfaceDataLoader
    from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
    from wisent.core.steering_methods.steering_object import load_steering_object

    print(f"\n{'='*80}")
    print(f"🎯 GENERATING RESPONSES FROM TASK")
    print(f"{'='*80}")
    print(f"   Task: {args.task}")
    print(f"   Model: {args.model}")
    print(f"   Num questions: {args.num_questions}")
    print(f"   Device: {args.device or 'auto'}")
    if args.steering_object:
        print(f"   Steering object: {args.steering_object}")
        print(f"   Steering strength: {args.steering_strength}")
        steering_strategy = getattr(args, 'steering_strategy', 'constant')
        print(f"   Steering strategy: {steering_strategy}")
    print(f"{'='*80}\n")

    # Load model
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
    input_file = getattr(args, 'input_file', None)
    if input_file and os.path.exists(input_file):
        from wisent.core.contrastive_pairs.core.pair import ContrastivePair
        from wisent.core.contrastive_pairs.core.response import PositiveResponse, NegativeResponse
        
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
        from wisent.core.data_loaders.loaders.huggingface_loader import HuggingFaceDataLoader
        
        load_limit = max(args.num_questions * 2, 20)
        load_kwargs = dict(
            split_ratio=0.8,
            seed=42,
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
                    sys.exit(1)

    # Generate responses
    print(f"🤖 Generating responses...\n")
    results = []

    for idx, pair in enumerate(pairs, 1):
        if args.verbose:
            print(f"Question {idx}/{len(pairs)}:")
            print(f"   Prompt: {pair.prompt[:100]}...")

        try:
            # Convert prompt to chat format
            messages = [
                {"role": "user", "content": pair.prompt}
            ]

            # Get inference config settings with CLI overrides
            gen_kwargs = get_generate_kwargs(max_new_tokens=args.max_new_tokens)
            if args.temperature is not None:
                gen_kwargs["temperature"] = args.temperature
            if args.top_p is not None:
                gen_kwargs["top_p"] = args.top_p

            # Get steering strategy
            steering_strategy = getattr(args, 'steering_strategy', 'constant')
            
            # Generate response with per-token steering strategy
            responses = model.generate(
                inputs=[messages],
                **gen_kwargs,
                use_steering=args.use_steering,
                steering_object=steering_object,
                steering_strength=args.steering_strength,
                steering_strategy=steering_strategy,
            )

            generated_text = responses[0] if responses else ""

            if args.verbose:
                print(f"   Generated: {generated_text[:100]}...")
                print()

            result_entry = {
                "question_id": idx,
                "prompt": pair.prompt,
                "generated_response": generated_text,
                "positive_reference": pair.positive_response.model_response,
                "negative_reference": pair.negative_response.model_response
            }
            if pair.metadata:
                if pair.metadata.get('correct_answers'):
                    result_entry['correct_answers'] = pair.metadata['correct_answers']
                if pair.metadata.get('incorrect_answers'):
                    result_entry['incorrect_answers'] = pair.metadata['incorrect_answers']
            results.append(result_entry)

        except Exception as e:
            print(f"   ❌ Error generating response for question {idx}: {e}")
            results.append({
                "question_id": idx,
                "prompt": pair.prompt,
                "generated_response": None,
                "error": str(e)
            })

    # Save results
    print(f"\n💾 Saving results...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    output_data = {
        "task": args.task,
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
        "responses": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"   ✓ Results saved to: {args.output}\n")

    # Print summary
    print(f"{'='*80}")
    print(f"✅ GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Total questions: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if 'error' not in r)}")
    print(f"   Failed: {sum(1 for r in results if 'error' in r)}")
    print(f"{'='*80}\n")
