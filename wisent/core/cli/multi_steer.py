"""Multi-steer command execution logic."""

import sys
import os
import torch

from wisent.core.models.inference_config import get_generate_kwargs


def execute_multi_steer(args):
    """Execute the multi-steer command - combine multiple steering vectors and apply to generation."""
    from wisent.core.multi_steering import MultiSteering, MultiSteeringError
    from wisent.core.models.wisent_model import WisentModel

    try:
        # Check if no vectors provided - unsteered baseline mode
        if not args.vector or len(args.vector) == 0:
            print(f"\n🎯 Unsteered Baseline Generation Mode")
            print(f"   Model: {args.model}")
            print(f"   No steering vectors provided - generating baseline response")

            # If prompt is provided, generate unsteered response
            if hasattr(args, 'prompt') and args.prompt:
                print(f"\n🤖 Loading model '{args.model}'...")
                model = WisentModel(args.model, device=args.device)

                # Get inference config settings with optional CLI overrides
                gen_kwargs = get_generate_kwargs(max_new_tokens=args.max_new_tokens)
                if getattr(args, 'temperature', None) is not None:
                    gen_kwargs["temperature"] = args.temperature
                if getattr(args, 'top_p', None) is not None:
                    gen_kwargs["top_p"] = args.top_p

                # Generate WITHOUT steering
                response = model.generate(
                    [[{"role": "user", "content": args.prompt}]],
                    **gen_kwargs,
                )[0]

                print(f"\nUnsteered baseline output:\n{response}\n")
            else:
                print(f"\n⚠️  No prompt provided. Use --prompt to generate output.")

            print(f"\n✅ Baseline generation completed successfully!\n")
            return

        print(f"\n🎯 Multi-Steering Mode")
        print(f"   Model: {args.model}")
        print(f"   Layer: {args.layer}")
        print(f"   Method: {args.method}")

        # Initialize multi-steering
        multi_steer = MultiSteering(device=args.device, method=args.method)

        # Load and combine vectors
        multi_steer.load_vectors(args.vector)

        # Override layer if specified in args
        if hasattr(args, 'layer') and args.layer:
            multi_steer.layer = int(args.layer)

        # Combine vectors
        normalize = getattr(args, 'normalize_weights', True)
        multi_steer.combine_vectors(normalize=normalize)

        # Save combined vector if requested
        if hasattr(args, 'save_combined') and args.save_combined:
            print(f"\n💾 Saving combined vector to '{args.save_combined}'...")
            os.makedirs(os.path.dirname(args.save_combined) or '.', exist_ok=True)
            torch.save({
                'steering_vector': multi_steer.combined_vector,
                'layer_index': multi_steer.layer,
                'method': args.method,
                'model': args.model,
                'weights': multi_steer.weights,
                'num_vectors': len(multi_steer.loaded_vectors),
                # Legacy keys for backward compatibility
                'vector': multi_steer.combined_vector,
                'layer': multi_steer.layer,
            }, args.save_combined)
            print(f"   ✓ Combined vector saved to: {args.save_combined}")

        # If prompt is provided, apply steering and generate
        if hasattr(args, 'prompt') and args.prompt:
            print(f"\n🤖 Loading model '{args.model}'...")
            model = WisentModel(args.model, device=args.device)

            # Generate with steering - pass CLI overrides or None to use config defaults
            output = multi_steer.apply_steering(
                model=model,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=getattr(args, 'temperature', None),
                top_p=getattr(args, 'top_p', None)
            )

            print(f"\nGenerated output:\n{output}\n")

        print(f"\n✅ Multi-steering completed successfully!\n")

    except MultiSteeringError as e:
        print(f"\n❌ Multi-steering error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
