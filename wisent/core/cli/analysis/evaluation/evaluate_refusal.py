"""Evaluate refusal command execution logic using UncensorBench."""

import json
import os
import re
import sys

import torch
from uncensorbench import UncensorBench
from uncensorbench.evaluator import KeywordEvaluator, SemanticEvaluator

from wisent.core.models.wisent_model import WisentModel
from wisent.core.models.inference_config import get_generate_kwargs


def execute_evaluate_refusal(args):
    """
    Execute the evaluate-refusal command using UncensorBench.

    Evaluates a model's refusal rate on a comprehensive set of prompts across
    15 topic categories using semantic similarity and keyword matching.
    """
    print(f"\n{'='*80}")
    print("EVALUATING MODEL REFUSAL RATE (UncensorBench)")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Evaluator: {args.evaluator}")
    print(f"   Max tokens: {args.max_new_tokens}")
    if args.topics:
        print(f"   Topics: {args.topics}")
    print(f"{'='*80}\n")

    # Initialize UncensorBench
    bench = UncensorBench()

    # Parse topics if provided
    topics = None
    if args.topics:
        topics = [t.strip() for t in args.topics.split(",")]

    # Get prompts
    if args.prompts:
        print(f"Loading prompts from {args.prompts}...")
        try:
            with open(args.prompts) as f:
                custom_prompts = json.load(f)
            if not isinstance(custom_prompts, list):
                custom_prompts = custom_prompts.get("prompts", [])
            print(f"   Loaded {len(custom_prompts)} custom prompts\n")
        except Exception as e:
            print(f"   Failed to load prompts: {e}")
            sys.exit(1)
        prompts_list = None  # Will use custom prompts
    else:
        prompts_list = list(bench.prompts(topics=topics))
        if args.num_prompts and args.num_prompts < len(prompts_list):
            prompts_list = prompts_list[: args.num_prompts]
        print(f"Using {len(prompts_list)} UncensorBench prompts\n")
        custom_prompts = None

    # Initialize evaluator
    print(f"Initializing {args.evaluator} evaluator...")
    if args.evaluator == "keyword":
        evaluator = KeywordEvaluator()
    else:
        try:
            evaluator = SemanticEvaluator()
        except ImportError:
            print("   SemanticEvaluator requires sentence-transformers, falling back to keyword")
            evaluator = KeywordEvaluator()
    print("   Evaluator ready\n")

    # Load model
    print(f"Loading model: {args.model}")
    try:
        model = WisentModel(args.model, device=getattr(args, 'device', None))
        tokenizer = model.tokenizer
        print(f"   Model loaded with {model.num_layers} layers\n")
    except Exception as e:
        print(f"   Failed to load model: {e}")
        sys.exit(1)

    # Determine what to iterate over
    if custom_prompts:
        items = custom_prompts
        total = len(items)
    else:
        items = prompts_list
        total = len(items)

    print(f"Evaluating {total} prompts...\n")

    results = []
    refusal_count = 0
    topic_stats = {}

    for i, item in enumerate(items, 1):
        # Get prompt text
        if custom_prompts:
            prompt_text = item if isinstance(item, str) else item.get("prompt", str(item))
            prompt_obj = None
            topic = "custom"
        else:
            prompt_obj = item
            prompt_text = prompt_obj.prompt
            topic = prompt_obj.topic

        # Format as chat message and generate
        messages = [{"role": "user", "content": prompt_text}]

        responses = model.generate(
            [messages],  # WisentModel expects list of conversations
            **get_generate_kwargs(max_new_tokens=args.max_new_tokens),
        )

        response = responses[0] if responses else ""

        # Strip <think>...</think> blocks from Qwen3 responses
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        response = re.sub(r"<think>.*", "", response, flags=re.DOTALL).strip()

        # Evaluate using UncensorBench
        if prompt_obj:
            eval_result = bench.evaluate_response(prompt_obj, response, evaluator)
            is_refusal = not eval_result.is_uncensored
            compliance_score = eval_result.compliance_score
            refusal_detected = eval_result.refusal_detected
        else:
            # For custom prompts, use basic refusal detection
            is_refusal = bench.detect_refusal(response)
            compliance_score = 0.0 if is_refusal else 1.0
            refusal_detected = is_refusal

        if is_refusal:
            refusal_count += 1

        # Track topic stats
        if topic not in topic_stats:
            topic_stats[topic] = {"total": 0, "refusals": 0, "scores": []}
        topic_stats[topic]["total"] += 1
        topic_stats[topic]["scores"].append(compliance_score)
        if is_refusal:
            topic_stats[topic]["refusals"] += 1

        result = {
            "prompt": prompt_text,
            "response": response[:500],
            "topic": topic,
            "refused": is_refusal,
            "compliance_score": compliance_score,
            "refusal_detected": refusal_detected,
        }
        if prompt_obj:
            result["prompt_id"] = prompt_obj.id
            result["subtopic"] = prompt_obj.subtopic
        results.append(result)

        # Print progress
        status = "REFUSED" if is_refusal else "COMPLIED"
        if args.verbose:
            print(f"[{i}/{total}] {status} (score={compliance_score:.2f})")
            print(f"   Topic: {topic}")
            print(f"   Prompt: {prompt_text[:60]}...")
            print(f"   Response: {response[:100]}...\n")
        else:
            print(f"[{i}/{total}] {status} (score={compliance_score:.2f}): {prompt_text[:50]}...")

    # Calculate metrics
    refusal_rate = refusal_count / total
    compliance_rate = 1 - refusal_rate
    avg_compliance = sum(r["compliance_score"] for r in results) / len(results) if results else 0

    # Print summary
    print(f"\n{'='*80}")
    print("REFUSAL EVALUATION RESULTS (UncensorBench)")
    print(f"{'='*80}")
    print(f"   Model: {args.model}")
    print(f"   Evaluator: {args.evaluator}")
    print(f"   Total prompts: {total}")
    print(f"   Refusals: {refusal_count}")
    print(f"   Compliances: {total - refusal_count}")
    print(f"   Refusal rate: {refusal_rate:.1%}")
    print(f"   Compliance rate: {compliance_rate:.1%}")
    print(f"   Avg compliance score: {avg_compliance:.3f}")
    print(f"{'='*80}")

    # Print topic breakdown
    if len(topic_stats) > 1:
        print("\nBy Topic:")
        print("-" * 40)
        for topic_name in sorted(topic_stats.keys()):
            stats = topic_stats[topic_name]
            topic_refusal_rate = stats["refusals"] / stats["total"] if stats["total"] > 0 else 0
            topic_avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0
            bar = "#" * int((1 - topic_refusal_rate) * 20) + "-" * int(topic_refusal_rate * 20)
            print(f"  {topic_name:25} {bar} {1-topic_refusal_rate:.1%} (avg: {topic_avg_score:.2f})")
        print(f"{'='*80}\n")

    # Save results
    if args.output:
        print(f"Saving results to {args.output}...")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        output_data = {
            "model": args.model,
            "evaluator": args.evaluator,
            "max_new_tokens": args.max_new_tokens,
            "topics": topics,
            "num_prompts": total,
            "refusal_count": refusal_count,
            "compliance_count": total - refusal_count,
            "refusal_rate": refusal_rate,
            "compliance_rate": compliance_rate,
            "average_compliance_score": avg_compliance,
            "by_topic": {
                topic_name: {
                    "total": stats["total"],
                    "refusals": stats["refusals"],
                    "refusal_rate": stats["refusals"] / stats["total"] if stats["total"] > 0 else 0,
                    "avg_compliance": sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else 0,
                }
                for topic_name, stats in topic_stats.items()
            },
            "results": results,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print("   Results saved\n")

    return {
        "refusal_rate": refusal_rate,
        "compliance_rate": compliance_rate,
        "average_compliance_score": avg_compliance,
        "refusal_count": refusal_count,
        "total": total,
        "by_topic": topic_stats,
        "results": results,
    }
