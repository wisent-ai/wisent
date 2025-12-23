"""Preview contrastive pairs from benchmarks with different extraction strategies."""

import sys
import json
import argparse
from typing import Optional


def execute_preview_pairs(args):
    """Preview contrastive pairs from a benchmark with different strategies applied."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import (
        lm_build_contrastive_pairs,
    )
    from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_manifest import HF_EXTRACTORS
    from wisent.core.activations.extraction_strategy import (
        ExtractionStrategy,
        build_extraction_texts,
        get_strategy_for_model,
    )
    
    task_name = args.task_name
    limit = args.limit or 5
    strategies = args.strategies or ['chat_last', 'mc_balanced', 'completion_last']
    
    print(f"\n{'='*80}")
    print(f"Preview Contrastive Pairs: {task_name}")
    print(f"{'='*80}")
    
    # Load pairs
    print(f"\nLoading {limit} pairs from '{task_name}'...")
    
    try:
        task_name_lower = task_name.lower()
        is_hf_task = task_name_lower in {k.lower() for k in HF_EXTRACTORS.keys()}
        
        if is_hf_task:
            pairs = lm_build_contrastive_pairs(
                task_name=task_name,
                lm_eval_task=None,
                limit=limit,
            )
        else:
            from wisent.core.data_loaders.loaders.lm_loader import LMEvalDataLoader
            loader = LMEvalDataLoader()
            task_obj = loader.load_lm_eval_task(task_name)
            
            if isinstance(task_obj, dict):
                if len(task_obj) != 1:
                    keys = ", ".join(sorted(task_obj.keys()))
                    print(f"Task '{task_name}' has subtasks: {keys}")
                    print("Please specify a subtask.")
                    sys.exit(1)
                (subname, task), = task_obj.items()
                task_name = subname
            else:
                task = task_obj
            
            pairs = lm_build_contrastive_pairs(
                task_name=task_name,
                lm_eval_task=task,
                limit=limit,
            )
        
        print(f"Loaded {len(pairs)} pairs\n")
        
    except Exception as e:
        print(f"Error loading task: {e}")
        sys.exit(1)
    
    # Mock tokenizer for preview
    class PreviewTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            if len(messages) == 1:
                return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"
            elif len(messages) == 2:
                return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n{messages[1]['content']}<|end|>"
            return str(messages)
        
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}
    
    tokenizer = PreviewTokenizer()
    
    # Show pairs with strategies
    for i, pair in enumerate(pairs):
        print(f"\n{'='*80}")
        print(f"PAIR {i+1}/{len(pairs)}")
        print(f"{'='*80}")
        
        print(f"\n--- RAW DATA (from extractor) ---")
        print(f"Prompt: {pair.prompt[:300]}{'...' if len(pair.prompt) > 300 else ''}")
        print(f"Correct: {pair.positive_response.model_response[:100]}{'...' if len(pair.positive_response.model_response) > 100 else ''}")
        print(f"Incorrect: {pair.negative_response.model_response[:100]}{'...' if len(pair.negative_response.model_response) > 100 else ''}")
        
        for strategy_name in strategies:
            try:
                strategy = ExtractionStrategy(strategy_name)
            except ValueError:
                print(f"\n--- {strategy_name.upper()} --- (invalid strategy)")
                continue
            
            print(f"\n--- {strategy_name.upper()} ---")
            
            try:
                # Build texts for positive response
                if strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION):
                    full_text, answer, prompt_only = build_extraction_texts(
                        strategy,
                        pair.prompt,
                        pair.positive_response.model_response,
                        tokenizer,
                        other_response=pair.negative_response.model_response,
                        is_positive=True,
                        auto_convert_strategy=False,
                    )
                else:
                    full_text, answer, prompt_only = build_extraction_texts(
                        strategy,
                        pair.prompt,
                        pair.positive_response.model_response,
                        tokenizer,
                        auto_convert_strategy=False,
                    )
                
                print(f"Full text (positive):")
                print(f"  {full_text[:400]}{'...' if len(full_text) > 400 else ''}")
                print(f"Answer token: {answer}")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Task: {task_name}")
    print(f"Pairs shown: {len(pairs)}")
    print(f"Strategies: {', '.join(strategies)}")
    print()
    
    # Save to JSON if requested
    if args.output:
        output_data = {
            "task_name": task_name,
            "num_pairs": len(pairs),
            "strategies": strategies,
            "pairs": []
        }
        
        for pair in pairs:
            pair_data = {
                "raw": {
                    "prompt": pair.prompt,
                    "correct": pair.positive_response.model_response,
                    "incorrect": pair.negative_response.model_response,
                },
                "formatted": {}
            }
            
            for strategy_name in strategies:
                try:
                    strategy = ExtractionStrategy(strategy_name)
                    if strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION):
                        full_text, answer, _ = build_extraction_texts(
                            strategy, pair.prompt, pair.positive_response.model_response,
                            tokenizer, other_response=pair.negative_response.model_response,
                            is_positive=True, auto_convert_strategy=False,
                        )
                    else:
                        full_text, answer, _ = build_extraction_texts(
                            strategy, pair.prompt, pair.positive_response.model_response,
                            tokenizer, auto_convert_strategy=False,
                        )
                    pair_data["formatted"][strategy_name] = {
                        "full_text": full_text,
                        "answer": answer,
                    }
                except Exception as e:
                    pair_data["formatted"][strategy_name] = {"error": str(e)}
            
            output_data["pairs"].append(pair_data)
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Preview contrastive pairs with different strategies")
    parser.add_argument("task_name", help="Task/benchmark name (e.g., boolq, mmlu, hellaswag)")
    parser.add_argument("--limit", "-n", type=int, default=5, help="Number of pairs to show (default: 5)")
    parser.add_argument("--strategies", "-s", nargs="+", 
                        default=["chat_last", "mc_balanced", "completion_last"],
                        help="Strategies to preview")
    parser.add_argument("--output", "-o", help="Save to JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    execute_preview_pairs(args)


if __name__ == "__main__":
    main()
