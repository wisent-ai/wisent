"""Test contrastive pairs generation for all supported benchmarks.

Generates example pairs for each benchmark and shows how they look
with different extraction strategies.
"""

import json
import signal
import sys
from pathlib import Path


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Timeout")


class MockTokenizer:
    """Mock tokenizer for previewing extraction strategies."""
    
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if len(messages) == 1:
            return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"
        elif len(messages) == 2:
            return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n{messages[1]['content']}<|end|>"
        return str(messages)
    
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def format_pair_with_strategies(pair, tokenizer):
    """Format a contrastive pair with all extraction strategies.
    
    Returns dict with raw data and formatted versions for each strategy.
    """
    from wisent.core.activations.extraction_strategy import (
        ExtractionStrategy,
        build_extraction_texts,
    )
    
    result = {
        "raw": {
            "prompt": pair.prompt,
            "positive": pair.positive_response.model_response,
            "negative": pair.negative_response.model_response,
        },
        "strategies": {}
    }
    
    strategies = [
        "chat_last",
        "chat_mean", 
        "mc_balanced",
        "completion_last",
        "completion_mean",
        "mc_completion",
    ]
    
    for strategy_name in strategies:
        try:
            strategy = ExtractionStrategy(strategy_name)
            
            # Build texts for positive response
            if strategy in (ExtractionStrategy.MC_BALANCED, ExtractionStrategy.MC_COMPLETION):
                pos_full, pos_answer, pos_prompt = build_extraction_texts(
                    strategy,
                    pair.prompt,
                    pair.positive_response.model_response,
                    tokenizer,
                    other_response=pair.negative_response.model_response,
                    is_positive=True,
                    auto_convert_strategy=False,
                )
                neg_full, neg_answer, neg_prompt = build_extraction_texts(
                    strategy,
                    pair.prompt,
                    pair.negative_response.model_response,
                    tokenizer,
                    other_response=pair.positive_response.model_response,
                    is_positive=False,
                    auto_convert_strategy=False,
                )
            else:
                pos_full, pos_answer, pos_prompt = build_extraction_texts(
                    strategy,
                    pair.prompt,
                    pair.positive_response.model_response,
                    tokenizer,
                    auto_convert_strategy=False,
                )
                neg_full, neg_answer, neg_prompt = build_extraction_texts(
                    strategy,
                    pair.prompt,
                    pair.negative_response.model_response,
                    tokenizer,
                    auto_convert_strategy=False,
                )
            
            result["strategies"][strategy_name] = {
                "positive": {
                    "full_text": pos_full,
                    "answer_token": pos_answer,
                    "prompt_only": pos_prompt,
                },
                "negative": {
                    "full_text": neg_full,
                    "answer_token": neg_answer,
                    "prompt_only": neg_prompt,
                }
            }
        except Exception as e:
            result["strategies"][strategy_name] = {"error": str(e)}
    
    return result


def test_all_benchmarks(timeout_per_task: int = 30, limit: int = 2):
    """Test contrastive pairs generation for all supported benchmarks.
    
    Args:
        timeout_per_task: Timeout in seconds per benchmark
        limit: Number of pairs to generate per benchmark
    
    Returns:
        Dictionary with results including example pairs with all strategies
    """
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
    from wisent.core.benchmark_registry import get_all_benchmarks, get_broken_tasks
    
    all_benchmarks = get_all_benchmarks()
    broken = set(get_broken_tasks())
    
    # Filter out broken benchmarks
    benchmarks = [b for b in all_benchmarks if b not in broken]
    
    print(f"Testing {len(benchmarks)} benchmarks (excluded {len(broken)} broken)")
    print(f"Timeout per task: {timeout_per_task}s, limit: {limit} pairs")
    print()
    
    tokenizer = MockTokenizer()
    
    results = {
        "total": len(benchmarks),
        "ok": 0,
        "failed": 0,
        "timeout": 0,
        "benchmarks": {}
    }
    
    for i, benchmark in enumerate(benchmarks):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_per_task)
        
        try:
            pairs = build_contrastive_pairs(benchmark, limit=limit)
            signal.alarm(0)
            
            if pairs and len(pairs) > 0:
                results["ok"] += 1
                
                # Format pairs with all strategies
                formatted_pairs = []
                for pair in pairs:
                    formatted_pairs.append(format_pair_with_strategies(pair, tokenizer))
                
                results["benchmarks"][benchmark] = {
                    "status": "ok",
                    "num_pairs": len(pairs),
                    "pairs": formatted_pairs
                }
                print(f"[{i+1}/{len(benchmarks)}] OK: {benchmark} - {len(pairs)} pairs")
            else:
                results["failed"] += 1
                results["benchmarks"][benchmark] = {"status": "no_pairs", "num_pairs": 0}
                print(f"[{i+1}/{len(benchmarks)}] FAIL: {benchmark} - no pairs returned")
                
        except TimeoutError:
            signal.alarm(0)
            results["timeout"] += 1
            results["benchmarks"][benchmark] = {"status": "timeout"}
            print(f"[{i+1}/{len(benchmarks)}] TIMEOUT: {benchmark}")
            
        except Exception as e:
            signal.alarm(0)
            results["failed"] += 1
            error_msg = str(e)[:200]
            results["benchmarks"][benchmark] = {"status": "error", "error": error_msg}
            print(f"[{i+1}/{len(benchmarks)}] ERROR: {benchmark} - {error_msg[:100]}")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total tested: {results['total']}")
    print(f"OK: {results['ok']}")
    print(f"Failed: {results['failed']}")
    print(f"Timeout: {results['timeout']}")
    print(f"Success rate: {results['ok']/results['total']*100:.1f}%")
    print("=" * 60)
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test contrastive pairs for all supported benchmarks")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="Timeout per task in seconds (default: 30)")
    parser.add_argument("--limit", "-l", type=int, default=2, help="Number of pairs per benchmark (default: 2)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output JSON file for results")
    
    args = parser.parse_args()
    
    results = test_all_benchmarks(timeout_per_task=args.timeout, limit=args.limit)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")
    
    # Exit with error code if any failures
    sys.exit(0 if results["failed"] == 0 and results["timeout"] == 0 else 1)


if __name__ == "__main__":
    main()
