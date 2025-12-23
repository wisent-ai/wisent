#!/usr/bin/env python3
"""
Run geometry detection and classifier accuracy across tasks, strategies, and models.
Uses CLI commands instead of reimplementing logic.
"""
import subprocess
import os
import sys
import tempfile

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "Qwen/Qwen3-8B",
    "openai/gpt-oss-20b",
]

TASKS = ["truthfulqa_gen", "livecodebench", "humaneval"]

STRATEGIES = ["chat_last", "chat_mean", "chat_first", "chat_max_norm", "chat_weighted", "role_play", "mc_balanced"]

DEVICE = "mps"
NUM_PAIRS = 200

def run_cmd(cmd, print_output=True, allow_nonzero=False):
    """Run a CLI command and return output. Exits on error unless allow_nonzero."""
    # Replace 'wisent' with python module call to use correct interpreter
    if cmd[0] == "wisent":
        cmd = [sys.executable, "-m", "wisent.core.main"] + cmd[1:]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if print_output and result.stdout:
        print(result.stdout)
    if result.returncode != 0 and not allow_nonzero:
        print(f"  Error (exit code {result.returncode}):")
        print(result.stderr)
        sys.exit(1)
    return result

for model_name in MODELS:
    print(f"\n{'#'*80}")
    print(f"# Model: {model_name}")
    print(f"{'#'*80}")
    
    for task in TASKS:
        print(f"\n  === Task: {task} ===")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs_file = os.path.join(tmpdir, "pairs.json")
            
            # 1. Generate pairs from task
            run_cmd([
                "wisent", "generate-pairs-from-task",
                task,
                "--output", pairs_file,
                "--limit", str(NUM_PAIRS),
            ])
            
            for strategy in STRATEGIES:
                print(f"\n    --- Strategy: {strategy} ---")
                
                linearity_output = os.path.join(tmpdir, f"linearity_{strategy}.json")
                
                # 2. Run check-linearity for geometry detection
                # allow_nonzero=True because exit code 1 means non-linear (not an error)
                run_cmd([
                    "wisent", "check-linearity",
                    pairs_file,
                    "--model", model_name,
                    "--extraction-strategy", strategy,
                    "--output", linearity_output,
                    "--device", DEVICE,
                    "--max-pairs", str(NUM_PAIRS // 2),
                    "--verbose",
                ], allow_nonzero=True)
                
                # 3. Run tasks for classifier training and accuracy testing
                run_cmd([
                    "wisent", "tasks",
                    task,
                    "--model", model_name,
                    "--extraction-strategy", strategy,
                    "--device", DEVICE,
                    "--limit", str(NUM_PAIRS),
                ])

print("\nDone")
