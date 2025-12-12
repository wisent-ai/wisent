"""
Run lm-evaluation-harness HumanEval in Docker.

Usage:
    python run_humaneval.py                                      # defaults: gpt-neo-125M, 20 samples
    python run_humaneval.py --model Qwen/Qwen2.5-0.5B-Instruct   # custom model
    python run_humaneval.py --limit 50 --batch_size 8            # custom params
"""

import argparse
import subprocess
import os
from pathlib import Path


IMAGE_NAME = "lm-eval:humaneval"
SCRIPT_DIR = Path(__file__).parent


def build_image() -> None:
    """Build Docker image if it doesn't exist."""
    result = subprocess.run(
        ["docker", "image", "inspect", IMAGE_NAME],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"Building Docker image: {IMAGE_NAME}")
        subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, str(SCRIPT_DIR)],
            check=True,
        )


def run_humaneval(
    model: str = "EleutherAI/gpt-neo-125M",
    tasks: str = "humaneval",
    device: str = "cuda:0",
    batch_size: int = 4,
    limit: int | None = 20,
    output_dir: Path | None = None,
) -> None:
    """Run lm_eval HumanEval in Docker container."""

    build_image()

    # Setup directories
    output_dir = output_dir or SCRIPT_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hf_cache.mkdir(parents=True, exist_ok=True)

    # Build lm_eval arguments
    lm_eval_args = [
        "--model", "hf",
        "--model_args", f"pretrained={model}",
        "--tasks", tasks,
        "--device", device,
        "--batch_size", str(batch_size),
        "--confirm_run_unsafe_code",
        "--output_path", "/home/sandbox/output",
    ]

    if limit is not None:
        lm_eval_args.extend(["--limit", str(limit)])

    # Docker command
    docker_cmd = [
        "docker", "run", "--rm", "-it",
        "--gpus", "all",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "-v", f"{hf_cache}:/home/sandbox/.cache/huggingface",
        "-v", f"{output_dir}:/home/sandbox/output",
        "-e", "HF_ALLOW_CODE_EVAL=1",
        "-e", "HF_HOME=/home/sandbox/.cache/huggingface",
        IMAGE_NAME,
        *lm_eval_args,
    ]

    print("Running lm_eval in Docker...")
    print(f"  Image: {IMAGE_NAME}")
    print(f"  Model: {model}")
    print(f"  Tasks: {tasks}")
    print(f"  Output: {output_dir}")
    print()

    subprocess.run(docker_cmd, check=True)

    print()
    print(f"Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run HumanEval in Docker")
    parser.add_argument("--model", default="EleutherAI/gpt-neo-125M", help="HuggingFace model name")
    parser.add_argument("--tasks", default="humaneval", help="lm_eval task(s)")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--limit", type=int, default=20, help="Number of samples (None for all)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")

    args = parser.parse_args()

    run_humaneval(
        model=args.model,
        tasks=args.tasks,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
