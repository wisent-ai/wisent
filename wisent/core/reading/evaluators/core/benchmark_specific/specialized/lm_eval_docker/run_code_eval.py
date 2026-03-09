"""
Run lm-evaluation-harness code evaluation tasks in Docker (sandboxed).

Supported tasks: any, in particular:
    - humaneval, humaneval_instruct, humaneval_plus, humaneval_plus_instruct
    - mbpp, mbpp_instruct, mbpp_plus, mbpp_plus_instruct

Args:
    --tasks       lm_eval task(s), comma-separated (required)
    --model       HuggingFace model name (required)
    --batch_size  Batch size (default: 4)
    --device      Device to use (required)
    --output_dir  Output directory (default: ./output)
"""

import argparse
import subprocess
import os
from pathlib import Path
from wisent.core.utils.infra_tools.infra.core.hardware import eval_batch_size


IMAGE_NAME = "lm-eval:code-eval"
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


def run_code_eval(
    model: str,
    tasks: str,
    device: str,
    batch_size: int,
    output_dir: Path,
) -> None:
    """Run lm_eval code evaluation tasks in sandboxed Docker container."""
    build_image()

    # Setup directories
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
    parser = argparse.ArgumentParser(description="Run code evaluation tasks in Docker")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--tasks", required=True, help="lm_eval task(s), comma-separated")
    parser.add_argument("--device", required=True, help="Device to use")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-detected if omitted)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory")

    args = parser.parse_args()

    batch_size = args.batch_size if args.batch_size is not None else eval_batch_size()
    output_dir = args.output_dir if args.output_dir is not None else SCRIPT_DIR / "output"

    run_code_eval(
        model=args.model,
        tasks=args.tasks,
        device=args.device,
        batch_size=batch_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
