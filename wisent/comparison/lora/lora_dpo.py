"""
LoRA fine-tuning using DPO (Direct Preference Optimization).

Unlike SFT which trains on positive examples only, DPO trains on
preference pairs (chosen vs rejected) to directly optimize for preferences.

Implementation split into _helpers/lora_dpo_train.py and
_helpers/lora_dpo_eval.py to keep files under 300 lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wisent.core.utils.config_tools.constants import (
    LORA_DEFAULT_DROPOUT,
    DPO_DEFAULT_BETA, DEFAULT_SPLIT_RATIO,
    COMPARISON_DEFAULT_BATCH_SIZE,
    COMPARISON_EVAL_BATCH_SIZE, COMPARISON_NUM_PAIRS, COMPARISON_STEERING_LAYER,
    LORA_DPO_LEARNING_RATE, LORA_DPO_NUM_EPOCHS,
)
from wisent.comparison._helpers.lora_dpo_train import (
    create_dpo_dataset,
    train_lora_dpo,
)
from wisent.comparison._helpers.lora_dpo_eval import (
    evaluate_lora_dpo,
)

__all__ = [
    "create_dpo_dataset",
    "train_lora_dpo",
    "evaluate_lora_dpo",
    "main",
]


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate LoRA adapter using DPO")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=COMPARISON_NUM_PAIRS, help="Number of preference pairs")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--lora-r", type=int, required=True, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=LORA_DEFAULT_DROPOUT, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=LORA_DPO_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=LORA_DPO_NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=COMPARISON_DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Max total sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=None, help="Max prompt length")
    parser.add_argument("--beta", type=float, default=DPO_DEFAULT_BETA, help="DPO beta (controls KL penalty)")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    # Eval args
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIO, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", default="auto", help="Eval batch size")
    parser.add_argument("--eval-max-batch-size", type=int, default=COMPARISON_EVAL_BATCH_SIZE, help="Max eval batch size")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    # DPO-LoRA + Steering args
    parser.add_argument("--with-steering", action="store_true", help="Also evaluate DPO-LoRA + steering")
    parser.add_argument("--steering-method", default="caa", choices=["caa", "fgaa"], help="Steering method")
    parser.add_argument("--steering-layers", default=str(COMPARISON_STEERING_LAYER), help="Layers for steering vector")
    parser.add_argument("--steering-num-pairs", type=int, default=COMPARISON_NUM_PAIRS, help="Number of pairs for steering")
    parser.add_argument("--steering-scales", default="1.0,2.0,4.0", help="Comma-separated steering scales")
    parser.add_argument("--extraction-strategy", default="mc_balanced", help="Extraction strategy for steering")

    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.task}_lora_dpo_adapter"

    # Train
    train_lora_dpo(
        task=args.task,
        model_name=args.model,
        output_path=output_path,
        num_pairs=args.num_pairs,
        device=args.device,
        keep_intermediate=args.keep_intermediate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
    )

    # Evaluate
    if not args.skip_eval:
        eval_batch_size = args.eval_batch_size
        if eval_batch_size != "auto":
            eval_batch_size = int(eval_batch_size)

        # Parse steering scales
        steering_scales = [float(s.strip()) for s in args.steering_scales.split(",")]

        evaluate_lora_dpo(
            model_name=args.model,
            lora_path=output_path,
            task=args.task,
            train_ratio=args.train_ratio,
            device=args.device,
            batch_size=eval_batch_size,
            max_batch_size=args.eval_max_batch_size,
            limit=args.eval_limit,
            output_dir=args.output_dir,
            # Training metadata
            num_train_pairs=args.num_pairs,
            num_epochs=args.num_epochs,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            learning_rate=args.learning_rate,
            beta=args.beta,
            max_length=args.max_length,
            max_prompt_length=args.max_prompt_length,
            # Steering parameters
            with_steering=args.with_steering,
            steering_method=args.steering_method,
            steering_layers=args.steering_layers,
            steering_num_pairs=args.steering_num_pairs,
            steering_scales=steering_scales,
            extraction_strategy=args.extraction_strategy,
        )


if __name__ == "__main__":
    main()
