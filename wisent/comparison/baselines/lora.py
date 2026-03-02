"""LoRA fine-tuning method for comparison experiments.
Trains a LoRA adapter on benchmark tasks using supervised fine-tuning (SFT)
on positive responses from contrastive pairs.
Re-exports from lora_train and lora_eval submodules."""
from wisent.core.utils.config_tools.constants import (
    LORA_DEFAULT_DROPOUT,
    COMPARISON_NUM_PAIRS, DEFAULT_SPLIT_RATIO, COMPARISON_EVAL_BATCH_SIZE,
    COMPARISON_STEERING_LAYER, COMPARISON_LORA_LEARNING_RATE,
    COMPARISON_NUM_EPOCHS_DEFAULT, COMPARISON_TRAINING_BATCH_SIZE,
)
from wisent.comparison.lora_train import prepare_sft_dataset, get_target_modules, train_lora_adapter
from wisent.comparison.lora_eval import apply_lora_to_model, remove_lora, evaluate_lora

__all__ = ["prepare_sft_dataset", "get_target_modules", "train_lora_adapter",
           "apply_lora_to_model", "remove_lora", "evaluate_lora"]

def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Train and evaluate LoRA adapter on benchmark task")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=COMPARISON_NUM_PAIRS, help="Number of training examples")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--lora-r", type=int, required=True, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, required=True, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=LORA_DEFAULT_DROPOUT, help="LoRA dropout")
    parser.add_argument("--learning-rate", type=float, default=COMPARISON_LORA_LEARNING_RATE, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=COMPARISON_NUM_EPOCHS_DEFAULT, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=COMPARISON_TRAINING_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIO, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", default="auto", help="Eval batch size (int or 'auto')")
    parser.add_argument("--eval-max-batch-size", type=int, default=COMPARISON_EVAL_BATCH_SIZE, help="Max eval batch size for auto")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--with-steering", action="store_true", help="Also evaluate LoRA + steering")
    parser.add_argument("--steering-method", default="caa", choices=["caa", "fgaa"], help="Steering method")
    parser.add_argument("--steering-layers", default=str(COMPARISON_STEERING_LAYER), help="Layers for steering vector")
    parser.add_argument("--steering-num-pairs", type=int, default=COMPARISON_NUM_PAIRS, help="Number of pairs for steering")
    parser.add_argument("--steering-scales", default="1.0,2.0,4.0", help="Comma-separated steering scales")
    parser.add_argument("--extraction-strategy", default="mc_balanced", help="Extraction strategy for steering")
    args = parser.parse_args()
    output_path = Path(args.output_dir) / f"{args.task}_lora_adapter"
    train_lora_adapter(
        task=args.task, model_name=args.model, output_path=output_path,
        num_pairs=args.num_pairs, device=args.device, keep_intermediate=args.keep_intermediate,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate, num_epochs=args.num_epochs,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    if not args.skip_eval:
        eval_batch_size = args.eval_batch_size
        if eval_batch_size != "auto":
            eval_batch_size = int(eval_batch_size)
        steering_scales = [float(s.strip()) for s in args.steering_scales.split(",")]
        evaluate_lora(
            model_name=args.model, lora_path=output_path, task=args.task,
            train_ratio=args.train_ratio, device=args.device,
            batch_size=eval_batch_size, max_batch_size=args.eval_max_batch_size,
            limit=args.eval_limit, output_dir=args.output_dir,
            num_train_pairs=args.num_pairs, num_epochs=args.num_epochs,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            learning_rate=args.learning_rate, with_steering=args.with_steering,
            steering_method=args.steering_method, steering_layers=args.steering_layers,
            steering_num_pairs=args.steering_num_pairs, steering_scales=steering_scales,
            extraction_strategy=args.extraction_strategy,
        )

if __name__ == "__main__":
    main()
