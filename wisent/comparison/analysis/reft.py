"""
ReFT (Representation Fine-Tuning) method for comparison experiments.

Trains a LoReFT intervention on benchmark tasks using supervised fine-tuning (SFT)
on positive responses from contrastive pairs.

Re-exports from reft_train and reft_eval submodules.
"""
from wisent.core.utils.config_tools.constants import GRADIENT_ACCUMULATION_STEPS_DEFAULT
from wisent.comparison.reft_train import prepare_reft_dataset, train_reft_adapter
from wisent.comparison.reft_eval import apply_reft_to_model, remove_reft, evaluate_reft

__all__ = ["prepare_reft_dataset", "train_reft_adapter", "apply_reft_to_model", "remove_reft", "evaluate_reft"]


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train and evaluate ReFT intervention on benchmark task")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", required=True, help="lm-eval task name")
    parser.add_argument("--trait-label", required=True, help="Label for the trait being steered")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=None, help="Number of training examples (required)")
    parser.add_argument("--device", required=True, help="Device")
    parser.add_argument("--low-rank-dimension", type=int, default=None, help="LoReFT rank (required)")
    parser.add_argument("--intervention-layers", default=None, help="Comma-separated intervention layers (required)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (required)")
    parser.add_argument("--num-epochs", type=int, default=None, help="Number of epochs (required)")
    parser.add_argument("--batch-size", type=int, required=True, help="Training batch size")
    parser.add_argument("--weight-decay", type=float, required=True, help="Weight decay for training optimizer")
    parser.add_argument("--max-length", type=int, default=None, help="Max sequence length")
    parser.add_argument("--logging-steps", type=int, required=True, help="Logging frequency in training steps")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    parser.add_argument("--train-ratio", type=float, required=True, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", required=True, help="Eval batch size (int or 'auto')")
    parser.add_argument("--eval-max-batch-size", type=int, default=None, help="Max eval batch size for auto (required)")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--with-steering", action="store_true", help="Also evaluate ReFT + steering")
    parser.add_argument("--steering-method", required=True, choices=["caa", "fgaa"], help="Steering method")
    parser.add_argument("--steering-layers", default=None, help="Layers for steering vector (required)")
    parser.add_argument("--steering-num-pairs", type=int, default=None, help="Number of pairs for steering (required)")
    parser.add_argument("--steering-scales", required=True, help="Comma-separated steering scales")
    parser.add_argument("--extraction-strategy", required=True, help="Extraction strategy for steering")
    parser.add_argument("--log-interval", type=int, required=True, help="Progress logging interval for LL evaluation")
    parser.add_argument("--min-norm-threshold", type=float, required=True, help="Minimum norm threshold for control vector health diagnostics")
    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.task}_reft_intervention"
    intervention_layers = [int(l.strip()) for l in args.intervention_layers.split(",")]

    train_reft_adapter(
        task=args.task, model_name=args.model, output_path=output_path,
        trait_label=args.trait_label, device=args.device,
        weight_decay=args.weight_decay,
        num_pairs=args.num_pairs, keep_intermediate=args.keep_intermediate,
        low_rank_dimension=args.low_rank_dimension, intervention_layers=args.intervention_layers,
        learning_rate=args.learning_rate, num_epochs=args.num_epochs,
        batch_size=args.batch_size, logging_steps=args.logging_steps,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_DEFAULT,
        max_length=args.max_length,
    )
    if not args.skip_eval:
        eval_batch_size = args.eval_batch_size
        if eval_batch_size != "auto":
            eval_batch_size = int(eval_batch_size)
        steering_scales = [float(s.strip()) for s in args.steering_scales.split(",")]
        evaluate_reft(
            model_name=args.model, reft_path=output_path, task=args.task,
            extraction_strategy=args.extraction_strategy,
            device=args.device,
            batch_size=eval_batch_size, max_batch_size=args.eval_max_batch_size,
            log_interval=args.log_interval,
            train_ratio=args.train_ratio,
            limit=args.eval_limit, output_dir=args.output_dir,
            num_train_pairs=args.num_pairs, num_epochs=args.num_epochs,
            low_rank_dimension=args.low_rank_dimension, intervention_layers=intervention_layers,
            learning_rate=args.learning_rate,
            with_steering=args.with_steering, steering_method=args.steering_method,
            steering_layers=args.steering_layers, steering_num_pairs=args.steering_num_pairs,
            steering_scales=steering_scales,
            min_norm_threshold=args.min_norm_threshold,
        )


if __name__ == "__main__":
    main()
