"""
ReFT (Representation Fine-Tuning) method for comparison experiments.

Trains a LoReFT intervention on benchmark tasks using supervised fine-tuning (SFT)
on positive responses from contrastive pairs.

Re-exports from reft_train and reft_eval submodules.
"""
from wisent.comparison.reft_train import prepare_reft_dataset, train_reft_adapter
from wisent.comparison.reft_eval import apply_reft_to_model, remove_reft, evaluate_reft

__all__ = ["prepare_reft_dataset", "train_reft_adapter", "apply_reft_to_model", "remove_reft", "evaluate_reft"]


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train and evaluate ReFT intervention on benchmark task")
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--task", default="boolq", help="lm-eval task name")
    parser.add_argument("--output-dir", default="/home/ubuntu/output", help="Output directory")
    parser.add_argument("--num-pairs", type=int, default=50, help="Number of training examples")
    parser.add_argument("--device", default="cuda:0", help="Device")
    parser.add_argument("--low-rank-dimension", type=int, default=4, help="LoReFT rank (default: 4)")
    parser.add_argument("--intervention-layers", default=None, help="Comma-separated intervention layers (default: auto)")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train/test split ratio")
    parser.add_argument("--eval-batch-size", default="auto", help="Eval batch size (int or 'auto')")
    parser.add_argument("--eval-max-batch-size", type=int, default=64, help="Max eval batch size for auto")
    parser.add_argument("--eval-limit", type=int, default=None, help="Limit eval examples")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument("--with-steering", action="store_true", help="Also evaluate ReFT + steering")
    parser.add_argument("--steering-method", default="caa", choices=["caa", "fgaa"], help="Steering method")
    parser.add_argument("--steering-layers", default="12", help="Layers for steering vector")
    parser.add_argument("--steering-num-pairs", type=int, default=50, help="Number of pairs for steering")
    parser.add_argument("--steering-scales", default="1.0,2.0,4.0", help="Comma-separated steering scales")
    parser.add_argument("--extraction-strategy", default="mc_completion", help="Extraction strategy for steering")
    args = parser.parse_args()

    output_path = Path(args.output_dir) / f"{args.task}_reft_intervention"
    if args.intervention_layers:
        intervention_layers = [int(l.strip()) for l in args.intervention_layers.split(",")]
    else:
        intervention_layers = [get_default_layer(args.model)]

    train_reft_adapter(
        task=args.task, model_name=args.model, output_path=output_path,
        num_pairs=args.num_pairs, device=args.device, keep_intermediate=args.keep_intermediate,
        low_rank_dimension=args.low_rank_dimension, intervention_layers=args.intervention_layers,
        learning_rate=args.learning_rate, num_epochs=args.num_epochs,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    if not args.skip_eval:
        eval_batch_size = args.eval_batch_size
        if eval_batch_size != "auto":
            eval_batch_size = int(eval_batch_size)
        steering_scales = [float(s.strip()) for s in args.steering_scales.split(",")]
        evaluate_reft(
            model_name=args.model, reft_path=output_path, task=args.task,
            train_ratio=args.train_ratio, device=args.device,
            batch_size=eval_batch_size, max_batch_size=args.eval_max_batch_size,
            limit=args.eval_limit, output_dir=args.output_dir,
            num_train_pairs=args.num_pairs, num_epochs=args.num_epochs,
            low_rank_dimension=args.low_rank_dimension, intervention_layers=intervention_layers,
            learning_rate=args.learning_rate,
            with_steering=args.with_steering, steering_method=args.steering_method,
            steering_layers=args.steering_layers, steering_num_pairs=args.steering_num_pairs,
            steering_scales=steering_scales, extraction_strategy=args.extraction_strategy,
        )


if __name__ == "__main__":
    main()
