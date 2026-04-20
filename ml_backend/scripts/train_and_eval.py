"""CLI runner for end-to-end training + evaluation + gate check."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.pipeline import run_training_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate sketch-to-face model with quality gate")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--loss-type", type=str, default="triplet", choices=["triplet", "contrastive"])
    parser.add_argument("--target-accuracy", type=float, default=0.94)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-reindex", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--gate-metric", type=str, default="accuracy", choices=["accuracy", "rank1_accuracy"])
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--triplet-margin", type=float, default=0.5)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-batch-hard-triplet", action="store_true")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_type": args.loss_type,
        "target_accuracy": args.target_accuracy,
        "top_k": args.top_k,
        "seed": args.seed,
        "reindex_after_train": not args.no_reindex,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "gate_metric": args.gate_metric,
        "resume_from_checkpoint": not args.no_resume,
        "triplet_margin": args.triplet_margin,
        "use_augmentation": not args.no_augmentation,
        "batch_hard_triplet": not args.no_batch_hard_triplet,
    }

    report = run_training_pipeline(project_root=project_root, config=config)
    print(json.dumps(report, indent=2))

    return 0 if report.get("gate", {}).get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
