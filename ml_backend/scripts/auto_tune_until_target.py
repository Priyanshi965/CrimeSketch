"""Run multiple training configs and stop when target gate passes."""

import argparse
import itertools
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.pipeline import run_training_pipeline


def parse_list(value: str, cast):
    return [cast(v.strip()) for v in value.split(",") if v.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-tune training until target metric is reached")
    parser.add_argument("--epochs", default="20,30,40")
    parser.add_argument("--batch-sizes", default="8,16")
    parser.add_argument("--learning-rates", default="1e-4,5e-5")
    parser.add_argument("--loss-types", default="triplet,contrastive")
    parser.add_argument("--target-accuracy", type=float, default=0.94)
    parser.add_argument("--gate-metric", type=str, default="accuracy", choices=["accuracy", "rank1_accuracy"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-reindex", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--triplet-margin", type=float, default=0.5)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-batch-hard-triplet", action="store_true")

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    epochs_list = parse_list(args.epochs, int)
    bs_list = parse_list(args.batch_sizes, int)
    lr_list = parse_list(args.learning_rates, float)
    loss_list = [v.strip() for v in args.loss_types.split(",") if v.strip()]

    configs = list(itertools.product(loss_list, epochs_list, bs_list, lr_list))
    history = []

    for idx, (loss_type, epochs, batch_size, lr) in enumerate(configs, start=1):
        cfg = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "loss_type": loss_type,
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

        print(f"\n[sweep] trial {idx}/{len(configs)} config={cfg}", flush=True)
        report = run_training_pipeline(project_root=project_root, config=cfg)

        summary = {
            "trial": idx,
            "timestamp": datetime.now().isoformat(),
            "config": cfg,
            "status": report.get("status"),
            "gate": report.get("gate"),
            "test": report.get("test"),
            "validation": report.get("validation"),
        }
        history.append(summary)

        with open(os.path.join(project_root, "ml_backend", "models", "sweep_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        passed = bool(report.get("gate", {}).get("passed"))
        metric = report.get("gate", {}).get("metric")
        actual = report.get("gate", {}).get("actual")
        target = report.get("gate", {}).get("target")
        print(f"[sweep] result: {metric}={actual:.4f} target={target:.4f} passed={passed}", flush=True)

        if passed:
            print("[sweep] target reached. stopping early.", flush=True)
            return 0

    print("[sweep] target not reached in configured trials.", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
