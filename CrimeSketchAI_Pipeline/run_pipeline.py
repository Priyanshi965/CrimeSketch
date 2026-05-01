"""
run_pipeline.py — CrimeSketch AI · IEEE FICV 2026
═══════════════════════════════════════════════════
Main entry point. Runs the FULL evaluation pipeline locally:
  1. Load (or generate) paired sketch-photo dataset
  2. Train models that require training (Custom CNN, PCA+SVM)
  3. Extract embeddings for all models
  4. Compute ALL metrics programmatically (Rank-N, CMC, ROC, PR, latency, etc.)
  5. Save results/metrics.json
  6. Generate interactive HTML dashboard from computed results

Usage:
  # Synthetic mode (auto-generates sketches from photos in data/photos/)
  python run_pipeline.py --synthetic --n_subjects 500

  # Real paired dataset (data/sketches/ + data/photos/ with matching filenames)
  python run_pipeline.py --data_dir ./data

  # Run only specific models
  python run_pipeline.py --synthetic --models crimesketch facenet arcface pca_svm

  # Skip training (use if models already trained)
  python run_pipeline.py --synthetic --skip_training

  # GPU acceleration
  python run_pipeline.py --synthetic --device cuda
"""

import os
import sys
import json
import argparse
import datetime
import traceback

import numpy as np

# ── Ensure project root is on path ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="CrimeSketch AI — Full Local Evaluation Pipeline")
    p.add_argument("--data_dir",      type=str,
                   default=r"C:\Users\User\OneDrive\Desktop\ippr\datasets\organized",
                   help="Root dataset directory (contains dataset1/ and dataset2/)")
    p.add_argument("--synthetic",     action="store_true",
                   help="Force synthetic sketch generation (overrides real dataset)")
    p.add_argument("--n_subjects",    type=int, default=500,
                   help="Max subjects to use (synthetic mode)")
    p.add_argument("--device",        type=str, default=None,
                   help="'cuda' or 'cpu' (auto-detect if omitted)")
    p.add_argument("--models",        nargs="+", default=None,
                   help="Model keys to run (default: all). E.g. crimesketch facenet arcface")
    p.add_argument("--skip_training", action="store_true",
                   help="Skip training step for CNN/SVM models")
    p.add_argument("--output_dir",    type=str, default="results",
                   help="Directory for results JSON and HTML dashboard")
    p.add_argument("--max_rank",      type=int, default=20,
                   help="Maximum CMC rank to compute")
    p.add_argument("--latency_trials",type=int, default=50,
                   help="Number of trials for latency measurement")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--sketch_style",  type=str, default="pencil",
                   choices=["pencil", "edge"],
                   help="Sketch simulation style for synthetic mode")
    p.add_argument("--fast",          action="store_true",
                   help="Fast mode: fewer CNN epochs, fewer latency trials, "
                        "smaller test slice — full pipeline in ~5 minutes on CPU")
    return p.parse_args()


def ablation_study(train_pairs, test_pairs, device, use_faiss=True):
    """
    Run ablation study: evaluate CrimeSketch AI backbone under
    different preprocessing configurations.
    Returns dict of {config_name: rank1_accuracy}.
    """
    from src.models import CrimeSketchAI
    from src.evaluate import rank_n_accuracy

    configs = [
        ("No Preprocessing",       False, False),
        ("Face Alignment Only",    False, True),
        ("CLAHE Only",             True,  False),
        ("CLAHE + Align (Full)",   True,  True),
    ]

    gallery_images = [p["photo_array"] for p in test_pairs]
    gallery_ids    = [p["subject_id"]  for p in test_pairs]
    probe_images   = [p["sketch_array"] for p in test_pairs]
    probe_ids      = [p["subject_id"]   for p in test_pairs]

    results = {}
    print("\n── Ablation Study ──────────────────────────────────────────")
    for name, use_clahe, use_align in configs:
        try:
            model = CrimeSketchAI(device=device, use_clahe=use_clahe,
                                  use_align=use_align)
            g_embs = model.extract_embeddings(gallery_images)
            p_embs = model.extract_embeddings(probe_images)
            acc = rank_n_accuracy(p_embs, g_embs, probe_ids, gallery_ids,
                                  n=1, use_faiss=use_faiss)
            results[name] = round(float(acc), 2)
            print(f"  {name:<35} Rank-1: {acc:.2f}%")
        except Exception as e:
            print(f"  {name:<35} ERROR: {e}")
            results[name] = 0.0
    return results


def main():
    args = parse_args()
    run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 65)
    print("  CrimeSketch AI — Local Training & Evaluation Pipeline")
    print("  IEEE International Conference on Forensic Intelligence")
    print("  and Computer Vision (FICV) 2026")
    print("=" * 65)
    print(f"  Run date : {run_date}")
    print(f"  Data dir : {args.data_dir}")
    print(f"  Mode     : {'Synthetic' if args.synthetic else 'Real dataset'}")
    print(f"  Subjects : {args.n_subjects if args.synthetic else 'all in data_dir'}")
    print(f"  Output   : {args.output_dir}")
    print("=" * 65)

    # ── Device ────────────────────────────────────────────────────────────
    import torch
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # ── Fast mode overrides ────────────────────────────────────────────────
    if args.fast:
        args.latency_trials = 10
        args.max_rank       = 10
        print("\n" + "─"*65)
        print("  ⚡ FAST MODE — optimised for speed on CPU")
        print("  CNN epochs reduced · test slice capped · 10 latency trials")
        print("─"*65)

    print(f"\n[Pipeline] Device: {device}")
    if device == "cuda":
        print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")

    # ── Dataset ───────────────────────────────────────────────────────────
    from src.dataset import load_paired_dataset, split_dataset
    pairs = load_paired_dataset(
        data_dir=args.data_dir,
        synthetic=args.synthetic,
        max_subjects=args.n_subjects if args.synthetic else None,
        sketch_style=args.sketch_style,
        seed=args.seed,
    )

    if len(pairs) == 0:
        print("\n[ERROR] No paired samples found.")
        print(f"  Expected dataset at: {args.data_dir}")
        print("  Structure required:")
        print("    dataset1/photos/   Original_Face_XXXX.jpg")
        print("    dataset1/sketches/ Pencil_Face_XXXX.jpg")
        print("    dataset2/photos/   f-005-01.jpg ...")
        print("    dataset2/sketches/ 00001.jpg ...")
        sys.exit(1)

    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, train_ratio=0.70, val_ratio=0.15, seed=args.seed)

    # In fast mode cap test split so embedding extraction is quick
    if args.fast and len(test_pairs) > 60:
        test_pairs = test_pairs[:60]
        print(f"[Fast] Test split capped to {len(test_pairs)} samples for speed.")

    print(f"\n[Pipeline] Dataset ready:")
    print(f"  Total  : {len(pairs)} subjects")
    print(f"  Train  : {len(train_pairs)}")
    print(f"  Val    : {len(val_pairs)}")
    print(f"  Test   : {len(test_pairs)}")

    # ── Models ────────────────────────────────────────────────────────────
    from src.models import get_all_models
    all_models = get_all_models(device=device, fast=args.fast)

    if args.models:
        # Filter to requested models
        all_models = {k: v for k, v in all_models.items()
                      if k in args.models}
        if not all_models:
            print(f"\n[ERROR] None of the requested models found: {args.models}")
            print(f"  Available: {list(get_all_models().keys())}")
            sys.exit(1)
        print(f"\n[Pipeline] Running selected models: {list(all_models.keys())}")
    else:
        print(f"\n[Pipeline] Running all {len(all_models)} models")

    # ── Benchmark ─────────────────────────────────────────────────────────
    from src.evaluate import run_full_benchmark
    results = run_full_benchmark(
        models=all_models,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        gallery_pairs=None,          # use test photos as gallery
        skip_training=args.skip_training,
        fast=args.fast,
    )

    # ── Ablation Study ────────────────────────────────────────────────────
    print("\n[Pipeline] Running ablation study...")
    try:
        ablation = ablation_study(train_pairs, test_pairs, device)
        results["ablation"] = ablation
    except Exception as e:
        print(f"[Pipeline] Ablation study failed: {e}")
        results["ablation"] = {}

    # ── Run info ──────────────────────────────────────────────────────────
    run_info = {
        "run_date":     run_date,
        "dataset_mode": "synthetic" if args.synthetic else "real",
        "n_total":      len(pairs),
        "n_train":      len(train_pairs),
        "n_val":        len(val_pairs),
        "n_test":       len(test_pairs),
        "device":       device,
        "models_run":   list(results.keys()),
        "seed":         args.seed,
    }
    results["_run_info"] = run_info

    # ── Save JSON ─────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[Pipeline] Results saved → {json_path}")

    # ── Generate Dashboard ────────────────────────────────────────────────
    from src.visualize import generate_dashboard
    from src.plot_figures import generate_all_figures

    # Remove non-model keys before passing to visualizer
    vis_results = {k: v for k, v in results.items()
                   if not k.startswith("_") and k != "ablation"}

    dashboard_path = os.path.join(args.output_dir, "dashboard.html")
    generate_dashboard(
        results=vis_results,
        output_path=dashboard_path,
        run_info=run_info,
    )

    # ── Generate static PNG figures ───────────────────────────────────────
    print("\n[Pipeline] Generating static figure images...")
    figures_dir = generate_all_figures(
        json_path=json_path,
        out_dir=os.path.join(args.output_dir, "figures"),
    )

    # ── Print final summary ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE — All metrics computed locally")
    print("=" * 65)
    print(f"  JSON results : {os.path.abspath(json_path)}")
    print(f"  Dashboard    : {os.path.abspath(dashboard_path)}")
    print(f"  Figures      : {os.path.abspath(os.path.join(args.output_dir, 'figures'))}")
    print("=" * 65)

    # Print final table
    print(f"\n  {'Model':<30} {'Rank-1':>8} {'Rank-5':>8} {'AUC':>8} {'Latency':>10}")
    print(f"  {'-'*64}")
    for key, res in vis_results.items():
        if "error" in res or "rank_accuracies" not in res:
            continue
        r = res["rank_accuracies"]
        tag = " ★" if "★" in res["model_name"] else ""
        print(f"  {res['model_name'] + tag:<30} "
              f"{r.get('rank_1', 0):>7.2f}%  "
              f"{r.get('rank_5', 0):>7.2f}%  "
              f"{res['roc']['auc']:>7.4f}  "
              f"{res['latency']['mean_ms']:>8.2f} ms")

    if results.get("ablation"):
        print(f"\n  Ablation Study (CrimeSketch AI backbone, Rank-1 %):")
        for cfg, acc in results["ablation"].items():
            print(f"    {cfg:<35} {acc:.2f}%")
    print()


if __name__ == "__main__":
    main()
